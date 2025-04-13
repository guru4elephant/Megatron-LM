#!/usr/bin/env python3

import math
from typing import Optional, Callable, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.router import Top2Router
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.custom_layers.layer_norm import RMSNorm

class DeepSeekV3SelfAttention(MegatronModule):
    """DeepSeekV3自注意力模块"""
    
    def __init__(self, config: TransformerConfig, layer_number: int = 0):
        super().__init__(config)
        
        self.config = config
        self.layer_number = layer_number
        
        # 获取各种参数
        hidden_size = config.hidden_size
        attention_heads = config.num_attention_heads
        kv_channels = config.kv_channels
        if kv_channels is None:
            kv_channels = hidden_size // attention_heads
        
        # 投影权重
        self.projection_size = kv_channels * attention_heads
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.projection_size,
            gather_output=False,
            bias=config.add_bias_linear,
            config=config,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.projection_size,
            gather_output=False,
            bias=config.add_bias_linear,
            config=config,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.projection_size,
            gather_output=False,
            bias=config.add_bias_linear,
            config=config,
        )
        self.o_proj = RowParallelLinear(
            self.projection_size,
            hidden_size,
            input_is_parallel=True,
            bias=config.add_bias_linear,
            config=config,
        )
        
        # 其他参数
        self.num_attention_heads = attention_heads
        self.hidden_size_per_attention_head = kv_channels
        self.hidden_size = hidden_size
        
        # 旋转位置编码
        self.rotary_dim = int(kv_channels * config.rotary_percent)
        self.rotary_emb = RotaryPositionalEmbedding(self.rotary_dim)
        
        # 是否使用attention mask
        self.use_flash_attention = getattr(config, "use_flash_attention", True)
        
        # 设置低精度注意力计算
        precision = getattr(config, "params_dtype", torch.float32)
        self.attn_dropout = config.attention_dropout if hasattr(config, "attention_dropout") else 0.0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """前向传播"""
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 计算查询、键、值向量
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重组维度，分离头和维度
        head_dim = self.hidden_size_per_attention_head
        query = query.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
        
        # 应用旋转位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=query.device).unsqueeze(0)
        
        query, key = self.rotary_emb(query, key, position_ids)
        
        # 处理过去的键值对用于解码
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # 如果需要缓存，则构建当前键值对
        if use_cache:
            present = (key, value)
        else:
            present = None
            
        # 应用FlashAttention或普通注意力
        if self.use_flash_attention and attention_mask is None:
            # 使用Flash Attention
            attention_output = self._flash_attention(query, key, value, attention_mask)
        else:
            # 传统注意力计算
            attention_output = self._traditional_attention(query, key, value, attention_mask)
        
        # 投影回原始维度
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.projection_size)
        output = self.o_proj(attention_output)
        
        outputs = (output, present)
        return outputs
        
    def _flash_attention(self, query, key, value, attention_mask):
        """使用Flash Attention进行高效注意力计算"""
        
        batch_size, num_heads, seq_length, head_dim = query.shape
        
        # 导入Flash Attention (runtime dependency)
        try:
            from flash_attn import flash_attn_func
            from flash_attn.flash_attention import FlashAttention
            
            # 转换为需要的形状
            q, k, v = [x.reshape(batch_size * num_heads, seq_length, head_dim) for x in [query, key, value]]
            
            # 调用Flash Attention
            context = flash_attn_func(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, causal=True)
            
            # 恢复形状
            context = context.reshape(batch_size, num_heads, seq_length, head_dim)
            return context
            
        except ImportError:
            print("Flash Attention not available, falling back to traditional attention")
            return self._traditional_attention(query, key, value, attention_mask)
            
    def _traditional_attention(self, query, key, value, attention_mask):
        """传统注意力计算"""
        
        batch_size, num_heads, seq_length, head_dim = query.shape
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 归一化得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(query)
        
        # 应用dropout（如果有）
        if self.attn_dropout > 0.0 and self.training:
            attention_probs = F.dropout(attention_probs, p=self.attn_dropout)
        
        # 计算context向量
        context = torch.matmul(attention_probs, value)
        return context


class DeepSeekV3MoEMLP(MegatronModule):
    """带有激活函数和混合专家(MoE)的前馈网络"""
    
    def __init__(self, config: TransformerConfig, layer_number: int = 0):
        super().__init__(config)
        
        self.config = config
        self.layer_number = layer_number
        
        # 获取参数
        hidden_size = config.hidden_size
        ffn_hidden_size = config.ffn_hidden_size
        
        # 定义路由器
        num_experts = config.num_moe_experts
        self.router = Top2Router(
            hidden_size,  # 输入维度
            num_experts,  # 专家数量
            k=config.moe_top_k if hasattr(config, "moe_top_k") else 2,  # 选择的专家数量
            capacity_factor=1.0,  # 专家容量因子
            eval_capacity_factor=1.0,  # 评估时的专家容量因子
            min_capacity=4,  # 最小容量
            noisy_gate_policy=None,  # 噪声门策略
            config=config,
        )
        
        # 定义MoE层
        self.experts = []
        for i in range(num_experts):
            self.experts.append(
                MLP(
                    config=config,
                    layer_number=layer_number,
                    moe_expert_index=i,  # 指定专家索引
                )
            )
        
        self.moe_layer = MoELayer(
            self.router,
            self.experts,
            hidden_size,  # 输出维度
            num_experts,  # 专家数量
            config=config
        )
    
    def forward(self, hidden_states):
        """前向传播"""
        
        # 应用MoE层
        output, _, _ = self.moe_layer(hidden_states)
        return output


class DeepSeekV3Block(MegatronModule):
    """DeepSeekV3基本模块，包含自注意力和前馈网络"""
    
    def __init__(
        self, 
        config: TransformerConfig, 
        layer_number: int = 0, 
        use_moe: bool = False
    ):
        super().__init__(config)
        
        self.config = config
        self.layer_number = layer_number
        self.use_moe = use_moe
        
        # 获取参数
        hidden_size = config.hidden_size
        
        # 定义层
        self.ln_1 = RMSNorm(hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = DeepSeekV3SelfAttention(config, layer_number)
        self.ln_2 = RMSNorm(hidden_size, eps=config.layernorm_epsilon)
        
        # 根据是否使用MoE选择不同的前馈网络
        if use_moe:
            self.mlp = DeepSeekV3MoEMLP(config, layer_number)
        else:
            self.mlp = MLP(config=config, layer_number=layer_number)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """前向传播"""
        
        # 应用预层归一化
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # 自注意力
        attention_output, present = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        
        # 残差连接
        hidden_states = residual + attention_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        
        # 残差连接
        hidden_states = residual + feed_forward_output
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present,)
            
        return outputs


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码实现"""
    
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
    
    def _build_cache(self, seq_len: int, device, dtype):
        """为给定序列长度构建缓存"""
        
        # 生成缓存和索引
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # 计算旋转角度的余弦和正弦值
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin
    
    def _apply_rotary_embedding(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """应用旋转位置编码"""
        
        # 将输入分为两半处理
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 : self.dim]
        
        # 应用旋转
        rotate_x1 = cos * x1 - sin * x2
        rotate_x2 = sin * x1 + cos * x2
        
        # 合并并保持其他维度不变
        rotated_x = torch.cat((rotate_x1, rotate_x2), dim=-1)
        
        # 复制还是保持不变
        if x.shape[-1] > self.dim:
            x = torch.cat([rotated_x, x[..., self.dim:]], dim=-1)
        else:
            x = rotated_x
            
        return x
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """前向传播"""
        
        # 获取序列长度
        batch_size, num_heads, seq_length, _ = q.shape
        
        # 构建缓存
        max_len = position_ids.max() + 1
        cos, sin = self._build_cache(max_len, q.device, q.dtype)
        
        # 根据位置ID获取对应的cos/sin
        position_ids = position_ids.view(-1)  # 展开位置索引
        cos = cos[position_ids].view(batch_size, seq_length, 1, -1)  # 获取对应位置的cos
        sin = sin[position_ids].view(batch_size, seq_length, 1, -1)  # 获取对应位置的sin
        
        # 将cos/sin扩展到与头数匹配
        cos = cos.permute(0, 2, 1, 3).contiguous()  # [batch, 1, seq, dim] -> [batch, 1, seq, dim]
        sin = sin.permute(0, 2, 1, 3).contiguous()  # [batch, 1, seq, dim] -> [batch, 1, seq, dim]
        
        # 确保维度匹配 (如果需要)
        if cos.shape[1] == 1:
            cos = cos.expand(-1, num_heads, -1, -1)
            sin = sin.expand(-1, num_heads, -1, -1)
        
        # 应用旋转位置编码
        q_embed = self._apply_rotary_embedding(q, cos, sin)
        k_embed = self._apply_rotary_embedding(k, cos, sin)
        
        return q_embed, k_embed 