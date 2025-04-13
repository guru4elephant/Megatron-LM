#!/usr/bin/env python3

import math
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.custom_layers.layer_norm import RMSNorm

from dsv3_layers import DeepSeekV3Block, RotaryPositionalEmbedding
from dsv3_config import DeepSeekV3Config, get_args


class DeepSeekV3Model(MegatronModule):
    """DeepSeekV3 模型实现"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
        self.config = config
        
        # 获取基础参数
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # 模型组件
        self.embed_tokens = ColumnParallelLinear(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            gather_output=True,
            config=config
        )
        
        # 特殊构建: 混合专家层
        self.layers = nn.ModuleList()
        
        # 获取MoE层的配置信息
        moe_frequency = config.moe_frequency if hasattr(config, "moe_frequency") else 0
        moe_expert_count = config.num_moe_experts if hasattr(config, "num_moe_experts") else 0
        use_moe = moe_frequency > 0 and moe_expert_count > 0
        
        # 构建主干网络层
        for i in range(config.num_layers):
            # 检查当前层是否应该是MoE层
            is_moe_layer = use_moe and (i + 1) % moe_frequency == 0
            
            # 构建当前层
            self.layers.append(
                DeepSeekV3Block(
                    config=config,
                    layer_number=i,
                    use_moe=is_moe_layer
                )
            )
            
        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        
        # 输出层
        self.lm_head = RowParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            input_is_parallel=True,
            config=config
        )
        
        # 权重绑定（可选）
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight
            
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 使用截断正态分布初始化
            std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.embed_tokens
        
    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.embed_tokens = value
            
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """模型前向传播"""
        
        # 默认参数处理
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # 确保至少有一种输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定input_ids和inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("必须提供input_ids或inputs_embeds")
        
        # 生成位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # 过去的键值对处理
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)  # 过去序列的长度
        
        # 准备注意力掩码
        if attention_mask is not None:
            # 扩展注意力掩码 [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # 掩码转换: 0 -> -10000, 1 -> 0
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # 初始化变量
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_presents = () if use_cache else None
        
        # 遍历所有层
        for i, layer in enumerate(self.layers):
            # 输出隐藏状态（如果需要）
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # 获取当前层的过去键值对（如果有）
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[i]
                
            # 处理当前层
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            
            # 解析层输出
            hidden_states = layer_outputs[0]
            
            # 缓存当前键值对（如果需要）
            if use_cache:
                all_presents += (layer_outputs[1],)
        
        # 最终层归一化
        hidden_states = self.norm(hidden_states)
        
        # 最终隐藏状态（如果需要）
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        # 计算语言模型头部输出
        lm_logits = self.lm_head(hidden_states)
        
        # 构造返回值
        if not return_dict:
            outputs = (lm_logits,)
            if use_cache:
                outputs += (all_presents,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            return outputs
            
        # 返回字典格式
        return {
            "logits": lm_logits,
            "past_key_values": all_presents if use_cache else None,
            "hidden_states": all_hidden_states if output_hidden_states else None,
        }


class DeepSeekV3ForCausalLM(MegatronModule):
    """用于因果语言建模的DeepSeekV3模型"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
        self.model = DeepSeekV3Model(config)
        self.config = config
        
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None,
        **kwargs
    ):
        """为生成准备输入"""
        
        # 计算输入长度
        input_shape = input_ids.shape
        
        # 处理position_ids
        position_ids = None
        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            
        # 如果有过去键值对，只使用最后一个token
        if past_key_values is not None:
            # 检查是否需要扩展注意力掩码
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], 
                    dim=-1
                )
                
            # 此时只需要最后一个token
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # 如果没有提供position_ids，则计算新的
            if position_ids is None:
                past_length = past_key_values[0][0].size(2)
                position_ids = torch.LongTensor([[past_length]], device=input_ids.device)
        else:
            # 无过去键值对，使用序列id
            if position_ids is None:
                position_ids = torch.arange(
                    0, input_shape[-1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
                
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """前向传播函数"""
        
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 提取logits
        logits = outputs["logits"] if return_dict else outputs[0]
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 将logits移动到CPU计算损失以节省内存
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        # 返回值
        if not return_dict:
            outputs = (logits,) + outputs[1:] if isinstance(outputs, tuple) else (logits,) + tuple(outputs.values())[1:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
            
        # 返回字典格式
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values", None),
            "hidden_states": outputs.get("hidden_states", None),
        }
        
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.get_input_embeddings()
        
    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.model.set_input_embeddings(value)


def load_pretrained_model(model_path: str, device: str = "cuda") -> DeepSeekV3ForCausalLM:
    """加载预训练模型
    
    Args:
        model_path: 模型权重路径
        device: 设备，默认为cuda
        
    Returns:
        加载好权重的模型
    """
    # 参数解析
    args = get_args()
    args.model_path = model_path
    
    # 配置设置
    config = DeepSeekV3Config()
    transformer_config = config.create_transformer_config()
    
    # 创建模型
    model = DeepSeekV3ForCausalLM(transformer_config)
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model


if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    
    # 创建配置
    config = DeepSeekV3Config()
    transformer_config = config.create_transformer_config()
    
    # 创建模型
    model = DeepSeekV3ForCausalLM(transformer_config)
    print(f"创建DeepSeekV3模型，参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    
    # 模型简单测试
    batch_size = 2
    seq_length = 16
    
    # 创建随机输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device="cuda")
    attention_mask = torch.ones((batch_size, seq_length), device="cuda")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"输出logits形状: {outputs['logits'].shape}")
    print("模型测试完成!") 