# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
#                                             MiniMind 配置类
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# 导入transformers库中的预训练配置基类
from transformers import PretrainedConfig


# MiniMind模型的配置类，继承自PretrainedConfig
# 这个类用于存储模型的所有超参数（hyperparameters）
class MiniMindConfig(PretrainedConfig):
    # 定义模型类型为"minimind"，用于在transformers库中识别这个模型
    model_type = "minimind"

    # 初始化函数，定义模型的所有配置参数
    def __init__(
            self,
            dropout: float = 0.0,  # dropout比例，用于防止过拟合，0.0表示不使用dropout
            bos_token_id: int = 1,  # 句子开始标记(Beginning Of Sentence)的ID
            eos_token_id: int = 2,  # 句子结束标记(End Of Sentence)的ID
            hidden_act: str = 'silu',  # 激活函数类型，silu是一种平滑的激活函数
            hidden_size: int = 512,  # 隐藏层的维度大小，也就是每个词向量的长度
            intermediate_size: int = None,  # 前馈网络中间层的大小，None表示自动计算
            max_position_embeddings: int = 32768,  # 最大位置编码数，即模型能处理的最大序列长度
            num_attention_heads: int = 8,  # 注意力头的数量，用于多头注意力机制
            num_hidden_layers: int = 8,  # Transformer层的数量，即模型的深度
            num_key_value_heads: int = 2,  # KV缓存的头数，用于加速推理
            vocab_size: int = 6400,  # 词汇表大小，即模型能识别的不同词的数量
            rms_norm_eps: float = 1e-05,  # RMSNorm归一化的epsilon值，防止除零错误
            rope_theta: int = 1000000.0,  # RoPE位置编码的theta参数，控制位置编码的频率
            flash_attn: bool = True,  # 是否使用Flash Attention加速计算
            ####################################################
            # 以下是混合专家模型(MoE)的特定配置
            # 当use_moe为False时，以下参数无效
            ####################################################
            use_moe: bool = False,  # 是否使用混合专家模型
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量
            n_routed_experts: int = 4,  # 可路由的专家总数
            n_shared_experts: int = 1,  # 共享专家的数量（所有token都会使用）
            scoring_func: str = 'softmax',  # 专家选择的评分函数
            aux_loss_alpha: float = 0.1,  # 辅助损失的权重系数
            seq_aux: bool = True,  # 是否在序列级别计算辅助损失
            norm_topk_prob: bool = True,  # 是否对top-k概率进行归一化
            **kwargs  # 其他额外参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        
        # 将所有参数保存为类的属性，方便后续使用
        self.dropout = dropout  # 保存dropout比例
        self.bos_token_id = bos_token_id  # 保存开始标记ID
        self.eos_token_id = eos_token_id  # 保存结束标记ID
        self.hidden_act = hidden_act  # 保存激活函数类型
        self.hidden_size = hidden_size  # 保存隐藏层维度
        self.intermediate_size = intermediate_size  # 保存中间层大小
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置编码数
        self.num_attention_heads = num_attention_heads  # 保存注意力头数量
        self.num_hidden_layers = num_hidden_layers  # 保存层数
        self.num_key_value_heads = num_key_value_heads  # 保存KV头数量
        self.vocab_size = vocab_size  # 保存词汇表大小
        self.rms_norm_eps = rms_norm_eps  # 保存归一化epsilon
        self.rope_theta = rope_theta  # 保存RoPE参数
        self.flash_attn = flash_attn  # 保存是否使用Flash Attention
        ####################################################
        # 保存混合专家模型的配置参数
        ####################################################
        self.use_moe = use_moe  # 是否使用MoE
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家数量
        self.scoring_func = scoring_func  # 评分函数类型
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失权重
        self.seq_aux = seq_aux  # 是否使用序列级辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否归一化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
#                                             MiniMind 模型实现
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# 导入必要的库
import math  # 数学运算库
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch的神经网络模块
from transformers.activations import ACT2FN  # transformers库中的激活函数字典
from typing import Optional, Tuple, List, Union  # 类型提示
import torch.nn.functional as F  # PyTorch的函数式API
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig  # transformers的基类
from transformers.modeling_outputs import CausalLMOutputWithPast  # 模型输出格式


# RMSNorm归一化层
# 这是一种比LayerNorm更简单高效的归一化方法，常用于大语言模型
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        初始化RMSNorm层
        参数:
            dim: 输入特征的维度
            eps: 防止除零的小常数
        """
        super().__init__()  # 调用父类初始化
        self.eps = eps  # 保存epsilon值
        # 创建可学习的权重参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        执行RMS归一化的核心计算
        RMS = Root Mean Square（均方根）
        公式: x / sqrt(mean(x^2) + eps)
        """
        # x.pow(2): 对x的每个元素平方
        # .mean(-1, keepdim=True): 在最后一个维度上求平均，保持维度
        # + self.eps: 加上epsilon防止除零
        # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
        # x * ...: 将原始值乘以归一化因子
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入张量
        返回:
            归一化并缩放后的张量
        """
        # 先将x转为float类型进行归一化，然后乘以可学习的权重
        # type_as(x)将结果转回x的原始数据类型
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算RoPE(Rotary Position Embedding)旋转位置编码的频率
    RoPE是一种将位置信息编码到注意力机制中的方法
    
    参数:
        dim: 注意力头的维度
        end: 最大序列长度
        theta: 基础频率参数，控制位置编码的周期
    返回:
        freqs_cos: 预计算的余弦值
        freqs_sin: 预计算的正弦值
    """
    # 计算频率序列
    # torch.arange(0, dim, 2): 生成[0, 2, 4, ..., dim-2]
    # [: (dim // 2)]: 取前dim//2个元素
    # / dim: 归一化到[0, 1)范围
    # theta ** (...): 计算theta的幂次
    # 1.0 / (...): 取倒数得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成位置索引序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # torch.outer计算外积，生成位置和频率的所有组合
    # 结果形状: [end, dim//2]
    freqs = torch.outer(t, freqs).float()
    
    # 计算余弦值并拼接两次（用于后续的旋转操作）
    # torch.cat在最后一个维度拼接，形状变为: [end, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    
    # 计算正弦值并拼接两次
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    # 返回预计算的余弦和正弦值
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到查询(Q)和键(K)向量
    这是RoPE的核心操作，通过旋转向量来编码位置信息
    
    参数:
        q: 查询向量
        k: 键向量
        cos: 预计算的余弦值
        sin: 预计算的正弦值
        position_ids: 位置ID（可选）
        unsqueeze_dim: 需要扩展维度的位置
    返回:
        q_embed: 应用位置编码后的查询向量
        k_embed: 应用位置编码后的键向量
    """
    def rotate_half(x):
        """
        将向量的后半部分移到前面，前半部分移到后面，并对前半部分取负
        这是实现旋转的关键步骤
        例如: [a, b, c, d] -> [-c, -d, a, b]
        """
        # x.shape[-1] // 2: 获取向量长度的一半
        # x[..., x.shape[-1] // 2:]: 取后半部分
        # x[..., : x.shape[-1] // 2]: 取前半部分
        # -x[..., x.shape[-1] // 2:]: 对后半部分取负
        # torch.cat拼接: 将取负的后半部分和前半部分拼接
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 对查询向量应用旋转位置编码
    # 公式: q_embed = q * cos + rotate_half(q) * sin
    # unsqueeze(unsqueeze_dim)在指定维度增加一个维度以便广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    
    # 对键向量应用相同的旋转位置编码
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    # 返回编码后的查询和键向量
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值(KV)张量以匹配查询(Q)的头数
    这用于实现分组查询注意力(Grouped Query Attention, GQA)
    GQA可以减少KV缓存的内存占用，提高推理效率
    
    参数:
        x: 输入张量，形状为 [batch_size, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数，即每个KV头需要服务多少个Q头
    返回:
        重复后的张量，形状为 [batch_size, seq_len, num_kv_heads * n_rep, head_dim]
    
    等价于: torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    # 获取输入张量的各个维度
    bs, slen, num_key_value_heads, head_dim = x.shape
    
    # 如果不需要重复（n_rep=1），直接返回原张量
    if n_rep == 1:
        return x
    
    # 重复KV张量的步骤:
    # 1. x[:, :, :, None, :]: 在第4个维度插入一个新维度
    #    形状变为: [bs, slen, num_kv_heads, 1, head_dim]
    # 2. .expand(...): 在新维度上扩展n_rep次（不复制数据，只是改变视图）
    #    形状变为: [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. .reshape(...): 将num_kv_heads和n_rep两个维度合并
    #    最终形状: [bs, slen, num_kv_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制(Multi-Head Attention)
    这是Transformer的核心组件，用于让模型关注输入序列的不同位置
    """
    def __init__(self, args: MiniMindConfig):
        """
        初始化注意力层
        参数:
            args: 模型配置对象
        """
        super().__init__()
        
        # 确定KV头的数量，如果未指定则使用与Q头相同的数量
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        
        # 确保Q头数量能被KV头数量整除（用于分组查询注意力）
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        # 保存注意力头的数量
        self.n_local_heads = args.num_attention_heads  # Q头的数量
        self.n_local_kv_heads = self.num_key_value_heads  # KV头的数量
        
        # 计算每个KV头需要服务多少个Q头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        # 计算每个注意力头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # 定义Q、K、V的线性投影层（不使用偏置以减少参数）
        # Q投影: 将hidden_size维度映射到 num_heads * head_dim
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # K投影: 使用较少的KV头以节省内存
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # V投影: 与K使用相同数量的头
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # 输出投影层: 将多头注意力的输出映射回hidden_size维度
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # Dropout层用于正则化，防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重的dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的dropout
        self.dropout = args.dropout  # 保存dropout比例
        
        # 检查是否可以使用Flash Attention（PyTorch 2.0+的优化实现）
        # Flash Attention可以显著加速注意力计算并减少内存使用
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 如果不支持Flash Attention，可以打印警告（已注释）
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收预计算的cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # KV缓存
                use_cache=False,  # 是否使用缓存
                attention_mask: Optional[torch.Tensor] = None):  # 注意力掩码
        """
        注意力层的前向传播
        参数:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]
            position_embeddings: 位置编码的(cos, sin)元组
            past_key_value: 之前缓存的(K, V)，用于加速生成
            use_cache: 是否返回当前的KV缓存
            attention_mask: 注意力掩码，用于屏蔽某些位置
        返回:
            output: 注意力输出
            past_kv: 当前的KV缓存（如果use_cache=True）
        """
        # 获取输入的形状: batch_size, 序列长度, 隐藏维度
        bsz, seq_len, _ = x.shape
        
        # 通过线性层计算Q、K、V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 重塑Q、K、V的形状以分离出多个注意力头
        # 从 [bsz, seq_len, total_dim] 变为 [bsz, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码(RoPE)到Q和K
        cos, sin = position_embeddings  # 解包位置编码
        # 只使用当前序列长度对应的位置编码
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # KV缓存实现（用于加速自回归生成）
        # 如果有之前的缓存，将新的K、V拼接到缓存后面
        if past_key_value is not None:
            # 在序列维度(dim=1)上拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 如果需要缓存，保存当前的K、V；否则返回None
        past_kv = (xk, xv) if use_cache else None

        # 调整张量形状以进行注意力计算
        # transpose(1, 2): 将seq_len和num_heads维度交换
        # 从 [bsz, seq_len, num_heads, head_dim] 变为 [bsz, num_heads, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),  # Q: [bsz, n_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # K: 重复KV头以匹配Q头数量
            repeat_kv(xv, self.n_rep).transpose(1, 2)   # V: 同上
        )

        # 根据条件选择使用Flash Attention还是标准注意力
        if self.flash and seq_len != 1:
            # 使用Flash Attention（更快更省内存）
            # 训练时使用dropout，推理时不使用
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            
            # 如果提供了attention_mask，需要调整其形状
            if attention_mask is not None:
                # 扩展mask的维度以匹配注意力矩阵的形状
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            # 调用PyTorch的优化注意力函数
            # is_causal=True表示使用因果掩码（下三角矩阵），用于自回归生成
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 使用标准的注意力计算（手动实现）
            # 1. 计算注意力分数: Q @ K^T / sqrt(head_dim)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 2. 添加因果掩码（上三角设为负无穷，softmax后变为0）
            # torch.triu创建上三角矩阵，diagonal=1表示主对角线上方
            # 这确保每个位置只能看到它之前的位置（自回归特性）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # 增加batch和head维度

            # 3. 如果有额外的attention_mask，也添加进去
            if attention_mask is not None:
                # 扩展mask维度
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 将0/1 mask转换为加法mask（0变为0，1变为-1e9）
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # 4. 应用softmax得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            # 5. 应用dropout
            scores = self.attn_dropout(scores)
            
            # 6. 用注意力权重加权V: attention_weights @ V
            output = scores @ xv

        # 调整输出形状
        # transpose(1, 2): [bsz, n_heads, seq_len, head_dim] -> [bsz, seq_len, n_heads, head_dim]
        # reshape: 合并所有头的输出 -> [bsz, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        
        # 通过输出投影层和dropout
        output = self.resid_dropout(self.o_proj(output))
        
        # 返回输出和KV缓存
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络(Feed-Forward Network, FFN)
    这是Transformer中的另一个核心组件，用于对每个位置独立地进行非线性变换
    使用SwiGLU激活函数（Swish-Gated Linear Unit）
    """
    def __init__(self, config: MiniMindConfig):
        """
        初始化前馈网络
        参数:
            config: 模型配置对象
        """
        super().__init__()
        
        # 如果未指定中间层大小，自动计算
        if config.intermediate_size is None:
            # 通常设置为hidden_size的8/3倍（约2.67倍）
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到64的倍数（有利于硬件加速）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 定义三个线性投影层（SwiGLU需要三个投影）
        # gate_proj: 门控投影，用于激活函数
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down_proj: 下投影，将中间维度映射回hidden_size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # up_proj: 上投影，与gate_proj配合实现门控机制
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(config.dropout)
        
        # 激活函数（从配置中获取，通常是'silu'）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前馈网络的前向传播
        实现SwiGLU: FFN(x) = (Swish(gate_proj(x)) * up_proj(x)) @ down_proj
        
        参数:
            x: 输入张量
        返回:
            经过前馈网络处理后的张量
        """
        # SwiGLU的计算步骤:
        # 1. gate_proj(x): 通过门控投影
        # 2. act_fn(...): 应用激活函数（如SiLU/Swish）
        # 3. up_proj(x): 通过上投影
        # 4. 两者逐元素相乘（门控机制）
        # 5. down_proj(...): 投影回原始维度
        # 6. dropout(...): 应用dropout正则化
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    混合专家模型的门控网络(Mixture of Experts Gate)
    负责为每个token选择最合适的专家进行处理
    """
    def __init__(self, config: MiniMindConfig):
        """
        初始化MoE门控网络
        参数:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        
        # 每个token选择的专家数量（top-k）
        self.top_k = config.num_experts_per_tok
        # 可路由的专家总数
        self.n_routed_experts = config.n_routed_experts

        # 评分函数类型（用于计算专家选择概率）
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数（用于平衡专家负载）
        self.alpha = config.aux_loss_alpha
        # 是否在序列级别计算辅助损失
        self.seq_aux = config.seq_aux

        # 是否归一化top-k概率
        self.norm_topk_prob = config.norm_topk_prob
        # 门控网络的输入维度
        self.gating_dim = config.hidden_size
        
        # 门控权重矩阵: [num_experts, hidden_size]
        # 用于计算每个token对每个专家的亲和度分数
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        使用Kaiming均匀分布初始化权重
        这种初始化方法有助于训练稳定性
        """
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        门控网络的前向传播
        为每个token选择top-k个专家及其权重
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
        返回:
            topk_idx: 选中的专家索引 [batch_size * seq_len, top_k]
            topk_weight: 对应的专家权重 [batch_size * seq_len, top_k]
            aux_loss: 辅助损失（用于平衡专家负载）
        """
        # 获取输入形状
        bsz, seq_len, h = hidden_states.shape
        
        # 将输入展平为2D: [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个token对每个专家的logits（未归一化的分数）
        # logits形状: [batch_size * seq_len, num_experts]
        logits = F.linear(hidden_states, self.weight, None)
        
        # 根据评分函数计算概率分布
        if self.scoring_func == 'softmax':
            # 使用softmax将logits转换为概率
            scores = logits.softmax(dim=-1)
        else:
            # 如果使用了不支持的评分函数，抛出错误
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k个专家
        # topk_weight: 最高的k个概率值
        # topk_idx: 对应的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果需要，对top-k权重进行归一化
        # 这确保选中的专家权重和为1
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算权重和（加上小常数防止除零）
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 归一化
            topk_weight = topk_weight / denominator

        # 计算辅助损失（仅在训练时）
        # 辅助损失用于鼓励专家负载均衡，防止某些专家被过度使用
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 重塑索引以便计算损失
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级辅助损失
                # 重塑分数: [batch_size, seq_len, num_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                # 计算每个专家被选中的频率
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_累加每个专家被选中的次数
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                
                # 辅助损失 = 专家使用频率 * 专家平均分数
                # 这鼓励高分数的专家被更多使用，同时平衡负载
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token级辅助损失
                # 创建one-hot编码
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 计算每个专家被选中的比例
                ce = mask_ce.float().mean(0)
                # 计算每个专家的平均分数
                Pi = scores_for_aux.mean(0)
                # 专家负载因子
                fi = ce * self.n_routed_experts
                # 辅助损失: 鼓励负载均衡
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 推理时不计算辅助损失
            aux_loss = 0
        
        # 返回选中的专家索引、权重和辅助损失
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络(Mixture of Experts Feed-Forward Network)
    使用多个专家网络和门控机制，每个token由选定的专家处理
    这可以增加模型容量而不成比例地增加计算量
    """
    def __init__(self, config: MiniMindConfig):
        """
        初始化MoE前馈网络
        参数:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        
        # 创建多个专家网络（每个都是一个FeedForward网络）
        # 这些专家会根据输入动态选择
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)  # 创建n_routed_experts个专家
        ])
        
        # 门控网络，用于选择专家
        self.gate = MoEGate(config)
        
        # 如果配置了共享专家，创建它们
        # 共享专家会处理所有token，不经过门控选择
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        MoE前馈网络的前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
        返回:
            处理后的输出张量
        """
        # 保存输入的副本（用于共享专家和残差连接）
        identity = x
        # 保存原始形状
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        # topk_idx: 每个token选中的专家索引
        # topk_weight: 对应的权重
        # aux_loss: 辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 将输入展平为2D: [batch_size * seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # 将专家索引也展平
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式：使用简单但内存效率较低的方法
            # 将每个token重复num_experts_per_tok次（因为每个token会被多个专家处理）
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            
            # 创建输出张量
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 遍历所有专家，让每个专家处理分配给它的token
            for i, expert in enumerate(self.experts):
                # 找出分配给专家i的所有token
                # 让专家i处理这些token
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            
            # 重塑输出并应用专家权重
            # 将输出reshape为 [batch_size * seq_len, num_experts_per_tok, hidden_size]
            # 乘以权重后在专家维度上求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            
            # 恢复原始形状
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用更高效的方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果有共享专家，将它们的输出加到结果上
        # 共享专家处理所有token，不经过门控选择
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # 保存辅助损失（用于训练时的损失计算）
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE推理的优化实现
        通过批量处理每个专家的所有token来提高效率
        
        参数:
            x: 输入张量 [num_tokens, hidden_size]
            flat_expert_indices: 专家索引 [num_tokens * num_experts_per_tok]
            flat_expert_weights: 专家权重 [num_tokens * num_experts_per_tok, 1]
        返回:
            处理后的输出张量
        """
        # 创建输出缓存，初始化为零
        expert_cache = torch.zeros_like(x)
        
        # 对专家索引排序，这样相同专家的token会聚在一起
        idxs = flat_expert_indices.argsort()
        
        # 计算每个专家处理的token数量的累积和
        # 例如: [6, 15, 20, 26] 表示专家0处理6个token，专家1处理9个token等
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 计算原始token索引
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 示例说明:
        # 当tokens_per_expert = [6, 15, 20, 26]时，有4个专家
        # token_idxs = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...]
        # token_idxs[:6] -> [3, 7, 19, 21, 24, 25] 是专家0处理的token位置
        # token_idxs[6:15] -> [4, 5, 6, 10, 11, 12...] 是专家1处理的token位置
        # 依此类推...
        
        # 遍历每个专家
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家处理的token范围
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            
            # 如果这个专家没有分配到token，跳过
            if start_idx == end_idx:
                continue
            
            # 获取当前专家
            expert = self.experts[i]
            
            # 获取分配给这个专家的token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            
            # 提取这些token
            expert_tokens = x[exp_token_idx]
            
            # 让专家处理这些token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # 乘以对应的权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将结果累加到输出缓存的对应位置
            # scatter_add_实现了累加操作（因为一个token可能被多个专家处理）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        # 返回累加后的结果
        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind的Transformer块
    每个块包含一个自注意力层和一个前馈网络层
    使用Pre-Norm结构（在子层之前进行归一化）
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        初始化Transformer块
        参数:
            layer_id: 当前层的ID（从0开始）
            config: 模型配置对象
        """
        super().__init__()
        # 保存配置参数
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 自注意力层
        self.self_attn = Attention(config)

        # 层ID
        self.layer_id = layer_id
        
        # 注意力层之前的归一化层（Pre-Norm）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络之前的归一化层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络层：根据配置选择普通FFN或MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Transformer块的前向传播
        使用Pre-Norm + 残差连接的结构
        
        参数:
            hidden_states: 输入隐藏状态
            position_embeddings: 位置编码
            past_key_value: KV缓存
            use_cache: 是否使用缓存
            attention_mask: 注意力掩码
        返回:
            hidden_states: 输出隐藏状态
            present_key_value: 当前的KV缓存
        """
        # 保存输入用于残差连接
        residual = hidden_states
        
        # 自注意力子层
        # 1. 先进行归一化（Pre-Norm）
        # 2. 通过自注意力层
        # 3. 返回输出和KV缓存
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        
        # 残差连接：将注意力输出与输入相加
        hidden_states += residual
        
        # 前馈网络子层
        # 1. 先进行归一化（Pre-Norm）
        # 2. 通过前馈网络
        # 3. 残差连接：直接在这里相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        # 返回输出和KV缓存
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind的主模型类
    包含词嵌入层、多个Transformer块和最终的归一化层
    """
    def __init__(self, config: MiniMindConfig):
        """
        初始化MiniMind模型
        参数:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        
        # 保存词汇表大小和层数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 词嵌入层：将token ID映射为向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(config.dropout)
        
        # 创建多个Transformer块（模型的主体）
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算RoPE位置编码的cos和sin值
        # 这些值在整个训练/推理过程中保持不变，所以预先计算可以提高效率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,  # 每个注意力头的维度
            end=config.max_position_embeddings,  # 最大序列长度
            theta=config.rope_theta  # RoPE的theta参数
        )
        
        # 将预计算的值注册为buffer（不是参数，但会随模型保存/加载）
        # persistent=False表示不保存到state_dict中（可以重新计算）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,  # 输入的token ID
                attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # KV缓存列表
                use_cache: bool = False,  # 是否使用缓存
                **kwargs):  # 其他参数
        """
        模型的前向传播
        参数:
            input_ids: 输入token ID，形状 [batch_size, seq_length]
            attention_mask: 注意力掩码
            past_key_values: 之前的KV缓存
            use_cache: 是否返回KV缓存
        返回:
            hidden_states: 最终的隐藏状态
            presents: 当前的KV缓存列表
            aux_loss: MoE的辅助损失
        """
        # 获取输入形状
        batch_size, seq_length = input_ids.shape
        
        # 如果没有提供KV缓存，为每一层创建None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算起始位置（用于KV缓存）
        # 如果有缓存，起始位置是缓存的长度；否则是0
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入：将token ID转换为向量，然后应用dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列对应的位置编码
        # 从start_pos开始，取seq_length长度的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 存储每一层的KV缓存
        presents = []
        
        # 依次通过每个Transformer块
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 通过当前层
            hidden_states, present = layer(
                hidden_states,  # 输入隐藏状态
                position_embeddings,  # 位置编码
                past_key_value=past_key_value,  # 该层的KV缓存
                use_cache=use_cache,  # 是否使用缓存
                attention_mask=attention_mask  # 注意力掩码
            )
            # 保存当前层的KV缓存
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 如果使用了MoE，收集所有MoE层的辅助损失
        # 辅助损失用于平衡专家负载
        aux_loss = sum(
            layer.mlp.aux_loss  # 获取每个MoE层的辅助损失
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)  # 只处理MoE层
        )

        # 返回最终隐藏状态、KV缓存列表和辅助损失
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    用于因果语言建模的MiniMind模型
    这是完整的语言模型，包含MiniMindModel和语言模型头
    继承自PreTrainedModel以兼容transformers库
    继承自GenerationMixin以支持文本生成功能
    """
    # 指定配置类
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        """
        初始化因果语言模型
        参数:
            config: 模型配置对象，如果为None则使用默认配置
        """
        # 如果没有提供配置，使用默认配置
        self.config = config or MiniMindConfig()
        # 调用父类初始化
        super().__init__(self.config)
        
        # 创建MiniMind主模型
        self.model = MiniMindModel(self.config)
        
        # 语言模型头：将隐藏状态映射到词汇表大小的logits
        # 用于预测下一个token
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享：让词嵌入层和输出层共享权重
        # 这是一种常见的技巧，可以减少参数量并提高性能
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # 创建输出对象（用于返回结果）
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,  # 输入token ID
                attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # KV缓存
                use_cache: bool = False,  # 是否使用缓存
                logits_to_keep: Union[int, torch.Tensor] = 0,  # 保留多少个位置的logits
                **args):  # 其他参数
        """
        模型的前向传播
        参数:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            past_key_values: KV缓存
            use_cache: 是否返回KV缓存
            logits_to_keep: 保留多少个位置的logits（用于节省内存）
                           如果为0，保留所有位置
                           如果为正整数n，只保留最后n个位置
        返回:
            CausalLMOutputWithPast对象，包含logits、hidden_states、past_key_values等
        """
        # 通过主模型获取隐藏状态
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 确定要保留哪些位置的logits
        # 如果logits_to_keep是整数，创建一个切片对象
        # 例如：logits_to_keep=1 -> slice(-1, None) -> 只保留最后一个位置
        # 如果logits_to_keep是张量，直接使用它作为索引
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # 通过语言模型头计算logits
        # 只对选定的位置计算logits以节省计算和内存
        logits = self.lm_head(h[:, slice_indices, :])
        
        # 将结果存入输出对象
        self.OUT.__setitem__('last_hidden_state', h)  # 最后的隐藏状态
        self.OUT.__setitem__('logits', logits)  # 预测的logits
        self.OUT.__setitem__('aux_loss', aux_loss)  # MoE的辅助损失
        self.OUT.__setitem__('past_key_values', past_kvs)  # KV缓存
        
        # 返回输出对象
        return self.OUT
