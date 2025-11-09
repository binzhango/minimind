"""
LoRA (Low-Rank Adaptation) 实现
LoRA是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁边添加低秩分解矩阵来实现微调

核心思想：
对于预训练权重矩阵 W ∈ R^(d×k)，LoRA添加一个低秩更新：
W' = W + ΔW = W + B·A
其中：A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k)

优势：
1. 参数效率：只需训练很少的参数（r << d,k）
2. 无推理延迟：训练后可以将ΔW合并到W中
3. 模块化：可以为不同任务保存不同的LoRA权重
4. 保持原模型：原始预训练权重保持冻结
"""

import torch
from torch import optim, nn


# ========================================
# LoRA核心模块
# ========================================
class LoRA(nn.Module):
    """
    LoRA模块：实现低秩分解的适配器
    
    将权重更新分解为两个小矩阵的乘积：ΔW = B @ A
    这样可以大幅减少需要训练的参数数量
    
    例如：如果原始矩阵是 1024×1024 = 1,048,576 个参数
         使用rank=8的LoRA：(1024×8) + (8×1024) = 16,384 个参数
         参数量减少到原来的 1.56%！
    """
    def __init__(self, in_features, out_features, rank):
        """
        初始化LoRA模块
        
        参数:
            in_features: 输入特征维度（对应原始权重矩阵的输入维度）
            out_features: 输出特征维度（对应原始权重矩阵的输出维度）
            rank: LoRA的秩，控制低秩矩阵的中间维度
                  rank越小，参数越少，但表达能力越弱
                  rank越大，参数越多，但表达能力越强
                  通常选择 4, 8, 16, 32 等值
        """
        super().__init__()
        
        # 保存秩参数
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        
        # 定义低秩矩阵A：将输入从in_features维度降到rank维度
        # 形状: [rank, in_features]（因为是Linear层的权重）
        # 不使用bias以减少参数
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A（下投影）
        
        # 定义低秩矩阵B：将rank维度升回到out_features维度
        # 形状: [out_features, rank]
        # 不使用bias以减少参数
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B（上投影）
        
        # ========================================
        # 权重初始化策略（非常重要！）
        # ========================================
        
        # 矩阵A使用高斯分布初始化
        # 均值为0，标准差为0.02
        # 这样可以让初始的ΔW有一定的随机性，有助于训练
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        
        # 矩阵B初始化为全0
        # 这是LoRA的关键设计！
        # 原因：在训练开始时，ΔW = B @ A = 0 @ A = 0
        # 这意味着初始时LoRA不会改变原模型的输出
        # 模型从预训练权重的状态开始，然后逐渐学习任务特定的调整
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        LoRA的前向传播
        
        计算过程：x -> A -> B -> output
        即：output = B(A(x)) = (B @ A) @ x
        
        参数:
            x: 输入张量，形状 [..., in_features]
        
        返回:
            输出张量，形状 [..., out_features]
            这个输出会被加到原始层的输出上
        """
        # 先通过矩阵A降维：[..., in_features] -> [..., rank]
        # 再通过矩阵B升维：[..., rank] -> [..., out_features]
        # 等价于：x @ A^T @ B^T = x @ (B @ A)^T = x @ ΔW^T
        return self.B(self.A(x))


# ========================================
# LoRA应用函数
# ========================================
def apply_lora(model, rank=8):
    """
    将LoRA模块应用到模型的所有方阵线性层
    
    这个函数会遍历模型的所有模块，找到符合条件的线性层，
    然后为每个线性层添加一个LoRA适配器
    
    参数:
        model: 要应用LoRA的PyTorch模型
        rank: LoRA的秩，默认为8
              较小的rank（如4）：参数更少，训练更快，但表达能力较弱
              较大的rank（如16,32）：参数更多，但能学习更复杂的调整
    
    工作原理:
        原始层输出: y = W @ x
        添加LoRA后: y = W @ x + ΔW @ x = W @ x + (B @ A) @ x
        其中W保持冻结，只训练A和B
    """
    # 遍历模型的所有命名模块
    # name: 模块的名称（如 "model.layers.0.self_attn.q_proj"）
    # module: 模块对象本身
    for name, module in model.named_modules():
        # 检查模块是否满足应用LoRA的条件：
        # 1. isinstance(module, nn.Linear): 必须是线性层
        # 2. module.weight.shape[0] == module.weight.shape[1]: 必须是方阵
        #    方阵意味着输入和输出维度相同，这在Transformer中很常见
        #    例如：注意力机制中的Q、K、V投影，前馈网络的某些层
        #    注意：这个条件可以根据需要修改，比如也可以应用到非方阵层
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA模块
            # 输入维度 = 输出维度 = module.weight.shape[0]（因为是方阵）
            # 将LoRA模块移到与原模型相同的设备（CPU或GPU）
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            
            # 将LoRA模块作为属性添加到原始模块上
            # 这样可以通过 module.lora 访问LoRA模块
            setattr(module, "lora", lora)
            
            # 保存原始的forward方法
            # 这是原始线性层的前向传播函数
            original_forward = module.forward

            # ========================================
            # 创建新的forward方法（显式绑定）
            # ========================================
            # 这里使用了Python闭包的技巧
            # layer1=original_forward, layer2=lora 是默认参数
            # 这样做是为了"捕获"当前的original_forward和lora对象
            # 避免闭包中的变量引用问题（所有循环迭代共享同一个变量）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                """
                新的前向传播函数，结合原始层和LoRA层
                
                计算过程:
                1. layer1(x): 通过原始权重W计算 W @ x
                2. layer2(x): 通过LoRA计算 (B @ A) @ x
                3. 两者相加: W @ x + (B @ A) @ x
                
                参数:
                    x: 输入张量
                    layer1: 原始层的forward方法（通过默认参数绑定）
                    layer2: LoRA模块（通过默认参数绑定）
                
                返回:
                    原始输出 + LoRA输出
                """
                return layer1(x) + layer2(x)

            # 替换模块的forward方法
            # 现在调用module.forward(x)会执行 original_forward(x) + lora(x)
            module.forward = forward_with_lora
    
    # 注意：此函数没有返回值，它直接修改了传入的model对象
    # 应用LoRA后，模型的行为已经改变：
    # - 原始权重（W）保持冻结
    # - 只有LoRA参数（A和B）会被训练
    # - 前向传播时会自动加上LoRA的贡献


# ========================================
# LoRA加载函数
# ========================================
def load_lora(model, path):
    """
    从文件加载LoRA权重到模型
    
    这个函数只加载LoRA的权重（A和B矩阵），不影响原始模型权重
    这使得可以为同一个基础模型加载不同的LoRA权重，实现不同的任务
    
    参数:
        model: 已经应用了LoRA的模型（必须先调用apply_lora）
        path: LoRA权重文件的路径（.pth文件）
    
    使用场景:
        1. 加载之前训练好的LoRA权重继续训练
        2. 加载不同任务的LoRA权重进行推理
        3. 在多个LoRA之间快速切换
    
    示例:
        # 基础模型 + 医疗LoRA
        load_lora(model, "lora_medical.pth")
        # 切换到法律LoRA
        load_lora(model, "lora_legal.pth")
    """
    # 从文件加载state_dict
    # map_location=model.device 确保权重加载到正确的设备（CPU或GPU）
    # state_dict是一个字典，键是参数名，值是参数张量
    # 例如: {"model.layers.0.self_attn.q_proj.lora.A.weight": tensor(...), ...}
    state_dict = torch.load(path, map_location=model.device)
    
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 检查模块是否有lora属性
        # 只有通过apply_lora添加了LoRA的模块才有这个属性
        if hasattr(module, 'lora'):
            # ========================================
            # 提取当前模块的LoRA权重
            # ========================================
            # 使用字典推导式过滤出属于当前模块的LoRA参数
            # 
            # 例如，如果 name = "model.layers.0.self_attn.q_proj"
            # 我们要找的键是 "model.layers.0.self_attn.q_proj.lora.A.weight" 等
            # 
            # 步骤：
            # 1. 遍历state_dict中的所有键值对
            # 2. if f'{name}.lora.' in k: 检查键是否属于当前模块的LoRA
            # 3. k.replace(f'{name}.lora.', ''): 移除前缀，只保留相对名称
            #    例如: "model.layers.0.self_attn.q_proj.lora.A.weight" 
            #          -> "A.weight"
            # 4. 构建新的字典，键是相对名称，值是对应的张量
            lora_state = {
                k.replace(f'{name}.lora.', ''): v  # 移除模块名前缀
                for k, v in state_dict.items()  # 遍历所有参数
                if f'{name}.lora.' in k  # 只选择属于当前模块LoRA的参数
            }
            
            # 将提取的权重加载到LoRA模块
            # load_state_dict会将lora_state中的权重复制到module.lora的对应参数中
            # 例如: lora_state["A.weight"] -> module.lora.A.weight
            #       lora_state["B.weight"] -> module.lora.B.weight
            module.lora.load_state_dict(lora_state)
    
    # 注意：此函数只加载LoRA权重，原始模型权重保持不变
    # 加载后，模型可以立即使用新的LoRA权重进行推理或继续训练


# ========================================
# LoRA保存函数
# ========================================
def save_lora(model, path):
    """
    将模型的LoRA权重保存到文件
    
    这个函数只保存LoRA的权重（A和B矩阵），不保存原始模型权重
    这样可以大幅减小保存文件的大小
    
    参数:
        model: 已经应用了LoRA的模型
        path: 保存LoRA权重的文件路径（.pth文件）
    
    优势:
        1. 文件小：只保存LoRA参数，通常只有几MB到几十MB
           相比之下，完整模型可能有几百MB到几GB
        2. 模块化：可以为不同任务保存不同的LoRA权重
        3. 共享友好：可以轻松分享LoRA权重而不泄露完整模型
    
    示例:
        # 训练医疗领域的LoRA
        train_on_medical_data(model)
        save_lora(model, "lora_medical_512.pth")
        
        # 训练法律领域的LoRA
        train_on_legal_data(model)
        save_lora(model, "lora_legal_512.pth")
        
        # 两个文件都很小，但可以让基础模型适应不同领域
    """
    # 创建空字典用于存储所有LoRA权重
    state_dict = {}
    
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 检查模块是否有lora属性
        # 只有通过apply_lora添加了LoRA的模块才有这个属性
        if hasattr(module, 'lora'):
            # ========================================
            # 提取当前模块的LoRA权重并添加完整路径
            # ========================================
            # 
            # module.lora.state_dict() 返回LoRA模块的参数字典
            # 例如: {"A.weight": tensor(...), "B.weight": tensor(...)}
            # 
            # 我们需要给每个参数添加完整的模块路径前缀
            # 这样在加载时才能知道每个参数属于哪个模块
            # 
            # 例如，如果 name = "model.layers.0.self_attn.q_proj"
            # 转换过程:
            # "A.weight" -> "model.layers.0.self_attn.q_proj.lora.A.weight"
            # "B.weight" -> "model.layers.0.self_attn.q_proj.lora.B.weight"
            lora_state = {
                f'{name}.lora.{k}': v  # 添加完整路径前缀
                for k, v in module.lora.state_dict().items()  # 遍历LoRA的所有参数
            }
            
            # 将当前模块的LoRA权重添加到总的state_dict中
            # update方法会将lora_state中的所有键值对添加到state_dict
            state_dict.update(lora_state)
    
    # 将收集到的所有LoRA权重保存到文件
    # torch.save会将state_dict序列化并写入文件
    # 保存的文件可以用torch.load加载
    torch.save(state_dict, path)
    
    # 保存完成后的state_dict结构示例:
    # {
    #     "model.layers.0.self_attn.q_proj.lora.A.weight": tensor(...),
    #     "model.layers.0.self_attn.q_proj.lora.B.weight": tensor(...),
    #     "model.layers.0.self_attn.k_proj.lora.A.weight": tensor(...),
    #     "model.layers.0.self_attn.k_proj.lora.B.weight": tensor(...),
    #     ...
    # }
    
    # 注意：
    # 1. 只保存LoRA权重，原始模型权重不保存
    # 2. 文件大小取决于LoRA的rank和应用的层数
    # 3. 加载时需要先有相同结构的基础模型，然后应用LoRA，最后加载权重


# ========================================
# 使用示例和最佳实践
# ========================================
"""
完整的LoRA使用流程:

1. 准备基础模型
   model = load_pretrained_model()
   
2. 冻结原始参数
   for param in model.parameters():
       param.requires_grad = False
   
3. 应用LoRA
   apply_lora(model, rank=8)
   
4. 只训练LoRA参数
   optimizer = optim.AdamW([p for n, p in model.named_parameters() if 'lora' in n])
   
5. 训练
   for epoch in range(num_epochs):
       train_one_epoch(model, optimizer, train_loader)
   
6. 保存LoRA权重
   save_lora(model, "lora_weights.pth")
   
7. 后续使用
   # 加载基础模型
   model = load_pretrained_model()
   # 应用LoRA结构
   apply_lora(model, rank=8)
   # 加载训练好的LoRA权重
   load_lora(model, "lora_weights.pth")
   # 推理
   output = model(input)

参数选择建议:
- rank=4: 极少参数，适合简单任务或资源受限场景
- rank=8: 平衡选择，适合大多数任务
- rank=16: 更强表达能力，适合复杂任务
- rank=32+: 接近全参数微调的效果，但参数量仍然小得多

LoRA的优势总结:
1. 参数效率: 只训练1-2%的参数
2. 内存效率: 训练时只需存储LoRA梯度
3. 存储效率: 保存的文件很小
4. 任务切换: 可以快速切换不同任务的LoRA
5. 无推理延迟: 可以将ΔW合并到W中
6. 保护隐私: 可以分享LoRA而不泄露完整模型
"""
