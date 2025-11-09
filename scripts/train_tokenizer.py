"""
Tokenizer训练脚本
=================

什么是Tokenizer（分词器）？
--------------------------
Tokenizer是将自然语言文本转换为数字序列的工具，是大语言模型的"词典"。
就像我们查字典时，每个汉字都有一个页码，Tokenizer为每个词/字符分配一个唯一的ID。

例如：
"你好世界" -> [1234, 5678, 9012, 3456]
模型只能处理数字，所以需要Tokenizer进行转换。

为什么需要训练Tokenizer？
-------------------------
1. 压缩效率：好的Tokenizer可以用更少的token表示相同的文本
   - 差的分词：["你", "好", "世", "界"] -> 4个token
   - 好的分词：["你好", "世界"] -> 2个token
   
2. 词汇覆盖：确保常用词都在词表中，减少未知词(UNK)
   
3. 语言适配：不同语言需要不同的分词策略
   - 英文：按空格和标点分词
   - 中文：需要更细粒度的分词

4. 模型大小控制：词表大小直接影响模型参数量
   - 词表6400 vs 词表150000，embedding层参数相差23倍！

BPE算法简介
-----------
BPE (Byte Pair Encoding) 是一种子词分词算法：
1. 初始：每个字节是一个token
2. 迭代：找出最频繁出现的token对，合并为新token
3. 重复：直到达到目标词表大小

例如：
初始: ["l", "o", "w", "e", "r"]
第1次合并: "lo" 出现频繁 -> ["lo", "w", "e", "r"]
第2次合并: "low" 出现频繁 -> ["low", "e", "r"]
第3次合并: "er" 出现频繁 -> ["low", "er"]
最终: "lower" -> ["low", "er"]

优势：
- 可以处理任何文本（包括生僻字）
- 平衡了字符级和词级分词
- 常用词用一个token，罕见词拆分为多个token
"""

# 导入必要的库
import random  # 用于设置随机种子，确保结果可复现
import json  # 用于读取和写入JSON格式的数据
from tokenizers import (  # HuggingFace的tokenizers库，快速高效
    decoders,  # 解码器：将token ID转回文本
    models,  # 模型：定义分词算法（如BPE、WordPiece等）
    pre_tokenizers,  # 预分词器：在主分词前的预处理
    trainers,  # 训练器：训练tokenizer的配置
    Tokenizer,  # Tokenizer主类
)
import os  # 用于文件和目录操作

# 设置随机种子为42，确保每次运行结果一致
# 这在机器学习中很重要，可以让实验结果可复现
random.seed(42)


def train_tokenizer():
    """
    训练自定义的Tokenizer
    
    这个函数完成以下任务：
    1. 从数据集读取文本
    2. 使用BPE算法训练tokenizer
    3. 配置特殊token
    4. 保存训练好的tokenizer
    
    训练过程类似于"编字典"：
    - 统计哪些字/词组合最常出现
    - 为常用组合分配独立的ID
    - 最终得到一个6400词的"词典"
    """
    
    # ========================================
    # 步骤1: 定义数据读取函数
    # ========================================
    def read_texts_from_jsonl(file_path):
        """
        从JSONL文件中逐行读取文本数据
        
        JSONL格式说明：
        - 每行是一个独立的JSON对象
        - 格式: {"text": "这是一段文本"}
        - 优势: 可以逐行读取，不需要一次性加载整个文件到内存
        
        为什么使用生成器(yield)？
        - 内存效率：不需要一次性加载所有文本
        - 流式处理：边读边训练，适合大数据集
        - 例如：1GB的文本数据，使用生成器只需要几MB内存
        
        参数:
            file_path: JSONL文件路径
        
        生成:
            每次yield一段文本字符串
        """
        # 打开文件，使用utf-8编码（支持中文等多语言）
        with open(file_path, 'r', encoding='utf-8') as f:
            # 逐行读取文件
            for line in f:
                # 解析JSON字符串为Python字典
                # 例如: '{"text": "你好"}' -> {"text": "你好"}
                data = json.loads(line)
                # 使用yield返回文本内容
                # yield使这个函数成为生成器，每次调用返回一个值
                # 而不是一次性返回所有值
                yield data['text']

    # 指定训练数据路径
    # pretrain_hq.jsonl 包含高质量的预训练文本数据
    data_path = '../dataset/pretrain_hq.jsonl'

    # ========================================
    # 步骤2: 初始化Tokenizer
    # ========================================
    
    # 创建Tokenizer对象，使用BPE模型
    # BPE (Byte Pair Encoding) 是一种子词分词算法
    # 它可以在字符级和词级之间找到平衡
    tokenizer = Tokenizer(models.BPE())
    
    # 设置预分词器为ByteLevel
    # 预分词器的作用：在BPE之前对文本进行初步处理
    # 
    # ByteLevel预分词器的特点：
    # 1. 将文本转换为字节级表示
    # 2. 可以处理任何Unicode字符（包括emoji、特殊符号等）
    # 3. 不会产生未知字符(UNK)
    # 
    # add_prefix_space=False 的含义：
    # - False: 不在文本开头添加空格
    # - True: 会在开头添加空格（某些模型需要这样）
    # 例如：
    #   False: "Hello" -> "Hello"
    #   True:  "Hello" -> " Hello"
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ========================================
    # 步骤3: 定义特殊Token
    # ========================================
    
    # 特殊token是具有特殊含义的标记，不是普通文本
    # 它们用于标识文本的结构和边界
    special_tokens = [
        "<|endoftext|>",  # ID=0, 文本结束标记/填充标记/未知词标记
        "<|im_start|>",   # ID=1, 消息开始标记 (im = instant message)
        "<|im_end|>"      # ID=2, 消息结束标记
    ]
    
    # 为什么需要这些特殊token？
    # 
    # 1. <|endoftext|> (ID=0)
    #    - 作用1: 标记文档结束
    #      例如: "文章内容<|endoftext|>"
    #    - 作用2: 作为padding token填充短序列
    #      例如: [123, 456, 0, 0, 0] (后面的0是padding)
    #    - 作用3: 作为未知词的替代
    #      例如: 遇到训练时没见过的生僻字
    #    - 为什么是ID=0？方便初始化和padding
    #
    # 2. <|im_start|> (ID=1)
    #    - 标记对话消息的开始
    #    - 配合role使用，例如:
    #      "<|im_start|>user\n你好<|im_end|>"
    #      "<|im_start|>assistant\n你好！<|im_end|>"
    #    - 帮助模型理解对话结构
    #
    # 3. <|im_end|> (ID=2)
    #    - 标记对话消息的结束
    #    - 告诉模型一轮对话已完成
    #    - 在生成时，模型看到这个token会停止生成

    # ========================================
    # 步骤4: 配置BPE训练器
    # ========================================
    
    trainer = trainers.BpeTrainer(
        # vocab_size: 词表大小，即"词典"中有多少个词
        # 6400是一个精心选择的值：
        # - 太小(如1000): 分词太细，序列太长，效率低
        # - 太大(如50000): embedding层参数太多，模型头重脚轻
        # - 6400: 在模型大小和分词效率间取得平衡
        vocab_size=6400,
        
        # special_tokens: 特殊token列表
        # 这些token会被优先分配ID（从0开始）
        # 确保它们不会被BPE算法拆分
        special_tokens=special_tokens,
        
        # show_progress: 是否显示训练进度条
        # True: 显示进度，方便观察训练状态
        # False: 不显示，适合在脚本中静默运行
        show_progress=True,
        
        # initial_alphabet: 初始字母表
        # ByteLevel.alphabet() 返回所有256个字节值
        # 这确保任何字节都可以被编码，不会出现UNK
        # 这是ByteLevel分词的核心：从字节级开始构建词表
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # BPE训练过程（简化说明）：
    # 1. 初始化：每个字节是一个token (256个基础token)
    # 2. 统计：扫描所有文本，统计token对的出现频率
    # 3. 合并：找出最频繁的token对，合并为新token
    # 4. 重复：重复步骤2-3，直到词表达到6400
    # 5. 结果：得到一个包含6400个token的词表
    #
    # 例如训练过程：
    # 初始: ["你", "好", "世", "界"] 各自是独立token
    # 发现"你好"出现1000次 -> 合并为一个token
    # 发现"世界"出现800次 -> 合并为一个token
    # 最终: ["你好", "世界"] 各自是一个token

    # ========================================
    # 步骤5: 读取训练数据
    # ========================================
    
    # 调用之前定义的生成器函数
    # texts是一个生成器对象，不会立即加载所有数据
    # 只有在训练时才会逐步读取
    texts = read_texts_from_jsonl(data_path)

    # ========================================
    # 步骤6: 训练Tokenizer
    # ========================================
    
    # 使用迭代器训练tokenizer
    # 这个过程会：
    # 1. 遍历所有文本数据
    # 2. 统计token对的频率
    # 3. 执行BPE合并操作
    # 4. 构建最终的词表
    # 
    # 训练时间取决于：
    # - 数据量大小（pretrain_hq.jsonl约1.6GB）
    # - CPU性能
    # - 词表大小
    # 通常需要几分钟到十几分钟
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # 训练完成后，tokenizer包含：
    # - 词表：6400个token及其ID
    # - 合并规则：如何将字节组合成token
    # - 特殊token：<|endoftext|>, <|im_start|>, <|im_end|>

    # ========================================
    # 步骤7: 设置解码器
    # ========================================
    
    # 解码器的作用：将token ID序列转换回文本
    # 例如: [1234, 5678] -> "你好"
    # 
    # ByteLevel解码器：
    # - 将token转回字节
    # - 将字节解码为UTF-8字符串
    # - 处理特殊字符和空格
    # 
    # 编码和解码必须配对：
    # - 编码用ByteLevel预分词器
    # - 解码用ByteLevel解码器
    # 这样才能保证 decode(encode(text)) == text
    tokenizer.decoder = decoders.ByteLevel()

    # ========================================
    # 步骤8: 验证特殊Token的ID
    # ========================================
    
    # 使用assert确保特殊token被分配了正确的ID
    # 这些ID必须固定，因为模型训练时会依赖这些ID
    
    # 检查<|endoftext|>的ID是否为0
    # 为什么必须是0？
    # 1. 作为padding token，0便于初始化
    # 2. 很多框架默认padding_id=0
    # 3. 在attention mask中，0位置会被忽略
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    
    # 检查<|im_start|>的ID是否为1
    # 作为BOS (Beginning Of Sequence) token
    # 标记序列的开始
    assert tokenizer.token_to_id("<|im_start|>") == 1
    
    # 检查<|im_end|>的ID是否为2
    # 作为EOS (End Of Sequence) token
    # 标记序列的结束，模型生成时遇到此token会停止
    assert tokenizer.token_to_id("<|im_end|>") == 2
    
    # 如果任何一个assert失败，程序会报错
    # 这说明训练过程有问题，需要检查special_tokens的顺序

    # ========================================
    # 步骤9: 保存Tokenizer
    # ========================================
    
    # 指定保存目录
    tokenizer_dir = "../model/"
    
    # 创建目录（如果不存在）
    # exist_ok=True: 如果目录已存在，不报错
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # 保存tokenizer的主文件
    # tokenizer.json包含：
    # - 词表（所有token及其ID）
    # - 合并规则（BPE的合并操作）
    # - 预分词器配置
    # - 解码器配置
    # 这是一个JSON文件，可以用文本编辑器打开查看
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    
    # 保存模型文件
    # 这会保存BPE模型的额外信息
    # 包括vocab.json和merges.txt
    tokenizer.model.save("../model/")

    # ========================================
    # 步骤10: 创建Tokenizer配置文件
    # ========================================
    
    # 这个配置文件告诉transformers库如何使用我们的tokenizer
    # 它定义了各种行为和特殊token的用途
    config = {
        # add_bos_token: 是否自动在序列开头添加BOS token
        # False: 不自动添加，我们会在chat_template中手动控制
        "add_bos_token": False,
        
        # add_eos_token: 是否自动在序列结尾添加EOS token
        # False: 不自动添加，同样在chat_template中手动控制
        "add_eos_token": False,
        
        # add_prefix_space: 是否在文本前添加空格
        # False: 不添加，与ByteLevel预分词器的设置一致
        "add_prefix_space": False,
        
        # added_tokens_decoder: 特殊token的详细配置
        # 这是一个字典，键是token ID，值是token的属性
        "added_tokens_decoder": {
            # Token ID 0: <|endoftext|>
            "0": {
                "content": "<|endoftext|>",  # token的实际内容
                "lstrip": False,  # 不去除左侧空格
                "normalized": False,  # 不进行标准化（如小写转换）
                "rstrip": False,  # 不去除右侧空格
                "single_word": False,  # 不要求是单个词
                "special": True  # 标记为特殊token
            },
            # Token ID 1: <|im_start|>
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            # Token ID 2: <|im_end|>
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        
        # additional_special_tokens: 额外的特殊token列表
        # 空列表表示没有额外的特殊token
        "additional_special_tokens": [],
        
        # bos_token: Beginning Of Sequence token
        # 用于标记序列的开始
        "bos_token": "<|im_start|>",
        
        # clean_up_tokenization_spaces: 是否清理分词产生的多余空格
        # False: 保持原样，不清理
        "clean_up_tokenization_spaces": False,
        
        # eos_token: End Of Sequence token
        # 用于标记序列的结束，模型生成时遇到此token会停止
        "eos_token": "<|im_end|>",
        
        # legacy: 是否使用旧版行为
        # True: 兼容旧版本的tokenizer行为
        "legacy": True,
        
        # model_max_length: 模型支持的最大序列长度
        # 32768: 支持最长32K个token的序列
        # 这与模型的位置编码长度相关
        "model_max_length": 32768,
        
        # pad_token: Padding token
        # 用于填充短序列，使batch中的序列长度一致
        # 例如: [1,2,3,0,0] 后面的0就是padding
        "pad_token": "<|endoftext|>",
        
        # sp_model_kwargs: SentencePiece模型的参数
        # 空字典表示不使用SentencePiece（我们用的是BPE）
        "sp_model_kwargs": {},
        
        # spaces_between_special_tokens: 特殊token之间是否添加空格
        # False: 不添加空格
        "spaces_between_special_tokens": False,
        
        # tokenizer_class: tokenizer的类名
        # PreTrainedTokenizerFast: 使用快速版本的tokenizer
        "tokenizer_class": "PreTrainedTokenizerFast",
        
        # unk_token: Unknown token
        # 用于表示词表中不存在的token
        # 虽然ByteLevel理论上不会产生UNK，但仍需定义
        "unk_token": "<|endoftext|>",
        
        # chat_template: 对话模板
        # 这是一个Jinja2模板，定义如何格式化对话
        # 
        # 模板逻辑说明：
        # 1. 如果第一条消息是system角色，使用它作为系统提示
        #    否则使用默认提示"You are a helpful assistant"
        # 2. 遍历所有消息：
        #    - user消息: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
        #    - assistant消息: {content}<|im_end|>\n
        # 
        # 示例输出：
        # <|im_start|>system
        # You are a helpful assistant<|im_end|>
        # <|im_start|>user
        # 你好<|im_end|>
        # <|im_start|>assistant
        # 你好！有什么可以帮助你的吗？<|im_end|>
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # ========================================
    # 步骤11: 保存配置文件
    # ========================================
    
    # 将配置字典保存为JSON文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        # json.dump: 将Python对象序列化为JSON
        # ensure_ascii=False: 允许非ASCII字符（如中文）
        # indent=4: 使用4个空格缩进，使文件更易读
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    # 打印完成消息
    print("Tokenizer training completed and saved.")
    
    # 训练完成后，../model/目录下会有以下文件：
    # - tokenizer.json: 主tokenizer文件
    # - tokenizer_config.json: 配置文件
    # - vocab.json: 词表文件
    # - merges.txt: BPE合并规则
    # 
    # 这些文件可以被transformers库直接加载使用


def eval_tokenizer():
    """
    评估和测试训练好的Tokenizer
    
    这个函数验证tokenizer的功能：
    1. 能否正确加载
    2. chat_template是否工作正常
    3. 编码和解码是否一致
    4. 词表大小是否正确
    
    这就像测试我们编的"词典"是否好用
    """
    # 从transformers库导入AutoTokenizer
    # AutoTokenizer可以自动识别tokenizer类型并加载
    from transformers import AutoTokenizer

    # ========================================
    # 步骤1: 加载Tokenizer
    # ========================================
    
    # 从指定目录加载tokenizer
    # AutoTokenizer会读取：
    # - tokenizer.json
    # - tokenizer_config.json
    # - vocab.json
    # - merges.txt
    # 并自动配置好所有参数
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    # ========================================
    # 步骤2: 测试Chat Template
    # ========================================
    
    # 创建一个多轮对话示例
    # 这是标准的对话格式，包含三种角色：
    messages = [
        # system角色：定义AI的行为和性格
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        # user角色：用户的问题
        {"role": "user", "content": '你来自哪里？'},
        # assistant角色：AI的回答
        {"role": "assistant", "content": '我来自地球'}
    ]
    
    # 应用chat_template将对话转换为模型输入格式
    # tokenize=False: 只返回文本，不进行tokenize
    # 
    # 这个函数会使用我们在config中定义的chat_template
    # 将对话列表转换为带有特殊token的格式化文本
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False  # 先不tokenize，只看格式化后的文本
    )
    
    # 打印格式化后的文本
    # 预期输出类似：
    # <|im_start|>system
    # 你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
    # <|im_start|>user
    # 你来自哪里？<|im_end|>
    # <|im_start|>assistant
    # 我来自地球<|im_end|>
    print(new_prompt)

    # ========================================
    # 步骤3: 检查词表大小
    # ========================================
    
    # 获取tokenizer的实际词表长度
    # len(tokenizer)返回词表中token的总数
    # 应该等于或接近我们设置的vocab_size=6400
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)
    
    # 为什么可能不完全等于6400？
    # - 特殊token占用了一些位置
    # - BPE训练可能没有完全达到目标大小
    # - 但应该非常接近6400

    # ========================================
    # 步骤4: 测试编码功能
    # ========================================
    
    # 将文本编码为token ID序列
    # tokenizer(text)会：
    # 1. 应用预分词器（ByteLevel）
    # 2. 应用BPE分词
    # 3. 将token转换为ID
    # 4. 返回包含input_ids等信息的字典
    model_inputs = tokenizer(new_prompt)
    
    # 打印编码后的序列长度
    # 这告诉我们这段对话被分成了多少个token
    # 例如：可能是50个token
    print('encoder长度：', len(model_inputs['input_ids']))
    
    # 编码长度的意义：
    # - 越短越好：说明tokenizer压缩效率高
    # - 太长：可能需要更大的词表或更好的训练数据
    # - 这个长度直接影响模型的计算量和内存占用

    # ========================================
    # 步骤5: 测试解码功能
    # ========================================
    
    # 提取token ID序列
    input_ids = model_inputs['input_ids']
    
    # 将token ID序列解码回文本
    # skip_special_tokens=False: 保留特殊token（如<|im_start|>）
    # 如果设为True，特殊token会被移除
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # 验证解码结果是否与原始文本一致
    # 这是tokenizer正确性的关键测试
    # 如果 decode(encode(text)) != text，说明tokenizer有问题
    print('decoder和原始文本是否一致：', response == new_prompt)
    
    # 为什么这个测试很重要？
    # 1. 确保信息无损：编码-解码过程不丢失信息
    # 2. 验证配置正确：预分词器和解码器配置匹配
    # 3. 保证可用性：模型生成的token能正确转回文本
    # 
    # 如果这个测试失败，可能的原因：
    # - 预分词器和解码器不匹配
    # - 特殊字符处理有问题
    # - 空格处理不一致


def main():
    """
    主函数：执行tokenizer的训练和评估
    
    执行流程：
    1. train_tokenizer(): 训练并保存tokenizer
    2. eval_tokenizer(): 加载并测试tokenizer
    
    整个过程类似于：
    1. 编写一本词典（训练）
    2. 测试词典是否好用（评估）
    """
    # 步骤1: 训练tokenizer
    # 这个过程可能需要几分钟到十几分钟
    # 取决于数据量和CPU性能
    train_tokenizer()
    
    # 步骤2: 评估tokenizer
    # 验证训练结果是否正确
    eval_tokenizer()


# Python程序的入口点
# 当直接运行这个脚本时（python train_tokenizer.py）
# __name__会被设置为'__main__'
# 这时会执行main()函数
# 
# 如果这个文件被其他脚本import，__name__不是'__main__'
# main()就不会自动执行
if __name__ == '__main__':
    main()


# ========================================
# 总结：Tokenizer训练的完整流程
# ========================================
"""
1. 数据准备
   - 收集大量文本数据（pretrain_hq.jsonl）
   - 数据应该覆盖目标领域的常用词汇

2. 选择算法
   - BPE: 平衡字符级和词级，适合多语言
   - WordPiece: Google使用，适合英文
   - SentencePiece: 不依赖预分词，适合无空格语言

3. 配置参数
   - vocab_size: 词表大小（6400）
   - special_tokens: 特殊token列表
   - 预分词器: ByteLevel（字节级）

4. 训练过程
   - 统计token对频率
   - 迭代合并高频token对
   - 构建最终词表

5. 保存和配置
   - 保存tokenizer文件
   - 配置特殊token
   - 定义chat_template

6. 测试验证
   - 测试编码解码一致性
   - 验证词表大小
   - 测试chat_template

关键概念回顾：
--------------
- Token: 文本的基本单位，可以是字符、子词或词
- Vocabulary: 词表，所有token的集合
- Encoding: 文本 -> token ID序列
- Decoding: token ID序列 -> 文本
- Special Tokens: 具有特殊含义的token
- BPE: 字节对编码，一种子词分词算法
- Chat Template: 对话格式化模板

为什么需要好的Tokenizer？
------------------------
1. 效率：好的分词可以用更少的token表示文本
   - 减少序列长度 -> 减少计算量
   - 减少内存占用 -> 可以处理更长的文本

2. 性能：合理的词表可以提高模型效果
   - 常用词作为整体 -> 更好的语义理解
   - 覆盖生僻词 -> 减少UNK token

3. 模型大小：词表大小直接影响参数量
   - 词表6400: embedding层参数少
   - 词表150000: embedding层参数多23倍

4. 多语言支持：ByteLevel可以处理任何语言
   - 不需要预定义字符集
   - 支持emoji和特殊符号

实际应用建议：
------------
1. 数据选择：使用与目标任务相关的数据训练
2. 词表大小：根据模型大小和任务复杂度选择
3. 特殊token：根据任务需求定义（对话、代码等）
4. 测试充分：确保编码解码一致性
5. 版本管理：保存tokenizer配置，确保可复现
"""
