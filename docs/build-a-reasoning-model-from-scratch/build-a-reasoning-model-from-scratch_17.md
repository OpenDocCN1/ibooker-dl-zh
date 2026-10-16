# 索引

## 符号

- `.detach()`方法
- `@torch.inference_mode()`装饰器
- `.unsqueeze()`操作
- 以编程方式加载`.jsonl`文件
- `@inference_mode`
- `--depth 1`选项
- `==`运算符
- ` thinking`标签
- `.squeeze()`操作
- `@torch.no_grad`装饰器

## A

- 仅答案损失
- 自回归地
- 自回归文本生成
- 优势跟踪
- `argmax`函数, 第 2 处, 第 3 处
- **优势**
  - 从 rollouts 中准备学习信号
- **添加显式**
  - 更多修改、技巧和诀窍, 第 2 处
  - 训练模型输出 ` thinking` token, 第 2 处
  - 使用 ` thinking` token, 第 2 处
  - `avg_logprob_score`函数
  - `avg_logprob_answer`函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处, 第 6 处

## B

- 基于基准的评估
- `brevity_bonus`, 第 2 处
- 批量训练
- `build_examples`函数, 第 2 处
- `boxed_bonus`, 第 2 处
- BPE（字节对编码）

## C

- CoT（思维链）
- `checkpoint_every`
- 条件概率
- `compute_grpo_loss_with_kl`函数
- 计算图
- 传统大语言模型（LLM）
- `compute_grpo_loss`函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处, 第 6 处, 第 7 处, 第 8 处
- 思维链提示, 第 2 处
- `cross_entropy`函数, 第 2 处, 第 3 处
- **检查点**
  - 加载和评估已保存的模型检查点, 第 2 处
- **截断的策略比率**
  - 计算, 第 2 处
  - 使用截断的策略比率稳定序列级 GRPO, 第 2 处
  - 使用截断的策略比率进行训练
- `count_samples`函数, 第 2 处, 第 3 处
- 聊天机器人界面
- `compute_example_loss`辅助函数
- 封闭世界设定
- `compute_length`函数
- 交叉熵损失
- `calc_next_token_probas`函数, 第 2 处
- `csv_log_path`
- `calc_next_token_logprobas`函数, 第 2 处, 第 3 处
- `compute_grpo_loss_plus_format_reward`函数

## D

- **蒸馏推理模型**
  - 评估蒸馏后的模型, 第 2 处
  - 为蒸馏加载 MATH 训练数据集, 第 2 处
  - 加载预训练模型
  - 训练示例, 第 2 处
- `draft`
- `dtype`
- **蒸馏**
  - 计算训练和验证损失, 第 2 处
  - 评估蒸馏后的模型, 第 2 处
  - 为蒸馏加载 MATH 训练数据集, 第 2 处
  - 加载预训练模型
  - 后续步骤
  - 在快速发展领域中保持最新
  - 用于蒸馏的训练循环, 第 2 处
- **数据集**
  - 生成用于推理蒸馏的数据集
  - `datasets`库
  - 蒸馏, 第 2 处
  - 概述, 第 2 处
  - 训练循环, 第 2 处
- **蒸馏模型以实现高效推理**
  - 训练示例, 第 2 处
- `download_qwen3_small()`函数
- 蒸馏损失

## E

- **评估**
  - 加载评估数据集, 第 2 处
- **评估**
  - 构建数学验证器
  - 提取最终答案框, 第 2 处
  - 实现包装器以简化文本生成
  - 加载评估数据集, 第 2 处
  - 加载预训练模型以生成文本, 第 2 处
  - 规范化提取的答案, 第 2 处
  - 验证数学等价性, 第 2 处
- `evaluate_json.py` 脚本
- `evaluate_math500_stream` 函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处
- 评估准确率的计算
- `equality_check` 函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处
- 提取答案的规范化, 第 2 处
- 熵追踪, 第 2 处
- `encode` 方法
- 评估推理模型
- 构建数学验证器
- 提取最终答案方框, 第 2 处
- 实现包装器以简化文本生成
- 加载预训练模型以生成文本, 第 2 处
- 示例：玩具示例
- `extract_bonus`, 第 2 处
- `extract_final_candidate` 函数, 第 2 处, 第 3 处
- 涌现特性

## F

- `format_distilled_answer` 函数
- `filter_examples_by_max_len` 步骤
- **格式奖励**
  - 添加显式, 第 2 处
- 完整前向传播
- `fulltext_bonus`

## G

- `generate_text_stream_concat` 函数, 第 2 处
- `generate_text_stream_concat` 模型
- `get_last_boxed` 函数
- GRPO（广义奖励预测目标）
- `generate_text_basic_stream`
- `grade_answer` 函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处, 第 6 处, 第 7 处
- `generate_stats` 函数
- `get_device()` 函数
- **使用预训练 LLM 生成文本**
  - 编写最简文本生成函数, 第 2 处
- **GRPO（广义响应策略优化）**
  - 跟踪性能指标, 第 2 处, 第 3 处, 第 4 处
  - 对答案评分, 第 2 处
- **GRPO（带策略优化的广义 REINFORCE）流水线**
  - 通过 GRPO 损失进行策略更新
- **GRPO（群体相对策略优化）**
  - 改进
  - `generate_text_basic_stream` 函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处, 第 6 处, 第 7 处, 第 8 处
  - `generate_text_temp_stream_cache` 函数
  - `generate_text_temp_stream_cache`
- **GRPO（群组相对策略优化）**
  - 使用强化学习训练推理模型, 第 2 处
  - `generate_text_stream_concat_flex`
  - `generate_text_stream_cache` 函数
  - `generate_text_stream_concat_flex` 函数, 第 2 处, 第 3 处, 第 4 处, 第 5 处
  - `generate_text_stream_cache`
  - `generate_text_top_p_stream_cache` 函数
  - `generate_text_basic_stream_cache`
  - GRPO（群体相对策略优化）算法
  - 通过厨师类比获得 GRPO 高层直觉
  - GRPO 高层流程
  - 使用, 第 2 处
  - `gtruth_answer` 字段
- **GRPO（目标-奖励-提示优化）**
  - 使用序列对数概率为 rollout 评分, 第 2 处
- **GRPO（群组相对策略优化）训练循环**
  - 实现, 第 2 处
  - `grad_clip_norm`
  - `generate_text_basic_stream_cache` 函数, 第 2 处, 第 3 处, 第 4 处
- **使用……生成文本**
  - 编写最简文本生成函数, 第 2 处
  - 真实值
  - 贪心解码
- **GRPO（广义 rollout 策略优化）**
  - 跟踪性能指标, 第 2 处, 第 3 处, 第 4 处

## H

- 硬蒸馏
- 启发式
- Hugging Face 模型中心
- **硬件**
  - 需求与建议, 第 2 处
  - `heuristic_score` 函数, 第 2 处, 第 3 处

## I

- **改进 GRPO 以用于强化学习**
  - 添加显式格式奖励, 第 2 处
  - 更多修改、技巧和窍门, 第 2 处
  - 训练模型输出 `<think>` 标记, 第 2 处
  - 使用 `<think>` 标记, 第 2 处
- 指令微调
- 推理计算扩展
- **改进 GRPO 以用于……**
  - 添加显式格式奖励, 第 2 处
  - 使用 KL 项控制模型变化幅度, 第 2 处
  - 更多修改、技巧和窍门, 第 2 处
  - 使用裁剪的策略比率稳定序列级 GRPO, 第 2 处
  - 训练模型输出 `<think>` 标记, 第 2 处
  - 使用 `<think>` 标记, 第 2 处
- **借助……改进推理**
  - 思维链提示, 第 2 处
  - 加载预训练模型, 第 2 处
  - 通过温度参数重新缩放词元分数（logits）
- **推理**
  - 通过 KV 缓存加速, 第 2 处
  - 通过 PyTorch 模型编译加速, 第 2 处
  - 推理时扩展（inference scaling）
- **通过推理时扩展进行改进**
  - 向文本生成函数添加温度缩放
  - 思维链提示, 第 2 处
  - 通过温度缩放控制输出多样性
  - 加载预训练模型, 第 2 处
  - 通过温度参数重新缩放词元分数（logits）
  - 从概率分布中采样下一个词元
  - 选择 top-p 词元的子集
  - 推理时扩展, 第 2 处, 第 3 处, 第 4 处
  - 将温度缩放应用于文本生成函数
  - 通过温度缩放控制输出多样性
  - 似然度与概率
  - 对数概率, 第 2 个, 第 3 个, 第 4 个
  - 下一个词元的选择过程
  - 从概率分布中采样下一个词元
  - 自洽性采样, 第 2 个
  - 自我精炼, 第 2 个
  - 自我精炼循环, 第 2 个
  - 词元概率分数, 第 2 个, 第 3 个, 第 4 个
  - top-k 过滤, 第 2 个
  - top-p 过滤, 第 2 个
  - 为强化学习改进
  - 添加显式格式奖励, 第 2 个
  - 计算裁剪的策略比率, 第 2 个
  - 通过 KL 项控制模型变化程度, 第 2 个
  - 更多修改、提示与技巧, 第 2 个
  - 使用裁剪的策略比率稳定序列级 GRPO, 第 2 个
  - 训练模型生成 `<think>` 词元, 第 2 个
  - 使用裁剪的策略比率进行训练
  - 使用 `<think>` 词元, 第 2 个
  - `iterations` 参数
  - 推理时计算缩放, 第 2 个
  - 推理时
  - 将温度缩放应用于文本生成函数
  - 指令微调

## J

- 联合概率
- 基于评判的评估

## K

- KL 损失项
- KL（Kullback–Leibler）散度
- 通过 KL 项控制模型变化程度, 第 2 个
- 实现 KL 损失项, 第 2 个
- 使用 KL 损失项进行训练, 第 2 个
- `kl_loss`
- 知识蒸馏
- KV 缓存
- 通过其加速推理, 第 2 个
- KL 散度
- `KVCache` 对象
- `kl_coeff`

## L

- `load_math_train` 函数
- `lr`（学习率）, 第 2 个
- `log_every`
- logits
- 通过温度参数重新缩放词元分数（logits）
- LLM（大语言模型）
- 用于文本生成
- 通过训练与推理技术提升推理能力, 第 2 个
- 为其准备输入文本, 第 2 个
- 使用基于规则的评分器为响应打分, 第 2 个
- 训练流程, 第 2 个
- 对数概率评分（logprob 评分）方法
- 对数概率, 第 2 个
- 用其评估模型置信度, 第 2 个
- 用序列对数概率为 rollout 打分, 第 2 个
- `logprob`
- `Latex` 类, 第 2 个
- 对数概率（logprobs）
- `load_model_and_tokenizer`
- `load_state_dict` 方法
- 似然度, 第 2 个
- `load_math500_test` 函数
- LLM 即评判者
- 学习信号
- 从 rollout 中通过优势值准备
- `load_model_and_tokenizer` 函数

## M

- MATH（数学）数据集
- 加载蒸馏训练数据集, 第 2 个
- `message_thinking` 字段
- `mini_eval_demo` 函数, 第 2 个
- `mini_eval_demo(model, tokenizer, device)` 函数
- 多奖励训练
- 数学等价性验证, 第 2 个
- `message_content` 字段
- 最大化
- 中训阶段
- `Math` 类
- `max` 函数
- 最小化
- 数学验证器
- 模型蒸馏
- 概述, 第 2 个
- `max_new_tokens`

## N

- `next_ids` 张量
- 规范化提取出的答案, 第 2 个
- 核采样
- `num_rollouts`, 第 2 个
- `n_kept` 变量
- `next_token_probas`
- 下一个词元的选择过程
- 下一个词元的概率
- `next_token_logits` 张量, 第 2 个
- `normalize_text` 函数, 第 2 个, 第 3 个

## O

- 开放世界设定
- 概述
- RLHF
- RLVR

## P

- 模式匹配
- 逻辑推理与之对比
- PyTorch
- 通过模型编译加速推理, 第 2 个
- 概率分数
- `plt.bar` 条形图
- `pprint` 库
- `print` 函数
- `plt.plot` 折线图
- 策略更新
- 通过 GRPO 损失
- PPO（近端策略优化）
- 每个词元的概率
- 预训练阶段
- 过程奖励
- `prompt` 参数
- `pprint` 标准库
- 策略梯度损失的数学定义
- 提示模板
- 偏好调优, 第 2 个
- 加载预训练模型, 第 2 个
- 概率
- 与似然度对比
- 策略优化算法
- `plot_grpo_metrics` 函数
- 加载预训练模型, 第 2 个
- `pip` 包安装器, 第 2 个
- `pred_answer` 最终候选答案
- Proximal Policy Optimization Algorithms（近端策略优化算法）
- `plot_scores_bar` 函数
- `pyproject.toml` 文件
- `partial` 函数
- 后训练阶段
- `pprint` 函数

## Q

- `Qwen3Model` 类

## R

- 推理模型
- 推理训练
- `raw_prompt`
- `re` 库
- 使用 RLVR 的推理模型
- 加载并评估保存的模型检查点, 第 2 处
- RLHF（基于人类反馈的强化学习），第 2 处，第 3 处
- RL（强化学习），第 2 处
- 概述，第 2 处
- `reward_format` 函数
- 正则表达式
- **展开（rollouts）**
  - 通过优势值准备学习信号
  - 采样，第 2 处
  - 使用序列对数概率评分，第 2 处
- `run_demos_table` 函数
- `requirements.txt` 文件，第 2 处
- `run_demos_table()` 函数
- **奖励（rewards）**
  - 计算
- `reasoning-from-scratch` Python 包
- `reasoning-from-scratch` 库
- **推理（reasoning）**
  - 通过推理时扩展改进
- `reward_avg`
- **推理蒸馏（reasoning distillation）**
  - 为……生成数据集
- 基于规则的评分函数
- 推理模型，第 2 处
- LLM 与人类推理的对比
- 从头构建
- 定义，第 2 处
- 评估，第 2 处
- 未来发展方向
- 答案评分，第 2 处
- 通过训练和推理技术改进，第 2 处
- 逻辑推理与……的对比
- 模式匹配与逻辑推理的对比
- 自一致性采样，第 2 处
- 无需显式规则模拟推理
- 使用强化学习训练，第 2 处，第 3 处
- **强化学习（reinforcement learning）**
  - 改进 GRPO
  - 使用……训练推理模型，第 2 处，第 3 处
- `run_demos_table(tests)` 函数
- regex（正则表达式）
- rollout 采样
- 构建推理模型的路线图
- `reward_rlvr` 函数，第 2 处，第 3 处
- **使用强化学习的推理模型**
  - 通过 GRPO 损失进行策略更新
  - 在单个 GRPO 函数中整合所有内容，第 2 处
- results 字典，第 2 处，第 3 处
- RLVR（基于可验证奖励的强化学习），第 2 处
  - 使用 GRPO，第 2 处
- **RLVR（带验证和再训练的强化学习）**
  - 加载和评估保存的模型检查点，第 2 处
- `results_logprob` 字典
- 原始字符串

## S

- **序列对数概率（sequence log probabilities）**
  - 使用……对 rollouts 进行评分，第 2 处
- 自一致性，第 2 处
- `SumBackward0` 条目
- **扩展（scaling）**
  - 通过温度参数调整 token 分数（logits）
- **评分（scoring）**
  - 基于规则分数对 LLM 响应进行评分，第 2 处
  - 评分并迭代改进模型响应
- 序列似然
- 软蒸馏
- `sequence_logprob_and_entropy` 函数，第 2 处，第 3 处
- `softmax` 函数
- 顺序 LLM 文本生成过程，第 2 处
- 采样 rollouts，第 2 处
- 统计推断
- `sequence_logprob` 函数，第 2 处，第 3 处，第 4 处，第 5 处
- `split_into_parts` 函数
- 监督微调（SFT）
- `score_fn` 选项
- `self_consistency_vote` 函数，第 2 处，第 3 处，第 4 处，第 5 处，第 6 处，第 7 处
- `SymPy` 数学库
- `score_fn` 调用
- `self_refinement_loop` 函数，第 2 处
- `sympy_parser` 函数，第 2 处
- 压缩张量
- **对 rollouts 评分**
  - 使用序列对数概率，第 2 处
- SFT（监督微调），第 2 处
- `scorer`
- 自我精炼，第 2 处，第 3 处，第 4 处，第 5 处
  - 加载预训练模型，第 2 处
  - 循环，第 2 处
- 基于规则分数对 LLM 响应进行评分，第 2 处
- 评分并迭代改进模型响应，第 2 处
- 使用对数概率评估模型置信度，第 2 处，第 3 处，第 4 处
- `scale_logits_by_temperature` 函数
- 自一致性采样，第 2 处
- `sample_response` 函数，第 2 处，第 3 处，第 4 处

## T

- **文本生成（text generation）**
  - 通过 KV 缓存加速推理，第 2 处
  - 通过 PyTorch 模型编译加速推理，第 2 处
  - 通过 PyTorch 模型编译进行推理，第 2 处
  - 加载预训练模型，第 2 处
  - 设置编码环境，第 2 处
- **分词（tokenizing）**
  - 数据集
- top-p 设置，第 2 处
- token 级似然
- `torch.max()` 函数
- token 概率分数，第 2 处
- **使用 RL 训练**
  - RLHF
  - RLVR
  - 概述，第 2 处
- `torch.argmax` 函数
- `tokenizers` 库
- `torch.argmax()` 函数
- `torch.log_softmax` 函数，第 2 处
- top-p 采样器
- 测试时计算扩展
- **训练子集（training subset）**
  - 加载，第 2 处
- `tokenizers`
- top-p 采样
- `torch.cumsum` 函数
- `torch.softmax()` 函数
- 目标序列
- 训练示例，第 2 处
- 筛选数据集，第 2 处，第 3 处，第 4 处，第 5 处，第 6 处
- 对数据集进行格式化
- 对数据集进行拆分，第 2 处，第 3 处，第 4 处，第 5 处，第 6 处
- 分词器，第 2 处，第 3 处
- 对数据集进行分词，第 2 处，第 3 处
- **使用强化学习训练**
  - 计算奖励
  - 实现 GRPO 训练循环，第 2 处
  - 加载 MATH 训练子集，第 2 处
  - 加载预训练模型
  - 通过 GRPO 损失更新策略
  - 将一切整合到单个 GRPO 函数中，第 2 处
  - 采样轨迹，第 2 处
- **训练**
  - 使用强化学习的推理模型，第 2 处
  - 使用序列对数概率对轨迹评分，第 2 处
- `torch.compile`
- `top-p` 范围
- `top_p_filter` 函数，第 2 处
- **温度参数**
  - 通过……重新缩放词元分数（logits）
  - 温度设置，第 2 处，第 3 处
- `tokenizer-base.json` 文件
- `top_p` 设置
- `tokenizer.eos_token_id` 属性
- `top-k` 过滤
- **训练循环**
  - 用于蒸馏，第 2 处
- 词元概率
- 文本生成
  - 用于……的 LLM
  - 为……添加温度缩放
  - 硬件需求与建议，第 2 处
  - 加载预训练模型，第 2 处
  - 为 LLM 准备输入文本，第 2 处
  - 顺序式 LLM 文本生成过程，第 2 处，第 3 处，第 4 处
  - 搭建编码环境，第 2 处
- **训练推理模型**
  - 计算奖励
  - 通过优势从轨迹中准备学习信号
  - 使用序列对数概率对轨迹评分，第 2 处
  - `torch.softmax` 函数，第 2 处，第 3 处
  - `torch.log` 函数
- **跟踪性能指标**
  - 优势跟踪
  - 计算评估准确率，第 2 处
  - 熵跟踪，第 2 处
  - 执行训练运行，第 2 处
  - 检查训练运行，第 2 处，第 3 处，第 4 处
  - 绘制额外指标
- `train_distillation` 函数
- `train_rlvr_grpo()` 函数
- **训练损失**
  - 计算，第 2 处
- 分词器，第 2 处
- 词元对数概率
- 温度缩放，第 2 处
- **使用……训练推理模型**
  - 计算奖励
  - 实现 GRPO 训练循环，第 2 处
  - 加载 MATH 训练子集，第 2 处
  - 加载预训练模型
  - 通过 GRPO 损失更新策略
  - 通过优势从轨迹中准备学习信号
  - 将一切整合到单个 GRPO 函数中，第 2 处
  - 采样轨迹，第 2 处
  - 使用序列对数概率对轨迹评分，第 2 处
  - `torch.exp()` 函数
  - `top-p` 过滤器
  - 训练轮数，第 2 处
  - `torch.log_softmax()` 函数，第 2 处，第 3 处
  - `t_idx` 索引张量
  - `top-p` 过滤，第 2 处
  - `torch.multinomial` 函数，第 2 处
- **`top-p` 词元**
  - 选择……的子集
- `torch.prod` 函数
- `torch.multinomial()` 函数
- 测试时扩展
- **使用 RLVR 训练**
  - 加载并评估已保存的模型检查点，第 2 处
  - 温度设置
- **使用强化学习训练推理模型**
  - 将一切整合到单个 GRPO 函数中，第 2 处
- 词元，第 2 处
- 思考词元，第 2 处
- `torch` 库
- **使用 KL 损失项训练**
  - KL 方向与轨迹采样
  - 使用 KL 项提高训练稳定性

## U

- 扩展张量维度
- **使用 GRPO**
  - 通过厨师类比理解 GRPO 的高层直觉
  - GRPO 高层流程
- `uv` Python 包

## V

- 验证损失
- 可验证奖励
- **验证损失**
  - 计算，第 2 处
- 验证数学等价性，第 2 处

## W

- **使用强化学习**
  - 加载 MATH 训练子集，第 2 处
  - 采样轨迹，第 2 处
- **使用预训练 LLM**
  - 通过 KV 缓存加速推理，第 2 处
  - 通过 `PyTorch` 模型编译加速推理，第 2 处
