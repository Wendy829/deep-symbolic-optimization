# 独立RNN符号回归实现

## 项目简介

这是一个**完全独立**的PyTorch实现，用于采样和训练符号数学表达式。本实现不依赖于原始DSO框架，所有模块都是独立开发的。

**核心特性:**
- ✅ 纯PyTorch实现，无外部DSO依赖
- ✅ 基于RNN的自回归表达式生成
- ✅ 层次先验约束确保有效表达式
- ✅ 策略梯度训练（REINFORCE算法）
- ✅ 完整的中英文注释
- ✅ 易于理解和扩展

---

## 目录

1. [什么是符号回归？](#什么是符号回归)
2. [RNN如何生成表达式？](#rnn如何生成表达式)
3. [项目结构](#项目结构)
4. [快速开始](#快速开始)
5. [模块详解](#模块详解)
6. [训练流程](#训练流程)
7. [使用示例](#使用示例)
8. [常见问题](#常见问题)

---

## 什么是符号回归？

**符号回归（Symbolic Regression）** 是一种从数据中发现数学表达式的机器学习方法。

### 与传统回归的区别

**传统回归（如线性回归）:**
- 假设固定的函数形式，如 `y = ax + b`
- 只优化参数 `a` 和 `b`
- 如果真实关系不是线性的，效果很差

**符号回归:**
- 不假设函数形式
- 搜索整个表达式空间
- 可以发现复杂的数学关系

### 例子

给定数据点:
```
X = [1, 2, 3, 4]
y = [3, 7, 13, 21]
```

符号回归可能发现: `y = x² + 2x`

这比简单的线性拟合 `y = 5x - 2` 更准确！

---

## RNN如何生成表达式？

### 核心思想: 序列到序列

表达式可以表示为标记（token）序列。RNN学习生成这些序列。

### 表达式的树形表示

数学表达式本质上是树结构:

```
表达式: x + sin(y)

树形结构:
    +
   / \
  x  sin
      |
      y
```

### 前序遍历

我们将树转换为序列（前序遍历）:

```
前序遍历: [add, x, sin, y]
```

这样，表达式就变成了序列生成问题！

### 自回归生成

RNN逐步生成序列，每一步的输出依赖于前面所有步骤:

```
步骤1: RNN → 'add'     序列: [add]
步骤2: RNN → 'x'       序列: [add, x]
步骤3: RNN → 'sin'     序列: [add, x, sin]
步骤4: RNN → 'y'       序列: [add, x, sin, y] ✓ 完成
```

### 层次约束

为了确保生成有效的表达式，我们使用**层次先验**:

```python
# 例子: 在构建 sin( 之后
当前状态: [sin, ...]
允许的下一步: x, y (函数的参数)
禁止的下一步: +, -, * (运算符不能作为参数)
```

这大大减少了搜索空间，只生成有效的表达式！

---

## 项目结构

```
standalone_rnn/
│
├── __init__.py                      # 包初始化
├── token_library.py                 # 标记库：定义数学运算
├── expression_tree.py               # 表达式树：表示和执行表达式
├── prior.py                         # 先验约束：确保有效表达式
├── state_manager.py                 # 状态管理：编码观测特征
├── rnn_policy.py                    # RNN策略：核心采样网络
├── trainer.py                       # 训练器：策略梯度训练
├── example_symbolic_regression.py   # 完整示例
└── README_CN.md                     # 本文档
```

### 模块依赖关系

```
token_library → expression_tree
      ↓              ↓
    prior ←    state_manager
      ↓              ↓
      └──> rnn_policy ←┘
              ↓
          trainer
```

---

## 快速开始

### 安装依赖

```bash
pip install torch numpy matplotlib
```

### 运行示例

```bash
cd standalone_rnn
python example_symbolic_regression.py
```

### 10行代码使用

```python
from standalone_rnn import *

# 1. 创建组件
lib = create_default_library(n_input_vars=2)
prior = HierarchicalPrior(lib)
state_mgr = StateManager(lib)
policy = RNNPolicy(lib, prior, state_mgr)

# 2. 采样表达式
actions, obs, probs = policy.sample(batch_size=10)

# 3. 构建和评估
expr = Expression(actions[0], lib)
print(expr)  # 输出: add(x1, mul(x2, x2))
```

---

## 模块详解

### 1. token_library.py - 标记库

**作用:** 定义数学表达式的构建块

**核心类:**
- `Token`: 单个标记（运算符、函数、变量）
- `TokenLibrary`: 管理所有标记

**标记类型:**
```python
# 二元运算符 (arity=2)
add, sub, mul, div

# 一元函数 (arity=1)
sin, cos, exp, log, sqrt, square

# 变量 (arity=0)
x1, x2, ...
```

**例子:**
```python
lib = create_default_library(n_input_vars=2)
print(f"总标记数: {lib.n_tokens}")
print(f"函数标记: {lib.function_tokens}")
print(f"终端标记: {lib.terminal_tokens}")
```

### 2. expression_tree.py - 表达式树

**作用:** 表示和执行符号表达式

**核心类:**
- `Expression`: 符号表达式（存储为前序遍历）
- `ExpressionBuilder`: 构建表达式的辅助类

**关键方法:**
```python
# 标记索引: 0-9是函数, 10-11是x1和x2
# Token indices: 0-9 are functions, 10-11 are x1 and x2
expr = Expression([0, 10, 4, 11], lib)  # [add, x1, sin, x2]
y = expr.evaluate(X)                    # 在数据上评估
print(expr.to_string())                 # "add(x1, sin(x2))"
print(expr.complexity())                # 5.5
```

**实现细节:**
- 使用栈进行后序评估
- 自动完成未完成的表达式
- 安全的数值计算（避免溢出）

### 3. prior.py - 层次先验

**作用:** 约束表达式生成，确保有效性

**核心类:**
- `HierarchicalPrior`: 层次约束管理

**工作原理:**

```python
# 悬空节点追踪
开始: dangling = 1        # 需要根节点
选择 add: dangling = 2     # add需要2个子节点
选择 x1: dangling = 1      # x1是终端，减少1
选择 x2: dangling = 0      # 完成!
```

**关键方法:**
```python
prior = HierarchicalPrior(lib, max_length=15)

# 计算有效标记掩码
mask = prior.compute_prior(tokens_so_far, current_length)

# 检查表达式是否完成
is_done = prior.is_complete(tokens)
```

### 4. state_manager.py - 状态管理

**作用:** 为RNN编码上下文特征

**观测特征:**
1. **动作历史** (one-hot): 最后选择的标记
2. **父标记** (one-hot): 当前正在构建的函数
3. **兄弟标记** (one-hot): 刚添加的标记
4. **悬空计数** (标量): 还需要多少个标记

**观测向量结构:**
```
[action (n_tokens) | parent (n_tokens) | sibling (n_tokens) | dangling (1)]
```

**为什么需要这些特征？**

不仅要知道选了什么，还要知道**在哪里**选的！

```python
# 例子: 构建 add(x1, sin(?))
当前位置: sin的参数
父标记: sin
兄弟标记: x1
悬空: 1

这些信息帮助RNN知道现在应该选择什么类型的标记
```

### 5. rnn_policy.py - RNN策略

**作用:** 核心采样网络

**架构:**
```
输入: 观测特征
  ↓
LSTM/GRU: 捕获序列依赖
  ↓
线性层: 映射到动作空间
  ↓
Softmax: 生成概率分布
  ↓
输出: 下一个标记的概率
```

**关键方法:**

```python
policy = RNNPolicy(
    library=lib,
    prior=prior,
    state_manager=state_mgr,
    hidden_size=64,
    num_layers=1,
    cell_type='lstm'
)

# 采样表达式
actions, obs, probs = policy.sample(batch_size=10)

# 计算对数概率（用于训练）
log_probs = policy.compute_log_probs(actions, obs)

# 计算熵（鼓励探索）
entropy = policy.compute_entropy(obs)
```

**超参数说明:**
- `hidden_size`: RNN隐藏层大小（32-128通常够用）
- `num_layers`: RNN层数（1-2层通常够用）
- `cell_type`: 'lstm' 或 'gru'（LSTM通常更稳定）

### 6. trainer.py - 训练器

**作用:** 使用策略梯度训练RNN

**算法: REINFORCE**

```python
for iteration in range(num_iterations):
    # 1. 采样轨迹
    actions, obs, probs = policy.sample(batch_size)
    
    # 2. 评估奖励
    rewards = [reward_fn(expr) for expr in expressions]
    
    # 3. 计算策略梯度
    advantages = rewards - baseline
    loss = -mean(log_prob * advantage)
    
    # 4. 更新参数
    optimizer.step()
```

**奖励函数设计:**

```python
def reward_function(expr):
    # 1. 拟合质量
    y_pred = expr.evaluate(X)
    mse = mean((y - y_pred)**2)
    
    # 2. 复杂度惩罚
    complexity_penalty = 0.01 * expr.complexity()
    
    # 3. 总奖励
    return -mse - complexity_penalty
```

**训练技巧:**
1. **基线**: 使用均值作为基线减少方差
2. **熵正则化**: 鼓励探索，避免过早收敛
3. **梯度裁剪**: 防止梯度爆炸

---

## 训练流程

### 完整训练循环

```python
# 步骤1: 准备数据
X, y = generate_data()

# 步骤2: 创建组件
lib = create_default_library(n_input_vars=2)
prior = HierarchicalPrior(lib)
state_mgr = StateManager(lib)
policy = RNNPolicy(lib, prior, state_mgr)

# 步骤3: 定义奖励
def reward_fn(expr):
    y_pred = expr.evaluate(X)
    mse = np.mean((y - y_pred)**2)
    return -mse - 0.01 * expr.complexity()

# 步骤4: 创建训练器
trainer = PolicyGradientTrainer(
    policy=policy,
    reward_function=reward_fn,
    learning_rate=0.001,
    entropy_coef=0.01
)

# 步骤5: 训练
trainer.train(num_iterations=100, batch_size=20)

# 步骤6: 评估
best_exprs = trainer.sample_best_expressions(num_samples=100)
print(f"最佳表达式: {best_exprs[0][0]}")
```

### 训练进度示例

```
迭代 10/100
  平均奖励: -12.5432
  最大奖励: -8.2341
  最佳奖励: -8.2341
  策略损失: 2.3451
  熵: 1.8765

迭代 20/100
  平均奖励: -9.8765
  最大奖励: -5.6789
  最佳奖励: -5.6789
  策略损失: 1.9876
  熵: 1.6543

...

训练完成!
最佳表达式: add(square(x1), x2)
MSE: 0.0234
```

---

## 使用示例

### 示例1: 简单多项式

**目标:** 学习 `y = x²`

```python
from standalone_rnn import *
import numpy as np

# 生成数据
X = np.random.uniform(-2, 2, size=(100, 1))
y = X[:, 0] ** 2

# 创建组件
lib = create_default_library(n_input_vars=1)
prior = HierarchicalPrior(lib, max_length=10)
state_mgr = StateManager(lib, max_length=10)
policy = RNNPolicy(lib, prior, state_mgr, hidden_size=32)

# 定义奖励
def reward_fn(expr):
    y_pred = expr.evaluate(X)
    return -np.mean((y - y_pred)**2) - 0.01 * expr.complexity()

# 训练
trainer = PolicyGradientTrainer(policy, reward_fn)
trainer.train(num_iterations=50, batch_size=20)

# 结果
best = trainer.sample_best_expressions(10)
print(f"最佳表达式: {best[0][0]}")
```

### 示例2: 三角函数

**目标:** 学习 `y = sin(x)`

```python
X = np.random.uniform(-np.pi, np.pi, size=(100, 1))
y = np.sin(X[:, 0])

# ... 相同的设置 ...

# 训练后可能学到:
# 最佳表达式: sin(x1)
```

### 示例3: 组合函数

**目标:** 学习 `y = x1² + sin(x2)`

```python
X = np.random.uniform(-2, 2, size=(100, 2))
y = X[:, 0]**2 + np.sin(X[:, 1])

lib = create_default_library(n_input_vars=2)
# ... 其余设置 ...

# 训练后可能学到:
# 最佳表达式: add(square(x1), sin(x2))
```

---

## 常见问题

### Q1: 为什么训练这么慢？

**A:** 符号回归本质上是一个困难的搜索问题。改进方法:

1. **增加批次大小**: 更多样本 → 更好的梯度估计
2. **调整学习率**: 尝试 0.0001 到 0.01
3. **使用GPU**: 设置 `device='cuda'`
4. **减小搜索空间**: 限制 `max_length` 或减少标记

### Q2: 表达式不收敛到目标函数？

**A:** 这是正常的！符号回归寻找**等价**表达式，不一定完全相同。

```python
目标: y = 2x
学到: y = x + x  # 数学上等价!
```

如果完全不匹配:
1. 检查奖励函数是否正确
2. 增加训练迭代次数
3. 调整熵系数（更多探索）
4. 检查数据是否有噪声

### Q3: 生成的表达式太复杂？

**A:** 增加复杂度惩罚:

```python
def reward_fn(expr):
    mse = ...
    complexity = expr.complexity()
    return -mse - 0.1 * complexity  # 增大系数
```

### Q4: 如何添加新的运算符？

**A:** 在 `token_library.py` 中添加:

```python
library.add_token(Token(
    name='pow',
    arity=2,
    function=lambda a, b: np.power(a, np.clip(b, -5, 5)),
    complexity=2.0
))
```

### Q5: 可以用于多输出吗？

**A:** 当前实现支持单输出。对于多输出，需要:
1. 为每个输出训练单独的模型
2. 或修改奖励函数支持向量输出

### Q6: 如何保存和加载模型？

**A:** 使用PyTorch标准方法:

```python
# 保存
torch.save(policy.state_dict(), 'policy.pth')

# 加载
policy.load_state_dict(torch.load('policy.pth'))
```

### Q7: 如何可视化训练过程？

**A:** 使用训练历史:

```python
import matplotlib.pyplot as plt

plt.plot(trainer.train_history['rewards'])
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.show()
```

### Q8: 内存占用太大？

**A:** 优化方法:
1. 减小 `batch_size`
2. 减小 `hidden_size`
3. 减小 `max_length`
4. 使用梯度累积

```python
# 梯度累积示例
for i in range(num_mini_batches):
    loss = train_step(small_batch_size)
    loss = loss / num_mini_batches
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

---

## 高级话题

### 自定义先验约束

```python
class CustomPrior(HierarchicalPrior):
    def compute_prior(self, tokens_so_far, current_length):
        prior = super().compute_prior(tokens_so_far, current_length)
        
        # 例子: 禁止连续使用同一运算符
        if current_length > 0:
            last_token = tokens_so_far[0, current_length-1]
            prior[0, last_token] = 0.0
        
        return prior
```

### 自定义奖励函数

```python
def multi_objective_reward(expr):
    # 目标1: 拟合质量
    fit = -mse(expr, X, y)
    
    # 目标2: 简洁性
    simplicity = -expr.complexity()
    
    # 目标3: 可解释性（偏好常见运算）
    interpretability = count_common_ops(expr) / len(expr.tokens)
    
    # 加权组合
    return 0.7*fit + 0.2*simplicity + 0.1*interpretability
```

### 迁移学习

```python
# 在简单任务上预训练
simple_trainer.train(num_iterations=100, batch_size=20)

# 在复杂任务上微调
complex_trainer = PolicyGradientTrainer(
    policy=policy,  # 使用相同的策略
    reward_function=complex_reward_fn,
    learning_rate=0.0001  # 更小的学习率
)
complex_trainer.train(num_iterations=50, batch_size=20)
```

---

## 与原始DSO的对比

| 特性 | 原始DSO | 本实现 |
|-----|--------|-------|
| 依赖性 | 依赖DSO框架 | 完全独立 |
| 代码量 | 复杂，多文件 | 简洁，约2000行 |
| 易读性 | 需要理解整个框架 | 每个模块独立可读 |
| 文档 | 英文 | 双语（中英文） |
| 注释 | 部分 | 详尽（>30%） |
| 可扩展性 | 需要了解DSO | 直观，易于修改 |
| 学习曲线 | 陡峭 | 平缓 |

---

## 参考资源

### 论文

1. **Deep Symbolic Regression** (ICLR 2021)
   - 原始DSO论文
   - https://openreview.net/forum?id=m5Qsh0kBQG

2. **REINFORCE Algorithm**
   - Williams, 1992
   - 策略梯度的经典算法

### 代码

1. **原始DSO仓库**
   - https://github.com/brendenpetersen/deep-symbolic-optimization

2. **本实现**
   - 位于 `standalone_rnn/` 目录

### 其他资源

- 符号回归综述: [SRBench](https://cavalab.org/srbench/)
- PyTorch文档: https://pytorch.org/docs/
- 策略梯度教程: [Spinning Up in Deep RL](https://spinningup.openai.com/)

---

## 贡献指南

欢迎贡献！请遵循以下步骤:

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

**贡献方向:**
- 添加新的运算符
- 优化训练算法
- 改进文档
- 添加更多示例
- 性能优化

---

## 许可证

本项目遵循主仓库的许可证。

---

## 致谢

- 感谢原始DSO框架的作者
- 感谢PyTorch团队
- 感谢所有贡献者

---

## 联系方式

如有问题或建议，请:
- 开启 GitHub Issue
- 或发送邮件

---

**祝符号回归愉快！ Happy Symbolic Regression! 🎉**
