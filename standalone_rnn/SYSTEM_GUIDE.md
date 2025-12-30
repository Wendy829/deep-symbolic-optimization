# RNN符号表达式生成系统 - 通俗易懂指南
# RNN Symbolic Expression Generation System - Easy-to-Understand Guide

## 目录 / Table of Contents

1. [系统概述 / System Overview](#系统概述--system-overview)
2. [各个文件的作用 / Purpose of Each File](#各个文件的作用--purpose-of-each-file)
3. [RNN生成表达式的过程 / How RNN Generates Expressions](#rnn生成表达式的过程--how-rnn-generates-expressions)
4. [训练方式 / Training Method](#训练方式--training-method)
5. [完整示例 / Complete Example](#完整示例--complete-example)

---

## 系统概述 / System Overview

### 这个系统是做什么的？/ What Does This System Do?

这个系统使用**循环神经网络 (RNN)** 来**自动生成数学表达式**，用于符号回归任务。

This system uses a **Recurrent Neural Network (RNN)** to **automatically generate mathematical expressions** for symbolic regression tasks.

**举例 / Example:**
- 输入数据 / Input data: `x1 = [1, 2, 3]`, `x2 = [2, 4, 6]`, `y = [3, 6, 9]`
- 系统自动发现 / System automatically discovers: `y = x1 + x2`

### 为什么用RNN？/ Why Use RNN?

传统符号回归使用遗传算法，而RNN的优势是：
1. **可以学习**：通过梯度下降优化，而不是随机搜索
2. **采样高效**：学会生成"好"的表达式
3. **可微分**：使用强化学习训练

Traditional symbolic regression uses genetic algorithms, while RNN advantages are:
1. **Learnable**: Optimizes through gradient descent, not random search
2. **Efficient sampling**: Learns to generate "good" expressions
3. **Differentiable**: Trains using reinforcement learning

---

## 各个文件的作用 / Purpose of Each File

### 核心模块 / Core Modules

#### 1. **token_library.py** - Token库 / Token Library
**作用 / Purpose:** 定义数学符号的"词汇表"

**比喻 / Analogy:** 就像汉语拼音表，定义了所有可用的"字"

**包含内容 / Contains:**
- **运算符 / Operators**: `+, -, *, /` (加减乘除)
- **函数 / Functions**: `sin, cos, exp, log, sqrt` (三角函数、指数、对数、开方)
- **变量 / Variables**: `x1, x2, ...` (输入变量)

**示例 / Example:**
```python
# Token索引 / Token indices:
# 0: add (+)     5: cos
# 1: sub (-)     6: exp
# 2: mul (*)     7: log
# 3: div (/)     8: sqrt
# 4: sin         9: square
# 10: x1         11: x2
```

**为什么重要 / Why Important:** RNN输出的是token的索引（数字），需要这个库来解释它们的含义。

---

#### 2. **expression_tree.py** - 表达式树 / Expression Tree
**作用 / Purpose:** 将token序列转换为可计算的表达式

**比喻 / Analogy:** 就像把拼音"jia fa"转换成实际的词"加法"

**核心功能 / Core Functions:**

**a) 前序遍历存储 / Pre-order Traversal Storage:**
```python
# 表达式: add(x1, mul(x2, 3))
# 前序遍历: [add, x1, mul, x2, 3]
#
# 树结构:      add
#            /     \
#          x1      mul
#                 /   \
#               x2     3
```

**b) 递归求值 / Recursive Evaluation:**
```python
# 如何计算 add(x1, mul(x2, 3))
# 1. 读取 'add' -> 需要2个子节点
# 2. 读取 'x1' -> 获取x1的值
# 3. 读取 'mul' -> 需要2个子节点
# 4. 读取 'x2' -> 获取x2的值
# 5. 读取 '3' -> 常数3
# 6. 计算 mul(x2, 3)
# 7. 计算 add(x1, result)
```

**为什么重要 / Why Important:** RNN生成的是token序列，这个模块让它们变成可执行的数学表达式。

---

#### 3. **prior.py** - 先验约束 / Prior Constraints
**作用 / Purpose:** 确保生成的表达式**语法正确**

**比喻 / Analogy:** 就像语法检查器，防止你写出"加加乘"这样无意义的表达式

**核心概念: Dangling (待填充节点) / Core Concept: Dangling Nodes**

Dangling表示"还需要多少个子节点才能完成表达式"

Dangling represents "how many more child nodes are needed to complete the expression"

**示例 / Example:**
```python
# 构建过程 / Construction Process:
dangling = 1  # 开始需要填充根节点 / Start: need to fill root

# Step 1: 添加 'add' (需要2个参数)
dangling = dangling - 1 + 2 = 2  # 填充1个节点，添加2个新的待填充节点

# Step 2: 添加 'x1' (叶子节点)
dangling = dangling - 1 = 1  # 填充1个节点，不添加新的

# Step 3: 添加 'x2' (叶子节点)
dangling = dangling - 1 = 0  # 完成! / Complete!

# 最终表达式: add(x1, x2) ✓
```

**约束规则 / Constraint Rules:**
- 如果 `dangling = 0`：只能结束，不能再添加token
- 如果 `dangling > 0`：可以添加任何有效token
- 如果 `dangling = 1`：不能添加需要多个参数的函数（否则表达式会太长）

**为什么重要 / Why Important:** 防止RNN生成无效表达式，节省计算时间。

---

#### 4. **state_manager.py** - 状态管理器 / State Manager
**作用 / Purpose:** 将表达式的当前状态编码为RNN的输入特征

**比喻 / Analogy:** 就像GPS告诉你"你在哪里，还需要走多远"

**观察特征 / Observation Features:**
```python
# 对于正在构建的表达式 [add, x1, ...]
observation = [
    parent_token,        # 父节点是什么: add (token 0)
    sibling_token,       # 兄弟节点是什么: x1 (token 10)
    dangling,            # 还需要多少节点: 1
    # ... 其他特征 / other features
]
```

**为什么重要 / Why Important:** RNN需要知道"现在在哪里"才能决定"下一步生成什么token"。

---

#### 5. **rnn_policy.py** - RNN策略网络 / RNN Policy Network
**作用 / Purpose:** 这是**核心**！使用LSTM/GRU网络逐步生成表达式

**比喻 / Analogy:** 就像一个会写诗的人，一个字一个字地写，每次根据前面的内容决定下一个字。

**网络结构 / Network Architecture:**
```
输入特征 / Input features (observation)
    ↓
嵌入层 / Embedding layer (把token转成向量)
    ↓
LSTM/GRU层 / LSTM/GRU layers (记忆上下文)
    ↓
全连接层 / Fully connected layer
    ↓
Softmax (输出每个token的概率)
    ↓
采样 / Sampling (选择一个token)
```

**生成过程 / Generation Process (详见下一节)**

**为什么重要 / Why Important:** 这是实际"创造"表达式的模块，包含了所有可学习的参数。

---

#### 6. **trainer.py** - 训练器 / Trainer
**作用 / Purpose:** 使用**强化学习**训练RNN，让它学会生成好的表达式

**比喻 / Analogy:** 就像训练一只狗，做对了给奖励，做错了惩罚（但不直接告诉它哪里错了）

**训练算法: REINFORCE (策略梯度) / Training Algorithm: REINFORCE (Policy Gradient)**

**为什么重要 / Why Important:** 没有训练器，RNN只会随机生成表达式。训练后，它学会生成"拟合数据好"的表达式。

---

### 辅助文件 / Auxiliary Files

#### 7. **example_symbolic_regression.py** - 完整示例
**作用:** 演示如何解决一个完整的符号回归问题（如发现 `y = x1^2 + x2`）

#### 8. **quickstart.py** - 快速入门
**作用:** 5分钟快速演示，展示最基本用法

#### 9. **test_sampling.py** - 采样测试
**作用:** 测试RNN能否正确采样表达式（不涉及训练）

#### 10. **test_training.py** - 训练测试
**作用:** 测试完整的训练流程是否正常工作

---

## RNN生成表达式的过程 / How RNN Generates Expressions

### 逐步演示 / Step-by-Step Demonstration

假设我们想生成表达式 `add(x1, mul(x2, x2))` 即 `x1 + x2*x2`

Suppose we want to generate the expression `add(x1, mul(x2, x2))`, i.e., `x1 + x2*x2`

#### **初始化 / Initialization**
```python
dangling = 1            # 需要填充根节点 / Need to fill root node
expression = []         # 空表达式 / Empty expression
hidden_state = None     # RNN初始隐藏状态 / Initial RNN hidden state
```

---

#### **第1步 / Step 1: 选择根节点 / Select Root Node**

**输入到RNN / Input to RNN:**
```python
observation = [
    parent: None,       # 没有父节点 / No parent
    sibling: None,      # 没有兄弟 / No sibling
    dangling: 1         # 需要1个节点 / Need 1 node
]
```

**RNN输出 / RNN Output:**
```python
# 对每个token计算概率 / Probability for each token
probabilities = [
    P(add)   = 0.35,    # ← 选这个 / Select this
    P(sub)   = 0.10,
    P(mul)   = 0.20,
    P(sin)   = 0.05,
    P(x1)    = 0.15,
    P(x2)    = 0.15
]
```

**先验约束应用 / Apply Prior Constraints:**
```python
# dangling=1时，可以选择任何token
# When dangling=1, can select any token
# (无需调整概率) / (No probability adjustment needed)
```

**采样 / Sampling:**
```python
selected_token = sample(probabilities) = 'add'  # 按概率采样 / Sample by probability
```

**更新状态 / Update State:**
```python
expression = [add]                    # 添加到表达式 / Add to expression
dangling = 1 - 1 + 2 = 2             # add需要2个参数 / add needs 2 arguments
hidden_state = updated_by_RNN        # RNN更新隐藏状态 / RNN updates hidden state
```

---

#### **第2步 / Step 2: 填充第一个参数 / Fill First Argument**

**输入到RNN / Input to RNN:**
```python
observation = [
    parent: add,        # 父节点是add / Parent is add
    sibling: None,      # 第一个子节点，无兄弟 / First child, no sibling
    dangling: 2         # 还需2个节点 / Need 2 more nodes
]
```

**RNN输出 / RNN Output:**
```python
probabilities = [
    P(add)   = 0.05,
    P(x1)    = 0.60,    # ← 选这个 / Select this
    P(x2)    = 0.30,
    ...
]
```

**采样 / Sampling:**
```python
selected_token = 'x1'
```

**更新状态 / Update State:**
```python
expression = [add, x1]
dangling = 2 - 1 = 1                 # x1是叶子，dangling减1 / x1 is leaf, dangling -1
```

---

#### **第3步 / Step 3: 填充第二个参数 / Fill Second Argument**

**输入到RNN / Input to RNN:**
```python
observation = [
    parent: add,        # 父节点是add / Parent is add
    sibling: x1,        # 兄弟节点是x1 / Sibling is x1
    dangling: 1         # 最后1个节点 / Last node needed
]
```

**RNN输出 / RNN Output:**
```python
probabilities = [
    P(mul)   = 0.50,    # ← 选这个 / Select this
    P(x1)    = 0.20,
    P(x2)    = 0.25,
    ...
]
```

**先验约束应用 / Apply Prior Constraints:**
```python
# dangling=1时，不能选arity≥2的函数（如add、mul）
# 但这里展示如果选了mul会发生什么...
# When dangling=1, cannot select functions with arity≥2 (like add, mul)
# But let's show what happens if mul is selected...

# 实际上，先验会把 P(add) 和 P(mul) 设为0
# Actually, prior would set P(add) and P(mul) to 0
# 我们这里假设选了mul（展示完整流程）
# We assume mul is selected here (to show full process)
```

**采样 / Sampling:**
```python
selected_token = 'mul'
```

**更新状态 / Update State:**
```python
expression = [add, x1, mul]
dangling = 1 - 1 + 2 = 2             # mul需要2个参数 / mul needs 2 arguments
```

---

#### **第4步 / Step 4: 填充mul的第一个参数**

**输入到RNN:**
```python
observation = [
    parent: mul,
    sibling: None,
    dangling: 2
]
```

**采样结果 / Sampling Result:**
```python
selected_token = 'x2'
expression = [add, x1, mul, x2]
dangling = 2 - 1 = 1
```

---

#### **第5步 / Step 5: 填充mul的第二个参数**

**输入到RNN:**
```python
observation = [
    parent: mul,
    sibling: x2,
    dangling: 1
]
```

**采样结果 / Sampling Result:**
```python
selected_token = 'x2'
expression = [add, x1, mul, x2, x2]
dangling = 1 - 1 = 0                 # 完成! / Complete!
```

---

#### **最终结果 / Final Result**

**前序遍历 / Pre-order Traversal:**
```python
tokens = [add, x1, mul, x2, x2]
```

**树结构 / Tree Structure:**
```
      add
     /   \
   x1     mul
         /   \
       x2     x2
```

**数学表达式 / Mathematical Expression:**
```
add(x1, mul(x2, x2))  →  x1 + x2*x2
```

---

### 关键要点 / Key Takeaways

1. **RNN是自回归的 / RNN is Autoregressive**
   - 每次只生成一个token / Generates one token at a time
   - 当前token依赖于之前所有token / Current token depends on all previous tokens

2. **先验确保有效性 / Prior Ensures Validity**
   - 每步都检查dangling / Checks dangling at each step
   - 屏蔽无效选择 / Masks invalid choices

3. **观察编码上下文 / Observation Encodes Context**
   - 告诉RNN"在哪里" / Tells RNN "where we are"
   - 包含父节点、兄弟节点、dangling等信息 / Includes parent, sibling, dangling, etc.

4. **采样是概率性的 / Sampling is Probabilistic**
   - 不是每次都选择最高概率的token / Doesn't always select highest probability token
   - 允许探索不同的表达式 / Allows exploring different expressions

---

## 训练方式 / Training Method

### 强化学习: REINFORCE算法 / Reinforcement Learning: REINFORCE Algorithm

#### 为什么用强化学习？/ Why Reinforcement Learning?

**问题 / Problem:** 我们没有"正确答案"的标签（不知道最佳表达式是什么）

We don't have labels for "correct answers" (don't know what the best expression is)

**解决方案 / Solution:** 通过"奖励信号"指导学习

Guide learning through "reward signals"

---

### 训练流程 / Training Flow

#### **Step 1: 采样一批表达式 / Sample a Batch of Expressions**

```python
# 使用当前RNN策略采样20个表达式
# Sample 20 expressions using current RNN policy
expressions = policy.sample(batch_size=20)

# 示例 / Example:
# expr_1: add(x1, x2)
# expr_2: mul(sin(x1), x2)
# expr_3: x1
# ... (20个表达式 / 20 expressions)
```

---

#### **Step 2: 计算每个表达式的奖励 / Compute Reward for Each Expression**

```python
rewards = []
for expr in expressions:
    # 在数据上评估表达式 / Evaluate expression on data
    y_pred = expr.evaluate(X)  # 预测值 / Predictions
    
    # 计算均方误差 / Compute mean squared error
    mse = mean((y - y_pred)^2)
    
    # 归一化 / Normalize
    nmse = mse / variance(y)
    
    # 转换为奖励（DSO风格：正值，越高越好）
    # Convert to reward (DSO style: positive, higher is better)
    reward = 1 / (1 + sqrt(nmse))
    
    rewards.append(reward)
```

**示例奖励 / Example Rewards:**
```python
expr_1: add(x1, x2)        → reward = 0.85  # 拟合很好 / Fits well
expr_2: mul(sin(x1), x2)   → reward = 0.20  # 拟合不好 / Fits poorly
expr_3: x1                 → reward = 0.15  # 拟合很差 / Fits very poorly
...
```

---

#### **Step 3: 计算策略梯度 / Compute Policy Gradient**

这是REINFORCE算法的核心！/ This is the core of REINFORCE!

**目标 / Objective:** 增加高奖励表达式的概率，减少低奖励表达式的概率

Increase probability of high-reward expressions, decrease probability of low-reward expressions

**数学公式 / Mathematical Formula:**
```
∇L = - (reward - baseline) * ∇log P(expression)
```

**通俗解释 / Intuitive Explanation:**

```python
for each expression:
    if reward > baseline:  # 这个表达式比平均好 / This expression is better than average
        # 增加生成这种表达式的概率 / Increase probability of generating this type
        gradient = negative  # 梯度下降会增加概率 / Gradient descent increases probability
    
    else:  # reward < baseline  # 这个表达式比平均差 / Worse than average
        # 减少生成这种表达式的概率 / Decrease probability
        gradient = positive  # 梯度下降会减少概率 / Gradient descent decreases probability
```

**代码实现 / Code Implementation:**
```python
# 1. 计算基线（减少方差）/ Compute baseline (reduce variance)
baseline = mean(rewards)

# 2. 计算优势 / Compute advantage
advantages = rewards - baseline  # [-0.15, 0.65, -0.85, ...]

# 3. 计算对数概率 / Compute log probabilities
log_probs = policy.compute_log_probs(expressions)

# 4. 计算损失 / Compute loss
policy_loss = -mean(advantages * log_probs)

# 5. 反向传播 / Backpropagation
policy_loss.backward()
optimizer.step()
```

---

#### **Step 4: 添加熵正则化 / Add Entropy Regularization**

**为什么需要？/ Why Needed?**

防止RNN过早收敛到单一表达式（鼓励探索）

Prevent RNN from converging too early to a single expression (encourage exploration)

**熵的含义 / Meaning of Entropy:**
- **高熵 / High Entropy**: 概率分布均匀，不确定性高，探索多
- **低熵 / Low Entropy**: 概率集中在少数token，不确定性低，利用多

**代码 / Code:**
```python
entropy = policy.compute_entropy(expressions)
total_loss = policy_loss - entropy_coef * entropy  # 鼓励高熵 / Encourage high entropy
```

---

#### **Step 5: 更新参数 / Update Parameters**

```python
# PyTorch自动处理 / PyTorch handles automatically
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

RNN的权重更新后，下次采样时会倾向于生成高奖励的表达式！

After RNN weights are updated, next sampling will tend to generate high-reward expressions!

---

### 训练循环 / Training Loop

```python
for iteration in range(num_iterations):
    # 1. 采样 / Sample
    expressions = policy.sample(batch_size=20)
    
    # 2. 评估 / Evaluate
    rewards = [compute_reward(expr, X, y) for expr in expressions]
    
    # 3. 计算梯度 / Compute gradients
    loss = compute_policy_gradient(expressions, rewards)
    
    # 4. 更新参数 / Update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 5. 监控进度 / Monitor progress
    print(f"Iteration {iteration}: Mean Reward = {mean(rewards):.4f}")
```

---

### 训练过程示例 / Training Process Example

```
迭代 1 / Iteration 1:
  采样表达式 / Sampled expressions: 多数是随机的 / Mostly random
  平均奖励 / Mean reward: 0.15
  
迭代 10 / Iteration 10:
  采样表达式 / Sampled: 开始出现简单有效的 / Simple effective ones appear
  平均奖励 / Mean reward: 0.35
  
迭代 50 / Iteration 50:
  采样表达式 / Sampled: 大多数接近真实公式 / Most close to true formula
  平均奖励 / Mean reward: 0.75
  
迭代 100 / Iteration 100:
  采样表达式 / Sampled: 基本都是正确或相近的 / Mostly correct or similar
  平均奖励 / Mean reward: 0.90
```

---

## 完整示例 / Complete Example

### 问题 / Problem

**已知数据 / Given Data:**
```python
X = [[1, 2], [2, 4], [3, 6]]  # x1, x2
y = [3, 6, 9]                  # 目标 / Target
```

**目标 / Goal:** 发现公式 `y = x1 + x2`

**Objective:** Discover formula `y = x1 + x2`

---

### 代码 / Code

```python
# ============================================================
# 1. 准备组件 / Prepare Components
# ============================================================
from standalone_rnn import *
import numpy as np

# 创建token库 / Create token library
lib = create_default_library(n_input_vars=2)

# 创建RNN策略 / Create RNN policy
policy = RNNPolicy(
    library=lib,
    prior=HierarchicalPrior(lib),
    state_manager=StateManager(lib),
    hidden_size=32,
    num_layers=1
)

# ============================================================
# 2. 定义奖励函数 / Define Reward Function
# ============================================================
def reward_function(expression):
    """
    评估表达式质量 / Evaluate expression quality
    """
    try:
        # 在数据上求值 / Evaluate on data
        y_pred = expression.evaluate(X)
        
        # 计算NMSE / Compute NMSE
        mse = np.mean((y - y_pred)**2)
        nmse = mse / np.var(y)
        
        # 转换为奖励（DSO风格）/ Convert to reward (DSO style)
        reward = 1.0 / (1.0 + np.sqrt(nmse))
        
        # 复杂度惩罚 / Complexity penalty
        penalty = 0.01 * expression.complexity()
        
        return reward - penalty
    
    except:
        return 0.0  # 无效表达式 / Invalid expression

# ============================================================
# 3. 创建训练器 / Create Trainer
# ============================================================
trainer = PolicyGradientTrainer(
    policy=policy,
    reward_function=reward_function,
    learning_rate=0.001,
    entropy_coef=0.01
)

# ============================================================
# 4. 训练 / Train
# ============================================================
print("开始训练 / Starting training...")
trainer.train(
    num_iterations=50,
    batch_size=20
)

# ============================================================
# 5. 查看结果 / View Results
# ============================================================
print("\n最佳表达式 / Best expressions:")
best_expressions = trainer.sample_best_expressions(num_samples=100)

for i, (expr, reward) in enumerate(best_expressions[:5], 1):
    print(f"  {i}. {expr.to_string():30s} (奖励/reward: {reward:.4f})")
```

---

### 预期输出 / Expected Output

```
开始训练 / Starting training...

迭代 10/50: 平均奖励=0.25, 最佳奖励=0.45
Iteration 10/50: Mean Reward=0.25, Best Reward=0.45

迭代 20/50: 平均奖励=0.52, 最佳奖励=0.88
Iteration 20/50: Mean Reward=0.52, Best Reward=0.88

迭代 30/50: 平均奖励=0.75, 最佳奖励=0.95
Iteration 30/50: Mean Reward=0.75, Best Reward=0.95

迭代 40/50: 平均奖励=0.88, 最佳奖励=0.98
Iteration 40/50: Mean Reward=0.88, Best Reward=0.98

迭代 50/50: 平均奖励=0.92, 最佳奖励=0.99
Iteration 50/50: Mean Reward=0.92, Best Reward=0.99

训练完成 / Training complete!

最佳表达式 / Best expressions:
  1. add(x1, x2)                   (奖励/reward: 0.9900)  ← 完美! / Perfect!
  2. add(x2, x1)                   (奖励/reward: 0.9900)  ← 等价 / Equivalent
  3. sub(add(x1, x2), sub(x1, x1)) (奖励/reward: 0.9800)  ← 复杂但正确 / Complex but correct
  4. mul(add(x1, x2), x1)          (奖励/reward: 0.4500)  ← 不够好 / Not good enough
  5. x1                            (奖励/reward: 0.2000)  ← 太简单 / Too simple
```

---

## 总结 / Summary

### 系统工作流程 / System Workflow

```
1. 初始化 / Initialize
   ↓
2. RNN采样表达式 (逐token生成) / RNN samples expressions (token by token)
   ↓
3. 评估表达式 (计算奖励) / Evaluate expressions (compute rewards)
   ↓
4. 计算策略梯度 (REINFORCE) / Compute policy gradient (REINFORCE)
   ↓
5. 更新RNN参数 / Update RNN parameters
   ↓
6. 重复步骤2-5，直到收敛 / Repeat steps 2-5 until convergence
```

### 关键创新点 / Key Innovations

1. **自回归生成 / Autoregressive Generation**
   - 像写作一样逐token生成 / Generates token by token like writing
   - 使用RNN记忆上下文 / Uses RNN to remember context

2. **先验约束 / Prior Constraints**
   - 保证100%有效表达式 / Guarantees 100% valid expressions
   - 使用dangling追踪完成状态 / Uses dangling to track completion

3. **强化学习 / Reinforcement Learning**
   - 无需标签，只需奖励信号 / No labels needed, only reward signals
   - 通过策略梯度优化 / Optimizes through policy gradients

4. **与DSO对齐 / Aligned with DSO**
   - 使用相同的dangling计算 / Uses same dangling calculation
   - 使用相同的奖励约定 / Uses same reward convention
   - 完全兼容 / Fully compatible

---

### 进一步学习 / Further Learning

**运行示例脚本 / Run Example Scripts:**
```bash
python standalone_rnn/test_sampling.py    # 理解采样过程 / Understand sampling
python standalone_rnn/test_training.py    # 理解训练过程 / Understand training
python standalone_rnn/quickstart.py       # 快速演示 / Quick demo
```

**阅读文档 / Read Documentation:**
- `README_CN.md`: 中文详细教程 / Detailed Chinese tutorial
- `CHANGES_SUMMARY.md`: 修改历史 / Change history
- 代码注释 / Code comments: 所有函数都有详细说明 / All functions documented

---

## 常见问题 / FAQ

### Q1: 为什么用前序遍历存储表达式？
**A:** 因为它可以线性存储树结构，便于RNN顺序生成。

**A:** Because it can linearly store tree structure, convenient for RNN sequential generation.

### Q2: 奖励函数为什么用 1/(1+NRMSE) 而不是 -MSE？
**A:** 
- 正值更直观（越高越好）/ Positive values more intuitive (higher is better)
- NRMSE归一化后可比较不同数据集 / NRMSE normalized for comparing different datasets
- 与原始DSO对齐 / Aligned with original DSO

### Q3: 为什么需要熵正则化？
**A:** 
- 防止过早收敛 / Prevent premature convergence
- 鼓励探索多样表达式 / Encourage exploring diverse expressions
- 避免陷入局部最优 / Avoid local optima

### Q4: 训练需要多久？
**A:** 
- 简单问题: 30-50次迭代 / Simple problems: 30-50 iterations
- 复杂问题: 100-500次迭代 / Complex problems: 100-500 iterations
- 每次迭代: 几秒钟 / Each iteration: a few seconds

### Q5: 如何改进性能？
**A:**
- 增加隐藏层大小 / Increase hidden size
- 增加batch size / Increase batch size
- 调整学习率 / Tune learning rate
- 添加更多token / Add more tokens
- 改进奖励函数 / Improve reward function

---

**祝使用愉快！/ Happy coding!** 🎉
