# PyTorch RNN Policy Implementation Guide
# PyTorch RNN 策略实现指南

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English Documentation

### Overview

This document provides a comprehensive guide to the PyTorch implementation of the RNN-based policy for symbolic regression. The PyTorch version (`pytorch_rnn_policy.py`) is a faithful reproduction of the TensorFlow version (`rnn_policy.py`), maintaining all features, methods, structures, and behaviors.

### What is the RNN Policy?

The RNN Policy is a core component of the Deep Symbolic Optimization (DSO) framework. It uses a recurrent neural network to generate symbolic mathematical expressions by:

1. **Autoregressive Generation**: Sampling one token at a time, where each token depends on previously sampled tokens
2. **Hierarchical Constraints**: Using prior knowledge to guide sampling toward valid expressions
3. **Context-Aware Decisions**: Observing features like parent tokens, sibling tokens, and expression structure to make informed choices

### Key Concepts

#### 1. Expression Representation
Mathematical expressions are represented as trees in **pre-order traversal**:
- Example: `x + sin(y)` becomes `[add, x, sin, y]`
- The RNN learns to generate valid tree structures sequentially

#### 2. Autoregressive Sampling
- At each time step t, the RNN predicts a probability distribution over possible tokens
- The distribution is conditioned on all previous tokens (t-1, t-2, ..., 0)
- This allows the model to learn structural dependencies

#### 3. Prior Constraints
- **Hierarchical priors** mask invalid tokens at each step
- Example: After `sin(`, only valid arguments (not operators) can follow
- This dramatically reduces the search space

#### 4. State Features
The policy observes contextual information:
- **Parent token**: What function is currently being built
- **Sibling token**: What was just added
- **Dangling nodes**: How many more tokens are needed
- **Action history**: Previously selected tokens

### Architecture Comparison

#### TensorFlow Version (Original)
```python
class RNNPolicy(Policy):
    def __init__(self, sess, prior, state_manager, ...):
        # Uses TensorFlow 1.14 with tf.nn.raw_rnn
        # Builds static computation graph
        # Requires TensorFlow session management
```

#### PyTorch Version (New)
```python
class PyTorchRNNPolicy:
    def __init__(self, prior, state_manager, ...):
        # Uses PyTorch with native RNN modules
        # Dynamic computation graph
        # No session management needed
        # More Pythonic and easier to debug
```

### Feature Parity

| Feature | TensorFlow | PyTorch | Notes |
|---------|-----------|---------|-------|
| LSTM cells | ✅ | ✅ | Identical behavior |
| GRU cells | ✅ | ✅ | Identical behavior |
| Multi-layer RNN | ✅ | ✅ | Supports arbitrary depth |
| Linear output layer | ✅ | ✅ | Projects hidden states to actions |
| Prior constraints | ✅ | ✅ | Hierarchical masking |
| Action probability bounds | ✅ | ✅ | Ensures exploration |
| Novel sample generation | ✅ | ✅ | Avoids duplicate expressions |
| Entropy computation | ✅ | ✅ | For regularization |
| Negative log-likelihood | ✅ | ✅ | For policy gradient |
| State management | ✅ | ✅ | Observation encoding |

### Installation

The PyTorch implementation requires:
```bash
pip install torch>=1.7.0
```

Note: The original TensorFlow dependencies are still required for the rest of the DSO framework.

### Usage Examples

#### Basic Usage
```python
from dso.policy.pytorch_rnn_policy import PyTorchRNNPolicy
from dso.prior import make_prior
from dso.tf_state_manager import make_state_manager

# Initialize components
prior = make_prior(config["prior"])
state_manager = make_state_manager(config["state_manager"])

# Create PyTorch policy
policy = PyTorchRNNPolicy(
    prior=prior,
    state_manager=state_manager,
    max_length=30,
    cell='lstm',
    num_layers=1,
    num_units=32,
    device='cpu'  # or 'cuda' for GPU
)

# Sample expressions
actions, obs, priors = policy.sample(n=100)

# actions: (100, 30) - sampled token sequences
# obs: (100, obs_dim, 30) - observation sequences  
# priors: (100, 30, n_choices) - prior probabilities
```

#### Computing Probabilities
```python
from dso.memory import Batch

# Create a batch from sampled data
batch = Batch(
    actions=actions,
    obs=obs,
    priors=priors,
    lengths=lengths,
    rewards=rewards,
    on_policy=True
)

# Compute negative log-probabilities and entropy
neglogp, entropy = policy.compute_neglogp_and_entropy(batch)

# neglogp: (100,) - negative log P(sequence|policy)
# entropy: (100,) - policy entropy for each sequence
```

#### Training Integration
```python
import torch.optim as optim

# Get model parameters
params = policy.get_trainable_parameters()

# Create optimizer
optimizer = optim.Adam(params, lr=0.001)

# Training step
optimizer.zero_grad()

# Compute loss (e.g., policy gradient)
neglogp, entropy = policy.compute_neglogp_and_entropy(batch)
advantages = torch.FloatTensor(rewards - baseline)
loss = (torch.FloatTensor(neglogp) * advantages).mean()
loss -= 0.01 * torch.FloatTensor(entropy).mean()  # Entropy bonus

# Backward pass
loss.backward()
optimizer.step()
```

### Implementation Details

#### 1. LinearWrapper
Wraps an RNN cell and adds a linear output projection:
```python
class LinearWrapper(nn.Module):
    def __init__(self, cell, output_size):
        self.cell = cell  # LSTM/GRU
        self.output_projection = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden):
        output, hidden = self.cell(input_seq, hidden)
        logits = self.output_projection(output)
        return logits, hidden
```

#### 2. Multi-Layer RNN
Stacks multiple RNN layers:
```python
class MultiLayerRNN(nn.Module):
    def __init__(self, cell_type, input_size, num_layers, num_units):
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else num_units[i-1]
            if cell_type == 'lstm':
                layer = nn.LSTM(layer_input, num_units[i], num_layers=1)
            else:  # gru
                layer = nn.GRU(layer_input, num_units[i], num_layers=1)
            self.layers.append(layer)
```

#### 3. Sampling Algorithm
```python
def sample(self, n):
    # Initialize
    hidden = None
    obs = initial_obs
    prior = initial_prior
    
    # Generate sequence
    for t in range(max_length):
        # Encode observations
        input_features = self._get_network_input(obs)
        
        # Forward RNN
        logits, hidden = self.rnn(input_features, hidden)
        
        # Apply bounds and priors
        if action_prob_lowerbound > 0:
            logits = apply_lowerbound(logits)
        logits = logits + prior
        
        # Sample action
        probs = softmax(logits)
        action = sample_categorical(probs)
        
        # Update state
        obs, prior, done = task.get_next_obs(action, obs)
        
        if all(done):
            break
    
    return actions, obs, priors
```

#### 4. Novel Sample Generation
```python
def sample_novel(self, n):
    n_novel = 0
    attempts = 0
    
    while n_novel < n and attempts < max_attempts:
        # Sample batch
        actions, obs, priors = self.sample(n)
        
        # Check novelty
        for action in actions:
            key = finish_tokens(action).tostring()
            if key not in cache:
                collect_novel(action)
                n_novel += 1
        
        attempts += 1
    
    # Fill remaining with old samples if needed
    return combine_novel_and_old()
```

### Differences from TensorFlow

#### 1. Session Management
- **TensorFlow**: Requires `tf.Session()` and explicit `sess.run()`
- **PyTorch**: Direct computation with automatic differentiation

#### 2. Dynamic vs Static Graphs
- **TensorFlow**: Static graph built once, then executed
- **PyTorch**: Dynamic graph rebuilt each forward pass

#### 3. Debugging
- **TensorFlow**: Must use session or eager execution to inspect values
- **PyTorch**: Standard Python debugger works naturally

#### 4. Tensor Operations
- **TensorFlow**: `tf.concat()`, `tf.reduce_sum()`, etc.
- **PyTorch**: `torch.cat()`, `torch.sum()`, etc.

#### 5. RNN API
- **TensorFlow**: `tf.nn.raw_rnn()` for fine-grained control
- **PyTorch**: Standard `nn.LSTM()` / `nn.GRU()` modules

### Testing and Validation

To verify that the PyTorch implementation produces equivalent results:

```python
# Compare sampling behavior
tf_actions, tf_obs, tf_priors = tf_policy.sample(100)
pt_actions, pt_obs, pt_priors = pt_policy.sample(100)

# Actions should follow similar distributions
assert actions_are_similar(tf_actions, pt_actions)

# Probabilities should match (given same weights)
tf_probs = tf_policy.compute_probs(batch)
pt_probs = pt_policy.compute_probs(batch)
assert np.allclose(tf_probs, pt_probs, rtol=1e-3)
```

### Performance Considerations

1. **Memory**: PyTorch typically uses similar memory to TensorFlow
2. **Speed**: Performance is comparable for this architecture
3. **GPU**: Both support GPU acceleration efficiently
4. **Batching**: Both handle variable-length sequences well

### Limitations and Future Work

Current implementation:
- ✅ Core RNN functionality complete
- ✅ LSTM and GRU support
- ✅ Prior constraints
- ✅ Novel sampling
- ⚠️ Embeddings not fully implemented (uses one-hot encoding)
- ⚠️ Not integrated into main DSO training loop (standalone)

Future enhancements:
- Full embedding support for categorical inputs
- Integration with PyTorch-based policy optimizers
- Mixed precision training support
- Distributed training capabilities

### Troubleshooting

**Issue**: Dimensions mismatch in `_get_network_input`
- **Solution**: Check state_manager configuration matches library settings

**Issue**: Novel sampling returns too few samples
- **Solution**: Increase `max_attempts_at_novel_batch` parameter

**Issue**: Model not learning
- **Solution**: Verify learning rate, check gradient flow with `torch.autograd.grad`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use gradient accumulation

### References

1. Original DSO paper: [Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients](https://openreview.net/forum?id=m5Qsh0kBQG)
2. TensorFlow implementation: `dso/policy/rnn_policy.py`
3. PyTorch RNN documentation: https://pytorch.org/docs/stable/nn.html#rnn

---

<a name="chinese"></a>
## 中文文档

### 概述

本文档提供了用于符号回归的基于RNN策略的PyTorch实现的全面指南。PyTorch版本（`pytorch_rnn_policy.py`）是TensorFlow版本（`rnn_policy.py`）的忠实再现，保持所有特性、方法、结构和行为。

### 什么是RNN策略？

RNN策略是深度符号优化（DSO）框架的核心组件。它使用循环神经网络通过以下方式生成符号数学表达式：

1. **自回归生成**：一次采样一个标记，其中每个标记依赖于先前采样的标记
2. **层次约束**：使用先验知识引导采样朝向有效表达式
3. **上下文感知决策**：观察父标记、兄弟标记和表达式结构等特征以做出明智选择

### 关键概念

#### 1. 表达式表示
数学表达式表示为**前序遍历**的树：
- 示例：`x + sin(y)` 变为 `[add, x, sin, y]`
- RNN学习按顺序生成有效的树结构

#### 2. 自回归采样
- 在每个时间步t，RNN预测可能标记的概率分布
- 该分布以所有先前标记（t-1, t-2, ..., 0）为条件
- 这允许模型学习结构依赖关系

#### 3. 先验约束
- **层次先验**在每一步屏蔽无效标记
- 示例：在`sin(`之后，只能跟随有效参数（不是运算符）
- 这大大减少了搜索空间

#### 4. 状态特征
策略观察上下文信息：
- **父标记**：当前正在构建什么函数
- **兄弟标记**：刚刚添加了什么
- **悬空节点**：还需要多少个标记
- **动作历史**：先前选择的标记

### 架构比较

#### TensorFlow版本（原始）
```python
class RNNPolicy(Policy):
    def __init__(self, sess, prior, state_manager, ...):
        # 使用TensorFlow 1.14和tf.nn.raw_rnn
        # 构建静态计算图
        # 需要TensorFlow会话管理
```

#### PyTorch版本（新）
```python
class PyTorchRNNPolicy:
    def __init__(self, prior, state_manager, ...):
        # 使用PyTorch和原生RNN模块
        # 动态计算图
        # 不需要会话管理
        # 更符合Python习惯，更容易调试
```

### 特性对比

| 特性 | TensorFlow | PyTorch | 说明 |
|-----|-----------|---------|-----|
| LSTM单元 | ✅ | ✅ | 行为相同 |
| GRU单元 | ✅ | ✅ | 行为相同 |
| 多层RNN | ✅ | ✅ | 支持任意深度 |
| 线性输出层 | ✅ | ✅ | 将隐藏状态投影到动作 |
| 先验约束 | ✅ | ✅ | 层次屏蔽 |
| 动作概率界限 | ✅ | ✅ | 确保探索 |
| 新样本生成 | ✅ | ✅ | 避免重复表达式 |
| 熵计算 | ✅ | ✅ | 用于正则化 |
| 负对数似然 | ✅ | ✅ | 用于策略梯度 |
| 状态管理 | ✅ | ✅ | 观测编码 |

### 安装

PyTorch实现需要：
```bash
pip install torch>=1.7.0
```

注意：DSO框架的其余部分仍需要原始的TensorFlow依赖项。

### 使用示例

#### 基本使用
```python
from dso.policy.pytorch_rnn_policy import PyTorchRNNPolicy
from dso.prior import make_prior
from dso.tf_state_manager import make_state_manager

# 初始化组件
prior = make_prior(config["prior"])
state_manager = make_state_manager(config["state_manager"])

# 创建PyTorch策略
policy = PyTorchRNNPolicy(
    prior=prior,
    state_manager=state_manager,
    max_length=30,
    cell='lstm',
    num_layers=1,
    num_units=32,
    device='cpu'  # 或'cuda'用于GPU
)

# 采样表达式
actions, obs, priors = policy.sample(n=100)

# actions: (100, 30) - 采样的标记序列
# obs: (100, obs_dim, 30) - 观测序列
# priors: (100, 30, n_choices) - 先验概率
```

#### 计算概率
```python
from dso.memory import Batch

# 从采样数据创建批次
batch = Batch(
    actions=actions,
    obs=obs,
    priors=priors,
    lengths=lengths,
    rewards=rewards,
    on_policy=True
)

# 计算负对数概率和熵
neglogp, entropy = policy.compute_neglogp_and_entropy(batch)

# neglogp: (100,) - 负对数P(序列|策略)
# entropy: (100,) - 每个序列的策略熵
```

#### 训练集成
```python
import torch.optim as optim

# 获取模型参数
params = policy.get_trainable_parameters()

# 创建优化器
optimizer = optim.Adam(params, lr=0.001)

# 训练步骤
optimizer.zero_grad()

# 计算损失（例如，策略梯度）
neglogp, entropy = policy.compute_neglogp_and_entropy(batch)
advantages = torch.FloatTensor(rewards - baseline)
loss = (torch.FloatTensor(neglogp) * advantages).mean()
loss -= 0.01 * torch.FloatTensor(entropy).mean()  # 熵奖励

# 反向传播
loss.backward()
optimizer.step()
```

### 实现细节

#### 1. LinearWrapper（线性包装器）
包装RNN单元并添加线性输出投影：
```python
class LinearWrapper(nn.Module):
    def __init__(self, cell, output_size):
        self.cell = cell  # LSTM/GRU
        self.output_projection = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden):
        output, hidden = self.cell(input_seq, hidden)
        logits = self.output_projection(output)
        return logits, hidden
```

#### 2. 多层RNN
堆叠多个RNN层：
```python
class MultiLayerRNN(nn.Module):
    def __init__(self, cell_type, input_size, num_layers, num_units):
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else num_units[i-1]
            if cell_type == 'lstm':
                layer = nn.LSTM(layer_input, num_units[i], num_layers=1)
            else:  # gru
                layer = nn.GRU(layer_input, num_units[i], num_layers=1)
            self.layers.append(layer)
```

#### 3. 采样算法
```python
def sample(self, n):
    # 初始化
    hidden = None
    obs = initial_obs
    prior = initial_prior
    
    # 生成序列
    for t in range(max_length):
        # 编码观测
        input_features = self._get_network_input(obs)
        
        # RNN前向传播
        logits, hidden = self.rnn(input_features, hidden)
        
        # 应用边界和先验
        if action_prob_lowerbound > 0:
            logits = apply_lowerbound(logits)
        logits = logits + prior
        
        # 采样动作
        probs = softmax(logits)
        action = sample_categorical(probs)
        
        # 更新状态
        obs, prior, done = task.get_next_obs(action, obs)
        
        if all(done):
            break
    
    return actions, obs, priors
```

#### 4. 新样本生成
```python
def sample_novel(self, n):
    n_novel = 0
    attempts = 0
    
    while n_novel < n and attempts < max_attempts:
        # 采样批次
        actions, obs, priors = self.sample(n)
        
        # 检查新颖性
        for action in actions:
            key = finish_tokens(action).tostring()
            if key not in cache:
                collect_novel(action)
                n_novel += 1
        
        attempts += 1
    
    # 如果需要，用旧样本填充剩余部分
    return combine_novel_and_old()
```

### 与TensorFlow的差异

#### 1. 会话管理
- **TensorFlow**：需要`tf.Session()`和显式的`sess.run()`
- **PyTorch**：直接计算，自动微分

#### 2. 动态vs静态图
- **TensorFlow**：静态图构建一次，然后执行
- **PyTorch**：动态图每次前向传播时重建

#### 3. 调试
- **TensorFlow**：必须使用会话或eager执行来检查值
- **PyTorch**：标准Python调试器自然工作

#### 4. 张量操作
- **TensorFlow**：`tf.concat()`、`tf.reduce_sum()`等
- **PyTorch**：`torch.cat()`、`torch.sum()`等

#### 5. RNN API
- **TensorFlow**：`tf.nn.raw_rnn()`用于细粒度控制
- **PyTorch**：标准`nn.LSTM()` / `nn.GRU()`模块

### 测试和验证

要验证PyTorch实现产生等效结果：

```python
# 比较采样行为
tf_actions, tf_obs, tf_priors = tf_policy.sample(100)
pt_actions, pt_obs, pt_priors = pt_policy.sample(100)

# 动作应遵循相似的分布
assert actions_are_similar(tf_actions, pt_actions)

# 概率应匹配（给定相同权重）
tf_probs = tf_policy.compute_probs(batch)
pt_probs = pt_policy.compute_probs(batch)
assert np.allclose(tf_probs, pt_probs, rtol=1e-3)
```

### 性能考虑

1. **内存**：PyTorch通常使用与TensorFlow相似的内存
2. **速度**：此架构的性能可比较
3. **GPU**：两者都有效支持GPU加速
4. **批处理**：两者都能很好地处理变长序列

### 限制和未来工作

当前实现：
- ✅ 核心RNN功能完整
- ✅ LSTM和GRU支持
- ✅ 先验约束
- ✅ 新样本采样
- ⚠️ 嵌入未完全实现（使用one-hot编码）
- ⚠️ 未集成到主DSO训练循环（独立）

未来增强：
- 完整支持分类输入的嵌入
- 与基于PyTorch的策略优化器集成
- 混合精度训练支持
- 分布式训练能力

### 故障排除

**问题**：`_get_network_input`中的维度不匹配
- **解决方案**：检查state_manager配置是否与库设置匹配

**问题**：新样本采样返回的样本太少
- **解决方案**：增加`max_attempts_at_novel_batch`参数

**问题**：模型不学习
- **解决方案**：验证学习率，使用`torch.autograd.grad`检查梯度流

**问题**：CUDA内存不足
- **解决方案**：减少批量大小或使用梯度累积

### 参考文献

1. 原始DSO论文：[Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients](https://openreview.net/forum?id=m5Qsh0kBQG)
2. TensorFlow实现：`dso/policy/rnn_policy.py`
3. PyTorch RNN文档：https://pytorch.org/docs/stable/nn.html#rnn
