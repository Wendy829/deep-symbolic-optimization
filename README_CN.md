# PyTorch RNN 策略实现 - 中文说明

## 项目概述

这是一个用于符号回归的深度符号优化（DSO）框架的PyTorch实现。该实现复现了原始TensorFlow版本的RNN策略网络，用于采样和生成数学表达式。

## 什么是符号回归？

符号回归是一种机器学习方法，旨在从数据中发现数学表达式。与传统的回归方法（如线性回归）不同，符号回归不假设特定的函数形式，而是搜索最能描述数据的数学表达式。

**示例：**
- 给定数据点：(1, 3), (2, 7), (3, 13), (4, 21)
- 符号回归可能发现：y = x² + 2x
- 这比简单的线性拟合更准确

## RNN在符号回归中的作用

本项目使用循环神经网络（RNN）来生成数学表达式。关键思想是：

1. **表达式即序列**：数学表达式可以表示为标记（token）序列
   - 例如：`sin(x) + y²` → `[add, sin, x, pow, y, 2]`

2. **自回归生成**：RNN一次生成一个标记，每个标记依赖于之前的标记
   - 这允许模型学习有效表达式的结构

3. **强化学习训练**：通过评估生成表达式的准确性来训练RNN
   - 好的表达式获得高奖励，引导RNN生成更好的表达式

## 项目结构

```
dso/dso/policy/
├── rnn_policy.py          # 原始TensorFlow实现
└── pytorch_rnn_policy.py  # 新的PyTorch实现 ⭐

PYTORCH_RNN_GUIDE.md       # 详细的双语使用指南
examples/
└── pytorch_rnn_example.py # 使用示例
README_CN.md              # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
# 安装PyTorch
pip install torch>=1.7.0

# 安装DSO包（如果还未安装）
cd dso
pip install -e .
```

### 2. 使用PyTorch RNN策略

```python
from dso.policy.pytorch_rnn_policy import PyTorchRNNPolicy

# 创建策略（需要先设置prior和state_manager）
policy = PyTorchRNNPolicy(
    prior=prior,                # 先验约束
    state_manager=state_manager, # 状态管理器
    max_length=30,              # 最大序列长度
    n_choices=len(library),     # 标记数量
    cell='lstm',                # 使用LSTM
    num_layers=1,               # 单层
    num_units=32,               # 32个隐藏单元
    device='cpu'                # 使用CPU（或'cuda'使用GPU）
)

# 采样100个表达式
actions, obs, priors = policy.sample(n=100)

# actions: (100, 30) - 100个长度为30的标记序列
# obs: (100, obs_dim, 30) - 对应的观测序列
# priors: (100, 30, n_choices) - 每步的先验概率
```

### 3. 查看示例

```bash
# 运行示例脚本
python examples/pytorch_rnn_example.py
```

## 核心概念

### 1. 前序遍历（Pre-order Traversal）

数学表达式表示为树结构，按前序遍历展开：

```
表达式: x + sin(y)

树结构:
    +
   / \
  x  sin
      |
      y

前序遍历: [add, x, sin, y]
```

### 2. 自回归采样（Autoregressive Sampling）

RNN逐步生成表达式：

```
步骤1: 采样 'add'     → [add]
步骤2: 采样 'x'       → [add, x]
步骤3: 采样 'sin'     → [add, x, sin]
步骤4: 采样 'y'       → [add, x, sin, y] ✓ 完成
```

每一步的采样都基于前面所有步骤的上下文。

### 3. 先验约束（Prior Constraints）

先验知识指导有效表达式的生成：

```python
# 例如，在'sin('之后：
# ✅ 可以: x, y, 常数 (有效的函数参数)
# ❌ 不可以: +, -, sin (无效的运算符)

# 先验将无效选项的概率设为0（或很小的值）
```

### 4. 层次状态特征（Hierarchical State Features）

策略观察多种上下文信息：

- **父标记**：当前正在构建的函数（如'sin'）
- **兄弟标记**：刚刚添加的标记
- **悬空节点**：还需要多少个标记才能完成表达式
- **动作历史**：之前选择的所有标记

这些特征帮助RNN做出明智的决策。

## 关键特性

### ✅ 完整功能对等

PyTorch版本实现了TensorFlow版本的所有功能：

- ✅ LSTM和GRU循环单元
- ✅ 多层RNN堆叠
- ✅ 线性输出投影
- ✅ 先验约束集成
- ✅ 动作概率下界（探索）
- ✅ 新样本生成（避免重复）
- ✅ 熵计算（正则化）
- ✅ 负对数似然（策略梯度）
- ✅ GPU加速支持

### 🎯 PyTorch优势

1. **更容易调试**
   - 动态计算图
   - 可以使用标准Python调试器
   - 更直观的错误信息

2. **更现代的API**
   - 符合PyTorch惯例
   - 无需会话管理
   - 与PyTorch生态系统无缝集成

3. **更好的文档**
   - 详细的双语注释
   - 清晰的概念解释
   - 丰富的使用示例

## 使用场景

### 场景1：符号回归任务

```python
# 给定数据：发现描述数据的数学表达式
X = np.random.randn(100, 2)
y = X[:, 0]**2 + np.sin(X[:, 1])

# 使用PyTorch RNN策略搜索表达式
policy = PyTorchRNNPolicy(...)
for epoch in range(num_epochs):
    # 采样候选表达式
    actions, obs, priors = policy.sample(n=batch_size)
    
    # 评估每个表达式的准确性
    programs = [from_tokens(a) for a in actions]
    rewards = [evaluate_program(p, X, y) for p in programs]
    
    # 更新策略以生成更好的表达式
    train_step(policy, actions, rewards)
```

### 场景2：物理定律发现

```python
# 从实验数据中发现物理定律
# 例如：开普勒第三定律 T² ∝ R³

# 设置库包含相关运算
library = make_library(["add", "mul", "div", "pow", "x1", "x2"])

# 使用先验偏好简单表达式
prior = make_prior(length_preference='short')

# 搜索最佳拟合定律
policy = PyTorchRNNPolicy(prior=prior, ...)
best_expr = search_for_law(policy, experimental_data)
```

### 场景3：与遗传编程结合

```python
# 结合RNN采样和遗传编程优化
rnn_policy = PyTorchRNNPolicy(...)

# RNN生成初始种群
initial_population = rnn_policy.sample(n=1000)

# 遗传编程细化表达式
refined_population = genetic_programming(initial_population)

# 使用改进的表达式训练RNN
train(rnn_policy, refined_population)
```

## 与TensorFlow版本的比较

### 相同之处
- 相同的网络架构（LSTM/GRU + 线性层）
- 相同的采样算法
- 相同的先验集成机制
- 相同的输出格式

### 不同之处

| 方面 | TensorFlow | PyTorch |
|-----|-----------|---------|
| 计算图 | 静态 | 动态 |
| 会话管理 | 需要tf.Session | 不需要 |
| 调试 | 需要特殊工具 | 标准调试器 |
| 代码风格 | 声明式 | 命令式 |
| 张量操作 | tf.concat, tf.reduce_sum | torch.cat, torch.sum |
| RNN API | tf.nn.raw_rnn | nn.LSTM, nn.GRU |

### 性能比较

对于相同的模型和批量大小：
- **内存使用**：基本相同
- **计算速度**：非常接近
- **GPU加速**：两者都很好
- **易用性**：PyTorch更友好

## 代码注释说明

代码中包含详细的双语注释：

```python
def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a batch of n symbolic expressions.
    采样n个符号表达式的批次
    
    This method generates a batch of expression sequences by sampling from
    the policy's probability distribution at each time step.
    
    该方法通过在每个时间步从策略的概率分布中采样来生成一批表达式序列。
    
    Process / 过程:
    --------------
    1. Initialize: Start with empty sequences and initial hidden states
       初始化：从空序列和初始隐藏状态开始
    2. For each time step: ...
       对于每个时间步：...
    """
```

这使得代码易于理解，适合学习和修改。

## 常见问题

### Q1: 我应该使用PyTorch版本还是TensorFlow版本？

**A:** 
- **使用PyTorch**，如果：
  - 你熟悉PyTorch
  - 需要更容易的调试
  - 想要更现代的API
  - 计划与其他PyTorch模型集成

- **使用TensorFlow**，如果：
  - 你熟悉TensorFlow 1.x
  - 需要与现有DSO训练流程完全兼容
  - 已有TensorFlow训练基础设施

### Q2: PyTorch版本的性能如何？

**A:** 性能与TensorFlow版本非常接近：
- 采样速度：相似
- 训练速度：相似
- GPU利用率：相似
- 内存使用：相似

主要优势在于易用性和可维护性，而非性能。

### Q3: 我可以在现有DSO项目中使用PyTorch版本吗？

**A:** 是的，但需要一些适配：
- PyTorch RNN策略是独立实现
- 接口与TensorFlow版本略有不同
- 需要调整训练循环以使用PyTorch优化器
- 完整集成需要更多工作

当前实现主要用于：
- 学习RNN策略的工作原理
- 原型设计和实验
- 作为未来完整PyTorch DSO的基础

### Q4: 如何调试采样过程？

**A:** PyTorch版本调试更容易：

```python
# 设置断点
import pdb

def sample(self, n):
    # ... 代码 ...
    pdb.set_trace()  # 在这里暂停
    logits, hidden = self.rnn(input_features, hidden)
    # 检查 logits、hidden 等
```

你可以：
- 打印中间张量
- 检查形状和值
- 单步执行代码
- 使用标准Python调试工具

### Q5: 如何扩展实现？

**A:** 代码设计为可扩展：

1. **添加新的RNN单元**：
```python
# 在 MultiLayerRNN.__init__ 中添加
elif cell_type == 'your_cell':
    layer = YourCustomRNNCell(...)
```

2. **自定义采样策略**：
```python
# 重写 sample 方法
class CustomPolicy(PyTorchRNNPolicy):
    def sample(self, n):
        # 你的自定义采样逻辑
        pass
```

3. **添加新特征**：
```python
# 修改 _get_network_input
def _get_network_input(self, obs):
    # 添加额外的特征
    features.append(your_custom_feature)
```

## 学习路径

### 初学者

1. **阅读概念**：理解前序遍历、自回归采样
2. **运行示例**：`python examples/pytorch_rnn_example.py`
3. **阅读代码**：从`sample()`方法开始
4. **小实验**：修改超参数（num_units, num_layers）

### 中级用户

1. **完整示例**：创建完整的符号回归任务
2. **训练循环**：实现策略梯度训练
3. **可视化**：绘制生成的表达式
4. **比较**：与TensorFlow版本对比结果

### 高级用户

1. **扩展架构**：添加注意力机制
2. **优化性能**：实现混合精度训练
3. **集成**：与现有DSO框架完全集成
4. **研究**：探索新的采样策略

## 进一步阅读

- **详细文档**：[PYTORCH_RNN_GUIDE.md](PYTORCH_RNN_GUIDE.md)
- **示例代码**：[examples/pytorch_rnn_example.py](examples/pytorch_rnn_example.py)
- **原始论文**：[Deep Symbolic Regression (ICLR 2021)](https://openreview.net/forum?id=m5Qsh0kBQG)
- **PyTorch文档**：[https://pytorch.org/docs/](https://pytorch.org/docs/)
- **符号回归综述**：[SRBench](https://cavalab.org/srbench/)

## 贡献和反馈

欢迎贡献和反馈！

- 报告问题：GitHub Issues
- 建议改进：Pull Requests
- 讨论想法：GitHub Discussions

## 许可证

遵循主项目的许可证（LLNL-CODE-647188）。

## 致谢

- 原始DSO框架的作者
- TensorFlow实现的贡献者
- PyTorch社区

---

**Happy Symbolic Regression! 祝符号回归愉快！** 🎉
