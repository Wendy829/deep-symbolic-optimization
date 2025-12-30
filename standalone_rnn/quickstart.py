#!/usr/bin/env python
"""
快速开始脚本 / Quick Start Script

这个脚本演示了如何用最少的代码使用独立RNN实现符号回归。
This script demonstrates how to use the standalone RNN for symbolic regression with minimal code.

运行 / Run:
    python quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from standalone_rnn.token_library import create_default_library
from standalone_rnn.prior import HierarchicalPrior
from standalone_rnn.state_manager import StateManager
from standalone_rnn.rnn_policy import RNNPolicy
from standalone_rnn.trainer import PolicyGradientTrainer
from standalone_rnn.expression_tree import Expression

def main(
    # RNN 参数 / RNN Parameters
    hidden_size=32,          # RNN隐藏层大小 / RNN hidden size
    num_layers=1,            # RNN层数 / Number of RNN layers  
    cell_type='lstm',        # RNN类型: 'lstm' 或 'gru' / RNN type: 'lstm' or 'gru'
    max_length=15,           # 最大表达式长度 / Maximum expression length
    
    # 训练参数 / Training Parameters
    batch_size=20,           # 批次大小 / Batch size
    num_iterations=30,       # 训练轮数 / Number of training iterations
    learning_rate=0.005,     # 学习率 / Learning rate
    temperature=1.0,         # 采样温度(>1增加探索) / Sampling temperature (>1 increases exploration)
    epsilon=0.0,             # Epsilon-greedy探索率 / Epsilon-greedy exploration rate
    entropy_coef=0.01,       # 熵系数 / Entropy coefficient
    
    # 数据参数 / Data Parameters
    n_samples=100,           # 样本数 / Number of samples
    noise_std=0.0,           # 噪声标准差 / Noise standard deviation
    
    # 任务参数 / Task Parameters
    use_task_class=True,     # 是否使用SymbolicRegressionTask / Whether to use SymbolicRegressionTask
    metric='inv_nrmse',      # 奖励指标 / Reward metric
    complexity_penalty=0.01  # 复杂度惩罚 / Complexity penalty
):
    """
    快速开始示例 / Quick Start Example
    
    演示如何使用独立的RNN进行符号回归
    Demonstrates how to use the standalone RNN for symbolic regression
    
    参数说明 / Parameter Description:
    --------------------------------
    可以通过修改上述参数来调试和实验不同的配置
    You can modify the above parameters to debug and experiment with different configurations
    
    示例 / Example:
    main(hidden_size=64, batch_size=50, num_iterations=100)
    """
    print("="*60)
    print("快速开始 / Quick Start")
    print("="*60)
    
    # 生成数据: y = x1^2 + x2
    # Generate data: y = x1^2 + x2
    print("\n1. 生成数据 / Generating data...")
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    y = X[:, 0]**2 + X[:, 1]
    if noise_std > 0:
        y += np.random.normal(0, noise_std, size=y.shape)
    print(f"   目标函数 / Target: y = x1^2 + x2")
    print(f"   样本数 / Samples: {n_samples}, 噪声 / Noise: {noise_std}")
    
    # 创建组件 / Create components
    print("\n2. 创建组件 / Creating components...")
    lib = create_default_library(n_input_vars=2)
    prior = HierarchicalPrior(lib, max_length=max_length)
    state_mgr = StateManager(lib, max_length=max_length)
    policy = RNNPolicy(lib, prior, state_mgr, 
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      cell_type=cell_type)
    print(f"   ✓ RNN策略: {sum(p.numel() for p in policy.parameters())} 参数")
    print(f"   ✓ 配置: hidden_size={hidden_size}, layers={num_layers}, cell={cell_type}")
    
    # 定义奖励函数 / Define reward function
    if use_task_class:
        # 使用SymbolicRegressionTask (推荐)
        # Use SymbolicRegressionTask (recommended)
        try:
            from standalone_rnn.task import SymbolicRegressionTask
            task = SymbolicRegressionTask(X, y, metric=metric, 
                                         complexity_penalty=complexity_penalty)
            reward_fn = task.reward
            print(f"   ✓ 使用SymbolicRegressionTask: metric={metric}")
        except ImportError:
            # 如果task.py不可用，回退到手动定义
            # Fall back to manual definition if task.py is not available
            use_task_class = False
    
    if not use_task_class:
        # 手动定义奖励函数（向后兼容）
        # Manually define reward function (backward compatible)
        def reward_fn(expr):
            try:
                y_pred = expr.evaluate(X)
                nmse = np.mean((y - y_pred)**2) / np.var(y)
                # 类似DSO的inv_nrmse: 1/(1+NRMSE), 范围[0,1]
                # Similar to DSO's inv_nrmse: 1/(1+NRMSE), range [0,1]
                reward = 1.0 / (1.0 + np.sqrt(nmse))
                # 添加复杂度惩罚 / Add complexity penalty
                reward -= complexity_penalty * expr.complexity()
                return reward
            except:
                return 0.0  # 失败时返回最低奖励 / Return lowest reward on failure
        
        print(f"   ✓ 奖励函数: 1/(1+NRMSE) - {complexity_penalty}*复杂度")
    
    # 训练 / Train
    print("\n3. 训练 / Training...")
    trainer = PolicyGradientTrainer(
        policy, reward_fn, 
        learning_rate=learning_rate,
        temperature=temperature,
        epsilon=epsilon,
        entropy_coef=entropy_coef
    )
    print(f"   ✓ 训练配置: lr={learning_rate}, temp={temperature}, eps={epsilon}")
    trainer.train(num_iterations=num_iterations, batch_size=batch_size, print_every=10)
    
    # 获取最佳表达式 / Get best expressions
    print("\n4. 结果 / Results...")
    best_exprs = trainer.sample_best_expressions(50)
    
    print(f"\n找到 {len(best_exprs)} 个表达式")
    print(f"Found {len(best_exprs)} expressions")
    print("\n前5个最佳 / Top 5:")
    for i, (expr, reward) in enumerate(best_exprs[:5]):
        print(f"  {i+1}. {expr} (奖励/reward: {reward:.4f})")
    
    if len(best_exprs) > 0:
        best_expr, best_reward = best_exprs[0]
        y_pred = best_expr.evaluate(X)
        mse = np.mean((y - y_pred)**2)
        print(f"\n最佳表达式 / Best:")
        print(f"  表达式 / Expression: {best_expr}")
        print(f"  MSE: {mse:.6f}")
        print(f"  复杂度 / Complexity: {best_expr.complexity()}")
    
    print("\n" + "="*60)
    print("完成! / Done!")
    print("="*60)

if __name__ == "__main__":
    main()
