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

def main():
    print("="*60)
    print("快速开始 / Quick Start")
    print("="*60)
    
    # 生成数据: y = x1^2 + x2
    # Generate data: y = x1^2 + x2
    print("\n1. 生成数据 / Generating data...")
    X = np.random.uniform(-2, 2, size=(100, 2))
    y = X[:, 0]**2 + X[:, 1]
    print(f"   目标函数 / Target: y = x1^2 + x2")
    
    # 创建组件 / Create components
    print("\n2. 创建组件 / Creating components...")
    lib = create_default_library(n_input_vars=2)
    prior = HierarchicalPrior(lib, max_length=15)
    state_mgr = StateManager(lib, max_length=15)
    policy = RNNPolicy(lib, prior, state_mgr, hidden_size=32)
    print(f"   ✓ RNN策略: {sum(p.numel() for p in policy.parameters())} 参数")
    
    # 定义奖励函数 / Define reward function
    def reward_fn(expr):
        try:
            y_pred = expr.evaluate(X)
            mse = np.mean((y - y_pred)**2)
            return -mse - 0.01 * expr.complexity()
        except:
            return -100.0
    
    print("   ✓ 奖励函数: -MSE - 0.01*复杂度")
    
    # 训练 / Train
    print("\n3. 训练 / Training...")
    trainer = PolicyGradientTrainer(policy, reward_fn, learning_rate=0.001)
    trainer.train(num_iterations=30, batch_size=20, print_every=10)
    
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
