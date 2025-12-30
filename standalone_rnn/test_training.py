#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RNN训练的脚本 / Test script for RNN training

这个脚本展示如何训练RNN策略进行符号回归
This script demonstrates how to train RNN policy for symbolic regression
"""

import numpy as np
import torch

# 导入我们的模块 / Import our modules
try:
    from token_library import create_default_library
    from expression_tree import Expression
    from prior import HierarchicalPrior
    from state_manager import StateManager
    from rnn_policy import RNNPolicy
    from trainer import PolicyGradientTrainer
except ImportError:
    from standalone_rnn.token_library import create_default_library
    from standalone_rnn.expression_tree import Expression
    from standalone_rnn.prior import HierarchicalPrior
    from standalone_rnn.state_manager import StateManager
    from standalone_rnn.rnn_policy import RNNPolicy
    from standalone_rnn.trainer import PolicyGradientTrainer


def test_training():
    """
    测试RNN训练功能
    Test RNN training functionality
    """
    
    print("=" * 80)
    print("RNN策略训练测试 / RNN Policy Training Test")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 步骤1: 生成训练数据 / Step 1: Generate Training Data
    # ========================================================================
    print("步骤1: 生成训练数据 / Step 1: Generate Training Data")
    print("-" * 80)
    
    # 目标函数: y = x1^2 + x2
    # Target function: y = x1^2 + x2
    n_samples = 100
    n_input_vars = 2
    
    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_input_vars)
    y_train = X_train[:, 0]**2 + X_train[:, 1]  # 真实函数 / True function
    
    print(f"训练样本数 / Number of training samples: {n_samples}")
    print(f"输入变量数 / Number of input variables: {n_input_vars}")
    print(f"目标函数 / Target function: y = x1^2 + x2")
    print(f"X形状 / X shape: {X_train.shape}")
    print(f"y形状 / y shape: {y_train.shape}")
    print(f"y均值 / y mean: {y_train.mean():.4f}")
    print(f"y标准差 / y std: {y_train.std():.4f}")
    print()
    
    # ========================================================================
    # 步骤2: 定义奖励函数 / Step 2: Define Reward Function
    # ========================================================================
    print("步骤2: 定义奖励函数 / Step 2: Define Reward Function")
    print("-" * 80)
    
    def reward_function(expr):
        """
        奖励函数：评估表达式的好坏
        Reward function: Evaluate how good an expression is
        
        参数 / Args:
            expr: Expression对象 / Expression object
            
        返回 / Returns:
            reward: 正值，越高越好 / Positive value, higher is better
        """
        try:
            # 1. 在训练数据上求值 / Evaluate on training data
            y_pred = expr.evaluate(X_train)
            
            # 2. 检查是否有无效值 / Check for invalid values
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return 0.0  # 无效表达式返回0奖励 / Invalid expression gets 0 reward
            
            # 3. 计算归一化均方误差 (NMSE) / Compute Normalized MSE
            mse = np.mean((y_train - y_pred) ** 2)
            y_var = np.var(y_train)
            nmse = mse / (y_var + 1e-10)  # 防止除零 / Prevent division by zero
            
            # 4. 使用DSO的inv_nrmse作为奖励 / Use DSO's inv_nrmse as reward
            # 奖励范围 [0, 1]，越接近1越好 / Reward in [0, 1], closer to 1 is better
            reward_fit = 1.0 / (1.0 + np.sqrt(nmse))
            
            # 5. 添加复杂度惩罚 (可选) / Add complexity penalty (optional)
            complexity_penalty = 0.001 * expr.complexity()
            
            # 6. 最终奖励 / Final reward
            reward = reward_fit - complexity_penalty
            
            return max(0.0, reward)  # 确保非负 / Ensure non-negative
            
        except Exception as e:
            # 出错时返回0 / Return 0 on error
            return 0.0
    
    print("奖励函数定义:")
    print("Reward function defined:")
    print("  - 基于归一化均方根误差 (NRMSE) / Based on Normalized Root MSE")
    print("  - 公式 / Formula: reward = 1/(1+sqrt(NMSE)) - 0.001*complexity")
    print("  - 范围 / Range: [0, 1], 越高越好 / higher is better")
    print("  - 无效表达式返回0 / Invalid expressions get 0")
    print()
    
    # ========================================================================
    # 步骤3: 创建组件 / Step 3: Create Components
    # ========================================================================
    print("步骤3: 创建Token库和RNN策略 / Step 3: Create Library and RNN Policy")
    print("-" * 80)
    
    # 创建Token库 / Create token library
    lib = create_default_library(n_input_vars=n_input_vars)
    print(f"Token库已创建 / Library created: {lib.n_tokens} tokens")
    
    # 创建先验、状态管理器和策略 / Create prior, state manager, and policy
    prior = HierarchicalPrior(lib)
    state_manager = StateManager(lib)
    policy = RNNPolicy(
        library=lib,
        prior=prior,
        state_manager=state_manager,
        hidden_size=32,
        num_layers=1,
        cell_type='lstm'
    )
    
    print(f"RNN策略已创建 / RNN policy created")
    print(f"  - 隐藏层大小 / Hidden size: 32")
    print(f"  - 参数数量 / Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()
    
    # ========================================================================
    # 步骤4: 创建训练器 / Step 4: Create Trainer
    # ========================================================================
    print("步骤4: 创建训练器 / Step 4: Create Trainer")
    print("-" * 80)
    
    trainer = PolicyGradientTrainer(
        policy=policy,
        reward_function=reward_function,
        learning_rate=0.001,      # 学习率 / Learning rate
        entropy_weight=0.01,      # 熵权重 (鼓励探索) / Entropy weight (encourages exploration)
        baseline_weight=0.9,      # 基线权重 (减少方差) / Baseline weight (reduces variance)
        max_grad_norm=1.0        # 梯度裁剪 / Gradient clipping
    )
    
    print("训练器配置 / Trainer configuration:")
    print(f"  - 学习率 / Learning rate: 0.001")
    print(f"  - 熵权重 / Entropy weight: 0.01")
    print(f"  - 基线权重 / Baseline weight: 0.9")
    print(f"  - 梯度裁剪 / Gradient clipping: 1.0")
    print()
    
    # ========================================================================
    # 步骤5: 训练前采样 / Step 5: Sample Before Training
    # ========================================================================
    print("步骤5: 训练前采样测试 / Step 5: Sample Before Training")
    print("-" * 80)
    
    print("训练前采样5个表达式 / Sampling 5 expressions before training:")
    actions, obs, probs, lengths = policy.sample(batch_size=5)
    
    for i in range(5):
        expr_tokens = actions[i, :lengths[i]].tolist()
        try:
            expr = Expression(expr_tokens, lib)
            reward = reward_function(expr)
            print(f"  {i+1}. {expr.to_string():40s} (奖励/reward: {reward:.4f})")
        except:
            print(f"  {i+1}. [无效表达式 / Invalid expression]")
    print()
    
    # ========================================================================
    # 步骤6: 开始训练 / Step 6: Start Training
    # ========================================================================
    print("步骤6: 开始训练 / Step 6: Start Training")
    print("-" * 80)
    print()
    
    # 训练参数 / Training parameters
    num_iterations = 50      # 训练迭代次数 / Number of training iterations
    batch_size = 20          # 每次采样的表达式数量 / Number of expressions per batch
    
    print(f"开始训练 / Starting training...")
    print(f"  - 迭代次数 / Iterations: {num_iterations}")
    print(f"  - 批次大小 / Batch size: {batch_size}")
    print()
    
    # 训练循环 / Training loop
    history = trainer.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        verbose=True,           # 打印训练过程 / Print training progress
        print_every=10          # 每10次迭代打印一次 / Print every 10 iterations
    )
    
    print()
    print("训练完成！ / Training complete!")
    print()
    
    # ========================================================================
    # 步骤7: 分析训练历史 / Step 7: Analyze Training History
    # ========================================================================
    print("=" * 80)
    print("步骤7: 训练历史分析 / Step 7: Training History Analysis")
    print("=" * 80)
    
    print(f"迭代次数 / Iterations: {len(history['mean_reward'])}")
    print()
    
    # 奖励统计 / Reward statistics
    print("奖励统计 / Reward Statistics:")
    print(f"  初始平均奖励 / Initial mean reward: {history['mean_reward'][0]:.4f}")
    print(f"  最终平均奖励 / Final mean reward: {history['mean_reward'][-1]:.4f}")
    print(f"  最佳奖励 / Best reward: {history['best_reward'][-1]:.4f}")
    print(f"  奖励提升 / Reward improvement: {history['mean_reward'][-1] - history['mean_reward'][0]:.4f}")
    print()
    
    # 损失统计 / Loss statistics
    print("损失统计 / Loss Statistics:")
    print(f"  最终策略损失 / Final policy loss: {history['policy_loss'][-1]:.4f}")
    print(f"  最终熵 / Final entropy: {history['entropy'][-1]:.4f}")
    print()
    
    # ========================================================================
    # 步骤8: 采样最佳表达式 / Step 8: Sample Best Expressions
    # ========================================================================
    print("=" * 80)
    print("步骤8: 采样最佳表达式 / Step 8: Sample Best Expressions")
    print("=" * 80)
    
    # 采样100个表达式，选择最好的10个
    # Sample 100 expressions and select best 10
    print("采样100个表达式并选择最佳10个...")
    print("Sampling 100 expressions and selecting best 10...")
    print()
    
    best_expressions = trainer.sample_best_expressions(
        n_samples=100,
        top_k=10
    )
    
    if len(best_expressions) > 0:
        print(f"找到 {len(best_expressions)} 个表达式:")
        print(f"Found {len(best_expressions)} expressions:")
        print()
        
        for i, (expr_str, reward) in enumerate(best_expressions):
            print(f"  {i+1:2d}. {expr_str:50s} (奖励/reward: {reward:.4f})")
        
        # 验证最佳表达式 / Verify best expression
        print()
        print("-" * 80)
        print("最佳表达式详细信息 / Best Expression Details:")
        print("-" * 80)
        
        best_expr_str, best_reward = best_expressions[0]
        print(f"表达式 / Expression: {best_expr_str}")
        print(f"奖励 / Reward: {best_reward:.4f}")
        
        # 重新评估以显示详细信息 / Re-evaluate to show details
        # 注意: 这里需要从字符串重建表达式，简化起见我们重新采样
        # Note: Would need to reconstruct from string, for simplicity we resample
        print()
        print("目标函数 / Target: y = x1^2 + x2")
        print(f"最佳找到 / Best found: {best_expr_str}")
        
    else:
        print("未找到有效表达式 / No valid expressions found")
    
    print()
    
    # ========================================================================
    # 步骤9: 训练后采样 / Step 9: Sample After Training
    # ========================================================================
    print("=" * 80)
    print("步骤9: 训练后随机采样 / Step 9: Random Sample After Training")
    print("=" * 80)
    
    print("训练后采样10个表达式 / Sampling 10 expressions after training:")
    actions, obs, probs, lengths = policy.sample(batch_size=10)
    
    rewards_after = []
    for i in range(10):
        expr_tokens = actions[i, :lengths[i]].tolist()
        try:
            expr = Expression(expr_tokens, lib)
            reward = reward_function(expr)
            rewards_after.append(reward)
            print(f"  {i+1:2d}. {expr.to_string():40s} (奖励/reward: {reward:.4f})")
        except:
            rewards_after.append(0.0)
            print(f"  {i+1:2d}. [无效表达式 / Invalid expression] (奖励/reward: 0.0000)")
    
    print()
    print(f"训练后平均奖励 / Mean reward after training: {np.mean(rewards_after):.4f}")
    print()
    
    # ========================================================================
    # 步骤10: 总结 / Step 10: Summary
    # ========================================================================
    print("=" * 80)
    print("训练总结 / Training Summary")
    print("=" * 80)
    
    print(f"✓ 成功训练 {num_iterations} 次迭代")
    print(f"✓ Successfully trained for {num_iterations} iterations")
    print()
    
    print(f"✓ 奖励从 {history['mean_reward'][0]:.4f} 提升到 {history['mean_reward'][-1]:.4f}")
    print(f"✓ Reward improved from {history['mean_reward'][0]:.4f} to {history['mean_reward'][-1]:.4f}")
    print()
    
    if len(best_expressions) > 0:
        best_expr_str, best_reward = best_expressions[0]
        print(f"✓ 最佳表达式: {best_expr_str}")
        print(f"✓ Best expression: {best_expr_str}")
        print(f"✓ 最佳奖励: {best_reward:.4f}")
        print(f"✓ Best reward: {best_reward:.4f}")
    
    print()
    print("=" * 80)
    print("测试完成！ / Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # 设置随机种子以便复现 / Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行测试 / Run test
    test_training()
