"""
完整示例: 符号回归任务 / Complete Example: Symbolic Regression Task

这个脚本展示如何使用独立的RNN实现来解决符号回归问题。
This script demonstrates how to use the standalone RNN implementation to solve symbolic regression problems.

任务 / Task:
----------
给定数据点 (X, y)，找到最佳的数学表达式 f(X) ≈ y
Given data points (X, y), find the best mathematical expression f(X) ≈ y

例子 / Example:
------------
目标函数: y = x1^2 + x2
Target function: y = x1^2 + x2
RNN会学习生成类似的表达式
RNN will learn to generate similar expressions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# 导入独立模块 / Import standalone modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standalone_rnn import (
    TokenLibrary, Token,
    Expression, ExpressionBuilder,
    HierarchicalPrior,
    StateManager,
    RNNPolicy,
    PolicyGradientTrainer,
    create_default_library
)


def generate_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成符号回归的训练数据
    Generate training data for symbolic regression
    
    目标函数: y = x1^2 + x2
    Target function: y = x1^2 + x2
    
    返回 / Returns:
    -------------
    X: np.ndarray, shape (n_samples, 2)
        输入数据
        Input data
    
    y: np.ndarray, shape (n_samples,)
        目标输出
        Target output
    """
    # 随机采样输入 / Random sample inputs
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    
    # 计算目标输出 / Compute target output
    y = X[:, 0] ** 2 + X[:, 1]
    
    return X, y


def create_reward_function(X: np.ndarray, y: np.ndarray):
    """
    创建奖励函数用于训练
    Create reward function for training
    
    奖励 = -MSE - 复杂度惩罚
    Reward = -MSE - complexity penalty
    
    参数 / Parameters:
    ----------------
    X: np.ndarray
        输入数据
        Input data
    
    y: np.ndarray
        目标输出
        Target output
    
    返回 / Returns:
    -------------
    reward_fn: Callable
        奖励函数
        Reward function
    """
    def reward_fn(expr: Expression) -> float:
        """
        评估表达式的奖励
        Evaluate reward of expression
        
        奖励组成 / Reward components:
        1. 拟合质量: -MSE (均方误差)
           Fitting quality: -MSE (mean squared error)
        2. 复杂度惩罚: -0.01 * complexity
           Complexity penalty: -0.01 * complexity
        """
        try:
            # 评估表达式 / Evaluate expression
            y_pred = expr.evaluate(X)
            
            # 计算MSE / Compute MSE
            mse = np.mean((y - y_pred) ** 2)
            
            # 计算复杂度惩罚 / Compute complexity penalty
            complexity_penalty = 0.01 * expr.complexity()
            
            # 总奖励 / Total reward
            # 使用负MSE，因为我们要最小化误差
            # Use negative MSE because we want to minimize error
            reward = -mse - complexity_penalty
            
            # 避免NaN / Avoid NaN
            if np.isnan(reward) or np.isinf(reward):
                reward = -100.0
            
            return reward
        except Exception as e:
            # 如果表达式评估失败，返回大负奖励
            # If expression evaluation fails, return large negative reward
            return -100.0
    
    return reward_fn


def visualize_results(expressions_and_rewards, X, y, save_path: str = None):
    """
    可视化结果
    Visualize results
    
    参数 / Parameters:
    ----------------
    expressions_and_rewards: List[Tuple[Expression, float]]
        表达式和奖励的列表
        List of expressions and rewards
    
    X: np.ndarray
        输入数据
        Input data
    
    y: np.ndarray
        目标输出
        Target output
    
    save_path: str, optional
        保存图像的路径
        Path to save figure
    """
    if len(expressions_and_rewards) == 0:
        print("没有表达式可视化 / No expressions to visualize")
        return
    
    # 取最佳表达式 / Get best expression
    best_expr, best_reward = expressions_and_rewards[0]
    
    # 评估最佳表达式 / Evaluate best expression
    try:
        y_pred = best_expr.evaluate(X)
        
        # 创建图像 / Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 预测 vs 真实
        # Left plot: Predicted vs True
        ax1.scatter(y, y_pred, alpha=0.5)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax1.set_xlabel('真实值 / True Values')
        ax1.set_ylabel('预测值 / Predicted Values')
        ax1.set_title(f'最佳表达式 / Best Expression\n{best_expr}\n奖励/Reward: {best_reward:.4f}')
        ax1.grid(True)
        
        # 右图: 奖励分布
        # Right plot: Reward distribution
        rewards = [r for _, r in expressions_and_rewards]
        ax2.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(best_reward, color='r', linestyle='--', linewidth=2, label='最佳/Best')
        ax2.set_xlabel('奖励 / Reward')
        ax2.set_ylabel('频数 / Frequency')
        ax2.set_title('奖励分布 / Reward Distribution')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到 / Figure saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        print(f"可视化失败 / Visualization failed: {e}")


def main():
    """主函数 / Main function"""
    
    print("="*70)
    print("独立RNN符号回归示例 / Standalone RNN Symbolic Regression Example")
    print("="*70)
    
    # ============================================
    # 步骤1: 生成数据 / Step 1: Generate Data
    # ============================================
    print("\n步骤1: 生成数据 / Step 1: Generating Data")
    print("-" * 70)
    
    X_train, y_train = generate_data(n_samples=100)
    print(f"训练数据形状 / Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"目标函数 / Target function: y = x1^2 + x2")
    print(f"X范围 / X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"y范围 / y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # ============================================
    # 步骤2: 创建组件 / Step 2: Create Components
    # ============================================
    print("\n步骤2: 创建组件 / Step 2: Creating Components")
    print("-" * 70)
    
    # 创建标记库 / Create token library
    lib = create_default_library(n_input_vars=2)
    print(f"标记库 / Token library: {lib.n_tokens} tokens")
    print(f"  函数 / Functions: {[lib.get_token(i).name for i in lib.function_tokens[:3]]}...")
    print(f"  变量 / Variables: {[lib.get_token(i).name for i in lib.terminal_tokens]}")
    
    # 创建先验 / Create prior
    max_length = 15
    prior = HierarchicalPrior(lib, max_length=max_length)
    print(f"\n层次先验 / Hierarchical prior: max_length={max_length}")
    
    # 创建状态管理器 / Create state manager
    state_mgr = StateManager(lib, max_length=max_length)
    print(f"状态管理器 / State manager: obs_dim={state_mgr.obs_dim}")
    
    # 创建策略 / Create policy
    policy = RNNPolicy(
        library=lib,
        prior=prior,
        state_manager=state_mgr,
        hidden_size=64,
        num_layers=1,
        cell_type='lstm',
        device='cpu',
        max_length=max_length
    )
    print(f"\nRNN策略 / RNN policy:")
    print(f"  类型 / Type: {policy.cell_type.upper()}")
    print(f"  隐藏大小 / Hidden size: {policy.hidden_size}")
    print(f"  参数数量 / Parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # ============================================
    # 步骤3: 创建奖励函数和训练器
    # Step 3: Create Reward Function and Trainer
    # ============================================
    print("\n步骤3: 创建奖励函数和训练器")
    print("Step 3: Creating Reward Function and Trainer")
    print("-" * 70)
    
    reward_fn = create_reward_function(X_train, y_train)
    print("奖励函数 / Reward function: -MSE - 0.01 * complexity")
    
    trainer = PolicyGradientTrainer(
        policy=policy,
        reward_function=reward_fn,
        learning_rate=0.001,
        entropy_coef=0.005,
        baseline_type='mean'
    )
    print(f"训练器 / Trainer: REINFORCE with baseline")
    print(f"  学习率 / Learning rate: 0.001")
    print(f"  熵系数 / Entropy coef: 0.005")
    
    # ============================================
    # 步骤4: 训练策略 / Step 4: Train Policy
    # ============================================
    print("\n步骤4: 训练策略 / Step 4: Training Policy")
    print("-" * 70)
    
    num_iterations = 50
    batch_size = 20
    print(f"训练配置 / Training config:")
    print(f"  迭代次数 / Iterations: {num_iterations}")
    print(f"  批次大小 / Batch size: {batch_size}")
    print(f"  总样本数 / Total samples: {num_iterations * batch_size}")
    
    print("\n开始训练... / Starting training...")
    trainer.train(num_iterations=num_iterations, batch_size=batch_size, print_every=10)
    
    # ============================================
    # 步骤5: 评估结果 / Step 5: Evaluate Results
    # ============================================
    print("\n步骤5: 评估结果 / Step 5: Evaluating Results")
    print("-" * 70)
    
    # 采样最佳表达式 / Sample best expressions
    num_test_samples = 100
    print(f"采样 {num_test_samples} 个表达式并排序...")
    print(f"Sampling {num_test_samples} expressions and ranking...")
    
    best_exprs = trainer.sample_best_expressions(num_samples=num_test_samples)
    
    print(f"\n找到 {len(best_exprs)} 个有效表达式")
    print(f"Found {len(best_exprs)} valid expressions")
    
    # 显示前10个最佳表达式 / Display top 10 best expressions
    print("\n前10个最佳表达式 / Top 10 Best Expressions:")
    print("-" * 70)
    for i, (expr, reward) in enumerate(best_exprs[:10]):
        print(f"{i+1:2d}. 奖励/Reward: {reward:7.4f} | 复杂度/Complexity: {expr.complexity():5.1f} | {expr}")
    
    # ============================================
    # 步骤6: 可视化 / Step 6: Visualize
    # ============================================
    print("\n步骤6: 可视化结果 / Step 6: Visualizing Results")
    print("-" * 70)
    
    try:
        visualize_results(best_exprs, X_train, y_train, 
                         save_path='standalone_rnn/results.png')
    except Exception as e:
        print(f"注意: 可视化需要matplotlib / Note: Visualization requires matplotlib")
        print(f"错误 / Error: {e}")
    
    # ============================================
    # 总结 / Summary
    # ============================================
    print("\n" + "="*70)
    print("总结 / Summary")
    print("="*70)
    
    if len(best_exprs) > 0:
        best_expr, best_reward = best_exprs[0]
        print(f"\n最佳表达式 / Best Expression:")
        print(f"  {best_expr}")
        print(f"  奖励 / Reward: {best_reward:.4f}")
        print(f"  复杂度 / Complexity: {best_expr.complexity():.1f}")
        
        # 计算MSE / Compute MSE
        y_pred = best_expr.evaluate(X_train)
        mse = np.mean((y_train - y_pred) ** 2)
        print(f"  MSE: {mse:.6f}")
        
        print(f"\n目标函数 / Target function: y = x1^2 + x2")
        print(f"学习到的 / Learned: {best_expr}")
    
    print("\n训练历史 / Training History:")
    print(f"  最终平均奖励 / Final mean reward: {trainer.train_history['rewards'][-1]:.4f}")
    print(f"  最佳奖励 / Best reward: {trainer.train_history['best_reward']:.4f}")
    print(f"  最终熵 / Final entropy: {trainer.train_history['entropy'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("示例完成 / Example Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
