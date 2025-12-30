"""
策略梯度训练器 / Policy Gradient Trainer

实现REINFORCE算法用于训练RNN策略。
Implements REINFORCE algorithm for training RNN policy.

REINFORCE算法 / REINFORCE Algorithm:
----------------------------------
1. 采样轨迹 τ ~ π(θ)
   Sample trajectories τ ~ π(θ)

2. 计算回报 R(τ)
   Compute returns R(τ)

3. 计算策略梯度:
   Compute policy gradient:
   ∇J(θ) = E[∇log π(a|s) * (R - baseline)]

4. 更新参数:
   Update parameters:
   θ ← θ + α * ∇J(θ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Callable, Optional, List, Tuple, Dict

# Handle imports for both package and direct script execution
# 处理包导入和直接脚本执行的导入
try:
    from .rnn_policy import RNNPolicy
    from .expression_tree import Expression
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from standalone_rnn.rnn_policy import RNNPolicy
    from standalone_rnn.expression_tree import Expression


class PolicyGradientTrainer:
    """
    策略梯度训练器
    Policy Gradient Trainer
    
    使用REINFORCE算法训练RNN策略网络
    Trains RNN policy network using REINFORCE algorithm
    """
    
    def __init__(self,
                 policy: RNNPolicy,
                 reward_function: Callable,
                 learning_rate: float = 0.001,
                 entropy_coef: float = 0.01,
                 baseline_type: str = 'mean'):
        """
        参数 / Parameters:
        ----------------
        policy: RNNPolicy
            要训练的策略网络
            Policy network to train
            
        reward_function: Callable
            奖励函数，接收Expression对象，返回标量奖励
            Reward function, takes Expression object, returns scalar reward
            
            **注意 / Note:** 奖励应该设计为"越高越好"。
            原始DSO项目使用正值奖励（如 1/(1+NMSE)，范围[0,1]）。
            本实现也可使用负值奖励（如 -MSE），但需要理解"越高越好"
            意味着更接近0的负数更好。
            
            Rewards should be designed as "higher is better".
            Original DSO uses positive rewards (e.g., 1/(1+NMSE), range [0,1]).
            This implementation can also use negative rewards (e.g., -MSE), but
            understand "higher is better" means less negative is better.
            
        learning_rate: float
            学习率
            Learning rate
            
        entropy_coef: float
            熵系数，用于鼓励探索
            Entropy coefficient, encourages exploration
            
        baseline_type: str
            基线类型: 'mean', 'none'
            Baseline type: 'mean', 'none'
        """
        self.policy = policy
        self.reward_function = reward_function
        self.entropy_coef = entropy_coef
        self.baseline_type = baseline_type
        
        # 优化器 / Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # 训练统计 / Training statistics
        self.train_history = {
            'rewards': [],
            'policy_loss': [],
            'entropy': [],
            'best_reward': -np.inf
        }
    
    def train_step(self, batch_size: int) -> Dict:
        """
        执行一步训练
        Execute one training step
        
        参数 / Parameters:
        ----------------
        batch_size: int
            批次大小
            Batch size
        
        返回 / Returns:
        -------------
        stats: Dict
            训练统计信息
            Training statistics
            
        训练步骤 / Training Steps:
        ----------------------
        1. 采样表达式 / Sample expressions
        2. 评估奖励 / Evaluate rewards
        3. 计算策略损失 / Compute policy loss
        4. 计算熵损失 / Compute entropy loss
        5. 反向传播和更新 / Backpropagate and update
        """
        self.policy.train()
        
        # 步骤1: 采样表达式 / Step 1: Sample expressions
        # 注意：采样过程不需要梯度
        # Note: Sampling process doesn't need gradients
        with torch.no_grad():
            actions, observations, probs, lengths = self.policy.sample(batch_size, use_prior=True)
        
        # 步骤2: 评估奖励 / Step 2: Evaluate rewards
        rewards = np.zeros(batch_size)
        for i in range(batch_size):
            # 使用返回的实际长度 / Use returned actual length
            valid_len = lengths[i]
            
            try:
                expr = Expression(actions[i, :valid_len].tolist(), self.policy.library)
                reward = self.reward_function(expr)
                rewards[i] = reward
            except Exception as e:
                # 如果表达式无效，给予低奖励
                # If expression is invalid, give low reward
                rewards[i] = 0.0
        
        # 步骤3: 计算基线 / Step 3: Compute baseline
        if self.baseline_type == 'mean':
            baseline = np.mean(rewards)
        else:
            baseline = 0.0
        
        # 步骤4: 计算优势 / Step 4: Compute advantages
        advantages = rewards - baseline
        
        # 步骤5: 重新计算对数概率和熵（带梯度）
        # Step 5: Recompute log probs and entropy (with gradients)
        
        # 转换观测和动作为张量 / Convert observations and actions to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.policy.device)
        actions_tensor = torch.LongTensor(actions).to(self.policy.device)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(self.policy.device)
        
        # 前向传播获取logits / Forward pass to get logits
        logits, _ = self.policy.forward(obs_tensor)  # (batch, seq_len, n_tokens)
        
        # 计算log概率 / Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集选择的动作的log概率 / Gather log probs of taken actions
        action_log_probs = log_probs.gather(2, actions_tensor.unsqueeze(-1)).squeeze(-1)
        
        # 计算概率和熵 / Compute probs and entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # 步骤6: 计算策略损失 / Step 6: Compute policy loss
        # 损失 = -E[log π(a|s) * advantage]
        # Loss = -E[log π(a|s) * advantage]
        policy_loss = -(action_log_probs * advantages_tensor).mean()
        
        # 步骤7: 计算熵损失 / Step 7: Compute entropy loss
        # 熵损失（取负，因为我们要最大化熵）
        # Entropy loss (negated because we want to maximize entropy)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # 总损失 / Total loss
        total_loss = policy_loss + entropy_loss
        
        # 步骤8: 反向传播 / Step 8: Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪以防止梯度爆炸 / Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 记录统计 / Record statistics
        mean_reward = rewards.mean()
        self.train_history['rewards'].append(mean_reward)
        self.train_history['policy_loss'].append(policy_loss.item())
        self.train_history['entropy'].append(entropy.mean().item())
        
        if mean_reward > self.train_history['best_reward']:
            self.train_history['best_reward'] = mean_reward
        
        stats = {
            'mean_reward': mean_reward,
            'max_reward': rewards.max(),
            'min_reward': rewards.min(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'best_reward': self.train_history['best_reward']
        }
        
        return stats
    
    def train(self, num_iterations: int, batch_size: int, 
              print_every: int = 10) -> None:
        """
        训练策略网络
        Train policy network
        
        参数 / Parameters:
        ----------------
        num_iterations: int
            训练迭代次数
            Number of training iterations
            
        batch_size: int
            每次迭代的批次大小
            Batch size per iteration
            
        print_every: int
            打印统计的频率
            Frequency of printing statistics
        """
        print("="*60)
        print("开始训练 / Starting Training")
        print("="*60)
        
        for iteration in range(num_iterations):
            stats = self.train_step(batch_size)
            
            if (iteration + 1) % print_every == 0:
                print(f"\n迭代 / Iteration {iteration + 1}/{num_iterations}")
                print(f"  平均奖励 / Mean Reward: {stats['mean_reward']:.4f}")
                print(f"  最大奖励 / Max Reward: {stats['max_reward']:.4f}")
                print(f"  最佳奖励 / Best Reward: {stats['best_reward']:.4f}")
                print(f"  策略损失 / Policy Loss: {stats['policy_loss']:.4f}")
                print(f"  熵 / Entropy: {stats['entropy']:.4f}")
        
        print("\n" + "="*60)
        print("训练完成 / Training Complete")
        print("="*60)
    
    def sample_best_expressions(self, num_samples: int = 10) -> List[Tuple[Expression, float]]:
        """
        采样并返回最佳表达式
        Sample and return best expressions
        
        参数 / Parameters:
        ----------------
        num_samples: int
            要采样的表达式数量
            Number of expressions to sample
        
        返回 / Returns:
        -------------
        best_exprs: List[Tuple[Expression, float]]
            (表达式, 奖励) 的列表，按奖励降序排序
            List of (expression, reward) tuples, sorted by reward descending
        """
        self.policy.eval()
        
        expressions_and_rewards = []
        
        # 采样多批 / Sample multiple batches
        batch_size = min(num_samples, 32)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            actions, _, _, lengths = self.policy.sample(batch_size, use_prior=True)
            
            for i in range(batch_size):
                # 使用返回的实际长度 / Use returned actual length
                valid_len = lengths[i]
                
                try:
                    expr = Expression(actions[i, :valid_len].tolist(), self.policy.library)
                    reward = self.reward_function(expr)
                    expressions_and_rewards.append((expr, reward))
                except Exception:
                    pass
        
        # 按奖励排序 / Sort by reward
        expressions_and_rewards.sort(key=lambda x: x[1], reverse=True)
        
        return expressions_and_rewards[:num_samples]


if __name__ == "__main__":
    # 测试代码 / Test code
    # Imports already handled at top of file with try-except
    # 导入已在文件顶部通过try-except处理
    from standalone_rnn.token_library import create_default_library
    from standalone_rnn.prior import HierarchicalPrior
    from standalone_rnn.state_manager import StateManager
    from standalone_rnn.rnn_policy import RNNPolicy
    from standalone_rnn.expression_tree import Expression
    
    print("="*60)
    print("策略梯度训练器测试 / Policy Gradient Trainer Test")
    print("="*60)
    
    # 创建组件 / Create components
    lib = create_default_library(n_input_vars=2)
    prior = HierarchicalPrior(lib, max_length=10)
    state_mgr = StateManager(lib, max_length=10)
    policy = RNNPolicy(
        library=lib,
        prior=prior,
        state_manager=state_mgr,
        hidden_size=32,
        num_layers=1,
        cell_type='lstm',
        device='cpu',
        max_length=10
    )
    
    # 定义简单的奖励函数 / Define simple reward function
    # 目标: 鼓励简短的表达式
    # Goal: Encourage short expressions
    def reward_fn(expr: Expression) -> float:
        """
        简单奖励: 基于复杂度的正值奖励（类似DSO）
        Simple reward: positive reward based on complexity (like DSO)
        
        范围 [0, 1]，越高越好
        Range [0, 1], higher is better
        """
        # 将复杂度转换为正值奖励: 1 / (1 + complexity)
        # Convert complexity to positive reward: 1 / (1 + complexity)
        return 1.0 / (1.0 + expr.complexity())
    
    # 创建训练器 / Create trainer
    trainer = PolicyGradientTrainer(
        policy=policy,
        reward_function=reward_fn,
        learning_rate=0.001,
        entropy_coef=0.01,
        baseline_type='mean'
    )
    
    print("\n训练器配置 / Trainer Configuration:")
    print(f"  学习率 / Learning rate: 0.001")
    print(f"  熵系数 / Entropy coef: 0.01")
    print(f"  基线 / Baseline: mean")
    
    # 运行几步训练 / Run a few training steps
    print("\n\n运行训练步骤 / Running Training Steps")
    print("(使用简单奖励函数: 负的复杂度)")
    print("(Using simple reward function: negative complexity)")
    
    for i in range(3):
        stats = trainer.train_step(batch_size=4)
        print(f"\n步骤 {i+1} / Step {i+1}:")
        print(f"  平均奖励 / Mean Reward: {stats['mean_reward']:.4f}")
        print(f"  策略损失 / Policy Loss: {stats['policy_loss']:.4f}")
    
    # 采样最佳表达式 / Sample best expressions
    print("\n\n采样最佳表达式 / Sampling Best Expressions")
    best_exprs = trainer.sample_best_expressions(num_samples=5)
    
    print(f"找到 {len(best_exprs)} 个表达式 / Found {len(best_exprs)} expressions:")
    for i, (expr, reward) in enumerate(best_exprs):
        print(f"  {i+1}. {expr} (奖励/reward: {reward:.4f})")
