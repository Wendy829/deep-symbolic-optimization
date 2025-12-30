"""
RNN策略网络 / RNN Policy Network

实现基于RNN的策略网络，用于采样符号表达式。
Implements RNN-based policy network for sampling symbolic expressions.

核心功能 / Core Functions:
- 自回归采样: 逐步生成表达式标记
  Autoregressive sampling: Progressively generate expression tokens
- 概率计算: 计算生成序列的概率
  Probability computation: Calculate probabilities of generated sequences
- 熵计算: 用于策略梯度训练的熵正则化
  Entropy computation: For entropy regularization in policy gradient training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

# Handle imports for both package and direct script execution
# 处理包导入和直接脚本执行的导入
try:
    from .token_library import TokenLibrary
    from .prior import HierarchicalPrior
    from .state_manager import StateManager
except ImportError:
    # If running as script, add parent directory to path
    # 如果作为脚本运行，将父目录添加到路径
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from standalone_rnn.token_library import TokenLibrary
    from standalone_rnn.prior import HierarchicalPrior
    from standalone_rnn.state_manager import StateManager


class RNNPolicy(nn.Module):
    """
    RNN策略网络
    RNN Policy Network
    
    架构 / Architecture:
    -----------------
    输入 / Input: 观测特征 (action + parent + sibling + dangling)
    ↓
    RNN层 (LSTM/GRU): 捕获序列依赖
    ↓ 
    线性输出层: 映射到动作空间
    ↓
    Softmax: 生成概率分布
    ↓
    输出 / Output: 下一个标记的概率
    
    训练方法 / Training:
    -----------------
    使用策略梯度（REINFORCE算法）:
    Use policy gradient (REINFORCE algorithm):
    
    1. 采样表达式 / Sample expressions
    2. 评估奖励 / Evaluate rewards
    3. 计算策略梯度 / Compute policy gradient
       ∇J = E[∇log π(a|s) * R]
    4. 更新参数 / Update parameters
    """
    
    def __init__(self,
                 library: TokenLibrary,
                 prior: HierarchicalPrior,
                 state_manager: StateManager,
                 hidden_size: int = 32,
                 num_layers: int = 1,
                 cell_type: str = 'lstm',
                 device: str = 'cpu',
                 max_length: int = 30):
        """
        参数 / Parameters:
        ----------------
        library: TokenLibrary
            标记库
            Token library
            
        prior: HierarchicalPrior  
            层次先验
            Hierarchical prior
            
        state_manager: StateManager
            状态管理器
            State manager
            
        hidden_size: int
            RNN隐藏层大小
            RNN hidden size
            
        num_layers: int
            RNN层数
            Number of RNN layers
            
        cell_type: str
            RNN单元类型: 'lstm' 或 'gru'
            RNN cell type: 'lstm' or 'gru'
            
        device: str
            计算设备: 'cpu' 或 'cuda'
            Computing device: 'cpu' or 'cuda'
            
        max_length: int
            最大序列长度
            Maximum sequence length
        """
        super(RNNPolicy, self).__init__()
        
        self.library = library
        self.prior = prior
        self.state_manager = state_manager
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.device = device
        self.max_length = max_length
        
        self.n_tokens = len(library)
        self.obs_dim = state_manager.obs_dim
        
        # 构建RNN / Build RNN
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.obs_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True  # (batch, seq, feature)
            )
        elif cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.obs_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        
        # 输出层：隐藏状态 -> 动作logits
        # Output layer: hidden state -> action logits
        self.output_layer = nn.Linear(hidden_size, self.n_tokens)
        
        self.to(device)
    
    def forward(self, obs: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Forward pass
        
        参数 / Parameters:
        ----------------
        obs: torch.Tensor, shape (batch, seq_len, obs_dim)
            观测序列
            Observation sequence
            
        hidden: torch.Tensor, optional
            初始隐藏状态
            Initial hidden state
        
        返回 / Returns:
        -------------
        logits: torch.Tensor, shape (batch, seq_len, n_tokens)
            动作logits
            Action logits
            
        hidden: torch.Tensor
            最终隐藏状态
            Final hidden state
        """
        # RNN前向传播 / RNN forward pass
        rnn_out, hidden = self.rnn(obs, hidden)
        
        # 映射到动作空间 / Map to action space
        logits = self.output_layer(rnn_out)
        
        return logits, hidden
    
    def sample(self, batch_size: int, 
               use_prior: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        采样一批表达式
        Sample a batch of expressions
        
        参数 / Parameters:
        ----------------
        batch_size: int
            批次大小
            Batch size
            
        use_prior: bool
            是否使用先验约束
            Whether to use prior constraints
        
        返回 / Returns:
        -------------
        actions: np.ndarray, shape (batch_size, max_length)
            采样的标记序列
            Sampled token sequences
            
        observations: np.ndarray, shape (batch_size, max_length, obs_dim)
            对应的观测序列
            Corresponding observation sequences
            
        probs: np.ndarray, shape (batch_size, max_length, n_tokens)
            每步的概率分布
            Probability distributions at each step
        
        lengths: np.ndarray, shape (batch_size,)
            每个表达式的实际长度（完成点）
            Actual length of each expression (completion point)
        
        采样过程 / Sampling Process:
        --------------------------
        对于每个时间步t:
        For each time step t:
        1. 根据当前序列计算观测 / Compute observation from current sequence
        2. RNN生成logits / RNN generates logits
        3. 应用先验掩码（如果使用）/ Apply prior mask (if using)
        4. 计算概率分布 / Compute probability distribution
        5. 采样下一个标记 / Sample next token
        6. 更新序列 / Update sequence
        7. 检查是否完成 / Check if complete
        """
        self.eval()  # 设置为评估模式 / Set to eval mode
        
        # 初始化 / Initialize
        actions = np.zeros((batch_size, self.max_length), dtype=np.int32)
        observations = np.zeros((batch_size, self.max_length, self.obs_dim), dtype=np.float32)
        probs = np.zeros((batch_size, self.max_length, self.n_tokens), dtype=np.float32)
        
        # RNN初始隐藏状态 / RNN initial hidden state
        hidden = None
        
        # 标记哪些样本已完成 / Track which samples are complete
        done = np.zeros(batch_size, dtype=bool)
        
        with torch.no_grad():
            for t in range(self.max_length):
                # 计算当前观测 / Compute current observation
                if t == 0:
                    obs = self.state_manager.get_initial_obs(batch_size)
                else:
                    obs = self.state_manager.compute_obs(actions, current_length=t)
                
                observations[:, t, :] = obs
                
                # 转换为torch张量 / Convert to torch tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(1).to(self.device)  # (batch, 1, obs_dim)
                
                # RNN前向传播 / RNN forward pass
                logits, hidden = self.forward(obs_tensor, hidden)
                logits = logits.squeeze(1)  # (batch, n_tokens)
                
                # 转回numpy / Convert back to numpy
                logits_np = logits.cpu().numpy()
                
                # 应用先验掩码 / Apply prior mask
                if use_prior:
                    logits_np = self.prior.mask_invalid_tokens(logits_np, actions, t)
                
                # 计算概率 / Compute probabilities
                prob = torch.softmax(torch.FloatTensor(logits_np), dim=-1).numpy()
                
                # 确保概率严格归一化，处理数值误差
                # Ensure probabilities are strictly normalized, handle numerical errors
                prob = np.maximum(prob, 0)  # 确保非负 / Ensure non-negative
                prob_sum = prob.sum(axis=-1, keepdims=True)
                
                # 处理全0的情况（所有标记都被掩码）
                # Handle case where all tokens are masked (all zeros)
                prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)
                prob = prob / prob_sum  # 严格归一化 / Strict normalization
                
                probs[:, t, :] = prob
                
                # 采样动作 / Sample actions
                for i in range(batch_size):
                    if done[i]:
                        actions[i, t] = 0  # 填充 / Padding
                    else:
                        # 从概率分布采样 / Sample from probability distribution
                        # 再次检查并修正数值误差 / Double-check and fix numerical errors
                        p = prob[i]
                        p = np.maximum(p, 0)  # 确保非负 / Ensure non-negative
                        p_sum = p.sum()
                        
                        if p_sum > 0:
                            p = p / p_sum  # 严格归一化 / Strict normalization
                            action = np.random.choice(self.n_tokens, p=p)
                        else:
                            # 如果所有概率为0，随机选择一个终端标记
                            # If all probabilities are 0, randomly choose a terminal token
                            action = np.random.choice(self.library.terminal_tokens)
                        
                        actions[i, t] = action
                
                # 检查哪些样本已完成 / Check which samples are complete
                if use_prior:
                    done = self.prior.is_complete(actions[:, :t+1])
                
                # 如果所有样本都完成，提前退出
                # If all samples complete, exit early
                if np.all(done):
                    break
        
        # 计算每个表达式的实际长度（使用DSO的方法）
        # Compute actual length of each expression (using DSO's method)
        # DSO方法：使用cumsum追踪dangling nodes
        # DSO method: use cumsum to track dangling nodes
        lengths = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            # 获取此序列的arity
            # Get arities for this sequence
            arities = np.array([self.library.arities[int(actions[i, j])] for j in range(self.max_length)])
            
            # DSO方法: dangling = 1 + cumsum(arities - 1)
            # 当dangling达到0时，表达式完成
            # DSO method: dangling = 1 + cumsum(arities - 1)
            # When dangling reaches 0, expression is complete
            dangling = 1 + np.cumsum(arities - 1)
            
            # 找到第一个dangling为0的位置（表达式完成）
            # Find first position where dangling is 0 (expression complete)
            complete_indices = np.where(dangling == 0)[0]
            if len(complete_indices) > 0:
                lengths[i] = complete_indices[0] + 1
            else:
                # 如果没完成，使用所有非零token的长度
                # If not complete, use length of all non-zero tokens
                lengths[i] = self.max_length
        
        return actions, observations, probs, lengths
    
    def compute_log_probs(self, 
                         actions: np.ndarray,
                         observations: np.ndarray) -> np.ndarray:
        """
        计算给定动作序列的对数概率
        Compute log probabilities of given action sequences
        
        参数 / Parameters:
        ----------------
        actions: np.ndarray, shape (batch_size, seq_len)
            动作序列
            Action sequences
            
        observations: np.ndarray, shape (batch_size, seq_len, obs_dim)
            观测序列
            Observation sequences
        
        返回 / Returns:
        -------------
        log_probs: np.ndarray, shape (batch_size, seq_len)
            每步的对数概率
            Log probabilities at each step
        """
        self.eval()
        
        batch_size, seq_len = actions.shape
        
        # 转换为torch张量 / Convert to torch tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        with torch.no_grad():
            # RNN前向传播 / RNN forward pass
            logits, _ = self.forward(obs_tensor)  # (batch, seq_len, n_tokens)
            
            # 计算log概率 / Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 收集对应动作的log概率 / Gather log probs of taken actions
            action_log_probs = log_probs.gather(2, actions_tensor.unsqueeze(-1)).squeeze(-1)
            
        return action_log_probs.cpu().numpy()
    
    def compute_entropy(self, observations: np.ndarray) -> np.ndarray:
        """
        计算给定观测的熵
        Compute entropy given observations
        
        熵用于鼓励探索 / Entropy encourages exploration:
        H = -∑ p(a) log p(a)
        
        参数 / Parameters:
        ----------------
        observations: np.ndarray, shape (batch_size, seq_len, obs_dim)
            观测序列
            Observation sequences
        
        返回 / Returns:
        -------------
        entropy: np.ndarray, shape (batch_size, seq_len)
            每步的熵
            Entropy at each step
        """
        self.eval()
        
        # 转换为torch张量 / Convert to torch tensor
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        
        with torch.no_grad():
            # RNN前向传播 / RNN forward pass
            logits, _ = self.forward(obs_tensor)
            
            # 计算概率 / Compute probabilities
            probs = F.softmax(logits, dim=-1)
            
            # 计算熵 / Compute entropy
            # H = -∑ p log p
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
        
        return entropy.cpu().numpy()


if __name__ == "__main__":
    # 测试代码 / Test code
    import sys
    import os
    # Add parent directory to path for imports when running as script
    # 运行脚本时将父目录添加到路径以进行导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from standalone_rnn.token_library import create_default_library
    from standalone_rnn.prior import HierarchicalPrior
    from standalone_rnn.state_manager import StateManager
    
    print("="*60)
    print("RNN策略网络测试 / RNN Policy Network Test")
    print("="*60)
    
    # 创建组件 / Create components
    lib = create_default_library(n_input_vars=2)
    prior = HierarchicalPrior(lib, max_length=15)
    state_mgr = StateManager(lib, max_length=15)
    
    print(f"\n标记库 / Token library: {lib.n_tokens} tokens")
    print(f"观测维度 / Observation dim: {state_mgr.obs_dim}")
    
    # 创建策略 / Create policy
    policy = RNNPolicy(
        library=lib,
        prior=prior,
        state_manager=state_mgr,
        hidden_size=32,
        num_layers=1,
        cell_type='lstm',
        device='cpu',
        max_length=15
    )
    
    print(f"\n策略网络 / Policy network:")
    print(f"  RNN类型 / RNN type: {policy.cell_type.upper()}")
    print(f"  隐藏大小 / Hidden size: {policy.hidden_size}")
    print(f"  层数 / Layers: {policy.num_layers}")
    print(f"  参数数量 / Parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # 测试1: 采样表达式
    # Test 1: Sample expressions
    print("\n\n测试1: 采样表达式 / Sample Expressions")
    batch_size = 3
    actions, obs, probs, lengths = policy.sample(batch_size, use_prior=True)
    
    print(f"采样了 {batch_size} 个表达式 / Sampled {batch_size} expressions:")
    for i in range(batch_size):
        try:
            # 使用返回的实际长度 / Use returned actual length
            valid_len = lengths[i]
            
            # Filter out invalid token indices and get names
            # 过滤无效的标记索引并获取名称
            token_names = []
            for a in actions[i, :valid_len]:
                a_int = int(a)
                if 0 <= a_int < lib.n_tokens:
                    token_names.append(lib.get_token(a_int).name)
            print(f"  {i+1}. {token_names}")
        except Exception as e:
            print(f"  {i+1}. Error processing expression: {e}")
    
    # 测试2: 计算对数概率
    # Test 2: Compute log probabilities
    print("\n\n测试2: 计算对数概率 / Compute Log Probabilities")
    try:
        log_probs = policy.compute_log_probs(actions, obs)
        print(f"对数概率形状 / Log probs shape: {log_probs.shape}")
        print(f"平均对数概率 / Mean log prob: {log_probs.mean():.4f}")
    except Exception as e:
        print(f"Error computing log probabilities: {e}")
    
    # 测试3: 计算熵
    # Test 3: Compute entropy
    print("\n\n测试3: 计算熵 / Compute Entropy")
    try:
        entropy = policy.compute_entropy(obs)
        print(f"熵形状 / Entropy shape: {entropy.shape}")
        print(f"平均熵 / Mean entropy: {entropy.mean():.4f}")
        print(f"(高熵 = 更多探索，低熵 = 更确定的策略)")
        print(f"(High entropy = more exploration, low entropy = more certain policy)")
    except Exception as e:
        print(f"Error computing entropy: {e}")
