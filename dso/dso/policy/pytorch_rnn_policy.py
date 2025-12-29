"""
PyTorch implementation of the RNN-based policy for sampling symbolic expressions.
PyTorch实现的基于RNN的策略网络，用于采样符号表达式

This module provides a PyTorch reimplementation of the TensorFlow-based RNNPolicy,
maintaining the same features, structure, and behavior while providing clearer
explanations of the implementation.

该模块提供了TensorFlow版本RNNPolicy的PyTorch重新实现，
保持相同的特性、结构和行为，同时提供更清晰的实现说明。

Key Features / 主要特性:
- Generates probability distributions over pre-order traversals of symbolic expression trees
  生成符号表达式树前序遍历的概率分布
- Supports LSTM and GRU recurrent cells
  支持LSTM和GRU循环单元
- Incorporates hierarchical priors for constrained sampling
  包含用于约束采样的层次先验
- Can sample novel expressions not in cache
  可以采样缓存中不存在的新表达式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch


class LinearWrapper(nn.Module):
    """
    RNN cell wrapper that adds a linear output layer to map hidden states to action logits.
    RNN单元包装器，添加线性输出层将隐藏状态映射到动作logits
    
    This wrapper takes the output of an RNN cell and projects it to the output space
    (number of possible tokens/actions). It maintains the same interface as PyTorch RNN cells.
    
    该包装器接收RNN单元的输出并将其投影到输出空间（可能的标记/动作数量）。
    它保持与PyTorch RNN单元相同的接口。
    
    Parameters / 参数:
    ----------
    cell : nn.Module
        The base RNN cell (LSTM or GRU)
        基础RNN单元（LSTM或GRU）
    output_size : int
        Number of possible actions/tokens in the library
        库中可能的动作/标记数量
    """
    
    def __init__(self, cell: nn.Module, output_size: int):
        super(LinearWrapper, self).__init__()
        self.cell = cell
        self.output_size = output_size
        
        # Get the hidden size from the cell
        # 从单元获取隐藏大小
        if hasattr(cell, 'hidden_size'):
            hidden_size = cell.hidden_size
        else:
            # For MultiRNN or custom cells
            # 对于MultiRNN或自定义单元
            hidden_size = cell.layers[-1].hidden_size if hasattr(cell, 'layers') else cell.cell.hidden_size
            
        # Linear layer to project hidden states to action space
        # 线性层将隐藏状态投影到动作空间
        self.output_projection = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the wrapped RNN cell.
        通过包装的RNN单元的前向传播
        
        Parameters / 参数:
        ----------
        input_seq : torch.Tensor
            Input sequence of shape (seq_len, batch_size, input_size)
            形状为(seq_len, batch_size, input_size)的输入序列
        hidden : torch.Tensor, optional
            Initial hidden state
            初始隐藏状态
            
        Returns / 返回:
        -------
        logits : torch.Tensor
            Output logits of shape (seq_len, batch_size, output_size)
            形状为(seq_len, batch_size, output_size)的输出logits
        hidden : torch.Tensor
            Final hidden state
            最终隐藏状态
        """
        # Run through the RNN cell
        # 通过RNN单元运行
        output, hidden = self.cell(input_seq, hidden)
        
        # Project to action space
        # 投影到动作空间
        logits = self.output_projection(output)
        
        return logits, hidden


class MultiLayerRNN(nn.Module):
    """
    Multi-layer RNN that stacks multiple LSTM or GRU layers.
    多层RNN，堆叠多个LSTM或GRU层
    
    Parameters / 参数:
    ----------
    cell_type : str
        Type of RNN cell: 'lstm' or 'gru'
        RNN单元类型：'lstm'或'gru'
    input_size : int
        Size of input features
        输入特征大小
    num_layers : int
        Number of RNN layers to stack
        要堆叠的RNN层数
    num_units : List[int]
        Number of hidden units in each layer
        每层的隐藏单元数
    """
    
    def __init__(self, cell_type: str, input_size: int, num_layers: int, num_units: List[int]):
        super(MultiLayerRNN, self).__init__()
        
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_units = num_units
        self.hidden_size = num_units[-1]  # Hidden size of the last layer / 最后一层的隐藏大小
        
        # Create RNN layers / 创建RNN层
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else num_units[i-1]
            layer_hidden_size = num_units[i]
            
            if cell_type == 'lstm':
                layer = nn.LSTM(layer_input_size, layer_hidden_size, num_layers=1, batch_first=False)
            elif cell_type == 'gru':
                layer = nn.GRU(layer_input_size, layer_hidden_size, num_layers=1, batch_first=False)
            else:
                raise ValueError(f"Unsupported cell type: {cell_type}. Use 'lstm' or 'gru'.")
            
            self.layers.append(layer)
    
    def forward(self, input_seq: torch.Tensor, hidden: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """
        Forward pass through all RNN layers.
        通过所有RNN层的前向传播
        
        Parameters / 参数:
        ----------
        input_seq : torch.Tensor
            Input sequence
            输入序列
        hidden : List, optional
            List of hidden states for each layer
            每层的隐藏状态列表
            
        Returns / 返回:
        -------
        output : torch.Tensor
            Output from the last layer
            最后一层的输出
        hidden_list : List
            List of final hidden states for each layer
            每层的最终隐藏状态列表
        """
        if hidden is None:
            hidden = [None] * self.num_layers
            
        output = input_seq
        new_hidden = []
        
        # Pass through each layer / 通过每一层
        for i, layer in enumerate(self.layers):
            output, h = layer(output, hidden[i])
            new_hidden.append(h)
            
        return output, new_hidden


class PyTorchRNNPolicy:
    """
    PyTorch-based RNN policy for generating symbolic expressions.
    基于PyTorch的RNN策略，用于生成符号表达式
    
    This policy uses a recurrent neural network to generate probability distributions
    over sequences of tokens that represent symbolic expressions in pre-order traversal.
    
    该策略使用循环神经网络生成表示前序遍历符号表达式的标记序列的概率分布。
    
    Key Concepts / 关键概念:
    ----------------------
    1. Pre-order traversal: Mathematical expressions are represented as trees and 
       traversed in pre-order (root, left subtree, right subtree).
       前序遍历：数学表达式表示为树，并以前序遍历（根、左子树、右子树）。
       
    2. Autoregressive sampling: Each token is sampled conditioned on previous tokens,
       allowing the RNN to learn dependencies in valid expression structures.
       自回归采样：每个标记在给定先前标记的条件下采样，允许RNN学习有效表达式结构中的依赖关系。
       
    3. Prior constraints: Hierarchical priors guide sampling toward valid expressions
       by masking invalid tokens at each step.
       先验约束：层次先验通过在每一步屏蔽无效标记来引导采样朝向有效表达式。
       
    4. State features: The policy observes contextual features like parent token,
       sibling token, and dangling nodes to make informed decisions.
       状态特征：策略观察上下文特征，如父标记、兄弟标记和悬空节点，以做出明智的决策。
    
    Parameters / 参数:
    ------------------
    prior : JointPrior
        Prior distribution for constraining the search space
        用于约束搜索空间的先验分布
    state_manager : StateManager
        Manages state observations and feature encoding
        管理状态观测和特征编码
    max_length : int, default=30
        Maximum length of generated sequences
        生成序列的最大长度
    n_choices : int
        Number of tokens in the library (vocabulary size)
        库中的标记数（词汇表大小）
    cell : str, default='lstm'
        Type of RNN cell: 'lstm' or 'gru'
        RNN单元类型：'lstm'或'gru'
    num_layers : int, default=1
        Number of stacked RNN layers
        堆叠的RNN层数
    num_units : int or List[int], default=32
        Number of hidden units per layer
        每层的隐藏单元数
    initializer : str, default='zeros'
        Weight initialization method: 'zeros' or 'xavier'
        权重初始化方法：'zeros'或'xavier'
    action_prob_lowerbound : float, default=0.0
        Minimum probability for each action (exploration)
        每个动作的最小概率（探索）
    device : str, default='cpu'
        Device to run computations on: 'cpu' or 'cuda'
        运行计算的设备：'cpu'或'cuda'
    """
    
    def __init__(self,
                 prior,
                 state_manager,
                 max_length: int = 30,
                 n_choices: Optional[int] = None,
                 cell: str = 'lstm',
                 num_layers: int = 1,
                 num_units: int = 32,
                 initializer: str = 'zeros',
                 action_prob_lowerbound: float = 0.0,
                 sample_novel_batch: bool = False,
                 max_attempts_at_novel_batch: int = 10,
                 device: str = 'cpu'):
        
        self.prior = prior
        self.state_manager = state_manager
        self.max_length = max_length
        self.n_choices = n_choices if n_choices is not None else Program.library.L
        self.action_prob_lowerbound = action_prob_lowerbound
        self.sample_novel_batch = sample_novel_batch
        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.device = torch.device(device)
        
        # Convert num_units to list if it's an integer / 如果是整数，将num_units转换为列表
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        
        # Determine input size from state manager / 从状态管理器确定输入大小
        # This depends on what observations are included (parent, sibling, etc.)
        # 这取决于包含哪些观测（父节点、兄弟节点等）
        self.input_size = self._compute_input_size()
        
        # Build the RNN model / 构建RNN模型
        self.rnn = MultiLayerRNN(cell, self.input_size, num_layers, num_units)
        self.rnn_with_output = LinearWrapper(self.rnn, self.n_choices)
        self.rnn_with_output.to(self.device)
        
        # Initialize weights / 初始化权重
        self._initialize_weights(initializer)
        
        # For tracking novel samples / 用于跟踪新样本
        self.extended_batch = None
        self.valid_extended_batch = False
        
    def _compute_input_size(self) -> int:
        """
        Compute the input size based on state manager configuration.
        根据状态管理器配置计算输入大小
        
        The input size depends on which observations are included:
        输入大小取决于包含哪些观测：
        - Parent token (one-hot or embedding) / 父标记（one-hot或嵌入）
        - Sibling token (one-hot or embedding) / 兄弟标记（one-hot或嵌入）  
        - Action token (one-hot or embedding) / 动作标记（one-hot或嵌入）
        - Dangling nodes count (scalar) / 悬空节点计数（标量）
        
        Returns / 返回:
        -------
        input_size : int
            Total size of input features / 输入特征的总大小
        """
        input_size = 0
        
        # Check if state manager uses embeddings / 检查状态管理器是否使用嵌入
        use_embedding = hasattr(self.state_manager, 'embedding') and self.state_manager.embedding
        embedding_size = getattr(self.state_manager, 'embedding_size', 8) if use_embedding else 0
        
        # Add sizes for each observation type / 为每种观测类型添加大小
        if hasattr(self.state_manager, 'observe_parent') and self.state_manager.observe_parent:
            if use_embedding:
                input_size += embedding_size
            else:
                input_size += Program.library.n_parent_inputs
                
        if hasattr(self.state_manager, 'observe_sibling') and self.state_manager.observe_sibling:
            if use_embedding:
                input_size += embedding_size
            else:
                input_size += Program.library.n_sibling_inputs
                
        if hasattr(self.state_manager, 'observe_action') and self.state_manager.observe_action:
            if use_embedding:
                input_size += embedding_size
            else:
                input_size += Program.library.n_action_inputs
                
        if hasattr(self.state_manager, 'observe_dangling') and self.state_manager.observe_dangling:
            input_size += 1  # Dangling is a scalar / dangling是标量
            
        return input_size
        
    def _initialize_weights(self, initializer: str):
        """
        Initialize network weights according to the specified method.
        根据指定的方法初始化网络权重
        
        Parameters / 参数:
        ----------
        initializer : str
            Initialization method: 'zeros' or 'xavier'
            初始化方法：'zeros'或'xavier'
        """
        for name, param in self.rnn_with_output.named_parameters():
            if 'weight' in name:
                if initializer == 'zeros':
                    nn.init.zeros_(param)
                elif initializer == 'xavier' or initializer == 'var_scale':
                    nn.init.xavier_uniform_(param)
                else:
                    raise ValueError(f"Unknown initializer: {initializer}")
            elif 'bias' in name:
                nn.init.zeros_(param)
    
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
        2. For each time step: 
           对于每个时间步：
           a. Compute RNN output (logits) conditioned on previous tokens
              计算基于先前标记的RNN输出（logits）
           b. Apply prior constraints to mask invalid actions
              应用先验约束以屏蔽无效动作
           c. Sample action from the probability distribution
              从概率分布中采样动作
           d. Update observations based on sampled action
              根据采样的动作更新观测
        3. Return: Complete sequences with actions, observations, and priors
           返回：包含动作、观测和先验的完整序列
        
        Parameters / 参数:
        ----------
        n : int
            Number of expressions to sample (batch size)
            要采样的表达式数量（批量大小）
            
        Returns / 返回:
        -------
        actions : np.ndarray
            Sampled action sequences, shape (n, max_length)
            采样的动作序列，形状(n, max_length)
        obs : np.ndarray  
            Observation sequences, shape (n, obs_dim, max_length)
            观测序列，形状(n, obs_dim, max_length)
        priors : np.ndarray
            Prior probabilities at each step, shape (n, max_length, n_choices)
            每步的先验概率，形状(n, max_length, n_choices)
        """
        if self.sample_novel_batch:
            return self.sample_novel(n)
        
        # Set model to evaluation mode / 将模型设置为评估模式
        self.rnn_with_output.eval()
        
        with torch.no_grad():
            # Initialize storage for sequences / 初始化序列存储
            actions_list = []
            obs_list = []
            priors_list = []
            
            # Get initial observation and prior / 获取初始观测和先验
            task = Program.task
            initial_obs = task.reset_task(self.prior)
            initial_obs = np.tile(initial_obs, (n, 1))  # (n, obs_dim)
            
            # Get initial prior / 获取初始先验
            initial_prior = self.prior.initial_prior()
            initial_prior = np.tile(initial_prior, (n, 1))  # (n, n_choices)
            
            # Initialize hidden state / 初始化隐藏状态
            hidden = None
            
            # Current observation and prior / 当前观测和先验
            obs = initial_obs
            prior = initial_prior
            finished = np.zeros(n, dtype=bool)
            
            # Generate sequence step by step / 逐步生成序列
            for t in range(self.max_length):
                # Store current observation and prior / 存储当前观测和先验
                obs_list.append(obs.copy())
                priors_list.append(prior.copy())
                
                # Convert observations to network input / 将观测转换为网络输入
                # This uses the state manager to encode features / 这使用状态管理器编码特征
                input_features = self._get_network_input(obs)
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)  # (1, n, input_size)
                
                # Forward pass through RNN / 通过RNN前向传播
                logits, hidden = self.rnn_with_output(input_tensor, hidden)
                logits = logits.squeeze(0)  # (n, n_choices)
                
                # Apply action probability lower bound if specified / 如果指定，应用动作概率下界
                if self.action_prob_lowerbound > 0:
                    logits = self._apply_action_prob_lowerbound(logits)
                
                # Add prior to logits / 将先验添加到logits
                prior_tensor = torch.FloatTensor(prior).to(self.device)
                logits = logits + prior_tensor
                
                # Sample actions / 采样动作
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n,)
                action_np = action.cpu().numpy()
                
                actions_list.append(action_np)
                
                # Get next observation and prior from task / 从任务获取下一个观测和先验
                actions_so_far = np.array(actions_list).T  # (n, t+1)
                obs, prior, finished = task.get_next_obs(actions_so_far, obs, finished)
                
                # Check if all sequences are finished / 检查是否所有序列都已完成
                if np.all(finished):
                    break
            
            # Convert lists to arrays / 将列表转换为数组
            actions = np.array(actions_list).T  # (n, seq_len)
            obs = np.array(obs_list).transpose(1, 2, 0)  # (n, obs_dim, seq_len)
            priors = np.array(priors_list).transpose(1, 0, 2)  # (n, seq_len, n_choices)
            
            # Pad to max_length if needed / 如果需要，填充到max_length
            if actions.shape[1] < self.max_length:
                pad_width = self.max_length - actions.shape[1]
                actions = np.pad(actions, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                obs = np.pad(obs, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
                priors = np.pad(priors, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            
            return actions, obs, priors
    
    def sample_novel(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of n expressions that are not in the cache.
        采样不在缓存中的n个表达式批次
        
        This method attempts to generate novel expressions that haven't been seen before.
        If it cannot generate enough novel samples within the maximum attempts, it fills
        the remaining slots with previously seen samples.
        
        该方法尝试生成之前未见过的新表达式。如果在最大尝试次数内无法生成足够的新样本，
        则用先前见过的样本填充剩余插槽。
        
        Parameters / 参数:
        ----------
        n : int
            Number of novel expressions to sample
            要采样的新表达式数量
            
        Returns / 返回:
        -------
        unique_actions : np.ndarray
            Actions for novel samples / 新样本的动作
        unique_obs : np.ndarray
            Observations for novel samples / 新样本的观测
        unique_priors : np.ndarray
            Priors for novel samples / 新样本的先验
        """
        n_novel = 0
        old_actions, old_obs, old_priors = [], [], []
        new_actions, new_obs, new_priors = [], [], []
        n_attempts = 0
        
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            # Sample a batch / 采样一批
            actions, obs, priors = self.sample(n)
            n_attempts += 1
            
            # Check which samples are novel / 检查哪些样本是新的
            new_indices = []
            old_indices = []
            
            for idx, a in enumerate(actions):
                tokens = _finish_tokens(a)
                key = tokens.tostring()
                
                if key not in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                else:
                    old_indices.append(idx)
            
            # Collect new and old samples / 收集新旧样本
            if new_indices:
                new_actions.append(actions[new_indices])
                new_obs.append(obs[new_indices])
                new_priors.append(priors[new_indices])
            
            if old_indices:
                old_actions.append(actions[old_indices])
                old_obs.append(obs[old_indices])
                old_priors.append(priors[old_indices])
        
        # Combine samples / 组合样本
        n_remaining = n - n_novel
        
        # Concatenate all samples / 连接所有样本
        if old_actions:
            old_actions = np.concatenate(old_actions)
            old_obs = np.concatenate(old_obs)
            old_priors = np.concatenate(old_priors)
        else:
            old_actions = np.empty((0, self.max_length))
            old_obs = np.empty((0, obs.shape[1], self.max_length))
            old_priors = np.empty((0, self.max_length, self.n_choices))
        
        if new_actions:
            new_actions = np.concatenate(new_actions)
            new_obs = np.concatenate(new_obs)
            new_priors = np.concatenate(new_priors)
        else:
            new_actions = np.empty((0, self.max_length))
            new_obs = np.empty((0, obs.shape[1], self.max_length))
            new_priors = np.empty((0, self.max_length, self.n_choices))
        
        # Fill remaining slots with old samples if needed / 如果需要，用旧样本填充剩余插槽
        if n_remaining > 0 and len(old_actions) > 0:
            new_actions = np.concatenate([new_actions, old_actions[:n_remaining]])
            new_obs = np.concatenate([new_obs, old_obs[:n_remaining]])
            new_priors = np.concatenate([new_priors, old_priors[:n_remaining]])
        
        # Store extended batch for training / 存储扩展批次用于训练
        self.extended_batch = np.array([len(old_actions), old_actions, old_obs, old_priors], dtype=object)
        self.valid_extended_batch = True
        
        return new_actions, new_obs, new_priors
    
    def _get_network_input(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert observations to network input features.
        将观测转换为网络输入特征
        
        This method processes raw observations through the state manager to create
        the feature vector that will be fed to the RNN. Features may include one-hot
        encodings or embeddings of parent/sibling/action tokens, plus dangling count.
        
        该方法通过状态管理器处理原始观测以创建将馈送到RNN的特征向量。
        特征可能包括父/兄弟/动作标记的one-hot编码或嵌入，加上悬空计数。
        
        Parameters / 参数:
        ----------
        obs : np.ndarray
            Raw observations from the task, shape (batch_size, obs_dim)
            来自任务的原始观测，形状(batch_size, obs_dim)
            
        Returns / 返回:
        -------
        input_features : np.ndarray
            Processed input features, shape (batch_size, input_size)
            处理后的输入特征，形状(batch_size, input_size)
        """
        batch_size = obs.shape[0]
        features = []
        
        # Extract observation components / 提取观测组件
        # Typically: [action, parent, sibling, dangling, ...]
        # 通常：[动作，父节点，兄弟节点，悬空，...]
        action = obs[:, 0].astype(int)
        parent = obs[:, 1].astype(int) if obs.shape[1] > 1 else None
        sibling = obs[:, 2].astype(int) if obs.shape[1] > 2 else None
        dangling = obs[:, 3] if obs.shape[1] > 3 else None
        
        # Check if using embeddings / 检查是否使用嵌入
        use_embedding = hasattr(self.state_manager, 'embedding') and self.state_manager.embedding
        
        # Process each observation type / 处理每种观测类型
        if hasattr(self.state_manager, 'observe_action') and self.state_manager.observe_action:
            if use_embedding:
                # Use embedding lookup (not implemented here for simplicity)
                # 使用嵌入查找（为简单起见此处未实现）
                action_feat = np.eye(Program.library.n_action_inputs)[action]
            else:
                action_feat = np.eye(Program.library.n_action_inputs)[action]
            features.append(action_feat)
        
        if hasattr(self.state_manager, 'observe_parent') and self.state_manager.observe_parent and parent is not None:
            if use_embedding:
                parent_feat = np.eye(Program.library.n_parent_inputs)[parent]
            else:
                parent_feat = np.eye(Program.library.n_parent_inputs)[parent]
            features.append(parent_feat)
        
        if hasattr(self.state_manager, 'observe_sibling') and self.state_manager.observe_sibling and sibling is not None:
            if use_embedding:
                sibling_feat = np.eye(Program.library.n_sibling_inputs)[sibling]
            else:
                sibling_feat = np.eye(Program.library.n_sibling_inputs)[sibling]
            features.append(sibling_feat)
        
        if hasattr(self.state_manager, 'observe_dangling') and self.state_manager.observe_dangling and dangling is not None:
            dangling_feat = dangling.reshape(-1, 1)
            features.append(dangling_feat)
        
        # Concatenate all features / 连接所有特征
        if features:
            input_features = np.concatenate(features, axis=-1)
        else:
            input_features = np.zeros((batch_size, self.input_size))
        
        return input_features
    
    def _apply_action_prob_lowerbound(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply a lower bound to the probability of each action.
        为每个动作的概率应用下界
        
        This ensures a minimum exploration probability for all actions, preventing
        the policy from becoming too deterministic and getting stuck in local optima.
        
        这确保了所有动作的最小探索概率，防止策略变得过于确定性并陷入局部最优。
        
        Parameters / 参数:
        ----------
        logits : torch.Tensor
            Raw logits before applying prior / 应用先验之前的原始logits
            
        Returns / 返回:
        -------
        bounded_logits : torch.Tensor
            Logits after applying probability lower bound / 应用概率下界后的logits
        """
        probs = F.softmax(logits, dim=-1)
        bounded_probs = ((1 - self.action_prob_lowerbound) * probs + 
                        self.action_prob_lowerbound / self.n_choices)
        bounded_logits = torch.log(bounded_probs + 1e-10)  # Add epsilon for numerical stability / 添加epsilon以保证数值稳定性
        return bounded_logits
    
    def compute_neglogp_and_entropy(self, 
                                   batch: Batch, 
                                   entropy_gamma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute negative log-probabilities and entropy for a batch of sequences.
        计算序列批次的负对数概率和熵
        
        This method evaluates how likely the given action sequences are under the
        current policy, and computes the entropy (uncertainty) of the policy's
        probability distribution at each time step.
        
        该方法评估在当前策略下给定动作序列的可能性，并计算策略在每个时间步的概率分布的熵（不确定性）。
        
        Uses / 用途:
        -----------
        - Policy gradient training: neglogp is used to compute gradients
          策略梯度训练：neglogp用于计算梯度
        - Entropy regularization: encourages exploration
          熵正则化：鼓励探索
        - Importance sampling: for off-policy learning
          重要性采样：用于离线策略学习
        
        Parameters / 参数:
        ----------
        batch : Batch
            Batch containing actions, observations, priors, and lengths
            包含动作、观测、先验和长度的批次
        entropy_gamma : float, optional
            Discount factor for entropy over time / 熵随时间的折扣因子
            
        Returns / 返回:
        -------
        neglogp : np.ndarray
            Negative log-probability of each sequence, shape (batch_size,)
            每个序列的负对数概率，形状(batch_size,)
        entropy : np.ndarray
            Entropy of the policy distribution, shape (batch_size,)
            策略分布的熵，形状(batch_size,)
        """
        self.rnn_with_output.eval()
        
        with torch.no_grad():
            # Extract batch components / 提取批次组件
            actions = batch.actions  # (batch_size, seq_len)
            obs = batch.obs  # (batch_size, obs_dim, seq_len)
            priors = batch.priors  # (batch_size, seq_len, n_choices)
            lengths = batch.lengths  # (batch_size,)
            
            batch_size, seq_len = actions.shape
            
            # Prepare entropy decay / 准备熵衰减
            if entropy_gamma is None:
                entropy_gamma = 1.0
            entropy_gamma_decay = np.array([entropy_gamma**t for t in range(seq_len)], dtype=np.float32)
            
            # Convert observations to inputs / 将观测转换为输入
            obs_transposed = obs.transpose(2, 0, 1)  # (seq_len, batch_size, obs_dim)
            inputs_list = []
            for t in range(seq_len):
                obs_t = obs_transposed[t]  # (batch_size, obs_dim)
                input_t = self._get_network_input(obs_t)
                inputs_list.append(input_t)
            inputs = np.stack(inputs_list, axis=0)  # (seq_len, batch_size, input_size)
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)
            
            # Forward pass / 前向传播
            logits, _ = self.rnn_with_output(inputs_tensor, None)  # (seq_len, batch_size, n_choices)
            
            # Apply action probability lower bound / 应用动作概率下界
            if self.action_prob_lowerbound > 0:
                logits = self._apply_action_prob_lowerbound(logits)
            
            # Add priors / 添加先验
            priors_transposed = priors.transpose(1, 0, 2)  # (seq_len, batch_size, n_choices)
            priors_tensor = torch.FloatTensor(priors_transposed).to(self.device)
            logits = logits + priors_tensor
            
            # Compute probabilities and log probabilities / 计算概率和对数概率
            log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, batch_size, n_choices)
            probs = F.softmax(logits, dim=-1)
            
            # Compute negative log probability of actions / 计算动作的负对数概率
            actions_tensor = torch.LongTensor(actions.T).to(self.device)  # (seq_len, batch_size)
            log_probs_selected = torch.gather(log_probs, 2, actions_tensor.unsqueeze(-1)).squeeze(-1)  # (seq_len, batch_size)
            
            # Create mask based on lengths / 根据长度创建掩码
            mask = torch.arange(seq_len).unsqueeze(1).to(self.device) < torch.LongTensor(lengths).unsqueeze(0).to(self.device)
            mask = mask.float()  # (seq_len, batch_size)
            
            # Compute negative log probability (sum over time) / 计算负对数概率（时间求和）
            neglogp = -(log_probs_selected * mask).sum(dim=0)  # (batch_size,)
            
            # Compute entropy / 计算熵
            entropy_per_step = -(probs * log_probs).sum(dim=-1)  # (seq_len, batch_size)
            entropy_gamma_decay_tensor = torch.FloatTensor(entropy_gamma_decay).unsqueeze(1).to(self.device)
            entropy_mask = entropy_gamma_decay_tensor * mask
            entropy = (entropy_per_step * entropy_mask).sum(dim=0)  # (batch_size,)
            
            return neglogp.cpu().numpy(), entropy.cpu().numpy()
    
    def compute_probs(self, batch: Batch, log: bool = False) -> np.ndarray:
        """
        Compute the probability (or log-probability) of a batch of sequences.
        计算序列批次的概率（或对数概率）
        
        Parameters / 参数:
        ----------
        batch : Batch
            Batch of sequences to evaluate / 要评估的序列批次
        log : bool, default=False
            If True, return log-probabilities / 如果为True，返回对数概率
            
        Returns / 返回:
        -------
        probs : np.ndarray
            Probabilities or log-probabilities, shape (batch_size,)
            概率或对数概率，形状(batch_size,)
        """
        neglogp, _ = self.compute_neglogp_and_entropy(batch, entropy_gamma=None)
        
        if log:
            return -neglogp  # Return log probabilities / 返回对数概率
        else:
            return np.exp(-neglogp)  # Return probabilities / 返回概率
    
    def set_training_mode(self, mode: bool = True):
        """
        Set the model to training or evaluation mode.
        将模型设置为训练或评估模式
        
        Parameters / 参数:
        ----------
        mode : bool, default=True
            True for training mode, False for evaluation mode
            训练模式为True，评估模式为False
        """
        if mode:
            self.rnn_with_output.train()
        else:
            self.rnn_with_output.eval()
    
    def get_trainable_parameters(self):
        """
        Get all trainable parameters for optimization.
        获取所有可训练参数以进行优化
        
        Returns / 返回:
        -------
        parameters : iterator
            Iterator over trainable parameters / 可训练参数的迭代器
        """
        return self.rnn_with_output.parameters()
    
    def save_weights(self, path: str):
        """
        Save model weights to file.
        将模型权重保存到文件
        
        Parameters / 参数:
        ----------
        path : str
            Path to save the weights / 保存权重的路径
        """
        torch.save(self.rnn_with_output.state_dict(), path)
    
    def load_weights(self, path: str):
        """
        Load model weights from file.
        从文件加载模型权重
        
        Parameters / 参数:
        ----------
        path : str
            Path to load the weights from / 加载权重的路径
        """
        self.rnn_with_output.load_state_dict(torch.load(path, map_location=self.device))
