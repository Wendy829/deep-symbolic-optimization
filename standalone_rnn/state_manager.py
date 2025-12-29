"""
状态管理器 / State Manager

管理RNN在生成表达式时的观测特征。
Manages observation features for RNN during expression generation.

观测特征包括 / Observation features include:
- 父标记: 当前正在构建的函数
  Parent token: Function currently being built
- 兄弟标记: 刚刚添加的标记
  Sibling token: Token just added
- 悬空节点数: 还需要多少个标记
  Dangling count: How many more tokens needed
- 动作历史: 之前选择的标记
  Action history: Previously selected tokens
"""

import numpy as np
from typing import Tuple
from .token_library import TokenLibrary


class StateManager:
    """
    状态管理器：编码RNN的观测特征
    State Manager: Encodes observation features for RNN
    
    为什么需要观测特征? / Why observation features?
    ----------------------------------------
    RNN不仅需要知道之前选择了什么标记，还需要知道当前的结构上下文：
    RNN needs to know not just what tokens were chosen, but also structural context:
    
    1. 我们在树的哪个位置？（通过父标记知道）
       Where are we in the tree? (Known through parent token)
       
    2. 刚刚添加了什么？（通过兄弟标记知道）
       What was just added? (Known through sibling token)
       
    3. 还需要多少个节点？（通过悬空计数知道）
       How many more nodes needed? (Known through dangling count)
    
    这些特征帮助RNN做出更明智的决策。
    These features help RNN make more informed decisions.
    """
    
    def __init__(self, library: TokenLibrary, max_length: int = 30):
        """
        参数 / Parameters:
        ----------------
        library: TokenLibrary
            标记库
            Token library
            
        max_length: int
            最大表达式长度
            Maximum expression length
        """
        self.library = library
        self.max_length = max_length
        self.n_tokens = len(library)
        self.arities = np.array(library.arities, dtype=np.int32)
        
        # 观测维度 / Observation dimensions:
        # - 动作历史 (one-hot): n_tokens
        # - 父标记 (one-hot): n_tokens  
        # - 兄弟标记 (one-hot): n_tokens
        # - 悬空计数 (标量): 1
        self.obs_dim = 3 * self.n_tokens + 1
    
    def get_initial_obs(self, batch_size: int) -> np.ndarray:
        """
        获取初始观测（第一步之前）
        Get initial observation (before first step)
        
        参数 / Parameters:
        ----------------
        batch_size: int
            批次大小
            Batch size
        
        返回 / Returns:
        -------------
        obs: np.ndarray, shape (batch_size, obs_dim)
            初始观测向量
            Initial observation vectors
        """
        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        # 悬空计数初始化为1（需要根节点）
        # Dangling count initialized to 1 (need root node)
        obs[:, -1] = 1.0
        return obs
    
    def compute_obs(self, tokens: np.ndarray, 
                    current_length: int) -> np.ndarray:
        """
        基于当前标记序列计算观测
        Compute observations based on current token sequence
        
        参数 / Parameters:
        ----------------
        tokens: np.ndarray, shape (batch_size, max_length)
            当前标记序列
            Current token sequence
            
        current_length: int
            序列的当前长度
            Current length of sequence
        
        返回 / Returns:
        -------------
        obs: np.ndarray, shape (batch_size, obs_dim)
            观测向量
            Observation vectors
            
        观测向量结构 / Observation vector structure:
        -----------------------------------------
        [action_one_hot (n_tokens) | 
         parent_one_hot (n_tokens) | 
         sibling_one_hot (n_tokens) | 
         dangling_count (1)]
        """
        batch_size = tokens.shape[0]
        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        
        for i in range(batch_size):
            seq = tokens[i, :current_length]
            
            if current_length == 0:
                # 第一步：没有历史
                # First step: no history
                obs[i, -1] = 1.0  # 悬空=1
                continue
            
            # 1. 动作特征：最后一个动作的one-hot
            # 1. Action feature: one-hot of last action
            last_action = seq[-1]
            obs[i, last_action] = 1.0
            
            # 2. 父标记特征
            # 2. Parent token feature
            parent_idx = self._find_parent(seq)
            if parent_idx is not None:
                parent_token = seq[parent_idx]
                obs[i, self.n_tokens + parent_token] = 1.0
            
            # 3. 兄弟标记特征
            # 3. Sibling token feature  
            sibling_idx = self._find_sibling(seq)
            if sibling_idx is not None:
                sibling_token = seq[sibling_idx]
                obs[i, 2 * self.n_tokens + sibling_token] = 1.0
            
            # 4. 悬空计数特征
            # 4. Dangling count feature
            dangling = self._compute_dangling(seq)
            obs[i, -1] = float(dangling) / self.max_length  # 归一化
        
        return obs
    
    def _find_parent(self, tokens: np.ndarray) -> int:
        """
        找到当前位置的父标记索引
        Find parent token index for current position
        
        父标记是最近的、还有未填充子节点的函数标记
        Parent is the most recent function token that still has unfilled children
        
        算法 / Algorithm:
        --------------
        从后向前扫描，跟踪每个函数还需要多少个子节点
        Scan backwards, track how many children each function still needs
        """
        if len(tokens) <= 1:
            return None
        
        # 从最后一个标记开始向前查找
        # Search backwards from last token
        children_needed = 0
        
        for idx in range(len(tokens) - 1, -1, -1):
            token = tokens[idx]
            arity = self.arities[token]
            
            if arity > 0:  # 这是一个函数
                # This is a function
                if children_needed == 0:
                    # 这个函数是父节点（它的子节点还没完全填充）
                    # This function is the parent (its children not fully filled)
                    return idx
                else:
                    # 这个函数自己是某个父节点的子节点
                    # This function itself is a child of some parent
                    children_needed += (arity - 1)
            else:
                # 终端节点，减少需要的子节点数
                # Terminal node, reduce children needed
                children_needed -= 1
        
        return None
    
    def _find_sibling(self, tokens: np.ndarray) -> int:
        """
        找到兄弟标记索引（同一父节点下的前一个子节点）
        Find sibling token index (previous child under same parent)
        
        如果当前节点是父节点的第一个子节点，则没有兄弟
        If current node is first child of parent, no sibling
        """
        if len(tokens) <= 1:
            return None
        
        # 简化实现：返回前一个标记作为兄弟
        # Simplified implementation: return previous token as sibling
        # （在完整实现中，应该检查它们是否真的是兄弟）
        # (In full implementation, should check if they're truly siblings)
        return len(tokens) - 1
    
    def _compute_dangling(self, tokens: np.ndarray) -> int:
        """
        计算悬空节点数
        Compute number of dangling nodes
        
        悬空 = 1 + sum(arity - 1) for all tokens
        """
        if len(tokens) == 0:
            return 1
        
        token_arities = self.arities[tokens]
        dangling = 1 + np.sum(token_arities - 1)
        return int(dangling)


if __name__ == "__main__":
    # 测试代码 / Test code
    from .token_library import create_default_library
    
    print("="*60)
    print("状态管理器测试 / State Manager Test")
    print("="*60)
    
    # 创建库和状态管理器 / Create library and state manager
    lib = create_default_library(n_input_vars=2)
    state_mgr = StateManager(lib, max_length=10)
    
    print(f"\n标记库 / Token library: {lib}")
    print(f"观测维度 / Observation dim: {state_mgr.obs_dim}")
    print(f"  = 3 * {lib.n_tokens} (action + parent + sibling) + 1 (dangling)")
    
    # 测试1: 初始观测
    # Test 1: Initial observation
    print("\n\n测试1: 初始观测 / Initial Observation")
    init_obs = state_mgr.get_initial_obs(batch_size=1)
    print(f"初始观测形状 / Initial obs shape: {init_obs.shape}")
    print(f"悬空计数 / Dangling count: {init_obs[0, -1]}")
    
    # 测试2: 构建表达式时的观测
    # Test 2: Observations while building expression
    print("\n\n测试2: 构建 'add(x1, sin(x2))' 时的观测")
    print("Building 'add(x1, sin(x2))' and observing states")
    
    batch_size = 1
    tokens = np.zeros((batch_size, 10), dtype=np.int32)
    
    # 获取标记索引 / Get token indices
    add_idx = lib.get_index('add')
    x1_idx = lib.get_index('x1')
    sin_idx = lib.get_index('sin')
    x2_idx = lib.get_index('x2')
    
    # 步骤1: 选择 'add'
    # Step 1: Choose 'add'
    tokens[0, 0] = add_idx
    obs1 = state_mgr.compute_obs(tokens, current_length=1)
    print(f"\n步骤1 / Step 1: add")
    print(f"  最后动作 / Last action: {np.argmax(obs1[0, :lib.n_tokens])}")
    print(f"  悬空计数 / Dangling: {obs1[0, -1] * state_mgr.max_length:.0f}")
    
    # 步骤2: 选择 'x1'
    # Step 2: Choose 'x1'
    tokens[0, 1] = x1_idx
    obs2 = state_mgr.compute_obs(tokens, current_length=2)
    print(f"\n步骤2 / Step 2: x1")
    print(f"  最后动作 / Last action: {np.argmax(obs2[0, :lib.n_tokens])}")
    parent_idx = np.argmax(obs2[0, lib.n_tokens:2*lib.n_tokens])
    print(f"  父标记 / Parent: {lib.get_token(parent_idx).name}")
    print(f"  悬空计数 / Dangling: {obs2[0, -1] * state_mgr.max_length:.0f}")
    
    # 步骤3: 选择 'sin'
    # Step 3: Choose 'sin'
    tokens[0, 2] = sin_idx
    obs3 = state_mgr.compute_obs(tokens, current_length=3)
    print(f"\n步骤3 / Step 3: sin")
    print(f"  最后动作 / Last action: {np.argmax(obs3[0, :lib.n_tokens])}")
    parent_idx3 = np.argmax(obs3[0, lib.n_tokens:2*lib.n_tokens])
    print(f"  父标记 / Parent: {lib.get_token(parent_idx3).name}")
    print(f"  悬空计数 / Dangling: {obs3[0, -1] * state_mgr.max_length:.0f}")
    
    # 步骤4: 选择 'x2'
    # Step 4: Choose 'x2'
    tokens[0, 3] = x2_idx
    obs4 = state_mgr.compute_obs(tokens, current_length=4)
    print(f"\n步骤4 / Step 4: x2")
    print(f"  最后动作 / Last action: {np.argmax(obs4[0, :lib.n_tokens])}")
    parent_idx4 = np.argmax(obs4[0, lib.n_tokens:2*lib.n_tokens])
    print(f"  父标记 / Parent: {lib.get_token(parent_idx4).name}")
    print(f"  悬空计数 / Dangling: {obs4[0, -1] * state_mgr.max_length:.0f}")
    print(f"  (应该为0，表达式完成!)")
