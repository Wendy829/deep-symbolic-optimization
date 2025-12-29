"""
层次先验和约束 / Hierarchical Prior and Constraints

实现层次先验，用于引导RNN生成有效的表达式。
Implements hierarchical priors to guide RNN in generating valid expressions.

关键概念 / Key Concepts:
- 约束采样: 在每一步只允许有效的标记
  Constrained sampling: Only allow valid tokens at each step
- 例如: 在 sin( 之后，只能是变量或其他函数，不能是运算符
  Example: After sin(, only variables or other functions are allowed, not operators
"""

import numpy as np
from typing import List, Optional
from .token_library import TokenLibrary


class HierarchicalPrior:
    """
    层次先验：基于表达式结构约束标记选择
    Hierarchical Prior: Constrains token selection based on expression structure
    
    功能 / Functions:
    - 计算每步的有效标记掩码
      Compute valid token mask at each step
    - 跟踪悬空节点（还需要多少个子节点）
      Track dangling nodes (how many children are still needed)
    - 确保生成有效的表达式树
      Ensure valid expression trees are generated
    
    工作原理 / How it works:
    ---------------------
    在构建表达式时，我们跟踪"悬空"节点的数量：
    While building an expression, we track the number of "dangling" nodes:
    
    例子 / Example:
    1. 开始: dangling=1 (需要1个根节点)
       Start: dangling=1 (need 1 root node)
       可选 / Can choose: 任何标记 / any token
       
    2. 选择 add: dangling=2 (add需要2个子节点)
       Choose add: dangling=2 (add needs 2 children)
       可选 / Can choose: 任何标记 / any token
       
    3. 选择 x1: dangling=1 (x1是终端，减少1个悬空)
       Choose x1: dangling=1 (x1 is terminal, reduces dangling by 1)
       可选 / Can choose: 任何标记（但接近结束）/ any token (but close to end)
       
    4. dangling=0: 表达式完成
       dangling=0: expression complete
    """
    
    def __init__(self, library: TokenLibrary, max_length: int = 30):
        """
        参数 / Parameters:
        ----------------
        library: TokenLibrary
            标记库
            Token library
            
        max_length: int
            表达式的最大长度
            Maximum length of expression
        """
        self.library = library
        self.max_length = max_length
        self.arities = np.array(library.arities, dtype=np.int32)
        self.n_tokens = len(library)
        
        # 标记类型分组 / Token type grouping
        self.terminal_mask = self.arities == 0  # 终端标记掩码
        self.function_mask = self.arities > 0   # 函数标记掩码
        
    def get_initial_prior(self, batch_size: int) -> np.ndarray:
        """
        获取初始的先验概率（第一步）
        Get initial prior probabilities (first step)
        
        参数 / Parameters:
        ----------------
        batch_size: int
            批次大小
            Batch size
        
        返回 / Returns:
        -------------
        prior: np.ndarray, shape (batch_size, n_tokens)
            先验概率，所有标记都有效
            Prior probabilities, all tokens are valid
        """
        # 初始时所有标记都可以选择
        # Initially all tokens can be chosen
        prior = np.ones((batch_size, self.n_tokens), dtype=np.float32)
        return prior
    
    def compute_prior(self, tokens_so_far: np.ndarray, 
                     current_length: int) -> np.ndarray:
        """
        基于当前已生成的标记计算先验
        Compute prior based on currently generated tokens
        
        参数 / Parameters:
        ----------------
        tokens_so_far: np.ndarray, shape (batch_size, current_length)
            到目前为止生成的标记
            Tokens generated so far
            
        current_length: int
            当前序列长度
            Current sequence length
        
        返回 / Returns:
        -------------
        prior: np.ndarray, shape (batch_size, n_tokens)
            先验概率（0或1），表示每个标记是否有效
            Prior probabilities (0 or 1), indicating if each token is valid
        
        约束规则 / Constraint rules:
        ------------------------
        1. 如果 dangling == 0: 表达式已完成，不允许任何标记
           If dangling == 0: expression complete, no tokens allowed
           
        2. 如果接近最大长度: 只允许终端标记以快速完成
           If close to max length: only terminals allowed to finish quickly
           
        3. 如果 dangling > remaining_length: 只允许终端标记
           If dangling > remaining_length: only terminals allowed
           
        4. 否则: 所有标记都允许
           Otherwise: all tokens allowed
        """
        batch_size = tokens_so_far.shape[0]
        prior = np.ones((batch_size, self.n_tokens), dtype=np.float32)
        
        # 对每个样本计算悬空节点数
        # Compute dangling nodes for each sample
        for i in range(batch_size):
            tokens = tokens_so_far[i, :current_length]
            
            # 计算悬空节点数 / Calculate dangling nodes
            # 开始时dangling=1，每个标记使悬空变化 (arity - 1)
            # Start with dangling=1, each token changes dangling by (arity - 1)
            token_arities = self.arities[tokens]
            dangling = 1 + np.sum(token_arities - 1)
            
            # 剩余可用长度 / Remaining available length
            remaining_length = self.max_length - current_length
            
            # 规则1: 如果表达式已完成，禁止所有标记
            # Rule 1: If expression is complete, disable all tokens
            if dangling == 0:
                prior[i, :] = 0.0
                continue
            
            # 规则2: 如果悬空 > 剩余长度，只允许终端
            # Rule 2: If dangling > remaining, only allow terminals
            # （因为只有终端才能减少悬空）
            # (because only terminals can reduce dangling)
            if dangling >= remaining_length:
                prior[i, self.function_mask] = 0.0
                continue
            
            # 规则3: 如果接近最大长度（剩余<3），偏好终端
            # Rule 3: If close to max length (remaining<3), prefer terminals
            if remaining_length < 3:
                # 不是完全禁止函数，但给予终端更高的先验
                # Don't completely disable functions, but give terminals higher prior
                prior[i, self.function_mask] *= 0.1
        
        return prior
    
    def is_complete(self, tokens: np.ndarray) -> np.ndarray:
        """
        检查表达式是否完成
        Check if expressions are complete
        
        参数 / Parameters:
        ----------------
        tokens: np.ndarray, shape (batch_size, length)
            标记序列
            Token sequences
        
        返回 / Returns:
        -------------
        complete: np.ndarray, shape (batch_size,)
            布尔数组，表示每个表达式是否完成
            Boolean array indicating if each expression is complete
        """
        batch_size = tokens.shape[0]
        complete = np.zeros(batch_size, dtype=bool)
        
        for i in range(batch_size):
            # 找到第一个0（填充）的位置
            # Find position of first 0 (padding)
            valid_tokens = tokens[i][tokens[i] != 0]
            if len(valid_tokens) == 0:
                continue
            
            # 计算悬空 / Calculate dangling
            token_arities = self.arities[valid_tokens]
            dangling = 1 + np.sum(token_arities - 1)
            
            # 如果悬空为0，表达式完成
            # If dangling is 0, expression is complete
            complete[i] = (dangling == 0)
        
        return complete
    
    def mask_invalid_tokens(self, logits: np.ndarray, 
                           tokens_so_far: np.ndarray,
                           current_length: int) -> np.ndarray:
        """
        对无效标记应用掩码（设置为大负数）
        Apply mask to invalid tokens (set to large negative number)
        
        参数 / Parameters:
        ----------------
        logits: np.ndarray, shape (batch_size, n_tokens)
            RNN输出的logits
            Logits from RNN output
            
        tokens_so_far: np.ndarray, shape (batch_size, max_length)
            到目前为止的标记
            Tokens so far
            
        current_length: int
            当前长度
            Current length
        
        返回 / Returns:
        -------------
        masked_logits: np.ndarray, shape (batch_size, n_tokens)
            掩码后的logits
            Masked logits
        """
        # 计算先验 / Compute prior
        prior = self.compute_prior(tokens_so_far, current_length)
        
        # 对无效标记（prior=0）应用大负数
        # Apply large negative number to invalid tokens (prior=0)
        masked_logits = logits.copy()
        masked_logits[prior == 0] = -1e10
        
        return masked_logits


if __name__ == "__main__":
    # 测试代码 / Test code
    from .token_library import create_default_library
    
    print("="*60)
    print("层次先验测试 / Hierarchical Prior Test")
    print("="*60)
    
    # 创建库和先验 / Create library and prior
    lib = create_default_library(n_input_vars=2)
    prior = HierarchicalPrior(lib, max_length=10)
    
    print(f"\n标记库 / Token library: {lib}")
    print(f"终端标记 / Terminal tokens: {lib.terminal_tokens}")
    print(f"函数标记 / Function tokens: {lib.function_tokens}")
    
    # 测试1: 初始先验
    # Test 1: Initial prior
    print("\n\n测试1: 初始先验 / Initial Prior")
    init_prior = prior.get_initial_prior(batch_size=1)
    print(f"初始先验形状 / Initial prior shape: {init_prior.shape}")
    print(f"所有标记都有效? / All tokens valid? {np.all(init_prior == 1)}")
    
    # 测试2: 构建表达式时的先验
    # Test 2: Prior while building expression
    print("\n\n测试2: 构建 'add(x1, x2)' 时的先验")
    print("Building 'add(x1, x2)' with prior constraints")
    
    # 模拟构建过程 / Simulate building process
    batch_size = 1
    tokens = np.zeros((batch_size, 10), dtype=np.int32)
    
    # 步骤1: 选择 'add'
    # Step 1: Choose 'add'
    add_idx = lib.get_index('add')
    tokens[0, 0] = add_idx
    print(f"\n步骤1 / Step 1: 选择 add (index={add_idx})")
    p1 = prior.compute_prior(tokens, current_length=1)
    print(f"  悬空节点 / Dangling: 2")
    print(f"  有效标记数 / Valid tokens: {np.sum(p1[0] > 0)}/{lib.n_tokens}")
    
    # 步骤2: 选择 'x1'
    # Step 2: Choose 'x1'
    x1_idx = lib.get_index('x1')
    tokens[0, 1] = x1_idx
    print(f"\n步骤2 / Step 2: 选择 x1 (index={x1_idx})")
    p2 = prior.compute_prior(tokens, current_length=2)
    print(f"  悬空节点 / Dangling: 1")
    print(f"  有效标记数 / Valid tokens: {np.sum(p2[0] > 0)}/{lib.n_tokens}")
    
    # 步骤3: 选择 'x2'
    # Step 3: Choose 'x2'
    x2_idx = lib.get_index('x2')
    tokens[0, 2] = x2_idx
    print(f"\n步骤3 / Step 3: 选择 x2 (index={x2_idx})")
    p3 = prior.compute_prior(tokens, current_length=3)
    print(f"  悬空节点 / Dangling: 0 (完成!)")
    print(f"  有效标记数 / Valid tokens: {np.sum(p3[0] > 0)}/{lib.n_tokens}")
    print(f"  (应该为0，因为表达式已完成)")
    
    # 测试3: 检查完成状态
    # Test 3: Check completion status
    print("\n\n测试3: 检查完成状态 / Check Completion")
    complete = prior.is_complete(tokens[:, :3])
    print(f"表达式 [add, x1, x2] 是否完成? / Is complete? {complete[0]}")
    
    incomplete_tokens = np.array([[add_idx, x1_idx, 0, 0, 0, 0, 0, 0, 0, 0]])
    complete2 = prior.is_complete(incomplete_tokens[:, :2])
    print(f"表达式 [add, x1] 是否完成? / Is complete? {complete2[0]}")
