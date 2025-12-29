"""
表达式树和程序 / Expression Tree and Program

实现符号数学表达式的树形表示和执行。
Implements tree representation and execution of symbolic mathematical expressions.

关键概念 / Key Concepts:
- 前序遍历: 表达式表示为前序遍历的标记序列
  Pre-order traversal: Expressions are represented as pre-order traversal token sequences
- 例如: x + sin(y) = [add, x, sin, y]
  Example: x + sin(y) = [add, x, sin, y]
"""

import numpy as np
from typing import List, Optional, Tuple
from .token_library import TokenLibrary, Token


class Expression:
    """
    符号数学表达式
    Symbolic mathematical expression
    
    表达式存储为前序遍历的标记序列
    Expression is stored as a pre-order traversal token sequence
    
    例子 / Example:
    -------------
    表达式: x1 + sin(x2)
    Expression: x1 + sin(x2)
    
    树形结构 / Tree structure:
        add
       /   \
      x1   sin
            |
           x2
    
    前序遍历 / Pre-order: [add, x1, sin, x2]
    对应的标记索引 / Token indices: [0, 2, 4, 3] (假设库中的索引)
    """
    
    def __init__(self, tokens: List[int], library: TokenLibrary):
        """
        参数 / Parameters:
        ----------------
        tokens: List[int]
            标记索引的列表（前序遍历）
            List of token indices (pre-order traversal)
        
        library: TokenLibrary
            标记库引用
            Reference to token library
        """
        self.tokens = np.array(tokens, dtype=np.int32)
        self.library = library
        
        # 完成未完成的表达式（如果需要）
        # Complete unfinished expression if needed
        self.tokens = self._finish_tokens(self.tokens)
        
    def _finish_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        完成可能未完成的标记序列
        Complete a possibly unfinished token sequence
        
        如果表达式不完整（悬空节点），用随机终端填充
        If expression is incomplete (dangling nodes), fill with random terminals
        
        例子 / Example:
        -------------
        输入 / Input: [add, x1] (不完整，add需要2个参数但只有1个)
        输出 / Output: [add, x1, x2] (用x2填充完成)
        """
        arities = np.array([self.library.arities[t] for t in tokens])
        
        # 计算悬空节点数 / Calculate number of dangling nodes
        # 每个标记消耗1个节点，产生arity个新节点
        # Each token consumes 1 node, produces arity new nodes
        # 所以净变化是 arity - 1
        # So net change is arity - 1
        dangling = 1 + np.cumsum(arities - 1)
        
        # 检查是否有完成点 / Check if there's a completion point
        if np.any(dangling == 0):
            # 在第一个完成点截断 / Truncate at first completion point
            completion_idx = 1 + np.argmax(dangling == 0)
            tokens = tokens[:completion_idx]
        else:
            # 用终端节点填充未完成的部分 / Fill incomplete parts with terminals
            n_needed = dangling[-1]
            terminal_tokens = self.library.terminal_tokens
            random_terminals = np.random.choice(terminal_tokens, size=n_needed)
            tokens = np.concatenate([tokens, random_terminals])
        
        return tokens
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        在数据上评估表达式
        Evaluate expression on data
        
        参数 / Parameters:
        ----------------
        X: np.ndarray, shape (n_samples, n_features)
            输入数据
            Input data
            
        返回 / Returns:
        -------------
        y: np.ndarray, shape (n_samples,)
            表达式的输出值
            Output values of the expression
        
        实现方式 / Implementation:
        ----------------------
        使用递归方法评估前序遍历的表达式树
        Use recursive approach to evaluate pre-order traversed expression tree
        """
        # 使用列表来跟踪遍历位置（列表在闭包中是可变的）
        # Use list to track traversal position (lists are mutable in closures)
        eval_index = [0]
        
        def eval_recursive():
            """递归评估子树 / Recursively evaluate subtree"""
            if eval_index[0] >= len(self.tokens):
                return np.zeros(X.shape[0])
            
            token_idx = int(self.tokens[eval_index[0]])
            eval_index[0] += 1
            
            token = self.library.get_token(token_idx)
            
            if token.arity == 0:
                # 终端节点：变量
                # Terminal node: variable
                if token.name.startswith('x'):
                    # 提取变量索引，如 'x1' -> 0
                    # Extract variable index, e.g., 'x1' -> 0
                    var_idx = int(token.name[1:]) - 1
                    if var_idx < X.shape[1]:
                        return X[:, var_idx]
                    else:
                        return np.zeros(X.shape[0])
                else:
                    # 常量（如果有的话）
                    # Constants (if any)
                    return np.full(X.shape[0], 0.0)
            else:
                # 函数节点：递归评估所有参数
                # Function node: recursively evaluate all arguments
                args = []
                for _ in range(token.arity):
                    arg_value = eval_recursive()
                    args.append(arg_value)
                
                # 执行函数 / Execute function
                try:
                    return token(*args)
                except Exception:
                    return np.zeros(X.shape[0])
        
        try:
            return eval_recursive()
        except Exception:
            return np.zeros(X.shape[0])
    
    def to_string(self) -> str:
        """
        将表达式转换为可读字符串
        Convert expression to readable string
        
        返回 / Returns:
        -------------
        expr_str: str
            表达式的字符串表示，如 "add(x1, sin(x2))"
            String representation of expression, e.g., "add(x1, sin(x2))"
        """
        # 使用列表来跟踪遍历位置
        # Use list to track traversal position
        str_index = [0]
        
        def build_string_recursive():
            """递归构建字符串 / Recursively build string"""
            if str_index[0] >= len(self.tokens):
                return "empty"
            
            token_idx = int(self.tokens[str_index[0]])
            str_index[0] += 1
            
            token = self.library.get_token(token_idx)
            
            if token.arity == 0:
                # 终端节点 / Terminal node
                return token.name
            else:
                # 函数节点，收集参数 / Function node, collect arguments
                args = []
                for _ in range(token.arity):
                    arg_str = build_string_recursive()
                    args.append(arg_str)
                
                # 构建函数调用字符串 / Build function call string
                return f"{token.name}({', '.join(args)})"
        
        try:
            return build_string_recursive()
        except Exception:
            return "invalid_expr"
    
    def __str__(self):
        return self.to_string()
    
    def __repr__(self):
        token_names = [self.library.get_token(t).name for t in self.tokens]
        return f"Expression({token_names})"
    
    def complexity(self) -> float:
        """
        计算表达式的复杂度
        Calculate complexity of the expression
        
        复杂度 = 所有标记的复杂度之和
        Complexity = sum of complexities of all tokens
        """
        return sum(self.library.get_token(t).complexity for t in self.tokens)


class ExpressionBuilder:
    """
    辅助类用于构建和验证表达式
    Helper class for building and validating expressions
    """
    
    @staticmethod
    def from_prefix(token_names: List[str], library: TokenLibrary) -> Expression:
        """
        从前序遍历的标记名称列表创建表达式
        Create expression from list of token names in pre-order
        
        参数 / Parameters:
        ----------------
        token_names: List[str]
            标记名称列表，如 ['add', 'x1', 'sin', 'x2']
            List of token names, e.g., ['add', 'x1', 'sin', 'x2']
        
        library: TokenLibrary
            标记库
            Token library
        """
        token_indices = [library.get_index(name) for name in token_names]
        return Expression(token_indices, library)
    
    @staticmethod
    def random_expression(library: TokenLibrary, max_length: int = 10, 
                         p_terminal: float = 0.5) -> Expression:
        """
        生成随机表达式
        Generate random expression
        
        参数 / Parameters:
        ----------------
        library: TokenLibrary
            标记库
            Token library
        
        max_length: int
            最大表达式长度
            Maximum expression length
        
        p_terminal: float
            选择终端节点的概率（用于控制树的深度）
            Probability of choosing terminal node (controls tree depth)
        """
        tokens = []
        dangling = 1  # 开始时需要1个节点
        
        while dangling > 0 and len(tokens) < max_length:
            if dangling > max_length - len(tokens) or np.random.rand() < p_terminal:
                # 选择终端节点 / Choose terminal node
                token_idx = np.random.choice(library.terminal_tokens)
            else:
                # 选择函数节点 / Choose function node
                token_idx = np.random.choice(library.function_tokens)
            
            tokens.append(token_idx)
            token = library.get_token(token_idx)
            dangling += token.arity - 1
        
        return Expression(tokens, library)


if __name__ == "__main__":
    # 测试代码 / Test code
    from .token_library import create_default_library
    
    print("="*60)
    print("表达式树测试 / Expression Tree Test")
    print("="*60)
    
    # 创建库 / Create library
    lib = create_default_library(n_input_vars=2)
    
    # 测试1: 简单表达式 x1 + x2
    # Test 1: Simple expression x1 + x2
    print("\n测试1: x1 + x2")
    expr1 = ExpressionBuilder.from_prefix(['add', 'x1', 'x2'], lib)
    print(f"表达式 / Expression: {expr1}")
    print(f"标记序列 / Token sequence: {expr1.tokens}")
    print(f"复杂度 / Complexity: {expr1.complexity()}")
    
    # 评估表达式 / Evaluate expression
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = expr1.evaluate(X)
    print(f"输入 X / Input X:\n{X}")
    print(f"输出 y / Output y: {y}")
    print(f"预期 / Expected: [3, 7, 11]")
    
    # 测试2: 复杂表达式 sin(x1) + x2 * x2
    # Test 2: Complex expression sin(x1) + x2 * x2
    print("\n\n测试2: sin(x1) + x2 * x2")
    expr2 = ExpressionBuilder.from_prefix(['add', 'sin', 'x1', 'mul', 'x2', 'x2'], lib)
    print(f"表达式 / Expression: {expr2}")
    print(f"标记序列 / Token sequence: {expr2.tokens}")
    print(f"复杂度 / Complexity: {expr2.complexity()}")
    
    y2 = expr2.evaluate(X)
    print(f"输出 y / Output y: {y2}")
    expected = np.sin(X[:, 0]) + X[:, 1] ** 2
    print(f"预期 / Expected: {expected}")
    
    # 测试3: 随机表达式
    # Test 3: Random expression
    print("\n\n测试3: 随机表达式 / Random Expression")
    expr3 = ExpressionBuilder.random_expression(lib, max_length=7)
    print(f"表达式 / Expression: {expr3}")
    print(f"标记序列 / Token sequence: {expr3.tokens}")
    y3 = expr3.evaluate(X)
    print(f"输出 y / Output y: {y3}")
