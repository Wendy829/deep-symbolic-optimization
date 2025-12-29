"""
符号标记库 / Token Library

定义数学表达式的基本构建块（标记）和标记库。
Define basic building blocks (tokens) and token library for mathematical expressions.

标记可以是：
Tokens can be:
- 运算符: add, sub, mul, div / Operators: add, sub, mul, div
- 函数: sin, cos, exp, log / Functions: sin, cos, exp, log
- 变量: x1, x2, ... / Variables: x1, x2, ...
- 常量: 数值常量 / Constants: numeric constants
"""

import numpy as np
from typing import Callable, List, Optional, Dict


class Token:
    """
    表示表达式的一个标记（构建块）
    Represents a token (building block) of an expression
    
    参数 / Parameters:
    ----------------
    name: str
        标记名称，如 'add', 'sin', 'x1'
        Token name, e.g., 'add', 'sin', 'x1'
    
    arity: int
        参数个数（0=终端节点，如变量；1+=函数/运算符）
        Number of arguments (0=terminal like variable; 1+=function/operator)
        
    function: Callable, optional
        执行标记操作的函数
        Function that executes the token's operation
        
    complexity: float
        标记的复杂度权重，用于偏好简单表达式
        Complexity weight of the token, for preferring simpler expressions
    """
    
    def __init__(self, name: str, arity: int, function: Optional[Callable] = None, 
                 complexity: float = 1.0):
        self.name = name
        self.arity = arity
        self.function = function
        self.complexity = complexity
        
    def __call__(self, *args):
        """执行标记的函数 / Execute the token's function"""
        if self.function is None:
            raise ValueError(f"Token {self.name} has no function defined")
        return self.function(*args)
    
    def __repr__(self):
        return f"Token({self.name}, arity={self.arity})"
    
    def __str__(self):
        return self.name


class TokenLibrary:
    """
    标记库：管理所有可用的标记
    Token Library: Manages all available tokens
    
    提供功能 / Provides:
    - 标记注册和检索 / Token registration and retrieval
    - 按类型组织标记（函数、变量等）/ Organize tokens by type (functions, variables, etc.)
    - 标记到索引的映射 / Token to index mapping
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.name_to_index: Dict[str, int] = {}
        self.names: List[str] = []
        self.arities: List[int] = []
        
    def add_token(self, token: Token) -> int:
        """
        添加标记到库中
        Add a token to the library
        
        返回 / Returns:
        -------------
        index: int
            标记的索引位置
            Index position of the token
        """
        index = len(self.tokens)
        self.tokens.append(token)
        self.name_to_index[token.name] = index
        self.names.append(token.name)
        self.arities.append(token.arity)
        return index
    
    def get_token(self, name_or_index) -> Token:
        """
        通过名称或索引获取标记
        Get token by name or index
        """
        if isinstance(name_or_index, int):
            return self.tokens[name_or_index]
        elif isinstance(name_or_index, str):
            index = self.name_to_index[name_or_index]
            return self.tokens[index]
        else:
            raise ValueError(f"Invalid token identifier: {name_or_index}")
    
    def get_index(self, name: str) -> int:
        """获取标记的索引 / Get token's index"""
        return self.name_to_index[name]
    
    @property
    def n_tokens(self) -> int:
        """标记总数 / Total number of tokens"""
        return len(self.tokens)
    
    @property
    def terminal_tokens(self) -> List[int]:
        """
        终端标记（arity=0，即变量和常量）的索引列表
        List of terminal token indices (arity=0, i.e., variables and constants)
        """
        return [i for i, arity in enumerate(self.arities) if arity == 0]
    
    @property
    def function_tokens(self) -> List[int]:
        """
        函数标记（arity>0）的索引列表
        List of function token indices (arity>0)
        """
        return [i for i, arity in enumerate(self.arities) if arity > 0]
    
    def __len__(self):
        return self.n_tokens
    
    def __repr__(self):
        return f"TokenLibrary({self.n_tokens} tokens)"


def create_default_library(n_input_vars: int = 2) -> TokenLibrary:
    """
    创建默认的标记库，包含常用数学运算
    Create default token library with common mathematical operations
    
    参数 / Parameters:
    ----------------
    n_input_vars: int
        输入变量的数量（x1, x2, ..., xn）
        Number of input variables (x1, x2, ..., xn)
    
    返回 / Returns:
    -------------
    library: TokenLibrary
        包含常用数学运算的标记库
        Token library with common mathematical operations
        
    标记包括 / Tokens include:
    - 二元运算符: add, sub, mul, div
    - 一元函数: sin, cos, exp, log, sqrt, square
    - 输入变量: x1, x2, ..., xn
    """
    library = TokenLibrary()
    
    # 二元运算符 / Binary operators
    # arity=2 表示需要两个参数
    # arity=2 means it needs two arguments
    library.add_token(Token(
        name='add',
        arity=2,
        function=lambda a, b: a + b,
        complexity=1.0
    ))
    
    library.add_token(Token(
        name='sub',
        arity=2,
        function=lambda a, b: a - b,
        complexity=1.0
    ))
    
    library.add_token(Token(
        name='mul',
        arity=2,
        function=lambda a, b: a * b,
        complexity=1.0
    ))
    
    library.add_token(Token(
        name='div',
        arity=2,
        function=lambda a, b: np.divide(a, b, out=np.ones_like(a), where=b!=0),
        complexity=1.0
    ))
    
    # 一元函数 / Unary functions
    # arity=1 表示需要一个参数
    # arity=1 means it needs one argument
    library.add_token(Token(
        name='sin',
        arity=1,
        function=lambda x: np.sin(x),
        complexity=2.0
    ))
    
    library.add_token(Token(
        name='cos',
        arity=1,
        function=lambda x: np.cos(x),
        complexity=2.0
    ))
    
    library.add_token(Token(
        name='exp',
        arity=1,
        function=lambda x: np.exp(np.clip(x, -10, 10)),  # 裁剪以避免溢出 / Clip to avoid overflow
        complexity=2.0
    ))
    
    library.add_token(Token(
        name='log',
        arity=1,
        function=lambda x: np.log(np.abs(x) + 1e-8),  # 添加小值避免log(0) / Add small value to avoid log(0)
        complexity=2.0
    ))
    
    library.add_token(Token(
        name='sqrt',
        arity=1,
        function=lambda x: np.sqrt(np.abs(x)),  # 使用绝对值以避免复数 / Use abs to avoid complex numbers
        complexity=2.0
    ))
    
    library.add_token(Token(
        name='square',
        arity=1,
        function=lambda x: x ** 2,
        complexity=1.5
    ))
    
    # 输入变量 / Input variables
    # arity=0 表示这是终端节点（叶子节点）
    # arity=0 means this is a terminal node (leaf node)
    for i in range(n_input_vars):
        var_name = f'x{i+1}'
        # 变量不需要函数，它们在表达式树中作为叶子节点
        # Variables don't need functions, they act as leaf nodes in the expression tree
        library.add_token(Token(
            name=var_name,
            arity=0,
            function=None,  # 变量没有函数，值由输入数据提供
            complexity=0.5
        ))
    
    return library


def create_simple_library() -> TokenLibrary:
    """
    创建简化的标记库，用于测试和演示
    Create simplified token library for testing and demonstration
    
    只包含: add, mul, x1, x2
    Contains only: add, mul, x1, x2
    """
    library = TokenLibrary()
    
    library.add_token(Token('add', arity=2, function=lambda a, b: a + b))
    library.add_token(Token('mul', arity=2, function=lambda a, b: a * b))
    library.add_token(Token('x1', arity=0, function=None))
    library.add_token(Token('x2', arity=0, function=None))
    
    return library


if __name__ == "__main__":
    # 测试代码 / Test code
    print("="*60)
    print("标记库测试 / Token Library Test")
    print("="*60)
    
    # 创建默认库 / Create default library
    lib = create_default_library(n_input_vars=2)
    print(f"\n创建的库 / Created library: {lib}")
    print(f"标记总数 / Total tokens: {lib.n_tokens}")
    print(f"终端标记 / Terminal tokens: {lib.terminal_tokens}")
    print(f"函数标记 / Function tokens: {lib.function_tokens}")
    
    print("\n所有标记 / All tokens:")
    for i, token in enumerate(lib.tokens):
        print(f"  {i}: {token.name} (arity={token.arity}, complexity={token.complexity})")
    
    # 测试标记功能 / Test token functions
    print("\n\n测试标记功能 / Testing token functions:")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    
    add_token = lib.get_token('add')
    print(f"add({x}, {y}) = {add_token(x, y)}")
    
    sin_token = lib.get_token('sin')
    print(f"sin({x}) = {sin_token(x)}")
    
    mul_token = lib.get_token('mul')
    print(f"mul({x}, {y}) = {mul_token(x, y)}")
