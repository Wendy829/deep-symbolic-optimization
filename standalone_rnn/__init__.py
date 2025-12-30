"""
独立的符号表达式采样和训练RNN实现 / Standalone RNN for Symbolic Expression Sampling and Training

这个包提供了一个完全独立的PyTorch实现，用于采样和训练符号数学表达式。
不依赖于原始DSO框架，所有模块都是独立实现的。

This package provides a fully standalone PyTorch implementation for sampling and training
symbolic mathematical expressions. Does not depend on the original DSO framework,
all modules are independently implemented.

主要组件 / Main Components:
- token_library: 符号标记库 / Token library for mathematical symbols
- expression_tree: 表达式树和程序 / Expression tree and program
- prior: 先验约束和层次结构 / Prior constraints and hierarchical structure
- state_manager: 状态管理器 / State manager for observations
- rnn_policy: RNN策略网络 / RNN policy network
- trainer: 训练器 / Trainer with policy gradient
"""

__version__ = "1.0.0"
__author__ = "Standalone RNN Implementation"

from .token_library import TokenLibrary, Token, create_default_library
from .expression_tree import Expression, ExpressionBuilder
from .prior import HierarchicalPrior
from .state_manager import StateManager
from .rnn_policy import RNNPolicy
from .trainer import PolicyGradientTrainer

__all__ = [
    'TokenLibrary',
    'Token',
    'create_default_library',
    'Expression',
    'ExpressionBuilder',
    'HierarchicalPrior',
    'StateManager',
    'RNNPolicy',
    'PolicyGradientTrainer'
]
