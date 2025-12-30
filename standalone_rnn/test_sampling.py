#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RNN采样表达式的脚本 / Test script for RNN expression sampling

这个脚本展示如何使用RNN策略采样数学表达式
This script demonstrates how to use RNN policy to sample mathematical expressions
"""

import numpy as np
import torch

# 导入我们的模块 / Import our modules
try:
    from token_library import TokenLibrary, create_default_library
    from expression_tree import Expression
    from prior import HierarchicalPrior
    from state_manager import StateManager
    from rnn_policy import RNNPolicy
except ImportError:
    from standalone_rnn.token_library import TokenLibrary, create_default_library
    from standalone_rnn.expression_tree import Expression
    from standalone_rnn.prior import HierarchicalPrior
    from standalone_rnn.state_manager import StateManager
    from standalone_rnn.rnn_policy import RNNPolicy


def test_sampling():
    """
    测试RNN采样表达式的功能
    Test RNN expression sampling functionality
    """
    
    print("=" * 80)
    print("RNN表达式采样测试 / RNN Expression Sampling Test")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 步骤1: 创建Token库 / Step 1: Create Token Library
    # ========================================================================
    print("步骤1: 创建Token库 / Step 1: Create Token Library")
    print("-" * 80)
    
    # 创建包含2个输入变量(x1, x2)的默认库
    # Create default library with 2 input variables (x1, x2)
    n_input_vars = 2
    lib = create_default_library(n_input_vars=n_input_vars)
    
    print(f"Token总数 / Total tokens: {lib.n_tokens}")
    print(f"函数数量 / Number of functions: {len(lib.functions)}")
    print(f"变量数量 / Number of variables: {n_input_vars}")
    print()
    
    # 打印所有Token / Print all tokens
    print("所有Token / All Tokens:")
    for i in range(lib.n_tokens):
        token = lib.get_token(i)
        print(f"  [{i:2d}] {token.name:10s} (arity={token.arity})")
    print()
    
    # ========================================================================
    # 步骤2: 创建层次化先验约束 / Step 2: Create Hierarchical Prior
    # ========================================================================
    print("步骤2: 创建层次化先验约束 / Step 2: Create Hierarchical Prior")
    print("-" * 80)
    
    # 先验约束确保只生成有效的表达式
    # Prior constraints ensure only valid expressions are generated
    prior = HierarchicalPrior(lib)
    
    print("先验约束已创建，将在采样时确保表达式有效性")
    print("Prior created, will ensure expression validity during sampling")
    print()
    
    # ========================================================================
    # 步骤3: 创建状态管理器 / Step 3: Create State Manager
    # ========================================================================
    print("步骤3: 创建状态管理器 / Step 3: Create State Manager")
    print("-" * 80)
    
    # 状态管理器编码当前表达式的特征供RNN使用
    # State manager encodes current expression features for RNN
    state_manager = StateManager(lib)
    
    print(f"观察特征维度 / Observation dimension: {state_manager.obs_dim}")
    print("特征包括 / Features include:")
    print("  - 父节点编码 / Parent encoding")
    print("  - 兄弟节点编码 / Sibling encoding")
    print("  - 悬空节点数量 / Dangling count")
    print()
    
    # ========================================================================
    # 步骤4: 创建RNN策略 / Step 4: Create RNN Policy
    # ========================================================================
    print("步骤4: 创建RNN策略 / Step 4: Create RNN Policy")
    print("-" * 80)
    
    # 创建RNN策略网络
    # Create RNN policy network
    policy = RNNPolicy(
        library=lib,
        prior=prior,
        state_manager=state_manager,
        hidden_size=32,          # LSTM隐藏层大小 / LSTM hidden size
        num_layers=1,            # LSTM层数 / Number of LSTM layers
        cell_type='lstm',        # 使用LSTM (也可以是'gru') / Use LSTM (can also be 'gru')
        max_length=30            # 表达式最大长度 / Max expression length
    )
    
    print(f"RNN类型 / RNN type: {policy.cell_type.upper()}")
    print(f"隐藏层大小 / Hidden size: {policy.hidden_size}")
    print(f"LSTM层数 / Number of layers: {policy.num_layers}")
    print(f"最大长度 / Max length: {policy.max_length}")
    print(f"参数总数 / Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()
    
    # ========================================================================
    # 步骤5: 采样表达式 / Step 5: Sample Expressions
    # ========================================================================
    print("步骤5: 采样表达式 / Step 5: Sample Expressions")
    print("-" * 80)
    
    # 采样一批表达式 / Sample a batch of expressions
    batch_size = 10
    print(f"采样 {batch_size} 个表达式 / Sampling {batch_size} expressions...")
    print()
    
    # sample()返回4个值 / sample() returns 4 values:
    # - actions: Token序列 / Token sequences
    # - obs: 观察特征 / Observation features
    # - probs: Token概率 / Token probabilities
    # - lengths: 表达式长度 / Expression lengths
    actions, obs, probs, lengths = policy.sample(batch_size=batch_size)
    
    print(f"actions形状 / actions shape: {actions.shape}")  # (batch_size, max_length)
    print(f"lengths形状 / lengths shape: {lengths.shape}")  # (batch_size,)
    print()
    
    # ========================================================================
    # 步骤6: 解析并显示表达式 / Step 6: Parse and Display Expressions
    # ========================================================================
    print("步骤6: 解析并显示表达式 / Step 6: Parse and Display Expressions")
    print("-" * 80)
    print()
    
    valid_count = 0  # 有效表达式计数 / Valid expression count
    
    for i in range(batch_size):
        # 获取这个表达式的Token序列 / Get token sequence for this expression
        expr_length = lengths[i]
        expr_tokens = actions[i, :expr_length].tolist()
        
        print(f"表达式 {i+1} / Expression {i+1}:")
        print(f"  长度 / Length: {expr_length}")
        print(f"  Token序列 / Token sequence: {expr_tokens}")
        
        # 创建Expression对象 / Create Expression object
        try:
            expr = Expression(expr_tokens, lib)
            print(f"  字符串形式 / String form: {expr.to_string()}")
            print(f"  复杂度 / Complexity: {expr.complexity()}")
            
            # 尝试在一些测试数据上求值 / Try evaluating on test data
            X_test = np.random.randn(5, n_input_vars)
            y_test = expr.evaluate(X_test)
            
            if not np.any(np.isnan(y_test)) and not np.any(np.isinf(y_test)):
                print(f"  求值结果 / Evaluation: 成功 / Success (前5个值 / first 5 values: {y_test[:5]})")
                valid_count += 1
            else:
                print(f"  求值结果 / Evaluation: 包含NaN或Inf / Contains NaN or Inf")
                
        except Exception as e:
            print(f"  错误 / Error: {str(e)}")
        
        print()
    
    # ========================================================================
    # 步骤7: 统计信息 / Step 7: Statistics
    # ========================================================================
    print("=" * 80)
    print("统计信息 / Statistics")
    print("=" * 80)
    print(f"总采样数 / Total sampled: {batch_size}")
    print(f"有效表达式 / Valid expressions: {valid_count}")
    print(f"成功率 / Success rate: {valid_count/batch_size*100:.1f}%")
    print()
    
    print(f"平均长度 / Average length: {np.mean(lengths):.2f}")
    print(f"长度范围 / Length range: [{np.min(lengths)}, {np.max(lengths)}]")
    print()
    
    # ========================================================================
    # 步骤8: 测试概率计算 / Step 8: Test Probability Calculation
    # ========================================================================
    print("=" * 80)
    print("步骤8: 测试对数概率计算 / Step 8: Test Log Probability Calculation")
    print("=" * 80)
    
    # 计算采样表达式的对数概率 / Compute log probabilities of sampled expressions
    log_probs = policy.compute_log_probs(actions, obs, probs, lengths)
    
    print(f"对数概率形状 / Log probs shape: {log_probs.shape}")  # (batch_size,)
    print(f"平均对数概率 / Mean log prob: {log_probs.mean().item():.4f}")
    print(f"对数概率范围 / Log prob range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
    print()
    
    # 显示每个表达式的对数概率 / Show log prob for each expression
    print("各表达式的对数概率 / Log probability for each expression:")
    for i in range(batch_size):
        print(f"  表达式 {i+1}: {log_probs[i].item():.4f}")
    print()
    
    # ========================================================================
    # 步骤9: 测试熵计算 / Step 9: Test Entropy Calculation
    # ========================================================================
    print("=" * 80)
    print("步骤9: 测试策略熵 / Step 9: Test Policy Entropy")
    print("=" * 80)
    
    # 计算策略熵 (衡量策略的随机性) / Compute policy entropy (measures randomness)
    entropy = policy.compute_entropy(probs, lengths)
    
    print(f"策略熵 / Policy entropy: {entropy.item():.4f}")
    print("熵越高，策略越随机 / Higher entropy means more random policy")
    print("熵越低，策略越确定 / Lower entropy means more deterministic policy")
    print()
    
    print("=" * 80)
    print("测试完成！ / Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # 设置随机种子以便复现 / Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行测试 / Run test
    test_sampling()
