#!/usr/bin/env python
"""
Test script to diagnose quickstart NaN issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from standalone_rnn.token_library import create_default_library
from standalone_rnn.prior import HierarchicalPrior
from standalone_rnn.state_manager import StateManager
from standalone_rnn.rnn_policy import RNNPolicy
from standalone_rnn.expression_tree import Expression

# Generate data: y = x1^2 + x2
X = np.random.uniform(-2, 2, size=(100, 2))
y = X[:, 0]**2 + X[:, 1]

# Create components
lib = create_default_library(n_input_vars=2)
prior = HierarchicalPrior(lib, max_length=15)
state_mgr = StateManager(lib, max_length=15)
policy = RNNPolicy(lib, prior, state_mgr, hidden_size=32)

print("="*60)
print("Test Expression Sampling and Evaluation")
print("="*60)

# Sample a few expressions
actions, observations, probs, lengths = policy.sample(batch_size=5, use_prior=True)

print(f"\nSampled {len(actions)} expressions")
print(f"Lengths: {lengths}")

# Define reward function
def reward_fn(expr):
    try:
        y_pred = expr.evaluate(X)
        nmse = np.mean((y - y_pred)**2) / np.var(y)
        reward = 1.0 / (1.0 + np.sqrt(nmse))
        reward -= 0.01 * expr.complexity()
        return reward
    except:
        return 0.0

# Test each expression
print("\nTesting each expression:")
for i in range(len(actions)):
    valid_len = lengths[i]
    print(f"\nExpression {i+1}:")
    print(f"  Length: {valid_len}")
    print(f"  Tokens: {actions[i, :valid_len]}")
    
    try:
        expr = Expression(actions[i, :valid_len].tolist(), lib)
        print(f"  String: {expr.to_string()}")
        print(f"  Complexity: {expr.complexity()}")
        
        # Try to evaluate
        y_pred = expr.evaluate(X)
        print(f"  Evaluation: SUCCESS, shape={y_pred.shape}")
        print(f"  y_pred sample: {y_pred[:5]}")
        
        # Calculate reward
        reward = reward_fn(expr)
        print(f"  Reward: {reward:.6f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)
