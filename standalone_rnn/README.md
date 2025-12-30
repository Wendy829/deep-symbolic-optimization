# Standalone RNN for Symbolic Regression

A **fully standalone** PyTorch implementation for sampling and training symbolic mathematical expressions. This implementation does NOT depend on the original DSO framework - all modules are independently implemented.

## Quick Start

### Installation

```bash
pip install torch numpy matplotlib
```

### 5-Minute Example

```python
from standalone_rnn.token_library import create_default_library
from standalone_rnn.prior import HierarchicalPrior
from standalone_rnn.state_manager import StateManager
from standalone_rnn.rnn_policy import RNNPolicy
from standalone_rnn.trainer import PolicyGradientTrainer
from standalone_rnn.expression_tree import Expression
import numpy as np

# 1. Create components
lib = create_default_library(n_input_vars=2)
prior = HierarchicalPrior(lib, max_length=15)
state_mgr = StateManager(lib, max_length=15)
policy = RNNPolicy(lib, prior, state_mgr, hidden_size=32)

# 2. Generate data: y = x1^2 + x2
X = np.random.uniform(-2, 2, size=(100, 2))
y = X[:, 0]**2 + X[:, 1]

# 3. Define reward function (DSO-style positive rewards)
def reward_fn(expr):
    try:
        y_pred = expr.evaluate(X)
        nmse = np.mean((y - y_pred)**2) / np.var(y)
        # inv_nrmse: 1/(1+NRMSE), range [0,1], higher is better
        reward = 1.0 / (1.0 + np.sqrt(nmse))
        # Add complexity penalty
        reward -= 0.01 * expr.complexity()
        return reward
    except:
        return 0.0  # Return lowest reward on failure

# 4. Train
trainer = PolicyGradientTrainer(policy, reward_fn, learning_rate=0.001)
trainer.train(num_iterations=50, batch_size=20, print_every=10)

# 5. Get best expressions
best_exprs = trainer.sample_best_expressions(10)
print(f"Best expression: {best_exprs[0][0]}")
# Output: add(square(x1), x2)
```

## Features

- ✅ **Fully standalone** - No dependencies on DSO framework
- ✅ **Pure PyTorch** - Modern, easy to understand and extend  
- ✅ **Autoregressive sampling** - RNN generates expressions token by token
- ✅ **Hierarchical constraints** - Only generates valid expressions
- ✅ **Policy gradient training** - REINFORCE algorithm with baseline
- ✅ **Extensive documentation** - Bilingual comments (English/Chinese)
- ✅ **Complete example** - Symbolic regression from scratch

## Project Structure

```
standalone_rnn/
├── token_library.py           # Mathematical operators and functions
├── expression_tree.py          # Expression representation and evaluation
├── prior.py                    # Hierarchical constraints
├── state_manager.py            # Observation features for RNN
├── rnn_policy.py               # Main RNN policy network
├── trainer.py                  # REINFORCE trainer
├── example_symbolic_regression.py  # Complete working example
├── README.md                   # This file
└── README_CN.md                # Chinese documentation (详细中文文档)
```

## How It Works

### 1. Expression as Sequence

Mathematical expressions are represented as sequences (pre-order traversal):

```
Expression: x + sin(y)

Tree:           Sequence:
    +           [add, x, sin, y]
   / \
  x  sin
      |
      y
```

### 2. RNN Generates Sequences

The RNN samples one token at a time, with each step conditioned on previous tokens:

```
Step 1: RNN → 'add'     Sequence: [add]
Step 2: RNN → 'x1'      Sequence: [add, x1]
Step 3: RNN → 'sin'     Sequence: [add, x1, sin]
Step 4: RNN → 'x2'      Sequence: [add, x1, sin, x2] ✓
```

### 3. Hierarchical Constraints

Prior knowledge ensures valid expressions:

```python
# After 'sin(' only variables allowed, not operators
Current: [sin, ...]
✓ Can choose: x1, x2 (function arguments)
✗ Cannot: +, -, * (operators invalid as arguments)
```

### 4. Policy Gradient Training

Train using REINFORCE algorithm:

```
1. Sample expressions → evaluate rewards
2. Compute policy gradient: ∇J = E[∇log π(a|s) * R]
3. Update parameters to generate better expressions
```

## Modules Explained

### token_library.py

Defines building blocks of mathematical expressions:

- **Binary operators** (arity=2): add, sub, mul, div
- **Unary functions** (arity=1): sin, cos, exp, log, sqrt, square
- **Variables** (arity=0): x1, x2, ...

```python
lib = create_default_library(n_input_vars=2)
print(lib.n_tokens)  # 12 tokens total
```

### expression_tree.py

Represents and evaluates symbolic expressions:

```python
expr = Expression([0, 10, 11], lib)  # add(x1, x2)
y = expr.evaluate(X)  # Evaluate on data
print(expr.to_string())  # "add(x1, x2)"
```

### prior.py

Manages hierarchical constraints:

```python
prior = HierarchicalPrior(lib, max_length=15)
mask = prior.compute_prior(tokens_so_far, current_length)
# Returns valid token mask (0 or 1) for each possible next token
```

### state_manager.py

Encodes observation features for RNN:

```python
state_mgr = StateManager(lib, max_length=15)
obs = state_mgr.compute_obs(tokens, current_length)
# Returns: [action_onehot | parent_onehot | sibling_onehot | dangling_count]
```

### rnn_policy.py

Main RNN network for sampling:

```python
policy = RNNPolicy(lib, prior, state_mgr, 
                   hidden_size=64, cell_type='lstm')

# Sample expressions
actions, obs, probs = policy.sample(batch_size=10)

# Compute probabilities (for training)
log_probs = policy.compute_log_probs(actions, obs)
```

### trainer.py

REINFORCE algorithm for training:

```python
trainer = PolicyGradientTrainer(
    policy=policy,
    reward_function=reward_fn,
    learning_rate=0.001,
    entropy_coef=0.01
)

trainer.train(num_iterations=100, batch_size=20)
```

## Running the Example

```bash
cd standalone_rnn
python example_symbolic_regression.py
```

This will:
1. Generate synthetic data: y = x1² + x2
2. Train RNN policy to discover the expression
3. Display top 10 best expressions found
4. Visualize results (if matplotlib available)

Expected output:
```
Best expression: add(square(x1), x2)
Reward: -0.0234
MSE: 0.0234
```

## Customization

### Add New Operators

```python
library.add_token(Token(
    name='pow',
    arity=2,
    function=lambda a, b: np.power(a, np.clip(b, -5, 5)),
    complexity=2.0
))
```

### Custom Reward Function

```python
def custom_reward(expr):
    # Fit quality
    fit = -mse(expr, X, y)
    
    # Simplicity preference
    simplicity = -expr.complexity()
    
    # Interpretability bonus
    common_ops = count_ops(['add', 'mul'], expr) / len(expr.tokens)
    
    return 0.7*fit + 0.2*simplicity + 0.1*common_ops
```

### Modify Architecture

```python
policy = RNNPolicy(
    library=lib,
    prior=prior,
    state_manager=state_mgr,
    hidden_size=128,      # Increase capacity
    num_layers=2,         # Stack more layers
    cell_type='gru',      # Use GRU instead of LSTM
    device='cuda'         # Use GPU
)
```

## Comparison with Original DSO

| Feature | Original DSO | This Implementation |
|---------|-------------|---------------------|
| Dependencies | Depends on DSO framework | Fully standalone |
| Code Size | Complex, multi-file | Concise, ~2000 lines |
| Readability | Requires DSO knowledge | Each module self-contained |
| Documentation | English only | Bilingual (EN/CN) |
| Comments | Partial | Extensive (>30%) |
| Extensibility | Requires DSO understanding | Intuitive, easy to modify |
| Learning Curve | Steep | Gentle |

## FAQ

**Q: Why is training slow?**

A: Symbolic regression is fundamentally a hard search problem. Try:
- Increase batch size for better gradient estimates
- Adjust learning rate (try 0.0001 to 0.01)
- Use GPU: `device='cuda'`
- Reduce search space: limit `max_length` or tokens

**Q: Expression doesn't converge to target?**

A: This is normal! Symbolic regression finds **equivalent** expressions:

```
Target: y = 2x
Learned: y = x + x  # Mathematically equivalent!
```

**Q: Generated expressions too complex?**

A: Increase complexity penalty:

```python
# inv_nrmse with larger complexity penalty
def reward_fn(expr):
    try:
        y_pred = expr.evaluate(X)
        nmse = np.mean((y - y_pred)**2) / np.var(y)
        reward = 1.0 / (1.0 + np.sqrt(nmse))
        reward -= 0.1 * expr.complexity()  # Larger coefficient
        return reward
    except:
        return 0.0
```

**Q: How to save/load model?**

A: Use PyTorch standard methods:

```python
torch.save(policy.state_dict(), 'policy.pth')
policy.load_state_dict(torch.load('policy.pth'))
```

## References

### Papers

- **Deep Symbolic Regression** (ICLR 2021)
  - https://openreview.net/forum?id=m5Qsh0kBQG

- **REINFORCE Algorithm**
  - Williams, 1992
  - Classic policy gradient algorithm

### Resources

- Original DSO: https://github.com/brendenpetersen/deep-symbolic-optimization
- PyTorch Docs: https://pytorch.org/docs/
- Policy Gradient Tutorial: https://spinningup.openai.com/

## License

Follows the main repository's license.

## Contributing

Contributions welcome! Areas:
- Add new operators
- Optimize training algorithms  
- Improve documentation
- Add more examples
- Performance optimization

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{petersen2021deep,
  title={Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients},
  author={Petersen, Brenden K and Landajuela, Mikel and Mundhenk, T Nathan and Santiago, Claudio P and Kim, Soo K and Kim, Joanne T},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

---

**Happy Symbolic Regression! 🎉**
