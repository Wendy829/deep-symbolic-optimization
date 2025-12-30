"""
符号回归任务 / Symbolic Regression Task

提供内置的奖励函数，用户无需每次自定义。
Provides built-in reward functions so users don't need to customize each time.

与DSO对齐的奖励函数 / DSO-aligned reward functions:
- inv_nrmse: 1/(1+NRMSE) [0,1], 越高越好 / higher is better
- neg_nmse: -NMSE, 越高越好（越接近0）/ higher is better (closer to 0)
- neg_mse: -MSE, 越高越好（越接近0）/ higher is better (closer to 0)
"""

import numpy as np
from typing import Callable, Optional, Tuple

# Handle imports for both package and direct script execution
try:
    from .expression_tree import Expression
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from standalone_rnn.expression_tree import Expression


class SymbolicRegressionTask:
    """
    符号回归任务类
    Symbolic Regression Task Class
    
    集成了常用的奖励函数，与DSO对齐。
    Integrates common reward functions, aligned with DSO.
    
    使用方法 / Usage:
    --------------
    ```python
    task = SymbolicRegressionTask(X_train, y_train, metric='inv_nrmse')
    reward = task.reward(expression)
    ```
    """
    
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 metric: str = 'inv_nrmse',
                 metric_params: Tuple = (1.0,),
                 complexity_penalty: float = 0.001,
                 protected: bool = True,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None):
        """
        参数 / Parameters:
        ----------------
        X_train: np.ndarray, shape (n_samples, n_features)
            训练输入数据
            Training input data
            
        y_train: np.ndarray, shape (n_samples,)
            训练目标数据
            Training target data
            
        metric: str
            奖励度量类型 / Reward metric type
            - 'inv_nrmse': 1/(1+NRMSE), DSO默认 / DSO default
            - 'neg_nmse': -NMSE
            - 'neg_mse': -MSE
            - 'neg_nrmse': -NRMSE
            - 'pearson': Pearson correlation coefficient
            
        metric_params: Tuple
            度量参数（用于inv_nrmse的缩放因子等）
            Metric parameters (e.g., scaling factor for inv_nrmse)
            
        complexity_penalty: float
            复杂度惩罚系数
            Complexity penalty coefficient
            
        protected: bool
            是否使用保护性运算（避免除零、log负数等）
            Whether to use protected operations (avoid division by zero, log of negative, etc.)
            
        X_test: Optional[np.ndarray]
            测试输入数据（可选）
            Test input data (optional)
            
        y_test: Optional[np.ndarray]
            测试目标数据（可选）
            Test target data (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.metric = metric
        self.metric_params = metric_params
        self.complexity_penalty = complexity_penalty
        self.protected = protected
        
        # 计算y的方差，用于归一化 / Compute variance of y for normalization
        self.y_train_var = np.var(y_train)
        if self.y_train_var == 0:
            self.y_train_var = 1.0  # 避免除零 / Avoid division by zero
            
    def reward(self, expr: Expression) -> float:
        """
        计算表达式的奖励
        Compute reward for an expression
        
        参数 / Parameters:
        ----------------
        expr: Expression
            要评估的表达式
            Expression to evaluate
            
        返回 / Returns:
        -------------
        reward: float
            奖励值（越高越好）
            Reward value (higher is better)
        """
        try:
            # 评估表达式 / Evaluate expression
            y_pred = expr.evaluate(self.X_train)
            
            # 检查预测值的有效性 / Check validity of predictions
            if not np.all(np.isfinite(y_pred)):
                return 0.0
            
            # 计算基础度量 / Compute base metric
            if self.metric == 'inv_nrmse':
                # DSO默认: 1/(1+NRMSE)
                # DSO default: 1/(1+NRMSE)
                mse = np.mean((self.y_train - y_pred)**2)
                nmse = mse / self.y_train_var
                nrmse = np.sqrt(nmse)
                scaling_factor = self.metric_params[0] if self.metric_params else 1.0
                reward = scaling_factor / (scaling_factor + nrmse)
                
            elif self.metric == 'neg_nmse':
                # 负的归一化MSE / Negative normalized MSE
                mse = np.mean((self.y_train - y_pred)**2)
                nmse = mse / self.y_train_var
                reward = -nmse
                
            elif self.metric == 'neg_mse':
                # 负的MSE / Negative MSE
                mse = np.mean((self.y_train - y_pred)**2)
                reward = -mse
                
            elif self.metric == 'neg_nrmse':
                # 负的NRMSE / Negative NRMSE
                mse = np.mean((self.y_train - y_pred)**2)
                nmse = mse / self.y_train_var
                nrmse = np.sqrt(nmse)
                reward = -nrmse
                
            elif self.metric == 'pearson':
                # Pearson相关系数 / Pearson correlation coefficient
                correlation = np.corrcoef(self.y_train, y_pred)[0, 1]
                if np.isnan(correlation):
                    return 0.0
                reward = correlation
                
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            # 添加复杂度惩罚 / Add complexity penalty
            complexity = expr.complexity()
            reward -= self.complexity_penalty * complexity
            
            return float(reward)
            
        except Exception as e:
            # 如果评估失败，返回最低奖励 / Return lowest reward on failure
            return 0.0
    
    def evaluate_test(self, expr: Expression) -> Optional[float]:
        """
        在测试集上评估表达式
        Evaluate expression on test set
        
        参数 / Parameters:
        ----------------
        expr: Expression
            要评估的表达式
            Expression to evaluate
            
        返回 / Returns:
        -------------
        score: Optional[float]
            测试集上的得分（如果有测试集）
            Score on test set (if test set available)
        """
        if self.X_test is None or self.y_test is None:
            return None
            
        try:
            y_pred = expr.evaluate(self.X_test)
            if not np.all(np.isfinite(y_pred)):
                return None
            
            # 使用相同的度量 / Use same metric
            if self.metric == 'inv_nrmse':
                mse = np.mean((self.y_test - y_pred)**2)
                y_test_var = np.var(self.y_test)
                if y_test_var == 0:
                    y_test_var = 1.0
                nmse = mse / y_test_var
                nrmse = np.sqrt(nmse)
                scaling_factor = self.metric_params[0] if self.metric_params else 1.0
                return scaling_factor / (scaling_factor + nrmse)
            else:
                # 其他度量类似 / Other metrics similar
                mse = np.mean((self.y_test - y_pred)**2)
                return -mse
                
        except Exception:
            return None


# 创建便捷函数 / Create convenience function
def create_task(X_train: np.ndarray,
                y_train: np.ndarray,
                metric: str = 'inv_nrmse',
                **kwargs) -> SymbolicRegressionTask:
    """
    创建符号回归任务的便捷函数
    Convenience function to create symbolic regression task
    
    参数 / Parameters:
    ----------------
    X_train: np.ndarray
        训练输入
        Training inputs
        
    y_train: np.ndarray
        训练目标
        Training targets
        
    metric: str
        奖励度量类型
        Reward metric type
        
    **kwargs: 其他参数传递给SymbolicRegressionTask
             Other arguments passed to SymbolicRegressionTask
    
    返回 / Returns:
    -------------
    task: SymbolicRegressionTask
        符号回归任务对象
        Symbolic regression task object
    """
    return SymbolicRegressionTask(X_train, y_train, metric=metric, **kwargs)


if __name__ == "__main__":
    # 测试代码 / Test code
    print("="*60)
    print("测试符号回归任务 / Testing Symbolic Regression Task")
    print("="*60)
    
    # 生成测试数据 / Generate test data
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(100, 2))
    y = X[:, 0]**2 + X[:, 1]  # y = x1^2 + x2
    
    # 创建任务 / Create task
    task = create_task(X, y, metric='inv_nrmse', complexity_penalty=0.001)
    print(f"\n✓ 任务创建成功 / Task created successfully")
    print(f"  度量 / Metric: {task.metric}")
    print(f"  复杂度惩罚 / Complexity penalty: {task.complexity_penalty}")
    
    # 测试几个表达式 / Test a few expressions
    from standalone_rnn.token_library import create_default_library
    
    lib = create_default_library(n_input_vars=2)
    
    # 正确的表达式: add(square(x1), x2)
    # Correct expression: add(square(x1), x2)
    # tokens: [add=0, square=9, x1=10, x2=11]
    expr1 = Expression([0, 9, 10, 11], lib)
    reward1 = task.reward(expr1)
    print(f"\n表达式 1 / Expression 1: {expr1}")
    print(f"  奖励 / Reward: {reward1:.6f}")
    
    # 简单但不准确的表达式: x1
    # Simple but inaccurate expression: x1
    expr2 = Expression([10], lib)
    reward2 = task.reward(expr2)
    print(f"\n表达式 2 / Expression 2: {expr2}")
    print(f"  奖励 / Reward: {reward2:.6f}")
    
    # 复杂的表达式: add(add(mul(x1, x1), x2), sin(x1))
    # Complex expression: add(add(mul(x1, x1), x2), sin(x1))
    expr3 = Expression([0, 0, 2, 10, 10, 11, 4, 10], lib)
    reward3 = task.reward(expr3)
    print(f"\n表达式 3 / Expression 3: {expr3}")
    print(f"  奖励 / Reward: {reward3:.6f}")
    
    print("\n" + "="*60)
    print("测试完成 / Testing Complete")
    print("="*60)
