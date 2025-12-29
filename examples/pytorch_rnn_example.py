"""
Example script demonstrating PyTorch RNN Policy usage
演示PyTorch RNN策略使用的示例脚本

This script shows how to:
该脚本展示如何：
1. Initialize the PyTorch RNN policy
   初始化PyTorch RNN策略
2. Sample symbolic expressions
   采样符号表达式
3. Compute probabilities and entropy
   计算概率和熵
4. Compare with TensorFlow implementation (optional)
   与TensorFlow实现进行比较（可选）
"""

# This example demonstrates the API without requiring imports
# 此示例演示API而不需要导入
# 
# To run with full functionality, first install dependencies:
# 要以完整功能运行，首先安装依赖项：
# pip install numpy torch
# 
# And ensure the DSO package is installed:
# 并确保安装了DSO包：
# pip install -e ./dso

def example_basic_sampling():
    """
    Basic example: Initialize policy and sample expressions
    基础示例：初始化策略并采样表达式
    """
    print("=" * 80)
    print("Example 1: Basic Sampling / 示例1：基本采样")
    print("=" * 80)
    
    # Note: This is a simplified example showing the API structure
    # A full working example requires the complete DSO environment
    # 注意：这是显示API结构的简化示例
    # 完整的工作示例需要完整的DSO环境
    
    print("""
    To use the PyTorch RNN policy in a real application:
    要在实际应用中使用PyTorch RNN策略：
    
    1. Set up the DSO environment with task, library, and prior
       设置带有任务、库和先验的DSO环境
       
    2. Create the policy:
       创建策略：
       
       from dso.policy.pytorch_rnn_policy import PyTorchRNNPolicy
       
       policy = PyTorchRNNPolicy(
           prior=prior,
           state_manager=state_manager,
           max_length=30,
           n_choices=len(library),
           cell='lstm',
           num_layers=1,
           num_units=32,
           device='cpu'
       )
    
    3. Sample expressions:
       采样表达式：
       
       actions, obs, priors = policy.sample(n=100)
       
       # actions.shape: (100, 30) - token sequences
       # actions.shape: (100, 30) - 标记序列
       
       # obs.shape: (100, obs_dim, 30) - observations
       # obs.shape: (100, obs_dim, 30) - 观测
       
       # priors.shape: (100, 30, n_choices) - prior probs
       # priors.shape: (100, 30, n_choices) - 先验概率
    
    4. Evaluate expressions:
       评估表达式：
       
       programs = [from_tokens(a) for a in actions]
       rewards = [program.evaluate(X, y) for program in programs]
    """)


def example_probability_computation():
    """
    Example: Compute probabilities and entropy for a batch
    示例：计算批次的概率和熵
    """
    print("=" * 80)
    print("Example 2: Probability Computation / 示例2：概率计算")
    print("=" * 80)
    
    print("""
    After sampling, you can compute probabilities and entropy:
    采样后，您可以计算概率和熵：
    
    from dso.memory import Batch
    
    # Create a batch
    # 创建批次
    batch = Batch(
        actions=actions,
        obs=obs,
        priors=priors,
        lengths=np.array([len(finish_tokens(a)) for a in actions]),
        rewards=rewards,
        on_policy=True
    )
    
    # Compute negative log-probabilities and entropy
    # 计算负对数概率和熵
    neglogp, entropy = policy.compute_neglogp_and_entropy(batch)
    
    # Use for policy gradient training
    # 用于策略梯度训练
    advantages = rewards - baseline
    loss = (neglogp * advantages).mean() - 0.01 * entropy.mean()
    
    # Or compute probabilities directly
    # 或直接计算概率
    probs = policy.compute_probs(batch, log=False)
    log_probs = policy.compute_probs(batch, log=True)
    """)


def example_training_integration():
    """
    Example: Integrate with PyTorch training loop
    示例：与PyTorch训练循环集成
    """
    print("=" * 80)
    print("Example 3: Training Integration / 示例3：训练集成")
    print("=" * 80)
    
    print("""
    To train the policy with PyTorch optimizers:
    要使用PyTorch优化器训练策略：
    
    import torch
    import torch.optim as optim
    
    # Get model parameters
    # 获取模型参数
    params = policy.get_trainable_parameters()
    optimizer = optim.Adam(params, lr=0.001)
    
    # Training loop
    # 训练循环
    for epoch in range(num_epochs):
        # Sample expressions
        # 采样表达式
        actions, obs, priors = policy.sample(n=batch_size)
        
        # Evaluate rewards
        # 评估奖励
        programs = [from_tokens(a) for a in actions]
        rewards = np.array([program.evaluate(X, y) for program in programs])
        
        # Create batch
        # 创建批次
        batch = Batch(...)
        
        # Compute loss
        # 计算损失
        neglogp, entropy = policy.compute_neglogp_and_entropy(batch)
        
        # Policy gradient loss
        # 策略梯度损失
        advantages = torch.FloatTensor(rewards - baseline)
        pg_loss = (torch.FloatTensor(neglogp) * advantages).mean()
        
        # Entropy regularization
        # 熵正则化
        entropy_loss = -0.01 * torch.FloatTensor(entropy).mean()
        
        # Total loss
        # 总损失
        loss = pg_loss + entropy_loss
        
        # Optimization step
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update baseline (e.g., moving average of rewards)
        # 更新基线（例如，奖励的移动平均）
        baseline = 0.9 * baseline + 0.1 * rewards.mean()
    """)


def example_novel_sampling():
    """
    Example: Sample novel expressions not in cache
    示例：采样缓存中没有的新表达式
    """
    print("=" * 80)
    print("Example 4: Novel Sampling / 示例4：新样本采样")
    print("=" * 80)
    
    print("""
    To sample only novel expressions:
    要仅采样新表达式：
    
    # Create policy with novel sampling enabled
    # 创建启用新样本采样的策略
    policy = PyTorchRNNPolicy(
        prior=prior,
        state_manager=state_manager,
        sample_novel_batch=True,
        max_attempts_at_novel_batch=10,
        ...
    )
    
    # Sample will automatically try to generate novel expressions
    # 采样将自动尝试生成新表达式
    actions, obs, priors = policy.sample(n=100)
    
    # The policy tries to avoid expressions already in Program.cache
    # 策略尝试避免Program.cache中已有的表达式
    
    # If unable to generate 100 novel samples within max_attempts,
    # it fills remaining slots with previously seen samples
    # 如果在max_attempts内无法生成100个新样本，
    # 它会用先前见过的样本填充剩余插槽
    
    # Access the extended batch with old samples for training
    # 访问包含旧样本的扩展批次用于训练
    if policy.valid_extended_batch:
        n_old, old_actions, old_obs, old_priors = policy.extended_batch
        print(f"Generated {100-n_old} novel and {n_old} old samples")
        print(f"生成了{100-n_old}个新样本和{n_old}个旧样本")
    """)


def example_model_persistence():
    """
    Example: Save and load model weights
    示例：保存和加载模型权重
    """
    print("=" * 80)
    print("Example 5: Model Persistence / 示例5：模型持久化")
    print("=" * 80)
    
    print("""
    Save and load trained models:
    保存和加载训练的模型：
    
    # Save model weights
    # 保存模型权重
    policy.save_weights('policy_weights.pth')
    
    # Later, load the weights
    # 稍后，加载权重
    new_policy = PyTorchRNNPolicy(...)
    new_policy.load_weights('policy_weights.pth')
    
    # The loaded policy will behave identically
    # 加载的策略将表现相同
    """)


def example_gpu_usage():
    """
    Example: Use GPU for faster computation
    示例：使用GPU加速计算
    """
    print("=" * 80)
    print("Example 6: GPU Usage / 示例6：GPU使用")
    print("=" * 80)
    
    print("""
    To use GPU acceleration:
    要使用GPU加速：
    
    import torch
    
    # Check if CUDA is available
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create policy on GPU
    # 在GPU上创建策略
    policy = PyTorchRNNPolicy(
        prior=prior,
        state_manager=state_manager,
        device=device,
        ...
    )
    
    # All computations will automatically run on GPU
    # 所有计算将自动在GPU上运行
    actions, obs, priors = policy.sample(n=1000)
    
    # For very large batches, GPU can provide significant speedup
    # 对于非常大的批次，GPU可以提供显著的加速
    """)


def comparison_with_tensorflow():
    """
    Example: Compare PyTorch and TensorFlow implementations
    示例：比较PyTorch和TensorFlow实现
    """
    print("=" * 80)
    print("Example 7: TensorFlow Comparison / 示例7：TensorFlow比较")
    print("=" * 80)
    
    print("""
    Compare the two implementations:
    比较两个实现：
    
    import tensorflow as tf
    from dso.policy.rnn_policy import RNNPolicy
    from dso.policy.pytorch_rnn_policy import PyTorchRNNPolicy
    
    # Create TensorFlow policy
    # 创建TensorFlow策略
    sess = tf.Session()
    tf_policy = RNNPolicy(
        sess=sess,
        prior=prior,
        state_manager=state_manager,
        cell='lstm',
        num_layers=1,
        num_units=32
    )
    sess.run(tf.global_variables_initializer())
    
    # Create PyTorch policy with same architecture
    # 创建相同架构的PyTorch策略
    pt_policy = PyTorchRNNPolicy(
        prior=prior,
        state_manager=state_manager,
        cell='lstm',
        num_layers=1,
        num_units=32,
        device='cpu'
    )
    
    # Sample from both
    # 从两者采样
    tf_actions, tf_obs, tf_priors = tf_policy.sample(n=100)
    pt_actions, pt_obs, pt_priors = pt_policy.sample(n=100)
    
    # Both should produce valid expressions
    # 两者都应该产生有效的表达式
    print("TensorFlow actions shape:", tf_actions.shape)
    print("PyTorch actions shape:", pt_actions.shape)
    
    # Statistical properties should be similar
    # 统计属性应该相似
    print("TF mean action:", tf_actions.mean())
    print("PT mean action:", pt_actions.mean())
    
    # With same initialization and random seed, distributions should match
    # 使用相同的初始化和随机种子，分布应该匹配
    """)


def main():
    """
    Main function to run all examples
    运行所有示例的主函数
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PyTorch RNN Policy Examples" + " " * 31 + "║")
    print("║" + " " * 20 + "PyTorch RNN策略示例" + " " * 39 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    examples = [
        example_basic_sampling,
        example_probability_computation,
        example_training_integration,
        example_novel_sampling,
        example_model_persistence,
        example_gpu_usage,
        comparison_with_tensorflow
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
            print("\n")
        except Exception as e:
            print(f"Error in example {i}: {e}")
            print("\n")
    
    print("=" * 80)
    print("Summary / 总结")
    print("=" * 80)
    print("""
    The PyTorch RNN Policy provides a clean, modern interface for symbolic
    regression with all the features of the original TensorFlow implementation.
    
    PyTorch RNN策略为符号回归提供了一个干净、现代的接口，
    具有原始TensorFlow实现的所有功能。
    
    Key advantages of PyTorch version:
    PyTorch版本的主要优势：
    - Easier to debug with standard Python debugger
      使用标准Python调试器更容易调试
    - More intuitive dynamic computation graphs
      更直观的动态计算图
    - Better integration with modern PyTorch ecosystem
      与现代PyTorch生态系统更好的集成
    - No session management overhead
      没有会话管理开销
    
    For detailed documentation, see PYTORCH_RNN_GUIDE.md
    有关详细文档，请参阅PYTORCH_RNN_GUIDE.md
    """)


if __name__ == "__main__":
    main()
