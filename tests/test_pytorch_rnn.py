"""
Validation script for PyTorch RNN Policy implementation
PyTorch RNN策略实现的验证脚本

This script performs basic validation tests to ensure the PyTorch implementation
is working correctly without requiring the full DSO environment.

该脚本执行基本验证测试，以确保PyTorch实现在不需要完整DSO环境的情况下正常工作。
"""

import sys
import os
import traceback

# Check if PyTorch is available / 检查PyTorch是否可用
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("✓ PyTorch is available / PyTorch可用")
    print(f"  Version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch not found. Install with: pip install torch>=1.7.0")
    print("✗ 未找到PyTorch。使用以下命令安装：pip install torch>=1.7.0")
    sys.exit(1)

# Add parent to path / 将父目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that the module can be imported / 测试模块是否可以导入"""
    print("\n" + "="*80)
    print("Test 1: Module Imports / 测试1：模块导入")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import (
            LinearWrapper, 
            MultiLayerRNN, 
            PyTorchRNNPolicy
        )
        print("✓ Successfully imported PyTorch RNN policy classes")
        print("✓ 成功导入PyTorch RNN策略类")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        print(f"✗ 导入失败：{e}")
        traceback.print_exc()
        return False


def test_linear_wrapper():
    """Test LinearWrapper class / 测试LinearWrapper类"""
    print("\n" + "="*80)
    print("Test 2: LinearWrapper / 测试2：LinearWrapper")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import LinearWrapper
        
        # Create a simple LSTM cell / 创建简单的LSTM单元
        lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=1, batch_first=False)
        
        # Wrap it / 包装它
        wrapped_cell = LinearWrapper(lstm, output_size=20)
        
        # Test forward pass / 测试前向传播
        batch_size = 4
        seq_len = 5
        input_tensor = torch.randn(seq_len, batch_size, 10)
        
        logits, hidden = wrapped_cell(input_tensor)
        
        # Check output shape / 检查输出形状
        expected_shape = (seq_len, batch_size, 20)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        print(f"✓ LinearWrapper works correctly")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"✓ LinearWrapper工作正常")
        print(f"  输入形状：{input_tensor.shape}")
        print(f"  输出形状：{logits.shape}")
        return True
    except Exception as e:
        print(f"✗ LinearWrapper test failed: {e}")
        print(f"✗ LinearWrapper测试失败：{e}")
        traceback.print_exc()
        return False


def test_multi_layer_rnn():
    """Test MultiLayerRNN class / 测试MultiLayerRNN类"""
    print("\n" + "="*80)
    print("Test 3: MultiLayerRNN / 测试3：MultiLayerRNN")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import MultiLayerRNN
        
        # Test LSTM / 测试LSTM
        rnn_lstm = MultiLayerRNN(
            cell_type='lstm',
            input_size=10,
            num_layers=2,
            num_units=[32, 16]
        )
        
        batch_size = 4
        seq_len = 5
        input_tensor = torch.randn(seq_len, batch_size, 10)
        
        output, hidden = rnn_lstm(input_tensor)
        
        expected_shape = (seq_len, batch_size, 16)  # Last layer has 16 units
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print(f"✓ LSTM MultiLayerRNN works")
        print(f"  Layers: 2")
        print(f"  Units: [32, 16]")
        print(f"  Output shape: {output.shape}")
        
        # Test GRU / 测试GRU
        rnn_gru = MultiLayerRNN(
            cell_type='gru',
            input_size=10,
            num_layers=1,
            num_units=[32]
        )
        
        output_gru, hidden_gru = rnn_gru(input_tensor)
        
        print(f"✓ GRU MultiLayerRNN works")
        print(f"  Layers: 1")
        print(f"  Units: [32]")
        print(f"  Output shape: {output_gru.shape}")
        
        print(f"✓ MultiLayerRNN工作正常")
        return True
    except Exception as e:
        print(f"✗ MultiLayerRNN test failed: {e}")
        print(f"✗ MultiLayerRNN测试失败：{e}")
        traceback.print_exc()
        return False


def test_parameter_initialization():
    """Test weight initialization / 测试权重初始化"""
    print("\n" + "="*80)
    print("Test 4: Weight Initialization / 测试4：权重初始化")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import LinearWrapper, MultiLayerRNN
        
        # Create model / 创建模型
        rnn = MultiLayerRNN('lstm', input_size=10, num_layers=1, num_units=[32])
        wrapped = LinearWrapper(rnn, output_size=20)
        
        # Count parameters / 计数参数
        total_params = sum(p.numel() for p in wrapped.parameters())
        trainable_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"✓ 模型创建成功")
        print(f"  总参数：{total_params:,}")
        print(f"  可训练参数：{trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        print(f"✗ 初始化测试失败：{e}")
        traceback.print_exc()
        return False


def test_device_placement():
    """Test CPU and GPU device placement / 测试CPU和GPU设备放置"""
    print("\n" + "="*80)
    print("Test 5: Device Placement / 测试5：设备放置")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import LinearWrapper, MultiLayerRNN
        
        # Test CPU / 测试CPU
        rnn_cpu = MultiLayerRNN('lstm', input_size=10, num_layers=1, num_units=[32])
        wrapped_cpu = LinearWrapper(rnn_cpu, output_size=20)
        wrapped_cpu.to('cpu')
        
        input_cpu = torch.randn(3, 2, 10)
        output_cpu, _ = wrapped_cpu(input_cpu)
        
        assert output_cpu.device.type == 'cpu', "Output should be on CPU"
        print(f"✓ CPU device works")
        print(f"✓ CPU设备工作正常")
        
        # Test GPU if available / 如果可用，测试GPU
        if torch.cuda.is_available():
            rnn_gpu = MultiLayerRNN('lstm', input_size=10, num_layers=1, num_units=[32])
            wrapped_gpu = LinearWrapper(rnn_gpu, output_size=20)
            wrapped_gpu.to('cuda')
            
            input_gpu = torch.randn(3, 2, 10).cuda()
            output_gpu, _ = wrapped_gpu(input_gpu)
            
            assert output_gpu.device.type == 'cuda', "Output should be on GPU"
            print(f"✓ GPU device works")
            print(f"✓ GPU设备工作正常")
        else:
            print(f"⊘ GPU not available, skipping GPU test")
            print(f"⊘ GPU不可用，跳过GPU测试")
        
        return True
    except Exception as e:
        print(f"✗ Device placement test failed: {e}")
        print(f"✗ 设备放置测试失败：{e}")
        traceback.print_exc()
        return False


def test_save_and_load():
    """Test model saving and loading / 测试模型保存和加载"""
    print("\n" + "="*80)
    print("Test 6: Save and Load / 测试6：保存和加载")
    print("="*80)
    
    try:
        from dso.policy.pytorch_rnn_policy import LinearWrapper, MultiLayerRNN
        import tempfile
        
        # Create model / 创建模型
        rnn = MultiLayerRNN('lstm', input_size=10, num_layers=1, num_units=[32])
        wrapped = LinearWrapper(rnn, output_size=20)
        
        # Save to temporary file / 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        torch.save(wrapped.state_dict(), temp_path)
        print(f"✓ Model saved to {temp_path}")
        print(f"✓ 模型已保存到 {temp_path}")
        
        # Load into new model / 加载到新模型
        rnn_new = MultiLayerRNN('lstm', input_size=10, num_layers=1, num_units=[32])
        wrapped_new = LinearWrapper(rnn_new, output_size=20)
        wrapped_new.load_state_dict(torch.load(temp_path))
        
        print(f"✓ Model loaded successfully")
        print(f"✓ 模型加载成功")
        
        # Clean up / 清理
        os.remove(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        print(f"✗ 保存/加载测试失败：{e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests / 运行所有验证测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PyTorch RNN Policy Validation Tests" + " " * 28 + "║")
    print("║" + " " * 15 + "PyTorch RNN策略验证测试" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    
    tests = [
        test_imports,
        test_linear_wrapper,
        test_multi_layer_rnn,
        test_parameter_initialization,
        test_device_placement,
        test_save_and_load
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            print(f"✗ 测试崩溃：{e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary / 总结
    print("\n" + "=" * 80)
    print("Summary / 总结")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"测试通过：{passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! / 所有测试通过！")
        print("✓ The PyTorch RNN implementation is working correctly.")
        print("✓ PyTorch RNN实现工作正常。")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        print(f"✗ {total - passed}个测试失败。")
        print("✗ Please check the errors above.")
        print("✗ 请检查上面的错误。")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
