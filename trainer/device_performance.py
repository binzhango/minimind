import torch
import time


def benchmark_mps(size, dtype=torch.float32, iterations=100):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available.")

    a = torch.randn(size, size, dtype=dtype, device='mps').contiguous()
    b = torch.randn(size, size, dtype=dtype, device='mps').contiguous()

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.mps.synchronize()

    # 基准测试
    torch.mps.synchronize()
    start_time = time.perf_counter()

    for _ in range(iterations):
        c = torch.matmul(a, b)

    torch.mps.synchronize()
    end_time = time.perf_counter()

    # 计算时间和性能
    elapsed_sec = end_time - start_time
    elapsed_ms = elapsed_sec * 1000  # 转换为毫秒
    flops = 2 * size ** 3 * iterations
    tflops = flops / (elapsed_sec * 1e12)  # 保持使用秒计算TFLOPS

    return tflops, elapsed_ms


def benchmark_cuda(size, dtype=torch.float32, iterations=100):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available.")

    a = torch.randn(size, size, dtype=dtype, device='cuda')
    b = torch.randn(size, size, dtype=dtype, device='cuda')

    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    # torch.cuda.synchronize()

    # 基准测试
    # torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(iterations):
        c = torch.matmul(a, b)

    # torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 计算时间和性能
    elapsed_sec = end_time - start_time
    elapsed_ms = elapsed_sec * 1000  # 转换为毫秒
    flops = 2 * size ** 3 * iterations
    tflops = flops / (elapsed_sec * 1e12)  # 保持使用秒计算TFLOPS

    return tflops, elapsed_ms


def main():
    print(f"Testing on:CUDA device")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    sizes = [1024, 2048, 4096, 8192, 16384]  # 测试尺寸
    for size in sizes:
        try:
            # tflops, elapsed_ms = benchmark_mps(size)
            tflops, elapsed_ms = benchmark_cuda(size)
            print(f"Matrix {size}x{size}: {tflops:.2f} TFLOPS | Time: {elapsed_ms:.2f}ms")
        except RuntimeError as e:
            print(f"Size {size} failed: {e}")
        except Exception as e:
            print(f"Size {size} error: {str(e)}")


if __name__ == "__main__":
    main()
    """
    Testing on: macOS MPS device
    PyTorch version: 2.3.0
    MPS available: True
    
    Matrix 1024x1024: 10.40 TFLOPS | Time: 20.65ms
    Matrix 2048x2048: 13.45 TFLOPS | Time: 127.76ms
    Matrix 4096x4096: 13.49 TFLOPS | Time: 1018.53ms
    Matrix 8192x8192: 12.82 TFLOPS | Time: 8573.45ms
    Matrix 16384x16384: 9.37 TFLOPS | Time: 93871.68ms
    
    
    
    Testing on:CUDA device (2080Ti)
    PyTorch version: 2.8.0+cu129
    CUDA available: True
    
    Matrix 1024x1024: 65.62 TFLOPS | Time: 3.27ms
    Matrix 2048x2048: 634.46 TFLOPS | Time: 2.71ms
    Matrix 4096x4096: 4447.00 TFLOPS | Time: 3.09ms
    Matrix 8192x8192: 34163.30 TFLOPS | Time: 3.22ms
    Matrix 16384x16384: 199933.93 TFLOPS | Time: 4.40ms
    
    """