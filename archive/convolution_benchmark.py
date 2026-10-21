import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matrix_operations import Matrix # type: ignore
from image_convolution import apply_convolution  

def run_convolution_benchmark():
    print("🚀 Initiating Convolution Engine Benchmark...\n")
    
    N = 300
    print(f"📦 Generating a {N}x{N} Matrix test case...")
    
    # Python Lists & Custom Object Setup
    py_image = [[1.0 for _ in range(N)] for _ in range(N)]
    py_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    cust_image = Matrix(N, N, data=py_image)
    cust_kernel = Matrix(3, 3, data=py_kernel)

    # NumPy C-Array Setup
    np_image = np.ones((N, N))
    np_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    print("-" * 40)

    # ==========================================
    # TEST 1: CUSTOM 4-LOOP PURE PYTHON ENGINE
    # ==========================================
    print("⏳ Running Custom Python Engine (Pure Math & Loops)...")
    start_time = time.perf_counter()
    
    apply_convolution(cust_image, cust_kernel)
    
    custom_time = time.perf_counter() - start_time
    print(f"🔗 Custom Engine Time: {custom_time:.4f} seconds")


    # ==========================================
    # TEST 2: NUMPY C-ENGINE (Zero-Copy Strides)
    # ==========================================
    print("\n⏳ Running NumPy Engine (C-Optimized Strides)...")
    start_time = time.perf_counter()
    
    windows = sliding_window_view(np_image, np_kernel.shape)
    
    np.sum(windows * np_kernel, axis=(2, 3))
    
    numpy_time = time.perf_counter() - start_time
    print(f"🚀 NumPy Engine Time:   {numpy_time:.6f} seconds")

    # ==========================================
    # THE VERDICT
    # ==========================================
    print("-" * 40)
    speed_diff = custom_time / numpy_time
    print(f"🏆 Verdict: NumPy C-Engine is {speed_diff:.2f}x faster!")

run_convolution_benchmark()