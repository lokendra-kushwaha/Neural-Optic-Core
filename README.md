# 🧠 Neural-Optic Core
> An N-Dimensional Tensor Processing and Convolutional Engine Built from Scratch.

## 📖 Overview
Neural-Optic Core is a custom-built, highly optimized tensor processing engine designed to demonstrate the raw mathematical foundations of Convolutional Neural Networks (CNNs) and Computer Vision. Rather than relying on high-level deep learning wrappers like PyTorch or TensorFlow, this project implements 3D Convolution, dynamic Max Pooling, and ReLU activations completely from scratch in pure Python.

## ✨ Key Features
* **Dual-Engine Architecture:**
  * **Architect Mode (Pure Python):** A purely algorithmic n-nested loop implementation showcasing the fundamental mathematics of stride, padding, and kernel transformations on raw 3D tensors.
  * **Turbo Mode (NumPy C-Backend):** A highly optimized, vectorized implementation utilizing `np.tensordot` and advanced matrix reshaping for lightning-fast feature extraction on full-resolution images without memory overflow.
* **Dynamic Kernel Filtering:** Switch instantly between standard mathematical kernels (Edge Detection, Box Blur, Ridge Detection, Sharpen) applied dynamically across multi-channel (RGB) configurations.
* **Production-Grade UI:** A responsive, custom-styled Streamlit web application featuring a fixed telemetry dashboard, side-by-side optic visualizers, and seamless Dark/Light mode integration.

## 📂 System Architecture

```text
Neural-Optic_Core/
├── core/
│   ├── __init__.py
│   └── tensor_operations.py       # Custom Mathematical N-Dimensional Tensor Object
├── layers/
│   ├── __init__.py
│   ├── image_convolution.py       # Custom & Vectorized Convolution Logic
│   ├── max_pooling.py             # Dynamic Downsampling Compression
│   └── relu_activation.py         # Non-Linear Activation Mappings
└── app.py                         # Advanced Front-End Visualizer
```

## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/lokendra-kushwaha/Neural-Optic-Core.git
cd Neural-Optic-Core
```

2. **Install the dependencies:**
```bash
pip install streamlit numpy pillow
```

3. **Initialize the Core Engine:**
```bash
streamlit run app.py
```

## 🔬 The Mathematics Inside
The core engine explicitly calculates sliding window operations without black-box functions. The spatial dimensions are mathematically managed and verified using the standard convolutional stride formula:

$$\text{Output Dimension} = \lfloor \frac{N - F}{\text{Stride}} \rfloor + 1$$

Where **N** represents the input image dimension, **F** is the dynamic filter (kernel) size, and **Stride** dictates the step-size. The architecture ensures strict dimensional integrity, maintaining the crucial `(Channels, Height, Width)` 3D format throughout the entire processing pipeline.

---
**Engineered with ❤️ by Lokendra Kushwaha** | © 2026 The Neural-Optic Core Project