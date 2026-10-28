import time
import numpy as np
from PIL import Image

# Import Custom Engine
from core.tensor_operations import Tensor
from layers.image_convolution import apply_convolution_py, apply_convolution_np
from layers.max_pooling import apply_max_pooling_py, apply_max_pooling_np
from layers.relu_activation import apply_relu_py, apply_relu_np

def load_and_preprocess_image(image_path, size=(32, 32)):
    # 1. Load Image and Resize (Crop/Shrink)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    
    # 2. Convert to Numpy Array (HWC Format: Height x Width x Channels)
    img_array = np.array(img)
    
    # 3. Transpose to CHW Format (Deep Learning Standard: Channels x Height x Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array

def test_custom_engine(img_array, kernel_list):
    print("\n--- RUNNING CUSTOM PURE PYTHON ENGINE ---")
    start_time = time.time()
    
    # Convert numpy array to Custom Tensor Object
    image_tensor = Tensor(img_array.tolist())
    kernel_tensor = Tensor(kernel_list)
    
    # Step 1: Convolution
    print("Running Convolution...")
    conv_out = apply_convolution_py(image_tensor, kernel_tensor)
    
    # Step 2: ReLU Activation
    print("Running ReLU...")
    relu_out = apply_relu_py(conv_out)
    
    # Step 3: Max Pooling (Dynamic 2x2)
    print("Running Max Pooling...")
    pool_out = apply_max_pooling_py(relu_out, pool_size=2)
    
    end_time = time.time()
    print(f"Custom Engine Time: {end_time - start_time:.4f} seconds")
    print(f"Final Output Shape: {pool_out.shape}")
    return pool_out

def test_numpy_engine(img_array, kernel_list):
    print("\n--- RUNNING NUMPY (TURBO) ENGINE ---")
    start_time = time.time()
    
    # Use native numpy arrays
    kernel_array = np.array(kernel_list)
    
    # Step 1: Convolution
    print("Running Convolution...")
    conv_out = apply_convolution_np(img_array, kernel_array)
    
    # Step 2: ReLU
    print("Running ReLU...")
    relu_out = apply_relu_np(conv_out)
    
    # Step 3: Max Pooling
    print("Running Max Pooling...")
    pool_out = apply_max_pooling_np(relu_out, pool_size=2)
    
    end_time = time.time()
    print(f"NumPy Engine Time: {end_time - start_time:.4f} seconds")
    print(f"Final Output Shape: {pool_out.shape}")
    return pool_out

if __name__ == "__main__":
    # Edge Detection 3x3x3 Kernel (Channels x Height x Width)
    kernel_3d = [
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]], # Red Channel
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]], # Green Channel
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]]  # Blue Channel
    ]
    
    IMAGE_PATH = r"C:\Users\loken\OneDrive\Pictures\919026024270.jpg"
    
    try:
        small_img = load_and_preprocess_image(IMAGE_PATH, size=(32, 32))
        test_custom_engine(small_img, kernel_3d)
        
        large_img = load_and_preprocess_image(IMAGE_PATH, size=(128, 128))
        test_numpy_engine(large_img, kernel_3d)
        
    except FileNotFoundError:
        print(f"File no found in '{IMAGE_PATH}'!")