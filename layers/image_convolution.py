from core.tensor_operations import Tensor

def apply_convolution_py(image, kernel):
    
    # 1. Understanding the 3D Shape (Channels, Height, Width)
    channels = image.shape[0]
    img_h = image.shape[1]
    img_w = image.shape[2]
    
    k_channels = kernel.shape[0]
    k_h = kernel.shape[1]
    k_w = kernel.shape[2]
    
    # Depth of kernel MUST match depth of image
    if channels != k_channels:
        raise ValueError("Filter depth (Channels) must match Image depth!")

    # 2. Correct Output Dimensions (Applying N - F + 1 on Height and Width)
    out_rows = img_h - k_h + 1
    out_cols = img_w - k_w + 1

    # Output Feature Map is always 3D
    result = Tensor.zeros((1, out_rows, out_cols))
    
    # 3. The Sliding Window (Torch moving across Height and Width)
    for i in range(out_rows):
        for j in range(out_cols):
            
            # STEP A: Extracting the 3D Patch
            patch_data = []
            for c in range(channels): # Loop through all depth channels (R, G, B)
                channel_layer = []
                for r in range(k_h):
                    row_data = []
                    for col in range(k_w):
                        # Use  __getitem__ chaining to extract exact scalar pixel
                        pixel = image[c][i + r][j + col]
                        row_data.append(pixel)
                    channel_layer.append(row_data)
                patch_data.append(channel_layer)
                
            # Convert the raw 3D list patch into our Custom Tensor Object
            patch_tensor = Tensor(patch_data)
            
            # STEP B: The Convolution Math 
            # 1. Element-wise multiplication (Tensor * Tensor)
            multiplied = patch_tensor * kernel 
            
            # 2. Sum up ALL elements in the 3D block to get ONE single scalar
            final_pixel_value = multiplied.sum()
            
            # 3. Store in our 2D output grid
            result[0][i][j] = final_pixel_value
            
    return result


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def apply_convolution_np(image, kernel):
    img_c, img_h, img_w = image.shape
    k_c, k_h, k_w = kernel.shape
    
    if img_c != k_c:
        raise ValueError("Filter depth must match Image depth!")

    # Extract all overlapping patches at once using C-level strides
    # Resulting shape: (1, out_h, out_w, img_c, k_h, k_w)
    windows = sliding_window_view(image, (img_c, k_h, k_w))
    
    # Remove the dummy first dimension
    windows = windows[0] 
    # Current shape: (out_h, out_w, img_c, k_h, k_w)
    
    # Compute dot product across the 3 kernel dimensions simultaneously
    # Axes mapping: match (img_c, k_h, k_w) from windows with kernel
    result = np.tensordot(windows, kernel, axes=([2, 3, 4], [0, 1, 2]))

    # This converts shape (H, W) to (1, H, W)
    result = np.expand_dims(result, axis=0)
    
    return result


if __name__ == "__main__":
    lst1 = [
        [
            [2, 2, 3],
            [6, 7, 8],
            [1, 2, 3],
            [2, 2, 3],
            [6, 7, 8],
            [1, 2, 3]
        ],
        [
            [3, 4, 5],
            [4, 4, 3],
            [1, 2, 3],
            [3, 4, 5],
            [4, 4, 3],
            [1, 2, 3]
        ],
        [
            [3, 4, 5],
            [4, 4, 3],
            [1, 2, 3],
            [3, 4, 5],
            [4, 4, 3],
            [1, 2, 3]
        ]
    ]

    lst2 = [
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ]

    image = Tensor(lst1)
    kernel = Tensor(lst2)

    print(apply_convolution_py(image, kernel).data)