from core.tensor_operations import Tensor
import numpy as np

def apply_max_pooling_py(image, pool_size=2):
    # Extract dimensions from the 3D Tensor
    channels = image.shape[0]
    img_h = image.shape[1]
    img_w = image.shape[2]

    # Calculate dynamic output dimensions
    out_rows = img_h // pool_size
    out_cols = img_w // pool_size

    # Initialize a 3D Tensor for the output
    result = Tensor.zeros((channels, out_rows, out_cols))

    # Iterate over each depth channel
    for c in range(channels):
        
        # Iterate over the new spatial dimensions
        for i in range(out_rows):
            for j in range(out_cols):

                slice_data = []
                
                # Dynamic extraction using pool_size as both the window size and the stride
                for r in range(pool_size):
                    for c_idx in range(pool_size):
                        
                        # The math: (current_step * pool_size) + internal_offset
                        row_idx = (i * pool_size) + r
                        col_idx = (j * pool_size) + c_idx
                        
                        pixel = image[c][row_idx][col_idx]
                        slice_data.append(pixel)

                # Find the maximum value in the dynamically sized patch
                max_val = slice_data[0]      
                for val in slice_data:
                    if val > max_val:
                        max_val = val
                
                # Assign to the output feature map
                result[c][i][j] = max_val

    return result


def apply_max_pooling_np(image, pool_size=2):
    channels, img_h, img_w = image.shape
    
    out_h = img_h // pool_size
    out_w = img_w // pool_size
    
    # Slice the image to drop remaining pixels if dimensions are not perfectly divisible
    cropped_image = image[:, :out_h * pool_size, :out_w * pool_size]
    
    # The Numpy Reshape Magic: Fold the grid into pool blocks
    reshaped_tensor = cropped_image.reshape(channels, out_h, pool_size, out_w, pool_size)
    
    # Take the maximum value across the pool_size axes (axis 2 and 4)
    result = reshaped_tensor.max(axis=(2, 4))
    
    return result