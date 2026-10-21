import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

from matrix_operations import Matrix # type: ignore

def apply_convolution(image, kernel):

    out_rows = image.row - kernel.row + 1
    out_cols = image.col - kernel.col + 1
    
    result = Matrix.zeros(out_rows, out_cols)
    
    for i in range(out_rows):
        for j in range(out_cols):
            
            slice_data = []
            
            for r in range(kernel.row):
                for c in range(kernel.col):
                    slice_data.append(image[i + r][j + c])
                    
            sliced_matrix = Matrix(kernel.row, kernel.col, data=slice_data)
            final_val = Matrix.sum(sliced_matrix * kernel)
            
            result[i][j] = final_val
            
    return result
        

if __name__ == "__main__":
    # 1. The Dummy Image (5x5 Matrix)
    image_list = [
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0]
    ]

    # 2. The Vertical Edge Detection Kernel (3x3 Matrix)
    kernel_list = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
    ]

    image = Matrix(5, 5, data=image_list)
    kernel = Matrix(3, 3, data=kernel_list)

    print(apply_convolution(image, kernel))
