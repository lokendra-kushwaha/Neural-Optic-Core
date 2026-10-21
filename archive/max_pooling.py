import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

from matrix_operations import Matrix # type: ignore

def apply_max_pooling(image):
    out_rows = image.row // 2
    out_cols = image.col // 2
    
    result = Matrix.zeros(out_rows, out_cols)
    
    for i in range(out_rows):
        for j in range(out_cols):
            
            slice_data = []
            
            for r in range(2):
                for c in range(2):
                    row_idx = (i * 2) + r
                    col_idx = (j * 2) + c
                    slice_data.append(image[row_idx][col_idx])

            max_val = slice_data[0]      
            for val in slice_data:
                if val > max_val:
                    max_val = val
            
            result[i][j] = max_val
            
    return result

if __name__ == "__main__":
    # The Dummy Image (6x6 Matrix)
    image_data = [
    [1, 5, 2, 4, 4, 8],
    [8, 3, 0, 1, 4, 7],
    [9, 7, 6, 2, 4, 3],
    [4, 5, 3, 8, 4, 3],
    [4, 3, 4, 6, 3, 9],
    [2, 6, 7, 8, 9, 8]
    ]

    image = Matrix(6, 6, data=image_data)
    print(apply_max_pooling(image))



    