import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

from matrix_operations import Matrix # type: ignore

def apply_relu(matrix):

    for r in range(matrix.row):
        for c in range(matrix.col):
            if matrix[r][c] < 0:
                matrix[r][c] = 0

    return matrix

if __name__ == "__main__":
    # The Dummy Image (3x3 Matrix)
    image_data = [
    [1, -5, 0],
    [-8, 3, 0],
    [9, 7, -6]
    ]

    image = Matrix(3, 3, data=image_data)
    print(apply_relu(image))