import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matrix_operations import Matrix # type: ignore
from image_convolution import apply_convolution
from relu_activation import apply_relu
from max_pooling import apply_max_pooling

print("🚀 Starting the Custom CNN Pipeline...")



# 1. Image Load & Grayscale Convert
img_array = mpimg.imread(r"C:\Users\loken\OneDrive\Pictures\919026024270.jpg")
grayscale_list = []
for i in range(len(img_array)):
    row = []
    for j in range(len(img_array[0])):
        r, g, b = img_array[i][j][:3] 
        gray_value = (int(r) + int(g) + int(b)) / 3.0
        row.append(gray_value)
    grayscale_list.append(row)
image_matrix = Matrix(len(grayscale_list), len(grayscale_list[0]), data=grayscale_list)

# 2. Sobel Filter
kernel_list = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
kernel = Matrix(3, 3, data=kernel_list)

# ==========================================
# 🧠 THE AI BRAIN (FORWARD PASS)
# ==========================================
print("⏳ Step 1: Running Convolution (Edge Detection)...")
conv_result = apply_convolution(image_matrix, kernel)

print("⏳ Step 2: Applying ReLU (Removing Negative Noise)...")
relu_result = apply_relu(conv_result)

print("⏳ Step 3: Running Max Pooling (Downsampling)...")
final_result = apply_max_pooling(relu_result)

print("✅ Pipeline Complete!")
# ==========================================

# 3. Visualize
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(grayscale_list, cmap='gray')
ax[0].set_title(f"1. Original ({image_matrix.row}x{image_matrix.col})")

ax[1].imshow(relu_result.matrix, cmap='gray')
ax[1].set_title(f"2. Conv + ReLU ({relu_result.row}x{relu_result.col})")

ax[2].imshow(final_result.matrix, cmap='gray')
ax[2].set_title(f"3. Max Pooled ({final_result.row}x{final_result.col})")

plt.show()