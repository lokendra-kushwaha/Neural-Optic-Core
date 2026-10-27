from core.tensor_operations import Tensor
import numpy as np

def apply_relu_py(tensor):
    return Tensor.apply(tensor, lambda x: x if x > 0 else 0)


def apply_relu_np(image_tensor):
    return np.maximum(0, image_tensor)