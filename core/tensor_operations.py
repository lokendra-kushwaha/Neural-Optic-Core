import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
logic_dir = os.path.abspath(os.path.join(current_dir, "../../", "Core_Math_Algorithms/01_Matrices_and_Determinants"))
sys.path.append(logic_dir)

from matrix_operations import Matrix # type: ignore


class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = self._calculate_shape(data)

    @classmethod
    def zeros(cls, shape):
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple.")

        data = cls._recursive_zeros(shape)

        new_tensor = cls(data=data)
        new_tensor.shape = shape

        return new_tensor

    @staticmethod
    def _recursive_zeros(shape):
        data = []
        if len(shape) == 1:
            for _ in range(shape[0]):
                data.append(0)
        else:
            for _ in range(shape[0]):
                inner_data = Tensor._recursive_zeros(shape[1:])
                data.append(inner_data)
            
        return data

    @classmethod
    def zeros_like(cls, tensor):
        return cls.zeros(tensor.shape)

    @classmethod
    def random(cls, Range, shape):
        if not isinstance(Range, tuple):
                raise ValueError("Range must be a tuple.")
        
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple.")

        data = cls._recursive_random(Range, shape)

        new_tensor = cls(data=data)
        new_tensor.shape = shape

        return new_tensor
    
    @staticmethod
    def _recursive_random(Range, shape):
        import random
        data = []
        if len(shape) == 1:
            for _ in range(shape[0]):
                data.append(random.randint(Range[0], Range[1]))
        else:
            for _ in range(shape[0]):
                inner_data = Tensor._recursive_random(Range, shape[1:])
                data.append(inner_data)
            
        return data

    @classmethod
    def Matrix_to_Tensor(cls, matrix):
        if not isinstance(matrix, Matrix):
            raise ValueError("matrix must be a object of Matrix class")

        new_tensor = cls(data=matrix.matrix)
        if matrix.row == None and matrix.col == None:
            rows = len(matrix.matrix)
            cols = len(matrix.matrix[0])

            new_tensor.shape = (rows, cols)
            return new_tensor

        new_tensor.shape = (matrix.row, matrix.col)
        return new_tensor

    def _calculate_shape(self, lst):
        shape = []
        while isinstance(lst, list):
            shape.append(len(lst))
            if len(lst) > 0:
                lst = lst[0]
            else:
                break
        return tuple(shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")
        
        # Start the recursive addition
        new_data = self._recursive_add(self.data, other.data)
        return Tensor(new_data)
        
    def _recursive_add(self, lst1, lst2):
        if isinstance(lst1, list) and isinstance(lst2, list):
            result = []
            for item1, item2 in zip(lst1, lst2):
                result.append(self._recursive_add(item1, item2))
            return result
        else:
            return lst1 + lst2

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")    

        # Start the recursive subtraction
        new_data = self._recursive_sub(self.data, other.data)
        return Tensor(new_data)

    def _recursive_sub(self, lst1, lst2):
        if isinstance(lst1, list) and isinstance(lst2, list):
            result = []
            for item1, item2 in zip(lst1, lst2):
                result.append(self._recursive_sub(item1, item2))
            return result
        else:
            return lst1 - lst2

    def __mul__(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")    

        # Start the recursive multiplication
        new_data = self._recursive_mul(self.data, other.data)
        return Tensor(new_data)
    
    def _recursive_mul(self, lst1, lst2):
        if isinstance(lst1, list) and isinstance(lst2, list):
            result = []
            for item1, item2 in zip(lst1, lst2):
                result.append(self._recursive_mul(item1, item2))
            return result
        else:
            return lst1 * lst2

    def __truediv__(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")    

        # Start the recursive division
        new_data = self._recursive_div(self.data, other.data)
        return Tensor(new_data)
    
    def _recursive_div(self, lst1, lst2):
        if isinstance(lst1, list) and isinstance(lst2, list):
            result = []
            for item1, item2 in zip(lst1, lst2):
                result.append(self._recursive_div(item1, item2))
            return result
        else:
            return lst1 / lst2

    def sum(self):
        return self._recursive_sum(self.data)

    def _recursive_sum(self, lst):
        total = 0
        if isinstance(lst, list):
            # If it's a list, go deeper and add the returned sum to the total
            for item in lst:
                total += self._recursive_sum(item)
            return total
        else:
            # BASE CASE: If it is a raw number (int or float), return it directly
            return lst

    def __gt__(self, other):
        # 1. Tensor vs Tensor
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            result = self._recursive_gt(self.data, other.data)
            
        # 2. Tensor vs Scalar
        elif isinstance(other, (int, float)):
            result = self._recursive_gt(self.data, other)
            
        else:
            raise TypeError("Comparison only supported with Tensor, int, or float")

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape
        return new_tensor

    def _recursive_gt(self, item, other):
        if isinstance(item, list):
            result = []
            
            if isinstance(other, list):
                for i1, i2 in zip(item, other):
                    result.append(self._recursive_gt(i1, i2))
            
            else:
                for i1 in item:
                    result.append(self._recursive_gt(i1, other))
                    
            return result
        
        else:
            return item > other

    def __lt__(self, other):
        # 1. Tensor vs Tensor
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            result = self._recursive_lt(self.data, other.data)
            
        # 2. Tensor vs Scalar
        elif isinstance(other, (int, float)):
            result = self._recursive_lt(self.data, other)
            
        else:
            raise TypeError("Comparison only supported with Tensor, int, or float")

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape
        return new_tensor

    def _recursive_lt(self, item, other):
        if isinstance(item, list):
            result = []
            
            if isinstance(other, list):
                for i1, i2 in zip(item, other):
                    result.append(self._recursive_lt(i1, i2))
            
            else:
                for i1 in item:
                    result.append(self._recursive_lt(i1, other))
                    
            return result
        
        else:
            return item < other

    def __ge__(self, other):
        # 1. Tensor vs Tensor
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            result = self._recursive_ge(self.data, other.data)
            
        # 2. Tensor vs Scalar
        elif isinstance(other, (int, float)):
            result = self._recursive_ge(self.data, other)
            
        else:
            raise TypeError("Comparison only supported with Tensor, int, or float")

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape
        return new_tensor

    def _recursive_ge(self, item, other):
        if isinstance(item, list):
            result = []
            
            if isinstance(other, list):
                for i1, i2 in zip(item, other):
                    result.append(self._recursive_ge(i1, i2))
            
            else:
                for i1 in item:
                    result.append(self._recursive_ge(i1, other))
                    
            return result
        
        else:
            return item >= other

    def __le__(self, other):
        # 1. Tensor vs Tensor
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            result = self._recursive_le(self.data, other.data)
            
        # 2. Tensor vs Scalar
        elif isinstance(other, (int, float)):
            result = self._recursive_le(self.data, other)
            
        else:
            raise TypeError("Comparison only supported with Tensor, int, or float")

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape
        return new_tensor

    def _recursive_le(self, item, other):
        if isinstance(item, list):
            result = []
            
            if isinstance(other, list):
                for i1, i2 in zip(item, other):
                    result.append(self._recursive_le(i1, i2))
            
            else:
                for i1 in item:
                    result.append(self._recursive_le(i1, other))
                    
            return result
        
        else:
            return item <= other

    def __eq__(self, other):
        # 1. Tensor vs Tensor
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shape must be same")
            result = self._recursive_eq(self.data, other.data)
            
        # 2. Tensor vs Scalar
        elif isinstance(other, (int, float)):
            result = self._recursive_eq(self.data, other)
            
        else:
            raise TypeError("Comparison only supported with Tensor, int, or float")

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape
        return new_tensor

    def _recursive_eq(self, item, other):
        if isinstance(item, list):
            result = []
            
            if isinstance(other, list):
                for i1, i2 in zip(item, other):
                    result.append(self._recursive_eq(i1, i2))
            
            else:
                for i1 in item:
                    result.append(self._recursive_eq(i1, other))
                    
            return result
        
        else:
            return item == other

    def apply(self, func):
        if not callable(func):
            raise TypeError("Argument must be a callable function or lambda.")

        result = self._recursive_apply(self.data, func)

        new_tensor = Tensor(data=result)
        new_tensor.shape = self.shape

        return new_tensor

    def _recursive_apply(self, lst, func):
        data = []
        if isinstance(lst, list):
            for item in lst:
                data.append(self._recursive_apply(item, func))
            return data

        else:
            return func(lst)

    def __getitem__(self, index):
        result = self.data[index]
        
        if isinstance(result, list):
            return Tensor(data=result)
        
        return result

    def __setitem__(self, index, value):
        if isinstance(value, Tensor):
            self.data[index] = value.data
        else:
            self.data[index] = value

    def __iter__(self):
        for item in self.data:
            if isinstance(item, list):
                yield Tensor(data=item)
            else:
                yield item

if __name__ == "__main__":
    lst1 = [
        [
            [2, 2, 3],
            [6, 7, 8]
        ],
        [
            [3, 4, 5],
            [4, 4, 3]
        ]
    ]

    lst2 = [
        [
            [1, 2, 3],
            [6, 7, 8]
        ],
        [
            [3, 4, 5],
            [3, 7, 9],
        ]
    ]

    T1 = Tensor(lst1)
    T2 = Tensor(lst2)
    print(T1)
    print(T2)

    T3 = T1 + T2
    print(T3.data)

    T4 = T1 - T2
    print(T4.data)


    T4 = Tensor.zeros((3, 2, 2))
    print(T4.data)

    T5 = Tensor.random((-10, 5), (3, 2, 2))
    print(T5.data)

    T6 = Tensor.random((-10, 10), (3, 2, 2))
    print(T6.data)

    print((T5 + T6).data)

    T7 = Tensor.zeros((2, 2))
    print(T7.data)

    M1 = Matrix(2, 2)
    print(M1)
    T8 = Tensor.Matrix_to_Tensor(M1)
    print(T8.data)

    lst3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    M2 = Matrix(3, 3, data=lst3)
    T9 = Tensor.Matrix_to_Tensor(M2)
    print(T9.data)

    val = Tensor.sum(T1)
    print(val)

    T10 = T1 > T2
    print(T10.data)

    T11 = T1 < T2
    print(T11.data)

    T12 = T1 >= T2
    print(T12.data)

    T13 = T1 <= T2
    print(T13.data)

    T14 = T1 == T2
    print(T14.data)

    def sum(x):
        return x**2
  
    T15 = Tensor.apply(T1, sum)
    print(T15.data)

    T16 = Tensor.zeros_like(T1)
    print(T16.data)