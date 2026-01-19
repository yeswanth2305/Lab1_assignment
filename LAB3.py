import numpy as np
def dot_product(vector_a, vector_b):
   
    dot_result = 0
    for i in range(len(vector_a)):
        dot_result += vector_a[i] * vector_b[i]
    return dot_result

def euclidean_norm(vector):
    sum_of_squares = 0
    for element in vector:
        sum_of_squares += element ** 2
    return sum_of_squares ** 0.5

vector_A = [1, 2, 3]
vector_B = [4, 5, 6]

numpy_A = np.array(vector_A)
numpy_B = np.array(vector_B)

manual_dot = dot_product(vector_A, vector_B)
manual_norm_A = euclidean_norm(vector_A)
manual_norm_B = euclidean_norm(vector_B)

numpy_dot = np.dot(numpy_A, numpy_B)
numpy_norm_A = np.linalg.norm(numpy_A)
numpy_norm_B = np.linalg.norm(numpy_B)

print("Manual Dot Product:", manual_dot)
print("NumPy Dot Product:", numpy_dot)

print("Manual Euclidean Norm of A:", manual_norm_A)
print("NumPy Euclidean Norm of A:", numpy_norm_A)

print("Manual Euclidean Norm of B:", manual_norm_B)
print("NumPy Euclidean Norm of B:", numpy_norm_B)