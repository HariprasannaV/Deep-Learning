import tensorflow as tf

# Checking TensorFlow version
print("TensorFlow version:", tf.__version__)

# Initializing constants
x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

# Multiplying two tensors
result = tf.multiply(x1, x2)
print(result)

# Adding a scalar to a list
x = [1, 2, 3, 4, 5]
y = 1
print(tf.add(x, y))

# Using binary + operator
x = tf.convert_to_tensor([1, 2, 3, 4, 5])
y = tf.convert_to_tensor(1)
print(x + y)

# Using binary * operator
x = tf.convert_to_tensor([1, 2, 3, 4, 5])
y = tf.convert_to_tensor(2)
print(x * y)

# Using binary - operator
x = tf.convert_to_tensor([1, 2, 3, 4, 5])
y = tf.convert_to_tensor(6)
print(x - y)

# Creating vectors and matrices
vector = tf.constant([[10], [10]])
print("Vector dimensions:", vector.ndim)

matrix = tf.constant([[1, 2], [3, 4]])
print(matrix)

# Performing matrix operations
matrix1 = tf.constant([[2, 4], [6, 8]])
print(matrix + matrix1)

matrix2 = tf.constant([[3, 4], [7, 8]])
print(matrix + matrix1 + matrix2)

print(matrix1 - matrix)

print(matrix1 * matrix)

print(matrix1 / matrix)

print(tf.transpose(matrix))

print("Dot product of matrices:", tf.tensordot(matrix, matrix, axes=1))
