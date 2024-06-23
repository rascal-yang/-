import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

# 定义输入范围
x = np.linspace(-10, 10, 100)

# 计算激活函数的输出
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_softmax = softmax(x)
y_leaky_relu = leaky_relu(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_softmax, label='Softmax')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.xlabel('x')
plt.ylabel('Activation')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()