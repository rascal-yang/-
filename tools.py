xxx = "D:/software/Python/Hand_written_digits_recognition/data/MNIST/raw"

import numpy as np

# 读取MNIST数据集
def load_mnist(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(-1, 28*28)
    return data / 255.0

def build_labels(y):
    m = len(y)  # 样本数量
    num_classes = 10  # 类别数量
    matrix = np.zeros((m, num_classes))  # 创建零矩阵
    matrix[np.arange(m), y.astype(int)] = 1  # 根据索引设置对应位置的元素为1

    return matrix

def load_labels(path):
    with open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

# 数据预处理
def preprocess_data():
    train_data = load_mnist(xxx+'/train-images-idx3-ubyte')
    train_labels = load_labels(xxx+'/train-labels-idx1-ubyte')
    test_data = load_mnist(xxx+'/t10k-images-idx3-ubyte')
    test_labels = load_labels(xxx+'/t10k-labels-idx1-ubyte')

    return train_data, train_labels, test_data, test_labels

# 激活函数及其梯度
def ReLU(z):
    # ReLU 激活函数
    return np.maximum(0, z)

def relu_gradient(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    # sigmoid 激活函数
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    sig = sigmoid(x)
    return sig * sig * np.exp(-x)

def softmax(x):
    exp_scores = np.exp(x)
    # softmax 对输出进行归一化 输出分类概率
    return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

def softmax_gradient(s):
    # 输入s是softmax函数的输出 假设其形状是(m, n) m是样本数 n是类别数
    # 我们需要计算每个样本的softmax梯度 因此输出将是一个形状为(m, n, n)的张量
    
    # 首先 获取必要的维度
    m, n = s.shape
    
    # 创建输出梯度张量 初始化为0
    grad_s = np.zeros((m, n, n))
    
    # 计算每个样本的梯度
    for i in range(m):
        for j in range(n):
            for k in range(n):
                if j == k:
                    grad_s[i, j, k] = s[i, j] * (1 - s[i, k])
                else:
                    grad_s[i, j, k] = -s[i, j] * s[i, k]
    
    return grad_s

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# 损失函数及其梯度
def cross_entropy_loss(Y_pred, Y_true):
    # 交叉熵损失函数
    m = Y_pred.shape[0]     # 样本数量
    log_probs = -np.log(Y_pred + 1e-10) * Y_true    # 修正概率矩阵
    loss = np.sum(log_probs) / m
    return loss

def cel_gradient(Y_pred, Y_true):
    # 交叉熵函数导数
    m = Y_pred.shape[0]     # 样本数量
    cel_g = (-1 * Y_true) / (Y_pred + 1e-10)
    return np.sum(cel_g) / m

def mean_squared_error(Y_pred, Y_true):
    # 均方差损失函数
    m = Y_pred.shape[0]     # 样本数量
    loss = np.sum((Y_pred - Y_true) ** 2) / (2 * m)
    return loss

def soft_cross_gradient(Y_pred, Y_true):
    '''交叉熵-softmax 导数'''
    return Y_pred - Y_true

def soft_mean_gradient(Y_pred, Y_true):
    '''均方误差-softmax导数'''
    '''
    均方误差对Softmax输入的梯度是 (y_pred - y_true) * softmax_derivative(z)
    其中 z 是Softmax函数的输入 y_pred 是Softmax的输出 
    y_true 是真实标签
    但由于Softmax梯度的计算比较复杂
    这里直接使用 (y_pred - y_true) 作为梯度通常也是可行的
    因为在误差表面上 这个简化的梯度仍然指向正确的方向（即误差减少的方向）
    '''
    grad = Y_pred - Y_true 
    return grad

# 损失函数正则化
def cross_entropy_loss_with_regularization(reg, Y_pred, Y_true, lambda_, *args):
    # 交叉熵损失函数（带正则化）
    m = Y_pred.shape[0]     # 样本数量
    log_probs = -np.log(Y_pred + 1e-10) * Y_true    # 修正概率矩阵
    cross_entropy_loss = np.sum(log_probs) / m

    regularization_term = 0
    if reg == 'L2':
        # 计算L2正则化项
        for weight_matrix in args:
            regularization_term += np.sum(np.square(weight_matrix))
    elif reg == 'L1':
        # 计算L2正则化项
        for weight_matrix in args:
            regularization_term += np.sum(np.abs(weight_matrix))
    else:
        regularization_term = 0

    # 添加正则化项到损失函数中
    loss = cross_entropy_loss + (lambda_ / (2 * m)) * regularization_term

    return loss

