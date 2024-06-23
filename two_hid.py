import numpy as np
import time
import matplotlib.pyplot as plt
from tools import *

# 定义神经网络模型
class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size,
                 lossmodel=1, Activation='relu', reg=''):
        self.W1 = np.random.randn(input_size, hidden_size_1) * 0.01   # （784, hid_size_1)
        self.b1 = np.zeros((1, hidden_size_1))                        # (1, hid_size_1)
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.01  # (hid_size_1, h2)
        self.b2 = np.zeros((1, hidden_size_2))                          # (1, h2)
        self.W3 = np.random.randn(hidden_size_2, output_size) * 0.01    #（h2, 10)
        self.b3 = np.zeros((1, output_size))                            # (1, 10)
        self.epoch = 0          # 训练的次数
        self.loss = []
        self.epochs = []
        self.start = 0.0        # 开始时间戳
        self.end = 0.0          # 结束时间戳
        self.lossmodel = lossmodel          # 选择损失函数
        self.lossname = ""
        self.Activation = Activation        # 选择激活函数

        self.lambda_ = 0.1      # 正则化超参
        self.reg = reg          # 正则化方式

        print('-----    init    -----')
        print('hid_1:   ', hid_size_1, '\thid2:   ', hid_size_2,
              '\tActivation:  ', self.Activation)

    def LOSS(self, Y_pred, Y_true):
        if self.lossmodel == 1:
            self.lossname = "Cross Entropy Loss With Regularization"
            return cross_entropy_loss_with_regularization(
                self.reg, Y_pred, Y_true, self.lambda_, self.W2, self.W1, self.W3)
        elif self.lossmodel == 2:
            self.lossname = "Mean Squared Loss"
            return mean_squared_error(Y_pred, Y_true)

    def activate(self, z):
        if self.Activation == 'relu':
            return ReLU(z)
        elif self.Activation == 'sigmoid':
            return sigmoid(z)
        elif self.Activation == 'tanh':
            return tanh(z)
        elif self.Activation == 'leaky_relu':
            return leaky_relu(z)
        
    def activate_grad(self, z):
        if self.Activation == 'relu':
            return relu_gradient(z)
        elif self.Activation == 'sigmoid':
            return sigmoid_gradient(z)
        elif self.Activation == 'tanh':
            return tanh_derivative(z)
        elif self.Activation == 'leaky_relu':
            return leaky_relu_derivative(z)        

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1      #z1:   (m, h1) x:(m, 784) 
        self.a1 = self.activate(self.z1)            # 激活函数 (m, h1）
        self.z2 = np.dot(self.a1, self.W2) + self.b2    # z2:   (m, h2)   
        self.a2 = self.activate(self.z2)            # (m, h2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3    # (m, 10)

        # softmax 对输出进行归一化，输出分类概率
        self.probs = softmax(self.z3)

    def backward(self, X, y, learning_rate):
        # X是训练数据(60000, 784) y是训练标签[5 0 4 ... 5 6 8]
        
        if self.epoch % 10 == 0:
            loss = self.LOSS(self.probs, y)
            print('epoch:   ',self.epoch, '\tloss:    ', loss)
            self.loss.append(loss)
            self.epochs.append(self.epoch)
        self.epoch += 1
        d = self.probs                      # 将模型的输出self.probs保存到变量d中

        if self.lossmodel == 1:
            # softmax激活函数 交叉熵损失函数
            d1 = soft_cross_gradient(d, y)
        elif self.lossmodel == 2:
            # softmax激活函数 均方误差损失函数
            d1 = soft_mean_gradient(d, y, self.z2)

        db3 = np.sum(d1, axis=0) 
        dW3 = np.dot(self.a2.T, d1)         # 计算输出层权重self.W2的梯度 (10, h2)

        d2 = np.dot(d1, self.W3.T)          # (m, h2)
        d3 = d2 * self.activate_grad(self.z2) 
        db2 = np.sum(d3, axis=0)            # 对矩阵的列求平均值 (1, h2)
        dW2 = np.dot(self.a1.T, d3)
        d4 = np.dot(d3, self.W2.T)
        d5 = d4 * self.activate_grad(self.z1)
        db1 = np.sum(d5, axis=0)
           
        dW1 = np.dot(X.T, d5)               # 计算输入层权重self.W1的梯度，即输入层误差delta2与输入数据X的转置矩阵的乘积

        if self.reg == 'L2':
            self.W1 -= learning_rate * (dW1 + self.lambda_* self.W1)  
            self.b1 -= learning_rate * db1  
            self.W2 -= learning_rate * (dW2 + self.lambda_* self.W2) 
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * (dW3 + self.lambda_* self.W2)
            self.b3 -= learning_rate * db3
        elif self.reg == 'L1':
            self.W1 -= learning_rate * (dW1 + self.lambda_* np.sign(self.W1))  
            self.b1 -= learning_rate * db1  
            self.W2 -= learning_rate * (dW2 + self.lambda_* np.sign(self.W2)) 
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * (dW3 + self.lambda_* np.sign(self.W3)) 
            self.b3 -= learning_rate * db3        
        else:       
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

    def train(self, X, y, num_epochs, learning_rate):
        self.start = time.time()
        for epoch in range(num_epochs+1):
            self.forward(X)
            self.backward(X, y, learning_rate)

        self.end = time.time()
        # 计算时间差
        exe_time = self.end - self.start
        minute = int(exe_time / 60)
        msc = exe_time % 60
        print('-----    训练结束    -----')
        print(f"Execution time:  {minute}min {msc:.2f}msc")
              
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.probs, axis=1)
    
    def draw_plot(self):
        # 绘制折线图和误差棒
        plt.errorbar(model.epochs, model.loss, yerr=[i*0.1 for i in model.loss], fmt='-o')

        # 显示每个数据点的误差数值
        for x, y in zip(model.epochs, model.loss):
            plt.annotate(f'{y:.2f}', (x, y + 0.05), xytext=(5, 5),
                          textcoords='offset points', rotation=20)

        # 绘制折线图
        plt.plot(model.epochs, model.loss, '-o')
        # 设置横轴和纵轴的标签
        plt.xlabel('Epochs')
        plt.ylabel(self.lossname)
        # 设置 x 轴刻度为 epochs 数组的数据
        step = 10
        plt.xticks(range(0, num_epochs+1, step), model.epochs)
        # Set the title of the graph
        plt.title('Training Error of the Neural Network')
        # 显示图表
        plt.show()

    def print_acc(self, test_data, test_labels):
        '''打印正确率'''
        predictions = self.predict(test_data)
        accuracy = np.mean(predictions == test_labels) * 100
        print(f'Test Accuracy:   {accuracy:.2f}%')

    def print_train_acc(self, train_data, train_labels):
        print('=====   训练集正确率   =====')
        # 随机选择索引
        indices = np.random.choice(len(train_data), size=10000, replace=False)

        # 从 train_data 和 train_labels 中选择对应索引的数据
        random_data = train_data[indices]
        random_labels = train_labels[indices]

        self.print_acc(random_data, random_labels)


if __name__ == '__main__':
    # 训练和测试
    train_data, train_labels, test_data, test_labels = preprocess_data()
    t_labels = build_labels(train_labels)

    input_size = 28 * 28
    hid_size_1 = 256
    hid_size_2 = 128
    output_size = 10
    num_epochs = 200
    learning_rate = 0.00001   # 交叉熵步长 0.00001  均方误差步长0.00007

    model = NeuralNetwork(input_size, hid_size_1, hid_size_2, output_size,
                          lossmodel=1, Activation="relu", reg='L1')

    
    model.train(train_data, t_labels, num_epochs=num_epochs, 
                learning_rate=learning_rate)
    model.print_acc(test_data=test_data, test_labels=test_labels)
    model.print_train_acc(train_data, train_labels)
    model.draw_plot()
