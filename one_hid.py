import numpy as np
import time
import matplotlib.pyplot as plt
from tools import *

# 定义神经网络模型
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 lossmodel=1, Activation='relu', reg='', 
                 momentum=0, Adagrad=0, RMSProp=0, Adam=0):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01   # （784, hid_size)
        self.b1 = np.zeros((1, hidden_size))                        # (1, hid_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # (hid_size, 10)
        self.b2 = np.zeros((1, output_size))                        # (1, 10)
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

        # 定义动量变量
        self.velocity_W1 = np.zeros_like(self.W1)
        self.velocity_b1 = np.zeros_like(self.b1)
        self.velocity_W2 = np.zeros_like(self.W2)
        self.velocity_b2 = np.zeros_like(self.b2)
        # 动量值 momentum=0 时无优化
        self.momentum = momentum

        # Adagrad=1 时启用Adagrad
        self.Adagrad = Adagrad
        self.RMSProp = RMSProp
        self.Adam = Adam
        self.G_W1 = np.zeros_like(self.W1)
        self.G_b1 = np.zeros_like(self.b1)
        self.G_W2 = np.zeros_like(self.W2)
        self.G_b2 = np.zeros_like(self.b2)
        
        print('-----    init    -----')
        print('hidden_size:   ', hidden_size,
              '\tActivation:  ', self.Activation)

    def LOSS(self, Y_pred, Y_true):
        if self.lossmodel == 1:
            if self.reg == 'L1':
                self.lossname = "Cross Entropy Loss With Regularization L1"
            elif self.reg == 'L2':
                self.lossname = "Cross Entropy Loss With Regularization L2"
            else:
                self.lossname = "Cross Entropy Loss"
            return cross_entropy_loss_with_regularization(
                self.reg, Y_pred, Y_true, self.lambda_, self.W2, self.W1)
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
        self.z1 = np.dot(X, self.W1) + self.b1      #z1:   (m, h) x:(m, 784) 
        self.a1 = self.activate(self.z1)  # ReLU 激活函数 (m, h）
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # z2:   (m, 10)

        # softmax 对输出进行归一化，输出分类概率
        self.probs = softmax(self.z2)

    def update_loss_epochs(self, y):
        '''更新loss 和 epochs 便于画图以及打印出来'''
        if self.epoch % 10 == 0:
            loss = self.LOSS(self.probs, y)
            print('epoch:   ',self.epoch, '\tloss:    ', loss)
            self.loss.append(loss)
            self.epochs.append(self.epoch)

        self.epoch += 1

    def backward(self, X, y, learning_rate):
        # X是训练数据(60000, 784) y是训练标签[5 0 4 ... 5 6 8]
        
        d = self.probs                      # 将模型的输出self.probs保存到变量d中

        if self.lossmodel == 1:
            # softmax激活函数 交叉熵损失函数
            d1 = soft_cross_gradient(d, y)      # (m, 10) 用简单方法求的梯度
            # a1 = cel_gradient(d, y)     # 先对交叉熵函数求导
            # a2 = softmax_gradient(self.z2)  #(m, 10) 再对softmax求导
            # d1 = a1 * a2
        elif self.lossmodel == 2:
            # softmax激活函数 均方误差损失函数
            d1 = soft_mean_gradient(d, y)

        db2 = np.sum(d1, axis=0) 
        dW2 = np.dot(d1.T, self.a1).T       # 计算输出层权重self.W2的梯度 (hid_size, 10)

        d2 = np.dot(d1, self.W2.T)          # (m, hid_size)
        d3 = d2 * self.activate_grad(self.z1)
        db1 = np.sum(d3, axis=0)            # 对矩阵的列求平均值 (1, hid_size)
           
        try:
            dW1 = np.dot(d3.T, X).T             # 计算输入层权重self.W1的梯度，即输入层误差delta2与输入数据X的转置矩阵的乘积
        except ValueError:
            dW1 = np.dot(d3.T, X.reshape(1, len(X))).T

        # 正则化
        if self.reg == 'L2':
            dW1 = (dW1 + self.lambda_* self.W1)    
            dW2 = (dW2 + self.lambda_* self.W2) 
        elif self.reg == 'L1':
            dW1 = (dW1 + self.lambda_* np.sign(self.W1))
            dW2 = (dW2 + self.lambda_* np.sign(self.W2))

        if self.momentum != 0:
            # momentum 优化
            # 更新动量变量
            self.velocity_W1 = self.momentum * self.velocity_W1 - learning_rate * dW1
            self.velocity_b1 = self.momentum * self.velocity_b1 - learning_rate * db1
            self.velocity_W2 = self.momentum * self.velocity_W2 - learning_rate * dW2
            self.velocity_b2 = self.momentum * self.velocity_b2 - learning_rate * db2

            self.W1 += self.velocity_W1
            self.b1 += self.velocity_b1
            self.W2 += self.velocity_W2
            self.b2 += self.velocity_b2

        elif self.Adagrad == 1:
            epsilon = 1e-8  # 为了数值稳定性而添加的小常数
            # 更新累积梯度
            self.G_W1 += dW1 ** 2
            self.G_b1 += db1 ** 2
            self.G_W2 += dW2 ** 2
            self.G_b2 += db2 ** 2

            # 更新参数
            self.W1 -= learning_rate * dW1 / (np.sqrt(self.G_W1) + epsilon)
            self.b1 -= learning_rate * db1 / (np.sqrt(self.G_b1) + epsilon)
            self.W2 -= learning_rate * dW2 / (np.sqrt(self.G_W2) + epsilon)
            self.b2 -= learning_rate * db2 / (np.sqrt(self.G_b2) + epsilon)

            '''步长会慢慢减小失去调节作用'''

        elif self.RMSProp == 1:
            p = 0.999
            epsilon = 1e-8
            self.G_W1 = p * self.G_W1 + (1-p) * (dW1 ** 2)
            self.G_b1 = p * self.G_b1 + (1-p) * (db1 ** 2)
            self.G_W2 = p * self.G_W2 + (1-p) * (dW2 ** 2)
            self.G_b2 = p * self.G_b2 + (1-p) * (db2 ** 2)

            # 更新参数
            self.W1 -= learning_rate * dW1 / (np.sqrt(self.G_W1) + epsilon)
            self.b1 -= learning_rate * db1 / (np.sqrt(self.G_b1) + epsilon)
            self.W2 -= learning_rate * dW2 / (np.sqrt(self.G_W2) + epsilon)
            self.b2 -= learning_rate * db2 / (np.sqrt(self.G_b2) + epsilon)       

        elif self.Adam == 1:
            '''
            取消了adam的偏差校正环节 loss收敛的更快更好
            
            取消偏差校正可能使得算法在训练初期对学习率的调整更加敏感
            从而有助于更快地找到合适的学习率
            '''
            p = 0.9999
            q = 0.9
            epsilon = 1e-8
            self.G_W1 = (p * self.G_W1 + (1-p) * (dW1 ** 2)) 
            self.G_b1 = (p * self.G_b1 + (1-p) * (db1 ** 2)) 
            self.G_W2 = (p * self.G_W2 + (1-p) * (dW2 ** 2)) 
            self.G_b2 = (p * self.G_b2 + (1-p) * (db2 ** 2)) 

            self.velocity_W1 = (q * self.velocity_W1 + (1-q) * dW1) 
            self.velocity_b1 = (q * self.velocity_b1 + (1-q) * db1) 
            self.velocity_W2 = (q * self.velocity_W2 + (1-q) * dW2) 
            self.velocity_b2 = (q * self.velocity_b2 + (1-q) * db2) 
            # 更新参数
            self.W1 -= learning_rate * self.velocity_W1 / (np.sqrt(self.G_W1) + epsilon)
            self.b1 -= learning_rate * self.velocity_b1 / (np.sqrt(self.G_b1) + epsilon)
            self.W2 -= learning_rate * self.velocity_W2 / (np.sqrt(self.G_W2) + epsilon)
            self.b2 -= learning_rate * self.velocity_b2 / (np.sqrt(self.G_b2) + epsilon) 

        else:
            # 无优化梯度更新
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

      
    def BGD(self, X, y, num_epochs, learning_rate):
        '''批量梯度下降 Batch Gradient Descent'''
        self.start = time.time()
        for epoch in range(num_epochs+1):
            self.forward(X)
            self.update_loss_epochs(y)
            self.backward(X, y, learning_rate)
            # self.epoch += 1

        self.end = time.time()
        # 计算时间差
        exe_time = self.end - self.start
        minute = int(exe_time / 60)
        msc = exe_time % 60
        print('-----    训练结束    -----')
        print(f"Execution time:  {minute}min {msc:.2f}msc")

    def MBGD(self, X, y, num_epochs, learning_rate, batch_):
        '''小批量梯度下降 Mini-Batch Gradient Descent MBGD'''
        self.start = time.time()
        # 将训练集随机打乱顺序
        permutation = np.random.permutation(X.shape[0])
        X_train_shuffled = X[permutation]
        y_train_shuffled = y[permutation]

        # 分割训练集为小批量 batch = 1为随机梯度下降SGD
        batch = batch_
        for epoch in range(num_epochs):
            for i in range(0, X.shape[0], batch):
                # 获取当前批量的输入和标签
                X_batch = X_train_shuffled[i:i + batch]
                y_batch = y_train_shuffled[i:i + batch]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                
            self.epoch += 1
            loss = self.LOSS(self.probs, y_batch)
            if self.epoch % 10 == 0:
                print('epoch:   ',epoch+1, '\tloss:    ', loss)
            self.loss.append(loss)
            self.epochs.append(self.epoch)
      

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
        plt.errorbar(self.epochs, self.loss, yerr=[i*0.1 for i in self.loss], fmt='-o')

        # 显示每个数据点的误差数值
        for x, y in zip(self.epochs, self.loss):
            plt.annotate(f'{y:.2f}', (x, y + 0.05), xytext=(5, 5),
                          textcoords='offset points', rotation=20)

        # 绘制折线图
        plt.plot(self.epochs, self.loss, '-o')
        # 设置横轴和纵轴的标签
        plt.xlabel('Epochs')
        plt.ylabel(self.lossname)
        # 设置 x 轴刻度为 epochs 数组的数据
        step = 10
        plt.xticks(range(0, num_epochs+1, step), self.epochs)
        # Set the title of the graph
        plt.title('Training Error of the Neural Network')
        # 显示图表
        plt.show()

    def print_acc(self, test_data, test_labels):
        '''打印正确率'''
        predictions = self.predict(test_data)
        accuracy = np.mean(predictions == test_labels) * 100
        print(f'Accuracy:   {accuracy:.2f}%')

    def print_train_acc(self, train_data, train_labels):
        print('-----   训练集正确率   -----')
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
    hid_size = 256
    output_size = 10
    num_epochs = 100
    batch_ = 6000

    # 批量下降 0.00001 小批量下降0.0001 Adagrad 0.1 RMSProp 0.001
    # 交叉熵步长 0.00001  均方误差步长0.00007
    learning_rate = 0.001

    # momentum=0.8
    model = NeuralNetwork(input_size, hid_size, output_size,
                          lossmodel=1, Activation="relu", reg='L1',RMSProp=1)

    model.MBGD(train_data, t_labels, num_epochs=num_epochs,
                learning_rate=learning_rate, batch_=batch_)
    # model.BGD(train_data, t_labels, num_epochs=num_epochs,
    #             learning_rate=learning_rate)
    model.print_acc(test_data=test_data, test_labels=test_labels)
    model.print_train_acc(train_data, train_labels)
    # model.draw_plot()
