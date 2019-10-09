import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
#np.newaxis指生成多一维 下面代码生成200行一列的代码
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise
#x为多行1列的矩阵
x = tf.compat.v1.placeholder(tf.float32,[None,1])
y = tf.compat.v1.placeholder(tf.float32,[None,1])

#定义神经网络中间层
#tf.random.normal()函数用于从服从指定正太分布的数值中取出指定个数的值
weight1 = tf.Variable(tf.random.normal([1,10]))
biases1 = tf.Variable(tf.zeros([1,10]))
layer1 = tf.matmul(x,weight1) + biases1
layer1_out = tf.nn.tanh(layer1)

weight2 = tf.Variable(tf.random.normal([10,1]))
biases2 = tf.Variable(tf.zeros([1,1]))
layer2 = tf.matmul(layer1_out,weight2) + biases2
prediction = tf.nn.tanh(layer2)

loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降进行训练
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y :y_data})
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r--',lw= 5)
    plt.show()

