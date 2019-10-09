import  tensorflow as tf
import  numpy as np
#随机100个数
x_data = np.random.rand(100)
#真实的Y值
y_data = 0.2 * x_data + 0.1

k = tf.Variable(0.)
b = tf.Variable(0.)
y = k* x_data + b

#定义loss函数
loss = tf.reduce_mean(tf.square(y - y_data))
#梯度下降的优化器
learning_rate = 0.2
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
#最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i % 20 == 0:
            print(i,sess.run([k,b]))

"""
输出：
0 [0.048551362, 0.08044155]
20 [0.14219503, 0.1316662]
40 [0.16913217, 0.116909765]
60 [0.18351659, 0.109029815]
80 [0.19119789, 0.104821905]
100 [0.19529966, 0.10257491]
120 [0.19749, 0.101375]
140 [0.19865966, 0.100734256]
160 [0.19928427, 0.10039209]
180 [0.19961779, 0.10020938]
200 [0.1997959, 0.100111805]


"""