import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 100
#一共多少个批次
n_batch = mnist.train.num_examples //batch_size

#初始化权值
def weight_variable(shape):
    init_value = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_value)

#初始化偏执
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

#卷积层
def conv2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#定义输入值x,y
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

#将x的格式转化为4维的向量[batch_size,in_height,in_with,in_channels]
x_image = tf.reshape(x,[-1,28,28,1])#-1表示未知

#初始化一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32]) #[filter_heith,filter_width,in_channel,out_channel] 5*5的采样窗口，从1个平面中抽取32个特征值
b_conv1 = bias_variable([32])#每个卷积核对应一个偏置

h_conv1 = tf.nn.relu(conv2(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64])#5*5的采样窗口，从32个平面中抽取64个特征值
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#28*28的图片经过第一次卷积还是28*28，因为padding = same,所以大小不改变，第一次池化后为14*14，因为卷积核为2*2，步长为2
#第二次卷积后为14*14，第二次池化后为7*7
#经过上面的操作后得到了64张7*7的平面


#创建全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#将池化层2的输出平化为1维的
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#第一个全连接的输出层
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

keep_drop = tf.placeholder(tf.float32)
h_fc1_drop_out = tf.nn.dropout(h_fc1,keep_prob=keep_drop)

#初始化第二个全连接层

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop_out,W_fc2 ) + b_fc2)

#使用交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer优化器，优化loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y,keep_drop:0.7})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_drop:1.0})
        print("Iter" + str(epoch) + ",testing accuracy = " + str(acc))

"""
Iter0,testing accuracy = 0.9565
2019-10-09 21:15:50.911649: W tensorflow/core/framework/allocator.cc:107] Allocation of 1003520000 exceeds 10% of system memory.
2019-10-09 21:15:51.961136: W tensorflow/core/framework/allocator.cc:107] Allocation of 501760000 exceeds 10% of system memory.
Iter1,testing accuracy = 0.9698
2019-10-09 21:17:04.039942: W tensorflow/core/framework/allocator.cc:107] Allocation of 1003520000 exceeds 10% of system memory.
Iter2,testing accuracy = 0.9735
Iter3,testing accuracy = 0.9797
Iter4,testing accuracy = 0.9803
Iter5,testing accuracy = 0.9854
Iter6,testing accuracy = 0.9853
Iter7,testing accuracy = 0.9888
Iter8,testing accuracy = 0.9882
Iter9,testing accuracy = 0.9871
Iter10,testing accuracy = 0.9902
Iter11,testing accuracy = 0.989
Iter12,testing accuracy = 0.9907
Iter13,testing accuracy = 0.9899
Iter14,testing accuracy = 0.991
Iter15,testing accuracy = 0.9915
Iter16,testing accuracy = 0.9912
Iter17,testing accuracy = 0.9912
Iter18,testing accuracy = 0.9911
Iter19,testing accuracy = 0.9908
Iter20,testing accuracy = 0.9914

"""