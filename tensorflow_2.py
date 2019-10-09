import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def tensor_example(batch_size_value = 100,learing_rate = 0.2,steps = 21,use_cross_entropy = False):
    #载入数据集
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    #每个批次的大小
    batch_size = batch_size_value
    #计算一共需要多少个批次
    n_batch = mnist.train.num_examples // batch_size

    #定义x,y的数据类型
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])


    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)

    if use_cross_entropy == False :
        #二次代价函数
        loss = tf.reduce_mean(tf.square(y - prediction))
    else:
        #使用交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    #使用梯度下降法
    learing_rate = learing_rate
    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)

    #初始化变量
    init = tf.global_variables_initializer()
    #结果存放在一个布尔型的列表中
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(steps):
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter" + str(epoch) + " Testing Accuracy = " + str(acc))

#tensor_example(batch_size_value = 1000,learing_rate = 0.2,steps = 21）
#Iter20Testing Accuracy = 0.9147 准确率为0.9147

#tensor_example(batch_size_value = 20,learing_rate = 0.2,steps = 21)
#将batch_size_value 修改为20，准确率提高了一个百分点 为Accuracy = 0.9268

#tensor_example(batch_size_value = 100,learing_rate = 0.2,steps = 101)
#将迭代的次数修改为101，准确率也提高了一个百分点 为Accuracy = 0.9258

#tensor_example(batch_size_value = 100,learing_rate = 0.1,steps = 21)
#减小学习率准确率下降了近一个百分点 Accuracy = 0.9057
#tensor_example(batch_size_value = 100,learing_rate = 0.5,steps = 21)
#加大学习率为0.5，准确度提高了一个百分点Accuracy = 0.9215

#tensor_example(batch_size_value = 100,learing_rate = 0.2,steps = 21)
#代价函数修改为交叉熵函数，准确率提升了一个百分点 Accuracy = 0.9218，而且迭代到第三次真确了就达到了0.9

#tensor_example(batch_size_value = 20,learing_rate = 0.5,steps = 100)
def tensor_example_layer2(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num = 200,use_cross_entropy = False,use_drop_out = True):
    #载入数据集
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    #每个批次的大小
    batch_size = batch_size_value
    #计算一共需要多少个批次
    n_batch = mnist.train.num_examples // batch_size

    #定义x,y的数据类型
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])

    W1 = tf.Variable(tf.truncated_normal([784, hide_num],stddev=0.1))
    b1 = tf.Variable(tf.zeros([hide_num]) + 0.1)
    W2 = tf.Variable(tf.truncated_normal([hide_num, 10],stddev=0.1))
    b2 = tf.Variable(tf.zeros([10]))
    layer1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
    if use_drop_out:
        layer1 = tf.nn.dropout(layer1,keep_prob=0.7)
    prediction = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

    if use_cross_entropy == False:
        # 二次代价函数
        loss = tf.reduce_mean(tf.square(y - prediction))
    else:
        # 使用交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 使用梯度下降法
    learing_rate = learing_rate
    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)
    #初始化变量
    init = tf.global_variables_initializer()
    #结果存放在一个布尔型的列表中
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(steps):
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter" + str(epoch) + " Testing Accuracy = " + str(acc))

#tensor_example_layer2(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num=2000)
#Iter20 Testing Accuracy = 0.9497 初始化x,y的时候如果把x,y都初始化为0，准确率特别低 ，所以初始化要使用tf.truncated_normal（）方法，stddv=0.1初始化

#tensor_example_layer2(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num=2000,use_cross_entropy=True)
#使用交叉熵代价函数准确率提高了两个百分点 Iter20 Testing Accuracy = 0.9641

#tensor_example_layer2(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num=2000,use_cross_entropy=True,use_drop_out=True)
#使用dropout防止过拟合  Iter20 Testing Accuracy = 0.9472

def tensor_example_layer3(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num = 200,use_cross_entropy = False,use_drop_out = True):
    #载入数据集
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    #每个批次的大小
    batch_size = batch_size_value
    #计算一共需要多少个批次
    n_batch = mnist.train.num_examples // batch_size

    #定义x,y的数据类型
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])

    W1 = tf.Variable(tf.truncated_normal([784, hide_num],stddev=0.1))
    b1 = tf.Variable(tf.zeros([hide_num]) + 0.1)
    W2 = tf.Variable(tf.truncated_normal([hide_num, hide_num], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hide_num]) + 0.1)
    W3 = tf.Variable(tf.truncated_normal([hide_num, 10],stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    layer1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
    if use_drop_out:
        layer1 = tf.nn.dropout(layer1,keep_prob=0.7)
    layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
    if use_drop_out:
        layer2 = tf.nn.dropout(layer2, keep_prob=0.7)
    prediction = tf.nn.softmax(tf.matmul(layer2,W3) + b3)
    if use_cross_entropy == False:
        # 二次代价函数
        loss = tf.reduce_mean(tf.square(y - prediction))
    else:
        # 使用交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 使用梯度下降法
    learing_rate = learing_rate
    #train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)
    #修改优化器
    train_step = tf.train.AdadeltaOptimizer(learing_rate).minimize(loss)
    #初始化变量
    init = tf.global_variables_initializer()
    #结果存放在一个布尔型的列表中
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(steps):
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter" + str(epoch) + " Testing Accuracy = " + str(acc))

#tensor_example_layer3(batch_size_value=20, learing_rate=0.5, steps=21, hide_num=2000, use_cross_entropy=True, use_drop_out=False)
"""
因为本身数据集比较大，我们的模型也不复杂，所以没有使用dropout,修改优化器把GradientDescentOptimizer（）修改为AdadeltaOptimizer（）
Iter0 Testing Accuracy = 0.9555
Iter1 Testing Accuracy = 0.9612
Iter2 Testing Accuracy = 0.97
Iter3 Testing Accuracy = 0.9723
Iter4 Testing Accuracy = 0.9749
Iter5 Testing Accuracy = 0.976
Iter6 Testing Accuracy = 0.9783
Iter7 Testing Accuracy = 0.9783
Iter8 Testing Accuracy = 0.9799
Iter9 Testing Accuracy = 0.9783
Iter10 Testing Accuracy = 0.9799
Iter11 Testing Accuracy = 0.9799
Iter12 Testing Accuracy = 0.9782
Iter13 Testing Accuracy = 0.9804
Iter14 Testing Accuracy = 0.9802
Iter15 Testing Accuracy = 0.98
Iter16 Testing Accuracy = 0.9805
Iter17 Testing Accuracy = 0.9805
Iter18 Testing Accuracy = 0.9807
Iter19 Testing Accuracy = 0.9805
Iter20 Testing Accuracy = 0.9807

"""

def tensor_example_layer3_update_lt(batch_size_value = 100,learing_rate = 0.2,steps = 21,hide_num = 200,use_cross_entropy = False,use_drop_out = True):
    #载入数据集
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    #每个批次的大小
    batch_size = batch_size_value
    #计算一共需要多少个批次
    n_batch = mnist.train.num_examples // batch_size

    #定义x,y的数据类型
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    learing_rate = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.truncated_normal([784, hide_num],stddev=0.1))
    b1 = tf.Variable(tf.zeros([hide_num]) + 0.1)
    W2 = tf.Variable(tf.truncated_normal([hide_num, hide_num], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hide_num]) + 0.1)
    W3 = tf.Variable(tf.truncated_normal([hide_num, 10],stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    layer1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
    if use_drop_out:
        layer1 = tf.nn.dropout(layer1,keep_prob=0.7)
    layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
    if use_drop_out:
        layer2 = tf.nn.dropout(layer2, keep_prob=0.7)
    prediction = tf.nn.softmax(tf.matmul(layer2,W3) + b3)
    if use_cross_entropy == False:
        # 二次代价函数
        loss = tf.reduce_mean(tf.square(y - prediction))
    else:
        # 使用交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 使用梯度下降法

    #train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)
    #修改优化器
    train_step = tf.train.AdadeltaOptimizer(learing_rate).minimize(loss)
    #初始化变量
    init = tf.global_variables_initializer()
    #结果存放在一个布尔型的列表中
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(steps):
            lt =0.001 * (0.95 ** epoch)
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,learing_rate: lt})
            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print("Iter" + str(epoch) + " Testing Accuracy = " + str(acc))

#tensor_example_layer3(batch_size_value=100, steps=21, hide_num=500, use_cross_entropy=True, use_drop_out=False)
"""
 Iter0 Testing Accuracy = 0.8325
Iter1 Testing Accuracy = 0.8464
Iter2 Testing Accuracy = 0.9353
Iter3 Testing Accuracy = 0.9415
Iter4 Testing Accuracy = 0.9444
Iter5 Testing Accuracy = 0.949
Iter6 Testing Accuracy = 0.9497
Iter7 Testing Accuracy = 0.9534
Iter8 Testing Accuracy = 0.9542
Iter9 Testing Accuracy = 0.9553
Iter10 Testing Accuracy = 0.9575
Iter11 Testing Accuracy = 0.96
Iter12 Testing Accuracy = 0.9606
Iter13 Testing Accuracy = 0.9618
Iter14 Testing Accuracy = 0.9636
Iter15 Testing Accuracy = 0.9635
Iter16 Testing Accuracy = 0.9647
Iter17 Testing Accuracy = 0.9647
Iter18 Testing Accuracy = 0.9671
Iter19 Testing Accuracy = 0.9664
Iter20 Testing Accuracy = 0.9688


"""

tensor_example_layer3(batch_size_value=20, steps=41, hide_num=1000, use_cross_entropy=True, use_drop_out=False)
"""
sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,learing_rate: lt}) 
每迭代完一次，修改一次学习率，因为开始离我们的目标比较远，学习率可以大一点，当越接近我们的目标值时，学习率要越小
Iter0 Testing Accuracy = 0.9455
Iter1 Testing Accuracy = 0.9554
Iter2 Testing Accuracy = 0.9605
Iter3 Testing Accuracy = 0.9651
Iter4 Testing Accuracy = 0.9664
Iter5 Testing Accuracy = 0.9704
Iter6 Testing Accuracy = 0.9712
Iter7 Testing Accuracy = 0.9727
Iter8 Testing Accuracy = 0.9745
Iter9 Testing Accuracy = 0.9733
Iter10 Testing Accuracy = 0.9741
Iter11 Testing Accuracy = 0.975
Iter12 Testing Accuracy = 0.9755
Iter13 Testing Accuracy = 0.976
Iter14 Testing Accuracy = 0.9767
Iter15 Testing Accuracy = 0.9776
Iter16 Testing Accuracy = 0.977
Iter17 Testing Accuracy = 0.9776
Iter18 Testing Accuracy = 0.9781
Iter19 Testing Accuracy = 0.9787
Iter20 Testing Accuracy = 0.9791
Iter21 Testing Accuracy = 0.9785
Iter22 Testing Accuracy = 0.9785
Iter23 Testing Accuracy = 0.9792
Iter24 Testing Accuracy = 0.9796
Iter25 Testing Accuracy = 0.9794
Iter26 Testing Accuracy = 0.9791
Iter27 Testing Accuracy = 0.9799
Iter28 Testing Accuracy = 0.9795
Iter29 Testing Accuracy = 0.9807
Iter30 Testing Accuracy = 0.98
Iter31 Testing Accuracy = 0.98
Iter32 Testing Accuracy = 0.9805
Iter33 Testing Accuracy = 0.9802
Iter34 Testing Accuracy = 0.9801
Iter35 Testing Accuracy = 0.9795
Iter36 Testing Accuracy = 0.9804
Iter37 Testing Accuracy = 0.9798
Iter38 Testing Accuracy = 0.9794
Iter39 Testing Accuracy = 0.9797
Iter40 Testing Accuracy = 0.9808
"""