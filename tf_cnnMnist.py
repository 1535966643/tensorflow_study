
# 卷积神经网络实现手写体识别

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
learning_rate = 0.01
train_step = 40
batch_size = 50
all_batch = int(mnist.train.num_examples / batch_size)

# 初始化权重值
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
# 初始化偏支值
def bias_variable(shape):
	initial = tf.zeros(shape)
	return tf.Variable(initial)

# 卷积层
def convolution(x, w):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
# 池化层
def maxPool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10], name='y')

''' 
	第一层卷积神经网络
'''
w1_convol = weight_variable([5,5,1,32])
b1 = bias_variable([32])
# 使用relu的激活函数处理卷积的结果
Y1_convol = tf.nn.relu(convolution(x_image, w1_convol) + b1)
# 进行池化(最大池化)
Y1_pool = maxPool(Y1_convol)

''' 
	第二层卷积神经网络
'''
w2_convol = weight_variable([5,5,32,64])
b2 = bias_variable([64])
# 使用relu的激活函数处理卷积的结果
Y2_convol = tf.nn.relu(convolution(Y1_pool, w2_convol) + b2)
# 进行池化(最大池化)
Y2_pool = maxPool(Y2_convol)

'''
	全连接
'''
w_fcn1 = weight_variable([7*7*64, 1024])
b_fcn1 = bias_variable([1024])

fcn_shape = tf.reshape(Y2_pool, [-1,7 * 7 * 64])
Y_fcn = tf.nn.relu(tf.matmul(fcn_shape, w_fcn1) + b_fcn1)

'''
	进行Dropout
'''
keep_prob = tf.placeholder('float32')
Y_fcn_drop = tf.nn.dropout(Y_fcn, keep_prob)

'''
	softmax 输出
'''
w_fcn2 = weight_variable([1024, 10])
b_fcn2 = bias_variable([10])

Y_out = tf.nn.softmax(tf.matmul(Y_fcn_drop, w_fcn2) + b_fcn2)

# 为了避免log(0)的值不稳定我们使用了新的损失函数
loss_function = -tf.reduce_sum(y * tf.log(Y_out))

# 使用优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#迭代优化模型
for i in range(1000):
    #每次取50个样本进行训练
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y:batch[1], keep_prob:1.0}) #模型中间不使用dropout
		print("step{}, training accuracy{}".format(i, train_accuracy))
	sess.run(optimizer, feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
	bb = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
	print(bb)








