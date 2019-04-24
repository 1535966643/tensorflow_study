
'''
	三层神经网络
'''

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
learning_rate = 0.01
train_step = 40
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)


x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')

H1_NN = 256
H2_NN = 64
H3_NN = 32

'''
	定义全连接层的函数
		inputs(输入数据), input_dim（输入神经元数量）,
		output_dim（输出神经元数量）, activation=None(激活函数)
'''
def fcn_layer(inputs, input_dim, output_dim, activation=None):

	W = tf.Variable(tf.truncated_normal([input_dim, output_dim],stddev=0.1))
	b = tf.Variable(tf.zeros([output_dim]))
	
	Xwb = tf.matmul(inputs, W) + b
	
	if activation is None:	# 如果不使用激活函数
		outputs = Xwb
	else:
		outputs = activation(Xwb)
	return outputs
		
# 这样构建隐藏层
h1 = fcn_layer(x, 784, H1_NN, tf.nn.relu)

#构建输出层
forWard = fcn_layer(h1, H2_NN, 10, None)
pred = tf.nn.softmax(forWard)



# 输入层
# truncated_normal：截断正态分布的随机数初始化
w1 = tf.Variable(tf.truncated_normal([784, H1_NN], stddev=0.1))
b1 = tf.Variable(tf.zeros([H1_NN]))

# 第一层网络
w2 = tf.Variable(tf.truncated_normal([H1_NN, H2_NN], stddev=0.1))
b2 = tf.Variable(tf.zeros([H2_NN]))

# 第二层网络
w3 = tf.Variable(tf.truncated_normal([H2_NN, H3_NN], stddev=0.1))
b3 = tf.Variable(tf.zeros([H3_NN]))

#  第三层网络
w4 = tf.Variable(tf.truncated_normal([H3_NN, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

# 第一层隐藏层的结果
Y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# 第二层隐藏层的结果
Y2 = tf.nn.relu(tf.matmul(Y1, w2) + b2)
# 第三层隐藏层的结果
Y3 = tf.nn.relu(tf.matmul(Y2, w3) + b3)

forward = tf.matmul(Y3, w4) + b4
pred = tf.nn.softmax(forward)

# 交叉熵损失函数
# 为了避免log(0)的值不稳定我们使用了新的损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# 使用优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(train_step):
	for batch in range(total_batch):
		xs, ys = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x:xs, y:ys})

	loss, acc = sess.run([loss_function, accuracy], feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
	if step % 5 == 0:
		print(step,'---loss--', loss,'---', acc)

sess.close()








