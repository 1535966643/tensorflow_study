
import tensorflow as tf
import numpy
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
'''
	构建函数模型
'''
mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
learning_rate = 0.01
train_step = 40
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

H1_NN = 256
H2_NN = 64
H3_NN = 32

x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')


def fcn_layer(inputs, input_dim, output_dim, activate=None):
	
	W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
	b = tf.Variable(tf.zeros([output_dim]))

	Y = tf.matmul(inputs, W) + b
	if activate is None:
		outputs = Y
	else:
		outputs = activate(Y)
	return outputs
 
# 定义了三层全连接
# 第一层
Y1 = fcn_layer(x, 784, H1_NN, tf.nn.relu)

# 第二层
Y2 = fcn_layer(Y1, H1_NN, H2_NN, tf.nn.relu)

# 第三层
# Y3 = fcn_layer(Y2, H2_NN, H3_NN, tf.nn.relu)

# 输出层
# forward = fcn_layer(Y3, H3_NN, 10, None)
forward = fcn_layer(Y2, H2_NN, 10, None)
pred = tf.nn.softmax(forward)

# 定义损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# 使用优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
corret_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuray = tf.reduce_mean(tf.cast(corret_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(train_step):
		for batch in range(total_batch):
			xs, ys = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={x:xs, y:ys})

		loss, acc = sess.run([loss_function, accuray], feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
		if (step+1) % 5 == 0:
			print(loss,'---',acc)
	print('--end--')
















