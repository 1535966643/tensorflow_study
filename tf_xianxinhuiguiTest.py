
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x_data = np.linspace(-1,1,100)
y_data = x_data * 3.1234 + 2.98 + np.random.randn(*x_data.shape) * 0.2


# 构建模型
train_step = 100
learning_rate = 0.05

w = tf.Variable(1.0, name='b')
b = tf.Variable(0.0, name='w')

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

y_c = tf.multiply(w, x) + b

# mse 这样的损失函数在多元函数中，容易出现‘陷入局部最小值’
loss = tf.reduce_mean(tf.square(y-y_c))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	for i in range(train_step):
		for j1, j2 in zip(x_data, y_data):
			sess.run(optimizer,feed_dict={x:j1, y:j2})
		if i % 10 == 0:
				print(sess.run(w),'---',sess.run(b))	
	# 测试模型
	x_test = 2
	print(sess.run(y_c, feed_dict={x:x_test}))










