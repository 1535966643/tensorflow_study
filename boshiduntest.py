'''
	波士顿放房价预测
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

learning_rate = 0.06
train_step = 20

# 准备数据
df = pd.read_csv('c:/boston.csv', header = 0)
df_value = np.array(df)

for x in range(12):
	df_value[:,x] = df_value[:,x] / (df_value[:,x].max() - df_value[:,x].min())

x_data = df_value[:,0:12]
y_data = df_value[:, 12]

x = tf.placeholder(tf.float32, [None, 12], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')


# 建立模型
with tf.name_scope('modle'):
	# random_normal:正态分布标准差
	w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name='w')

	b = tf.Variable(1.0, name='b')

	y_c = tf.matmul(x, w) + b

# 损失函数
loss_function = tf.reduce_mean(tf.pow(y - y_c, 2))
# 梯度下降的优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	loss_list = []
	for i in range(train_step):
		loss_sum = 0.0
		for j1, j2 in zip(x_data, y_data):
			xs = j1.reshape(1,12)
			ys = j2.reshape(1,1)

			_, loss = sess.run([optimizer, loss_function], feed_dict={x:xs, y:ys})
			loss_sum = loss_sum + loss
		# xv, yv = shuffle(x_data, y_data)
		loss_average = loss_sum / len(y_data)
		loss_list.append(loss_average)	
		# print(sess.run(w))
		# print(sess.run(b))
		# print(loss_average)

	# 测试数据
	n = np.random.randint(506)
	print(n)
	aa = x_data[n]
	x_text = aa.reshape(1,12)
	predict = sess.run(y_c, feed_dict={x:x_text})
	print(predict)
	target_value = y_data[n]
	print(target_value)

	# 可视化损失值
	plt.plot(loss_list)


# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#     shape: 输出张量的形状，必选
#     mean: 正态分布的均值，默认为0
#     stddev: 正态分布的标准差，默认为1.0
#     dtype: 输出的类型，默认为tf.float32
#     seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
#     name: 操作的名称
plt.show()