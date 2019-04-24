
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 等差数列生成100个数据
x_data = np.linspace(-1,1,100)		
# 产生x_data.shape个满足标准正太分布的随机数
y_data = x_data * 2.0 + 1.0 + np.random.randn(*x_data.shape) * 0.4
# y_data = x_data * 2.0 + 1.0

'''构建模型'''
train_step = 10
learning_rate = 0.05

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')


def model(w, x, b):
	return tf.multiply(x, w) + b

w = tf.Variable(1.0, name='w')
b = tf.Variable(0.0, name='b')

pred = model(w, x, b)

# 采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))

# 梯度下降的优化器-----去最小化损失函数
optmizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# with tf.Session() as sess:
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 将真实值给x, y
for sx in range(train_step):
	for k, t in zip(x_data, y_data):
		loss = sess.run([optmizer, loss_function], feed_dict = {x:k, y:t})
		
	boptemp = b.eval(session = sess)
	woptemp = w.eval(session = sess)
	# plt.plot(x_data, woptemp * x_data + boptemp)

print(sess.run(w))
print(sess.run(b))

plt.scatter(x_data, y_data, label='original data')
plt.plot(x_data, x_data * sess.run(w) + sess.run(b), label='after', color='r')
plt.legend(loc=2)

x_text = 3.21
predict = sess.run(pred, feed_dict={x:x_text})
print(predict)

plt.show()







