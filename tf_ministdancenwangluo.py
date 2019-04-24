'''
	全连接单影藏层的建模
'''
import tensorflow as tf
import numpy
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)

learning_rate = 0.03
train_step = 50
batch_size = 50
total_batch = int(mnist.train.num_examples/batch_size)

# 输入
x = tf.placeholder(tf.float32, [None,784], name='x')
y = tf.placeholder(tf.float32, [None,10], name='y')

# 第一个影藏层
H1_NN = 256
w1 = tf.Variable(tf.random_normal([784, H1_NN]))
b1 = tf.Variable(tf.zeros([H1_NN]))

Y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 构建输出层
w2 = tf.Variable(tf.random_normal([H1_NN, 10]))
b2 = tf.Variable(tf.zeros([10]))

forword = tf.matmul(Y1, w2) + b2
pred = tf.nn.softmax(forword)

# 交叉熵损失函数
# loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# 为了避免log(0)的值不稳定我们使用了新的损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forword, labels=y))


# 使用优化器
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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







