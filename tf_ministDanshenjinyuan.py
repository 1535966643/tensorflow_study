
import tensorflow as tf
import numpy
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
'''
	构建模型
'''
mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
learning_rate = 0.03
train_epochs = 500
total_batch = 100
batch_size = 50
display_step = 50

x = tf.placeholder(tf.float32, [None, 784], name = 'x')
y = tf.placeholder(tf.float32, [None, 10], name='y')

# 定义变量
W =  tf.Variable(tf.random_normal([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# 只有一个神经元
forword = tf.matmul(x, W) + b
# 分类的概率 (激活函数) softmax():分类的概率运算  
# ReLU:  tanh:  sigmod:
pred = tf.nn.softmax(forword)

# 交叉熵的损失函数
loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 检查预测：tf.argmax(pred, 1),     实际：tf.argmax(y, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epochs):
	for batch in range(total_batch):
		xs, ys = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x:xs, y:ys})
	
	loss, acc = sess.run([loss_function, accuracy], feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
	if (epoch + 1) % display_step == 0:
		print('---', epoch,'--loss::',loss,'--acc::',acc)

print(b,'end---',W)

accu_test = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
print('out_findlly--', accu_test)




