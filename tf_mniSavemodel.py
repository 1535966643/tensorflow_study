'''
	保存训练的模型
'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import os

mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
learning_rate = 0.01
train_step = 40
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
save_step = 5

x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')

H1_NN = 256
H2_NN = 64

def fcn_layer(inputs, input_dim, output_dim, activation=None):

	W = tf.Variable(tf.truncated_normal([input_dim, output_dim],stddev=0.1))
	b = tf.Variable(tf.zeros([output_dim]))
	
	Xwb = tf.matmul(inputs, W) + b
	
	if activation is None:	# 如果不使用激活函数
		outputs = Xwb
	else:
		outputs = activation(Xwb)
	return outputs
# 影藏层
Y1 = fcn_layer(x, 784, H1_NN, tf.nn.relu)
Y2 = fcn_layer(Y1, H1_NN, H2_NN, tf.nn.relu)

# 构建输出层
forward = fcn_layer(Y2, H2_NN, 10, None)
pred = tf.nn.softmax(forward)

# 为了避免log(0)的值不稳定我们使用了新的损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# 使用优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 保存训练的模型
model_dir = './mx_dir/'
# 如果没有该文件则创建
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

# 并存储模型
saver = tf.train.Saver()

# 开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(train_step):
	for batch in range(total_batch):
		xs, ys = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x:xs, y:ys})

	loss, acc = sess.run([loss_function, accuracy], feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
	if (step+1) % 5 == 0:
		print(step+1,'---loss--', loss,'---', acc)
	if (step+1) % 5 == 0:
		saver.save(sess, os.path.join(model_dir, 'mnist_model_{}.ckpt'.format(step+1)))
		print('mnist_model_{}.ckpt saved'.format(save_step+1))

# saver.save(sess, os.path.join(model_dir, 'mnist_model_final_model.ckpt'))

print('--end--')

sess.close()	





















