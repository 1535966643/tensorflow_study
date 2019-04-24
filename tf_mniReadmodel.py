
'''
	读取模型
'''
import tensorflow as tf
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

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

# 矩阵的0-1互换
def one_to_zero(img):
	img = img / 255
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] == 0.0:
				img[i][j] = 1.0
			else:
				img[i][j] = 0.0
	return img

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

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 读取的时候读取最新的保存文件，即为最后生成的文件
saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

model_dir = './mx_dir/'
# 去获取最新的存盘状态
model = tf.train.get_checkpoint_state(model_dir)

if model and model.model_checkpoint_path:
	# 这是一条恢复语句  把模型文件所有的变量保存到当前的sess中去
	saver.restore(sess, model.model_checkpoint_path)
	print('-----', model.model_checkpoint_path)

# 这里我们继续  断点续训
print('Accuracy', accuracy.eval(session = sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
# print(mnist.test.images, '====', mnist.test.labels)


# 测试自己的手写数字
img = cv2.imread('c:/number/6.png',0)
ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = cv2.resize(img, (28,28))
img = one_to_zero(img)
img = np.reshape(img, [-1, 784])

y = sess.run(pred, feed_dict={x: img}) #y类似一个二维表，因为只有一张图片所以只有一行，y[0]包含10个值，


print('Predict digit', np.argmax(y))#找出最大的值

