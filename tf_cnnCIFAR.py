
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as p
from sklearn.preprocessing import OneHotEncoder

# 导入数据集
def load_cifar_batch(filename):As.18435205162

	with open(filename, 'rb') as f:
		# 每个样本都是由标签和图像数据组成
		data_dict = p.load(f, encoding='bytes')
		images = data_dict[b'data']
		labels = data_dict[b'labels']
		# 调整课原始数据结构：BCWH
		images = images.reshape(10000,3,32,32)
		# 由于后面的需要，将通道熟移动到最后一个维度上
		images = images.transpose(0,2,3,1)
		labels = np.array(labels)
		return images, labels

def load_cifar_data(data_dir):
	images_train=[]
	labels_train=[]
	for i in range(5):
		f = os.path.join(data_dir, 'data_batch_%d'%(i+1))
		# print('f:',f)
		# 调用load_cifar_batch获得批量图像和对应标签
		image_batch,label_batch = load_cifar_batch(f)
		images_train.append(image_batch)
		labels_train.append(label_batch)

		Xtrain = np.concatenate(images_train)
		Ytrain = np.concatenate(labels_train)
		image_batch,label_batch
	Xtest, Ytest = load_cifar_batch(os.path.join(data_dir,'test_batch'))
	return Xtrain,Ytrain,Xtest,Ytest

data_dir = 'D:/cifar10Data/cifar-10-batches-py/'
Xtrain,Ytrain,Xtest,Ytest = load_cifar_data(data_dir)

# print('training data shape:',Xtrain.shape)
# print('training labels shape:',Ytrain.shape)
# print('test data shape:',Xtest.shape)
# print('test labels shape:',Ytest.shape)

# 定义标签字典，每个数字代表一个图像类别
label_dict = {
	0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
	5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'
}

# 将数字标准化
Xtrain_normalize = Xtrain.astype('float32') / 255.0
Xtest_normalize = Xtest.astype('float32') / 255.0

# 数据的预处理
encoder = OneHotEncoder(sparse=False)
yy = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
encoder.fit(yy)

# print(Ytrain) # 可以看看格式
Ytrain_reshape = Ytrain.reshape(-1,1)
# print(Ytrain_reshape)
Ytrain_onehot = encoder.transform(Ytrain_reshape)
Ytest_reshape = Ytest.reshape(-1,1)
Ytest_onehot = encoder.transform(Ytest_reshape)

# 定义权值
def weight(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name= 'W')

# 定义偏置
def bias(shape):
	return tf.Variable(tf.constant(0.1, shape=shape,), name='b')

# d定义卷积操作（步长为1）
def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

# 定义池化
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')


# 定义网络结构

# 输入层
# 图像为32X32,彩色图
with tf.name_scope('input_layer'):
	x = tf.placeholder('float32', shape=[None,32,32,3], name='x')

# 第一个卷积层
# 输出通道32，图像尺寸不变
with tf.name_scope('conv_1'):
	W1 = weight([3,3,3,32])	#这里就是在说卷积核的大小为3x3
	b1 = bias([32])
	conv_1 = conv2d(x, W1) + b1
	conv_1 = tf.nn.relu(conv_1)
# 第一个池化层
with tf.name_scope('pool_1'):
	pool_1 = max_pool_2x2(conv_1)

# 第二个卷积层
# 输出通道64，图像尺寸不变
with tf.name_scope('conv_2'):
	W2 = weight([3,3,32,64])	#这里就是在说卷积核的大小为3x3
	b2 = bias([64])
	conv_2 = conv2d(pool_1, W2) + b2
	conv_2 = tf.nn.relu(conv_2)
# 第二个池化层
with tf.name_scope('pool_2'):
	pool_2 = max_pool_2x2(conv_2)

# 定义全连接层
with tf.name_scope('fc'):
	W3 = weight([4096, 128]) # 128个神经元
	b3 = bias([128])
	flat = tf.reshape(pool_2,[-1, 4096])
	h = tf.nn.relu(tf.matmul(flat, W3) + b3)
	h_drop = tf.nn.dropout(h, keep_prob=0.7)

# 输出层
with tf.name_scope('output_layer'):
	W4 = weight([128, 10])
	b4 = bias([10])
	pred = tf.nn.softmax(tf.matmul(h_drop, W4) + b4)

with tf.name_scope('optimizer'):
	# 定义占位符
	y = tf.placeholder('float32', shape=[None, 10], name='label')
	# 定义损失函数
	loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	# 选择优化器
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

# 定义准确率
with tf.name_scope('evaluation'):
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

# 启动会话
train_epochs = 25
batch_size = 50
total_batch = int(len(Xtrain)/batch_size)
epoch_list = []
accuracy_list = []
loss_list = []

# 
epoch = tf.Variable(0, name='epoch', trainable=False)

sess = tf.Session()
sess.run(tf.global_variabl es_initializer())


# 断点续训
# 设置检查点存储目录
ckpt_dir = 'cifar10_log/'
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)

# 生成saver 
saver = tf.train.Saver(max_to_keep=1)

# 如果检查点文件，读取最新的检查点文件，恢复变量
ckpt = tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
	# （重要！！！）这是一条恢复语句  把模型文件所有的变量保存到当前的sess中去
	saver.restore(sess, ckpt)
else:
	print('training from scratch')

# 获取续训参数
start = sess.run(epoch)
print('Train start from {} epoch'.format(start+1))

# 迭代训练-------得到的是归一化的图像数据和标签数据
def get_train_batch(number, batch_size):
	return Xtrain_normalize[number*batch_size:(number+1)*batch_size], \
	Ytrain_onehot[number*batch_size:(number+1)*batch_size] 

for ep in range(train_epochs):
	for i in range(total_batch):
		batch_x, batch_y = get_train_batch(i, batch_size)
		sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
		# if i % 100 == 0:
		# 	print(i)
	loss, acc = sess.run([loss_function, accuracy], feed_dict={x:batch_x,y:batch_y})	
	epoch_list.append(ep+1)
	loss_list.append(loss)
	accuracy_list.append(acc)

	print('trein epoch:',ep, 'loss=',loss, 'Accuracy=',acc)

	# 保存检查点
	saver.save(sess, ckpt_dir+'cifar10_cnn_model.cpkt', global_step = ep+1)
	# sess.run(epoch.assign(ep+1))

print('end')

# 可视化损失值
fig = plt.gcf()
fig.set_size_inche(4,2)
plt.plot(epoch_list, loss_list, label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')

# 可视化准确率
plt.plot(epoch_list, loss_list, label='accuracy')
fig = plt.gcf()
fig.set_size_inche(4,2)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

# 模型的的预测以及评估
# 计算测试集上的准确率
test_total_batch = int(len(Xtest_normalize)/batch_size)
test_acc_sum = 0.0
for i in range(test_total_batch):
	test_imgae_batch = Xtest_normalize[i*batch_size:(i+1)*batch_size]
	test_label_batch = Ytest_onehot[i*batch_size:(i+1)*batch_size]
	test_batch_acc = sess.run(accuracy, feed_dict={x:test_imgae_batch,y:test_label_batch})
	test_acc_sum += test_batch_acc
test_ac = float(test_acc_sum/test_total_batch)
print('Test accuracy=', test_ac)


# 利用用模型进行预测
test_pred = sess.run(pred, feed_dict={x:Xtest_normalize[0:10]})
prediction_result = sess.run(tf.argmax(test_pred, 1))

# 可视化结果
plot_images_lables_prediction(Xtest, Ytest, prediction_result, 1, 10)




































