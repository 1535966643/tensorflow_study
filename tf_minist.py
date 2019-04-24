'''
	手写体识别
'''
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("D:/mnistData", one_hot=True)
# 训练集
print(mnist.train.num_examples)
# 验证集
print(mnist.validation.num_examples)
# 测试集
print(mnist.test.num_examples)
# 图像类型
print(mnist.train.images.shape)
# 标签类型
print(mnist.train.labels.shape)

# 简单显示一下手写体数字
def function_image():
	img = mnist.train.images[246]
	image = img.reshape(28,28)
	plt.imshow(image, plt.cm.gray)			

function_image()

# 独热编码::如2：[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(mnist.train.labels[246])
print(np.argmax(mnist.train.labels[246]))

print(mnist.train.images[0:10],'------')

# 批量取十条数据
bx, by = mnist.train.next_batch(batch_size=10)
print(bx.shape)
print(by.shape)


plt.show()



