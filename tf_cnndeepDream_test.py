'''
	经典的CNN-----VGGNet:vgg16
			-----Google Net	
'''
from __future__ import print_function
import os
from io import BytesIO
import numpy  as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

# 创建图和会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

model_fn = 'D:/tensorflow_inception_graphData/tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

# 定义输入图像的占位符
t_input = tf.placeholder(np.float32, name='input')

# 图像预处理--减均值
imagenet_mean = 117.0

# 头像那个预处理--增加维度
t_preproessed = tf.expand_dims(t_input - imagenet_mean, 0)

# 导入模型并且将与处理的图像送入网络
tf.import_graph_def(graph_def,{'input':t_preproessed})

# 找卷积层
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
# 输出卷积层的层数
print(len(layers))
# 输出所有的卷积层名称
print(layers)

# 渲染函数
def render_naive(t_obj, img0, iter_n=20, step=1.0):
	# t_obj:layer_output[],即为卷积层的某个通道值
	# img0:初始的噪声图
	# iter_n:迭代次数
	# step:用于控制每次迭代的步长，可以看做是学习率

	t_score = tf.reduce_mean(t_obj)   # 求平均值

	# 计算t_score对t_input的梯度
	t_grad = tf.gradients(t_score, t_input)[0]

	img = img0.copy()
	for i in range(iter_n):
		# 在sess中计算梯度， 以及当前的t_grad
		g, score = sess.run([t_grad, t_score],{t_input:img})

		# 对img应用梯度
		# 首先对梯度进行归一化处理
		g /= g.std() + 1e-8
		# 将正规划处理的梯度应用在图像上，step用于控制每次的迭代步长，次数为1
		img += g * step
		print('iter:%d'%(i+1), 'score(mean)=%f'%score)

	# 保存图片
	savePath = 'c:/deepdream.jpg'
	scipy.misc.toimage(img).save(savePath)
	print('img saved:%s'%savePath)


# 输出制定卷积层的参数  
# mixed4d_3x3_bottleneck_pre_relu 有144个通道
name = 'mixed4d_3x3_bottleneck_pre_relu'
print(name, '----', str(graph.get_tensor_by_name('import/'+ name +':0').get_shape()))

# 此处选择任意通道进行最大化
# layer_output = graph.get_tensor_by_name('import/%s:0'%name)

# # 定义噪声图像
# img_noise = np.random.uniform(size=(224,224,3)) + 100.0

# 生成原始的Deep Dream图像-----单通道
# channel = 139
# 较低层但通到卷积特征生成Deep Dream图像
# channel = 86
# 高层通道卷积特征生成Deep Dream图像
# channel = 118
# 生成原始Deep Dream图像-----所有通道通道(去掉)
# render_naive(layer_output, img_noise, iter_n=20)
# 调用render_naive函数渲染
# render_naive(layer_output[:,:,:,channel], img_noise, iter_n = 20)

###### 以背景图像为起点生成Deep Dream图像(目标)
name='mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0'%name)
image_test = PIL.Image.open('c:/0.jpg')
render_naive(layer_output, image_test, iter_n = 50)

# 保存并且显示图像
im = PIL.Image.open('c:/1.jpg')
im.show()
im.save('c:/naive_single_chn.jpg')


















