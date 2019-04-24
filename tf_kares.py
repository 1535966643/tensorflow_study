

from keras.layers import Dense, Dropout
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
import urllib.request
import os
import numpy
import pandas as pd
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint


# 泰坦尼克号数据处理
data_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
data_file_path = "data/titanic3.xls"
# 这里面根据网络下载文件，，，但是添加文件夹后无法识别（不知道为什么）
if not os.path.isfile(data_file_path):
	result = urllib.request.urlretrieve(data_url, data_file_path)
	print('downloaded:', result)
else:
	print(data_file_path, 'data file already save')

# 读入数据
df_data = pd.read_excel(data_file_path)
# 这是在查看数据摘要（关于数据的一些统计信息）
# print(df_data.describe())
#  数据筛选
selected_cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
# 从元数据取出这些数据
selected_df_data = df_data[selected_cols]
# print(selected_df_data)

# 进行数据的预处理
def prepare_data(df_data):
	df = df_data.drop(['name'], axis=1)	# 删除姓名列
	age_mean = df['age'].mean()
	df['age'] = df['age'].fillna(age_mean)		# fillna() 函数找出了所有的空值，并将平均值填入其中
	fare_mean = df['fare'].mean()
	df['fare'] = df['fare'].fillna(fare_mean)

	df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)	# 	吧sex的值转化为数字
	df['embarked'] = df['embarked'].fillna('S')	#  为缺省值填入S
	df['embarked'] = df['embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

	ndarry_data = df.values 	# 转化为ndarray数组

	# 第一列是标签值，， 后面都是特征值
	features = ndarry_data[:, 1:]	
	label = ndarry_data[:, 0]

	# 特征值标准化  使用了sklearn 库
	minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
	norm_festures = minmax_scale.fit_transform(features)
	# print(norm_festures[:3])

	return norm_festures,label


# 打乱数据(这是panda里面的数据打乱方式)
shuffled_df_data = selected_df_data.sample(frac = 1)
#  进行数据预处理
x_data, y_data = prepare_data(shuffled_df_data)
train_size = int(len(x_data) * 0.8)

# 这里我们的训练集为数据的80%
x_train = x_data[:train_size]
y_train = y_data[:train_size]

# 测试集为20%
x_test = x_data[train_size:]
y_test = y_data[train_size:]


# 建立Keras模型
model =  Sequential()

# 加入第一层，输入特征数据是7个神经元，第一个影藏层为64个神经元节点，都有偏置项
model.add(Dense(units = 64,		# 这层有64个神经元节点
				input_dim = 7,		# 输入层有7个神经元节点
				use_bias = True,		# 有偏置项
				kernel_initializer = 'uniform',		# 初始化权重使用什么方式
				bias_initializer = 'zeros',		# 对偏置初始化的值
				activation = 'relu'		# 所使用的激活函数
				))

model.add(Dense(
				units = 32,		# 第二个隐藏层有32个节点
				activation = 'sigmoid'		# 所使用的激活函数
				))
 
# model.add(Dropout(rate=0.3))		#为防止过拟合，我们采用了dropout 
# 指定输出层
model.add(Dense(
				units = 1,
				activation = 'sigmoid'
				))

model.summary()

# 设置模型的学习进程
model.compile(
				optimizer = Adam(0.003),		# 优化器的名称
				loss = 'binary_crossentropy',		
					# loss 是损失函数名
						# 用sigmoid作为激活函数，一般损失函数选用binary_crossentropy
						# softmax作为激活函数，一般损失函数选用categorical_crossentropy
				metrics = ['accuracy']		# 模型要训练和评估的度量值
				)


# 模型的训练
## 这是有返回对象
train_history = model.fit(
							x = x_train,
							y = y_train,
							validation_split = 0.2,		# 验证集所占的比例
							epochs = 100,				# 训练周期
							batch_size = 40,			# 一次带多少数据
							verbose = 2		# 训练过后才能显示模式，0：不显示  1：带进度条模型  2：每个epoch显示一行
							)



# 模型的可视化





#  模型的评估
evaluate_result = model.evaluate(x = x_test, y = y_test)
print('模型评估：：', evaluate_result)
# return 
exit()
# 模型的应用
jack_info = [0, 'jack', 3, 'male', 23, 1, 0, 5.000, 'S']
rose_info = [1, 'rose', 1, 'female', 20, 1, 0, 100.000, 'S']

#  创建新的旅客dataframe
new_passenger_pd = pd.DataFrame([jack_info, rose_info], columns=selected_cols)

# 将新创建的旅客信息加入到旧的dataframe
all_passenger_pd = selected_df_data.append(new_passenger_pd)
# print(all_passenger_pd[-3:])

# 执行预测
# 准备数据
x_features, y_label = prepare_data(all_passenger_pd)
# 利用模型计算旅客的生存概率
surv_probability = model.predict(x_features)		# 这个 .predict是keras自带的测试方法

# 在数据表最后插入一列生存概率
all_passenger_pd.insert(len(all_passenger_pd.columns), 'surv_probability', surv_probability)
# print(all_passenger_pd[-3:])		# 查看数据的生存概率


#  保存模型
# 设置回调参数，内置的回调还有：：：
		# learningRataScheduler()
		# EarlyStopping

logdir = './logs'	# 建立日志的目录
checkpoint_path = './checkpoint/Titanic.{epoch:02d}-{val_loss:.2f}.ckpt'		# epoch::第几轮   val_losss::损失

# 模型的设置
model.compile(
				optimizer = Adam(0.003),		# 优化器的名称
				loss = 'binary_crossentropy',		
					# loss 是损失函数名
						# 用sigmoid作为激活函数，一般损失函数选用binary_crossentropy
						# softmax作为激活函数，一般损失函数选用categorical_crossentropy
				metrics = ['accuracy']		# 模型要训练和评估的度量值
				)
callbacks = [
			TensorBoard(log_dir = logdir,
						histogram_freq = 2),	# 直方图存储频率

			ModelCheckpoint(								# 模型的检查点文件
							filepath = checkpoint_path,		# 文件名的取名规则
							save_weights_only = True,		# 仅仅保存的是模型中的权重值，没有保存结构
							verbose = 1,						# 
							period=5)						# 循环多少个周期，然后保存。。。。。。最后只是保存最后的五个
											
	]

train_history = model.fit(
							x = x_train,
							y = y_train,
							validation_split = 0.2,		# 验证集所占的比例
							epochs = 100,				# 训练周期
							batch_size = 40,			# 一次带多少数据
							callbacks = callbacks,
							verbose = 1		# 训练过后才能显示模式，0：不显示  1：带进度条模型  2：每个epoch显示一行
							)
# print(train_history.history)
# model.save_weights('my_model_weights.h5')
# 恢复模型
checkpoint_path = './checkpoint/Titanic.{epoch:02d}-{val_loss:.2f}.ckpt'

ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
''' 这是是视频上的讲'''
# checkpoint_dir = os.path.dirname(checkpoint_path)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights('my_model_weights.h5')		# 从保存的模型中恢复权重

loss, acc = model.evaluate(x_test, y_test)
print('模型的准确率：{:5.2f}%'.format(100 * acc))


# 
















