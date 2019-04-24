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


checkpoint_path = './checkpoint/Titanic.{epoch:02d}-{val_loss:.2f}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model =  Sequential()

model.load_weights(latest)		# 从保存的模型中恢复权重

loss, acc = model.evaluate(x_test, y_test)
print('模型的准确率：{：5.2f}%'.format(100 * acc))