#!/user/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  # 不显示警告
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model  # 泛型模型
from tensorflow.keras.layers import Dense, Input

train_long = pd.read_csv('train_long.csv')
data1 = train_long['question']
print('question.shape: ', data1.shape)
data2 = train_long['respond']
print('respond.shape: ', data2.shape)

def one_hot(data):  # one_hot编码
    # 创建一个tokenizer，配置为只考虑前2551个最常用的单词
    tokenizer = Tokenizer(2551)
    # 构建单词索引
    tokenizer.fit_on_texts(data)
    one_hot_results = tokenizer.texts_to_matrix(data, mode='binary')
    return one_hot_results

one_hot_results1 = one_hot(data1)
one_hot_results2 = one_hot(data2)

def auto_encoder(data):
    # 压缩特征维度至21维
    encoding_dim = 21
    # this is our input placeholder
    input_img = Input(shape=(data.shape[1],))

    # 编码层
    encoded = Dense(300, activation='relu')(input_img)
    encoded = Dense(150, activation='relu')(encoded)
    encoded = Dense(20, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    # 解码层
    decoded = Dense(20, activation='relu')(encoder_output)
    decoded = Dense(150, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='tanh')(decoded)

    # 构建自编码模型
    autoencoder = Model(inputs=input_img, outputs=decoded)
    # 构建编码模型
    encoder = Model(inputs=input_img, outputs=encoder_output)
    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    # training
    autoencoder.fit(data, data, epochs=1, batch_size=256, shuffle=True)

    return data

autoencode_data1 = auto_encoder(one_hot_results1)
autoencode_data2 = auto_encoder(one_hot_results2)

def cosine_similarity(x, y, norm=False):
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

similarity = []
for x, y in tqdm(zip(autoencode_data1, autoencode_data2)):
    similarity.append(cosine_similarity(x, y))

train_long['similarity'] = pd.Series(similarity)

train_long.to_csv('final_train_long.csv')


# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# 
# train = pd.read_csv('final_train_long.csv')
# X = train['similarity'].values.reshape(-1, 1)
# y = train['target'].values.reshape(-1, 1)
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
# 
# gdbt = GradientBoostingClassifier()
# gdbt.fit(X_train, y_train)
# X_predict = gdbt.predict(X_train)
# print('训练集准确率：', accuracy_score(y_train, X_predict))
# print('训练集f1_score：', f1_score(y_train, X_predict))
# y_predict = gdbt.predict(X_test)
# print('测试集准确率：', accuracy_score(y_test, y_predict))
# print('测试集f1_score：', f1_score(y_test, y_predict))
