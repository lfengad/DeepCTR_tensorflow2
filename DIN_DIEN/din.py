





import numpy as np
import json
import pickle as pkl
import random
import gzip
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from layers import Dice
from utils import DataIterator, prepare_data
class EmbeddingLayer(Layer):
    def __init__(self, user_count, item_count, cate_count, emb_dim, use_negsampling=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_negsampling = use_negsampling
        self.user_emb = Embedding(user_count, self.emb_dim, name="user_emb")
        self.item_emb = Embedding(item_count, self.emb_dim, name="item_emb")
        self.cate_emb = Embedding(cate_count, self.emb_dim, name="cate_emb")
        
    def call(self, user, item, cate, item_his, cate_his,
             noclick_item_his=[],  noclick_cate_hiss=[]):
        user_emb = self.user_emb(user) # (B, D)
        
        # 基本属性embedding:
        item_emb = self.item_emb(item) # (B, D)
        cate_emb = self.cate_emb(cate) # (B, D)
        item_join_emb = Concatenate(-1)([item_emb, cate_emb]) # (B, 2D)
        
        
        # 历史行为序列embedding:
        item_his_emb = self.item_emb(item_his) # (B, T, D)
        cate_his_emb = self.item_emb(cate_his) # (B, T, D)
        item_join_his_emb = Concatenate(-1)([item_his_emb, cate_his_emb]) # (B, T, 2D)
        item_his_emb_sum = tf.reduce_sum(item_join_his_emb, axis=1) # (B, D)
        
        if self.use_negsampling:
            # (B, T, neg_num, D)
            noclick_item_his_emb = self.item_emb(noclick_item_his) 
            # (B, T, neg_num, D)
            noclick_cate_his_emb = self.item_emb(noclick_cate_his) 
            # (B, T, neg_num, 2D)
            noclick_item_join_his_emb = Concatenate(-1)([noclick_item_his_emb, noclick_cate_his_emb])
            # (B, T, 2D)
            noclick_item_emb_neg_sum = tf.reduce_sum(noclick_item_join_his_emb, axis=2) 
            # (B, 2D)
            noclick_item_his_emb_sum = tf.reduce_sum(noclick_item_emb_neg_sum, axis=1) 
            
            return user_emb, item_join_emb, \
                    item_join_his_emb, item_his_emb_sum, \
                    noclick_item_join_his_emb, noclick_item_his_emb_sum 
            
        return user_emb, item_join_emb, \
                item_join_his_emb, item_his_emb_sum
        
class FCLayer(Layer):
    def __init__(self, hid_dims=[80, 40, 2], use_dice=False):
        super().__init__()
        self.hid_dims = hid_dims
        self.use_dice = use_dice
        self.fc = []
        self.dice = []
        for dim in self.hid_dims[:-1]:
            if use_dice:
                self.fc.append(Dense(dim, name=f'dense_{dim}'))
                self.dice.append(Dice())
            else:
                self.fc.append(Dense(dim, activation="sigmoid", 
                                     name=f'dense_{dim}'))
        self.fc.append(Dense(self.hid_dims[-1], name="dense_output"))
        
    def call(self, inputs):
        if self.use_dice:
            fc_out = inputs
            for i in range(len(self.dice)):
                fc_out = self.fc[i](fc_out)
                fc_out = self.dice[i](fc_out)
            fc_out = self.fc[-1](fc_out)
            return fc_out
        else: 
            fc_out = self.fc[0](inputs)
            for fc in self.fc[1:]:
                fc_out = fc(fc_out)
            return fc_out
# 计算注意力得分
class DINAttenLayer(Layer):
    def __init__(self, hid_dims=[80, 40, 1]):
        super().__init__()
        self.FCLayer = FCLayer(hid_dims)
        
    def call(self, query, facts, mask):
        """
        query: (B, 2D)
        facts: (B, T, 2D)
        mask: (B, T)
        """
        mask = tf.equal(mask, tf.ones_like(mask)) # (B, T)
        queries = tf.tile(query, [1, facts.shape[1]]) # (B, 2D*T)
        queries = tf.reshape(queries, [-1, facts.shape[1], facts.shape[2]]) # # (B, T, 2D)
        # print("queries", queries.shape)
        # (B, T, 2D*4)
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        
        fc_out = self.FCLayer(din_all) # (B, T, 1)
        score = fc_out # (B, T, 1)
        score = tf.reshape(score, [-1, 1, facts.shape[1]]) # (B, 1, T)
        
        key_masks = tf.expand_dims(mask, 1) # (B, 1, T)
        padding = tf.ones_like(score) * (-2**32 + 1)
        # True的地方为score，否则为极大的负数
        score = tf.where(key_masks, score, padding) # (B, 1, T)
        score = tf.nn.softmax(score)
        
        output = tf.matmul(score, facts) # (B, 1, 2D)
        output = tf.squeeze(output, 1) # (B, 2D)
        return output
        
# 得到历史行为的embedding表示
class DIN(Model):
    def __init__(self, user_count, item_count, cate_count, EMBEDDING_DIM, 
                 HIS_LEN = 100, use_negsampling = False, hid_dims=[200, 80, 2]):
        super().__init__()
        self.EmbLayer = EmbeddingLayer(user_count, item_count, cate_count, 
                                       EMBEDDING_DIM, use_negsampling)
        self.AttenLayer = DINAttenLayer()
        self.FCLayer = FCLayer(hid_dims, use_dice=True)
        
        
    def call(self, user, item, cate, item_his, cate_his, mask):
        # 得到embedding
        embs = self.EmbLayer(user, item, cate, item_his, cate_his)
        # (B, 2D) 
        user_emb, item_join_emb, item_join_his_emb, item_his_emb_sum = embs
        # 计算目标item与历史item的attention分数，然后加权求和，得到最终的embedding
        behavior_emb = self.AttenLayer(item_join_emb, item_join_his_emb, mask) # (B, 2D)
        
        # 全连接层
        inp = tf.concat([user_emb, item_join_emb, item_his_emb_sum, 
                         item_his_emb_sum, behavior_emb], axis=-1)
        output = self.FCLayer(inp)
        # logit = tf.nn.softmax(output)
        return output # , logit
        
base_path = "data/"
train_file = base_path + "local_train_splitByUser"
test_file = base_path + "local_test_splitByUser"
uid_voc = base_path + "uid_voc.pkl"
mid_voc = base_path + "mid_voc.pkl"
cat_voc = base_path + "cat_voc.pkl"
batch_size = 128
maxlen = 100

train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, 
                          batch_size, maxlen, shuffle_each_epoch=False)

n_uid, n_mid, n_cat = train_data.get_n() # 用户数，电影数，类别数
model = DIN(n_uid, n_mid, n_cat, 8)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

import time
# 训练模型

total = 0
cnt = 0
for i, (src, tgt) in enumerate(train_data):
    data = prepare_data(src, tgt, maxlen=100, return_neg=False)
    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = data
    st = time.time()
    with tf.GradientTape() as tape:
        output = model(uids, mids, cats, mid_his, cat_his, mid_mask)
        loss = tf.keras.losses.categorical_crossentropy(target, output)
        loss = tf.reduce_mean(loss)
        if i%100 == 0:
            print("batch %d loss %f" % (i, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    en = time.time()
    total += (en - st)
    cnt += 1
#    print("Iter time {} ms".format((en - st) * 1000))
    if i == 1000:
        break
print("Iter time {} ms".format((total) * 1000/ cnt))
