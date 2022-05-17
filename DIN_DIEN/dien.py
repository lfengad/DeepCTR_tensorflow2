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
        self.user_emb = Embedding(user_count, self.emb_dim,
                                  mask_zero=True, name="user_emb")
        self.item_emb = Embedding(item_count, self.emb_dim,
                                  mask_zero=True, name="item_emb")
        self.cate_emb = Embedding(cate_count, self.emb_dim,
                                  mask_zero=True, name="cate_emb")
        
    def call(self, user, item, cate, item_his, cate_his,
             noclick_item_his=[],  noclick_cate_his=[]):
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
            # 只取出第一个负样本构成序列，(B, T, 2D)
            noclick_item_join_his_emb = noclick_item_join_his_emb[:, :, 0, :] 
            # # (B, T, 2D)
            # noclick_item_join_his_emb = tf.squeeze(noclick_item_join_his_emb, 2)
            
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
        self.bn = BatchNormalization()
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
        inputs = self.bn(inputs)
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
        # (B, T, 2D*4)
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        
        fc_out = self.FCLayer(din_all) # (B, T, 1)
        score = fc_out # (B, T, 1)
        key_masks = tf.expand_dims(mask, 2) # (B, T) -> (B, T, 1)
        padding = tf.ones_like(score) * (-2**32 + 1)
        # True的地方为score，否则为极大的负数
        score = tf.where(key_masks, score, padding) # (B, T, 1)
        score = tf.nn.softmax(score) # (B, T, 1)
        
        return score
        
class AuxTrainLayer(Layer):
    def __init__(self, hid_dims=[100, 50, 1]):
        super().__init__()
        self.clk_fc = FCLayer(hid_dims)
        self.noclk_fc = FCLayer(hid_dims)
        
    def call(self, h_states, click_seq, noclick_seq, mask):
        mask = tf.cast(mask, tf.float32)
        seq_len = click_seq.shape[1] # T-1
        
        clk_input = tf.concat([h_states, click_seq], -1) # (B, T-1, 2D*2)
        clk_prob = tf.sigmoid(self.clk_fc(clk_input)) # (B, T-1, 1)
        # (B, T-1)
        clk_loss = - tf.reshape(tf.math.log(clk_prob), [-1, seq_len]) * mask 
        
        noclk_input = tf.concat([h_states, noclick_seq], -1) # (B, T-1, 2D*2)
        noclk_prob = tf.sigmoid(self.clk_fc(noclk_input)) # (B, T-1, 1)
        # (B, T-1)
        noclk_loss = - tf.reshape(tf.math.log(1.0 - noclk_prob), [-1, seq_len]) * mask
        # 不指定axis，则计算全部数值的平均值
        aux_loss = tf.reduce_mean(clk_loss + noclk_loss)
        return aux_loss
        
class AUGRUCell(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units
    
    
    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_output, atten_scores)
        # 因此，t时刻输入的x_t的维度为：
        dim_xt = input_shape[0][-1]
        
        # 重置门对t时刻输入数据x的权重参数：
        self.W_R_x = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='W_R_x')
        # 重置门对t时刻输入隐藏状态state的权重参数：
        self.W_R_s = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_R_s')
        # 重置门偏置项参数：
        self.W_R_b = tf.Variable(tf.random.normal(shape=[self.units]), name='W_R_b')
        
        
        # 更新门对t时刻输入数据x的权重参数：
        self.W_U_x = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='W_U_x')
        # 更新门对t时刻输入隐藏状态state的权重参数：
        self.W_U_s = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_U_s')
        # 更新门偏置项参数：
        self.W_U_b = tf.Variable(tf.random.normal(shape=[self.units]), name='W_U_b')
        
        
        # 候选隐藏状态 ~h_t 对t时刻输入数据x的权重参数：
        self.W_H_x = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='W_H_x')
        # 候选隐藏状态 ~h_t 对t时刻输入隐藏状态state的权重参数：
        self.W_H_s = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_H_s')
        # 候选隐藏状态 ~h_t 偏置项参数：
        self.W_H_b = tf.Variable(tf.random.normal(shape=[self.units]), name='W_H_b')
        
    
    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]
        """
        x_t: x_(t), shape=(B, 2D)
        states: hidden_state_(t-1), shape=(B, units)
        att_score: attention_score_(t),  shape=(B, 1)
        """
        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.W_R_x) + tf.matmul(states, self.W_R_s) + self.W_R_b)
        # 更新门
        u_t = tf.sigmoid(tf.matmul(x_t, self.W_U_x) + tf.matmul(states, self.W_U_s) + self.W_U_b)
        # 带有注意力的更新门
        a_u_t = tf.multiply(att_score, u_t)
        # 候选隐藏状态
        _h_t = tf.tanh(tf.matmul(x_t, self.W_H_x) + tf.matmul(tf.multiply(r_t, states), self.W_H_s) 
                       + self.W_H_b)
        # 输出值
        h_t = tf.multiply(1-a_u_t, states) + tf.multiply(a_u_t, _h_t)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t
        
        
        return h_t, next_state # 第一个表示output
        
        
# 得到历史行为的embedding表示
class DIEN(Model):
    def __init__(self, user_count, item_count, cate_count, EMBEDDING_DIM, 
                 HIS_LEN = 100, use_negsampling = True, hid_dims=[200, 80, 2]):
        super().__init__()
        
        self.rnn_dim = EMBEDDING_DIM*2
        
        self.EmbLayer = EmbeddingLayer(user_count, item_count, cate_count, 
                                       EMBEDDING_DIM, use_negsampling)
        
        self.GRU = GRU(self.rnn_dim, return_sequences=True)
        self.AuxTrainLayer = AuxTrainLayer()
        self.AttenLayer = DINAttenLayer()
        # self.AUGRU = AUGRU(EMBEDDING_DIM*2, return_state=True)
        self.AUGRU = RNN(AUGRUCell(self.rnn_dim))
        self.FCLayer = FCLayer(hid_dims, use_dice=True)
        
        
    def call(self, user, item, cate, item_his, cate_his, mask, no_m_his, no_c_his):
        # 转 0, 1 为 True, False 
        mask_bool = tf.cast(mask, tf.bool)
        # 得到embedding
        embs = self.EmbLayer(user, item, cate, item_his, cate_his, no_m_his, no_c_his)
        # (B, 2D) 
        user_emb, item_emb, his_emb, his_emb_sum, noclk_his_emb, noclk_his_emb_sum = embs
        
        
        # 第一层 GRU
        # tf2.2中的大坑：
        # 官方文档中第二个参数为mask，
        # 但是不指定参数名字mask=mask_bool的话，
        # 则mask_bool会当成参数initial_state的值
        gru_output = self.GRU(his_emb, mask=mask_bool) # (B, T, 2D)
        # 辅助损失函数
        aux_loss = self.AuxTrainLayer(gru_output[:, :-1, :], 
                                      his_emb[:, 1:, :],
                                      noclk_his_emb[:, 1:, :],
                                      mask[:, 1:]) # (B,)
        
        # 计算目标item与历史item的attention分数
        atten_scores = self.AttenLayer(item_emb, gru_output, mask) # (B, T, 1)
        
        # AUGRU
        behavior_emb = self.AUGRU((gru_output, atten_scores), mask=mask_bool) # (B, 2D) 
        
        # 全连接层
        inp = tf.concat([user_emb, item_emb, his_emb_sum, behavior_emb, 
                         noclk_his_emb_sum], axis=-1)
        output = self.FCLayer(inp)
        logit = tf.nn.softmax(output)
        return output, logit, aux_loss
    
    def train(self, user, item, cate, item_his, cate_his, mask, no_m_his, no_c_his, target):
        output, _, aux_loss = self.call(user, item, cate, item_his, cate_his, mask, no_m_his, no_c_his)
        loss = tf.keras.losses.categorical_crossentropy(target, output, from_logits=False)
        loss = tf.reduce_mean(loss)
        return loss, aux_loss
        
    def predict(self, user, item, cate, item_his, cate_his, mask):
        _, pred, _ = self.call(user, item, cate, item_his, cate_his, mask)
        return pred
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
model = DIEN(n_uid, n_mid, n_cat, 16)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

# 训练模型
for i, (src, tgt) in enumerate(train_data):
    data = prepare_data(src, tgt, maxlen=100, return_neg=True)
    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, no_m_his, no_c_his = data
    with tf.GradientTape() as tape:
        loss, aux_loss = model.train(uids, mids, cats, mid_his, cat_his, 
                                     mid_mask, no_m_his, no_c_his, target)
        if i%10 == 0:
            print("batch %d loss %f, aux loss %f" % (i, loss.numpy(), aux_loss.numpy()))
            
        loss = loss + aux_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    
    if i == 1000:
        break

 
 


