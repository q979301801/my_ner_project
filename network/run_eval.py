import math
import pickle

import tensorflow as tf
from absl import logging

from my_ner_project.network import model_help
from tensorflow.keras.layers import LSTM,Bidirectional,Dense,Conv2D,BatchNormalization,Activation,AvgPool2D,Dropout
import numpy as np
from tensorflow_addons.text import crf_log_likelihood,viterbi_decode
# from model.create_model import add_crf_layer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class add_crf_layer(tf.keras.layers.Layer):
    def __init__(self,num_tags,**kwargs):
        super(add_crf_layer, self).__init__(**kwargs)
        self._num_tags = num_tags

    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotUniform()
        self.crf_layer = self.add_weight(
            "crf_layer",
            shape=[self._num_tags, self._num_tags],
            initializer=initializer,
            dtype=tf.float32)
        super(add_crf_layer, self).build(input_shape)

    def call(self, inputs):
        outputs1 = inputs
        mid_ = tf.eye(self._num_tags)
        transition_params_out = tf.matmul(self.crf_layer,mid_)
        return transition_params_out


class my_crf_model(tf.keras.Model):
    def __init__(self,vocab_size,embedding_width,lstm_hiden_sieze,num_tags,
                 max_sequence_length,num_layers,
                 num_attention_heads,inner_dim,
                 output_range = None):
        self.vocab_size = vocab_size
        self.embedding_width = embedding_width
        inputx = tf.keras.layers.Input(shape=(None,),dtype=tf.int32,name='input_data')
        mask = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_mask')

        embedding_layer = model_help.OnDeviceEmbedding(vocab_size = vocab_size,
                                     embedding_width = embedding_width,use_one_hot = True)
        embedding_data = embedding_layer(inputx)

        position_embedding_layer = model_help.PositionEmbedding(
            max_length=max_sequence_length,
            name='position_embedding')
        position_embeddings = position_embedding_layer(embedding_data)

        embeddings = tf.keras.layers.Add()(
            [embedding_data, position_embeddings])

        embedding_norm_layer = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

        embeddings = embedding_norm_layer(embeddings)
        embedding_data = (tf.keras.layers.Dropout(rate=0.1)(embeddings))

        # dim = int(math.sqrt(embedding_width))
        # assert dim * dim == embedding_width
        # embedding_data = tf.expand_dims(embedding_data,axis=-1)
        # c1 = Conv2D(filters=8,kernel_size=(3,3),padding='same',activation=None)
        # embedding_data = c1(embedding_data)
        # embedding_data = BatchNormalization()(embedding_data)
        # embedding_data = Activation('relu')(embedding_data)
        # embedding_data = AvgPool2D(pool_size=(2,2),strides=1,padding='same')(embedding_data)
        # embedding_data = Dropout(0.1)(embedding_data)
        # embedding_data = tf.reduce_mean(embedding_data,axis=-1)

        # print('embedding_data',embedding_data)

        transformer_layers = []
        data = embedding_data
        attention_mask = model_help.SelfAttentionMask()(data, mask)
        encoder_outputs = []
        for i in range(num_layers):
            if i == num_layers - 1 and output_range is not None:
                transformer_output_range = output_range
            else:
                transformer_output_range = None
            layer = model_help.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation='gelu',
                output_dropout=0.1,
                attention_dropout=0.1,
                output_range=transformer_output_range,
                name='transformer/layer_%d' % i)
            transformer_layers.append(layer)
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        bid_lstm_output = encoder_outputs[-1]
        # last_encoder_output = encoder_outputs[-1]

        # lstm_shape = last_encoder_output.shape[1:]
        # bid_lstm = Bidirectional(LSTM(lstm_hiden_sieze, return_sequences=True,activation = 'tanh'), input_shape=lstm_shape)
        # bid_lstm_output = bid_lstm(last_encoder_output)

        # bid_lstm_output_norm_layer = tf.keras.layers.LayerNormalization(
        #     name='bid_lstm_output_norm_layer', axis=-1, epsilon=1e-12, dtype=tf.float32)
        # bid_lstm_output = bid_lstm_output_norm_layer(bid_lstm_output)

        d_out = Dense(units=num_tags)
        final_output = d_out(bid_lstm_output)
        crf_layers = add_crf_layer(num_tags = num_tags)
        transition_params = crf_layers(final_output)

        super(my_crf_model,self).__init__(inputs = [inputx,mask],outputs = [final_output,transition_params])



def loss1(final_output,real_tag,sequence_lengths,transition_params):
    log_likelihood, transition_params = crf_log_likelihood(inputs=final_output,
                                                       tag_indices=real_tag,
                                                       sequence_lengths=sequence_lengths,
                                                        transition_params= transition_params)
    # 最大化 log_likelihood 相当于最小化  -log_likelihood
    # print('log_likelihood',log_likelihood.shape)
    return tf.reduce_mean(-log_likelihood)



#
num_tags = 10
m1 = my_crf_model(vocab_size = 5000,embedding_width = 128,lstm_hiden_sieze = 64,num_tags = num_tags,
                  max_sequence_length = 50,num_layers = 1,num_attention_heads = 2,
                  inner_dim = 512)


f1=open(r'..\data\y_test.pkl','rb')
f=open(r'..\data\x_test.pkl','rb')

# f1=open(r'C:\Users\yy\Desktop\my_ner_weibo\chinese_nlp\NER\MSRA\data\y_valid.pkl','rb')
# f=open(r'C:\Users\yy\Desktop\my_ner_weibo\chinese_nlp\NER\MSRA\data\x_valid.pkl','rb')

# f1=open(r'C:\Users\yy\Desktop\my_ner_weibo\chinese_nlp\NER\MSRA\data\y_train.pkl','rb')
# f=open(r'C:\Users\yy\Desktop\my_ner_weibo\chinese_nlp\NER\MSRA\data\x_train.pkl','rb')
data = pickle.load(f)
x_valid = tf.cast(data,dtype=tf.int32)

data_y = pickle.load(f1)
y_valid = tf.cast(data_y,dtype=tf.int32)

seq_len = tf.where(x_valid > 0,1,0)
sequence_lengths = tf.reduce_sum(seq_len,axis=-1)

repeat_size = 1
batch_size = 1
# print(sequence_lengths)
new_sequence_lengths = tf.data.Dataset.from_tensor_slices(sequence_lengths).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100,seed=10).as_numpy_iterator()
# print(new_sequence_lengths.next())

# print(x_train)
new_x_train = tf.data.Dataset.from_tensor_slices(x_valid).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100,seed=10).as_numpy_iterator()
# print(new_x_train.next())

# print(y_train)
new_y_train = tf.data.Dataset.from_tensor_slices(y_valid).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100,seed=10).as_numpy_iterator()
# print(new_y_train.next())
total_step = sequence_lengths.shape[0] * repeat_size
repeat_step = int(total_step / batch_size)

checkpoint = tf.train.Checkpoint(
    model=m1 )

model_dir = r'..\model'
latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)

if latest_checkpoint_file:
  print('latest_checkpoint_file', latest_checkpoint_file)
  # logging.info('Checkpoint file %s found and restoring from '
  #              'checkpoint', latest_checkpoint_file)
  checkpoint.read(latest_checkpoint_file)
  # logging.info('Loading from checkpoint file completed')

all_number = 0
right_number = 0
rigth_chars = 0
all_chars = 0
for _ in range(repeat_step):
        x_train_input = tf.cast(new_x_train.next(), dtype=tf.int32)
        real_tag_input = tf.cast(new_y_train.next(), dtype=tf.int32)
        sequence_lengths_input = tf.cast(new_sequence_lengths.next(), dtype=tf.int32)
        masks_input = tf.sequence_mask(
            sequence_lengths_input, maxlen=x_train_input.shape[1], dtype=tf.int32
        )

        final_output,transition_params = m1([x_train_input,masks_input], training=False)
        score = tf.squeeze(final_output,axis=0)
        seq_out = viterbi_decode(score,transition_params)
        pre_output = seq_out[0]
        loss = loss1(final_output,real_tag_input,sequence_lengths_input
                     ,transition_params=transition_params )

        real_input = tf.squeeze(real_tag_input, axis=0)
        seq_lengs = sequence_lengths_input.numpy()[0]
        # print('seq_lengs', seq_lengs)
        # print('111',list(real_input.numpy())[:seq_lengs])
        # print('2222', pre_output[:seq_lengs])
        real_ten = tf.cast(list(real_input.numpy())[:seq_lengs],dtype=tf.int32)
        pre_ten = tf.cast(pre_output[:seq_lengs],dtype=tf.int32)

        # 准确率的计算
        d = tf.where(pre_ten > 0)
        pre_list = list(tf.squeeze(d, 1).numpy())
        pre_dic = []
        pre_index = []
        for index, x in enumerate(pre_list):
            if index == 0:
                pre_dic.append({})
                pre_dic[-1][x] = 1
                pre_index.append(x)
                continue
            if pre_index[-1] + 1 == x:
                for k, v in pre_dic[-1].items():
                    pre_dic[-1][k] = v + 1
                pre_index[-1] = x
            else:
                pre_dic.append({})
                pre_dic[-1][x] = 1
                pre_index.append(x)

        real_data = list(real_input.numpy())[:seq_lengs]
        pre_data = pre_output[:seq_lengs]
        for dics in pre_dic:
            for k, v in dics.items():
                pre_str = pre_data[k:k + v]
                real_str = real_data[k:k + v]
                if pre_str == real_str:
                    rigth_chars += 1
                all_chars += 1

        # 召回率的计算
        e = tf.where(real_ten > 0)
        real_list = list(tf.squeeze(e, 1).numpy())
        rel_dic = []
        rel_index = []
        for index, x in enumerate(real_list):
            if index == 0:
                rel_dic.append({})
                rel_dic[-1][x] = 1
                rel_index.append(x)
                continue
            if rel_index[-1] + 1 == x:
                for k, v in rel_dic[-1].items():
                    rel_dic[-1][k] = v + 1
                rel_index[-1] = x
            else:
                rel_dic.append({})
                rel_dic[-1][x] = 1
                rel_index.append(x)

        real_data = list(real_input.numpy())[:seq_lengs]
        pre_data = pre_output[:seq_lengs]
        for dics in rel_dic:
            for k, v in dics.items():
                pre_str = pre_data[k:k + v]
                real_str = real_data[k:k + v]
                if pre_str == real_str:
                    right_number += 1
                all_number += 1


        # print('loss',loss)
        if all_number % 100 == 0 :
            # print('right_number', right_number)
            # print('all_number', all_number)
            print('召回率', right_number / all_number)
            print('准确度',rigth_chars/all_chars)
            f1 = (2 * (right_number / all_number) * (rigth_chars/all_chars)) / (right_number / all_number + rigth_chars/all_chars)
            print('f1', f1)


print('召回率', right_number / all_number)
print('准确度', rigth_chars / all_chars)
print('f1', f1)