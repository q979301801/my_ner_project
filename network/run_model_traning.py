import datetime
import pickle

import tensorflow as tf
from absl import logging
import math
from model import model_help
from tensorflow.keras.layers import LSTM,Bidirectional,Dense,Conv2D,BatchNormalization,Activation,AvgPool2D,Dropout
import numpy as np
from tensorflow_addons.text import crf_log_likelihood,viterbi_decode
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

        # last_encoder_output = encoder_outputs[-1]
        bid_lstm_output = encoder_outputs[-1]

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


num_tags = 10
m1 = my_crf_model(vocab_size = 5000,embedding_width = 128,lstm_hiden_sieze = 64,num_tags = num_tags,
                  max_sequence_length = 50,num_layers = 1,num_attention_heads = 2,
                  inner_dim = 512)


# optimizer = tf.optimizers.SGD(learning_rate=0.0001)
# optimizer = tf.optimizers.Adam(learning_rate=0.005)
optimizer = tf.optimizers.SGD(learning_rate=0.002,momentum=0.9,nesterov= True)
# optimizer = tf.optimizers.Adadelta(learning_rate=0.1)
m1.optimizer = optimizer



f1=open(r'..\data\y_train.pkl','rb')
f=open(r'..\data\x_train.pkl','rb')
data = pickle.load(f)
x_train = tf.cast(data,dtype=tf.int32)

data_y = pickle.load(f1)
y_train = tf.cast(data_y,dtype=tf.int32)

seq_len = tf.where(x_train > 0,1,0)
sequence_lengths = tf.reduce_sum(seq_len,axis=-1)

repeat_size = 4
batch_size = 32
# print(sequence_lengths)
new_sequence_lengths = tf.data.Dataset.from_tensor_slices(sequence_lengths).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100000,seed=200).as_numpy_iterator()
# print(new_sequence_lengths.next())

# print(x_train)
new_x_train = tf.data.Dataset.from_tensor_slices(x_train).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100000,seed=200).as_numpy_iterator()
# print(new_x_train.next())

# print(y_train)
new_y_train = tf.data.Dataset.from_tensor_slices(y_train).repeat(repeat_size).batch(batch_size).shuffle\
    (buffer_size=100000,seed=200).as_numpy_iterator()
# print(new_y_train.next())
total_step = sequence_lengths.shape[0] * repeat_size
repeat_step = int(total_step / batch_size)



#
checkpoint = tf.train.Checkpoint(
    model=m1,optimizer=optimizer, global_step=optimizer.iterations)

model_dir = r'..\model'
latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = model_dir + '\\logs\\' + current_time + '\\train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

if latest_checkpoint_file:
  print('latest_checkpoint_file', latest_checkpoint_file)
  # logging.info('Checkpoint file %s found and restoring from '
  #              'checkpoint', latest_checkpoint_file)
  checkpoint.read(latest_checkpoint_file)
  # logging.info('Loading from checkpoint file completed')


# optimizer = tf.optimizers.Adam(learning_rate=0.001,beta_2=0.999)
# optimizer = tf.optimizers.SGD(learning_rate=0.005)
# optimizer = tf.optimizers.Adadelta(learning_rate=0.001)
optimizer = tf.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
m1.optimizer = optimizer

# print('m1.optimizer.get_confi',m1.optimizer.get_config()['learning_rate'])
training_vars = m1.trainable_variables
for i in range(repeat_step):
    with tf.GradientTape() as tape:
        x_train_input = tf.cast(new_x_train.next(), dtype=tf.int32)
        real_tag_input = tf.cast(new_y_train.next(), dtype=tf.int32)
        sequence_lengths_input = tf.cast(new_sequence_lengths.next(), dtype=tf.int32)
        masks_input = tf.sequence_mask(
            sequence_lengths_input, maxlen=x_train_input.shape[1], dtype=tf.int32
        )

        final_output,transition_params = m1([x_train_input,masks_input], training=True)
        loss = loss1(final_output,real_tag_input,sequence_lengths_input
                     ,transition_params=transition_params )

        # print('transition_params',transition_params)
        # training_vars.append(transition_params)
        grads = tape.gradient(loss, training_vars)
        optimizer.apply_gradients(zip(grads, training_vars))
        # print('transition_params2222', transition_params)

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step= i + 1)
        #     # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        #     tf.summary.scalar('learn_rate', optimizer.get_config()['learning_rate'], step=i + 1)
        #     for index, variables in enumerate(training_vars):
        #         tf.summary.histogram('variables_{}'.format(index), variables, step=i + 1)
        #     for index, grads in enumerate(grads) :
        #       if grads is not None:
        #        tf.summary.histogram('grads_{}'.format(index), grads, step=i + 1)

        print('loss',loss)

checkpoint = tf.train.Checkpoint(
    model=m1,optimizer=optimizer, global_step=optimizer.iterations)

model_dir = r'..\model'
latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
if latest_checkpoint_file:
  print('latest_checkpoint_file', latest_checkpoint_file)
  # logging.info('Checkpoint file %s found and restoring from '
  #              'checkpoint', latest_checkpoint_file)
  checkpoint.restore(latest_checkpoint_file)
  # logging.info('Loading from checkpoint file completed')
else:
    print('save', latest_checkpoint_file)
    checkpoint.save(model_dir + r'\new_model{}.ckpt'.format(current_time))


# inupts1 = tf.cast(np.random.randint(1,1000,(64,20)),dtype=tf.int32)
# real_tag = tf.cast(np.random.randint(0,10,(64,20)),dtype=tf.int32)
# sequence_lengths = tf.cast(np.random.randint(10,15,(64,)),dtype=tf.int32)

def my_loss(y_true,y_pred):
    final_output = y_pred
    # transition_params = y_pred[1]

    real_tag = y_true[:,:-1]
    sequence_lengths = y_true[:,-1]

    # initializer = tf.keras.initializers.GlorotUniform()
    # transition_params = initializer([10, 10])
    transition_params = tf.fill([10,10],0.5)
    print('final_output',final_output)
    print('transition_params', transition_params)
    # print('sequence_lengths', sequence_lengths)
    log_likelihood, transition_params = crf_log_likelihood(inputs=final_output,
                                                       tag_indices=real_tag,
                                                       sequence_lengths=sequence_lengths,
                                                        transition_params= transition_params)
    return tf.reduce_mean(log_likelihood)


#
# for i in range(100):
#     with tf.GradientTape() as tape:
#         model_outputs,transition_params = m1(inupts1, training=True)
#         loss = loss1(model_outputs,real_tag,sequence_lengths,transition_params=transition_params )
#         # print('transition_params',transition_params)
#         # training_vars.append(transition_params)
#         grads = tape.gradient(loss, training_vars)
#         optimizer.apply_gradients(zip(grads, training_vars))
#         # print('transition_params2222', transition_params)
#         print('loss',loss)
#
# m1.summary()
# m1.compile(optimizer='sgd',loss=my_loss)
# m1.fit(x = inupts1,y = y_true,batch_size = 32,epochs = 10)