import tensorflow as tf

from ATTENTION import SelfAttention, CoAttention
from BIGRU import BiGRU


def R_net(RU_emb, RI_emb, rnn_dim, batch_size, sequence_length, name):
    with tf.variable_scope(name):
        with tf.variable_scope("BiGRU"):
            seq_length_batch = tf.fill([batch_size], sequence_length)
            HU = BiGRU(RU_emb, rnn_dim, seq_length_batch, batch_size, "RU_GRU")
            HI = BiGRU(RI_emb, rnn_dim, seq_length_batch, batch_size, "RI_GRU")  # shape(bs,2u,m)
        with tf.variable_scope("CoAttention"):
            aUI_fw, aUI_bw, rUI_fw, rUI_bw = CoAttention(HI, HU, rnn_dim, batch_size, "UI")
    return HU, HI, aUI_fw, aUI_bw, rUI_fw, rUI_bw  # H shape(bs,2u,m), a shape=(bs,m), r shape=(bs,2u)


def S_net(H, aUI, rnn_dim, batch_size, sent_length, k, name):
    with tf.variable_scope(name):
        # H_split shape=(block_size,2u,sent_length),  block_size = tf.shape(H_split)[0]
        H_split = tf.transpose(tf.reshape(tf.transpose(H, (0, 2, 1)), (-1, sent_length, 2 * rnn_dim)), (0, 2, 1))
        ai = SelfAttention(H_split, rnn_dim, k)  # shape=(-1,sent_length)
        si_temp = tf.matmul(H_split, tf.expand_dims(ai, axis=2))  # shape=(block_size,2u,1)
        si = tf.squeeze(si_temp, axis=[2])  # shape=(block_size,2u)

        aUIi_temp = tf.reduce_sum(tf.reshape(aUI, (-1, sent_length)), axis=1, keep_dims=True)  # shape=(block_size,1)
        aUIi = tf.tile(aUIi_temp, (1, 2 * rnn_dim))
        S_temp = tf.multiply(aUIi, si)  # formula (6), shape=(block_size,2u)
        S = tf.reduce_sum(tf.reshape(S_temp, (batch_size, -1, 2 * rnn_dim)), axis=1)  # shape=(batch_size,2u)
    return S
