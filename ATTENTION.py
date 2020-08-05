import tensorflow as tf


def CoAttention(gru1, gru2, rnn_dim, batch_size, scope_name):  # gru1(bs,2u,m), gru2(bs,2u,n), m=n=sequence_length
    with tf.variable_scope(scope_name):
        M = tf.get_variable("M", shape=(2 * rnn_dim, 2 * rnn_dim),
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        G_temp = tf.matmul(tf.transpose(gru1, (0, 2, 1)), tf.tile(tf.expand_dims(M, 0), (batch_size, 1, 1)))
        G = tf.tanh(tf.matmul(G_temp, gru2))  # shape=(m,n)

        a_fw = tf.nn.softmax(tf.reduce_max(G, axis=2))  # row max,shape=(bs,m)
        a_bw = tf.nn.softmax(tf.reduce_max(G, axis=1))  # col max,shape=(bs,n)
        r_fw1 = tf.matmul(gru1, tf.expand_dims(a_fw, axis=2))  # shape=(bs,2u,1)
        r_bw2 = tf.matmul(gru2, tf.expand_dims(a_bw, axis=2))
        return a_fw, a_bw, tf.squeeze(r_fw1, (2,)), tf.squeeze(r_bw2, (2,))  # a shape=(bs,m),r shape=(bs,2u)


def SelfAttention(Hi, rnn_dim, k):  # Hi shape=(-1,2u,ni)
    block_size = tf.shape(Hi)[0]
    M_s = tf.get_variable("M_s", [k, 2 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    p = tf.get_variable("p", [1, k], initializer=tf.truncated_normal_initializer(stddev=0.1))
    M_s_expand = tf.tile(tf.expand_dims(M_s, 0), (block_size, 1, 1))
    p_expand = tf.tile(tf.expand_dims(p, 0), (block_size, 1, 1))
    # formula (5)
    pre_ai = tf.matmul(p_expand, tf.tanh(tf.matmul(M_s_expand, Hi)))  # shape=(block_size,1,ni)
    ai = tf.nn.softmax(tf.squeeze(pre_ai, [1]))  # shape=(block_size,ni)
    return ai
