import tensorflow as tf


def BiGRU(x, hidden_size, x_length, batch_size, name):  # s shape=(batch_size,m,word_dim)
    with tf.variable_scope(name):
        fw_cell = tf.contrib.rnn.GRUCell(hidden_size)
        bw_cell = tf.contrib.rnn.GRUCell(hidden_size)
        initial_state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
        initial_state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, x, x_length, initial_state_fw, initial_state_bw)
        out = tf.concat(outputs, axis=2)  # concatenate forward and backward
    return tf.transpose(out, (0, 2, 1))  # shape(bs,2u,m)
