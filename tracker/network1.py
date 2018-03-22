import tensorflow

msra_initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.zeros_initializer()
prelu_initializer = tf.constant_initializer(0.25)

def inference(inputs, num_unrolls, train, batch_size=None, prevLstmState=None, reuse=None):
    # Data should be in order BxTx2xHxWxC where T is the number of unrolls
    # Mean subtraction
    if batch_size is None:
        batch_size = int(inputs.get_shape().as_list()[0] / (num_unrolls * 2))

    variable_list = []

    if reuse is not None and not reuse:
        reuse = None

    with tf.variable_scope('re3', reuse=reuse):
        conv_layers = alexnet_conv_layers(inputs, batch_size, num_unrolls)

        # Embed Fully Connected Layer
        with tf.variable_scope('fc6'):
            fc6_out = tf_util.fc_layer(conv_layers, 2048)

            # (BxT)xC
            fc6_reshape = tf.reshape(fc6_out, tf.stack([batch_size, num_unrolls, fc6_out.get_shape().as_list()[-1]]))

        # LSTM stuff
        swap_memory = num_unrolls > 1
        with tf.variable_scope('lstm1'):
            lstm1 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            if prevLstmState is not None:
                state1 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[0], prevLstmState[1])
            else:
                state1 = lstm1.zero_state(batch_size, dtype=tf.float32)
            lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, fc6_reshape, initial_state=state1, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm1' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])

        with tf.variable_scope('lstm2'):
            lstm2 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            state2 = lstm2.zero_state(batch_size, dtype=tf.float32)
            if prevLstmState is not None:
                state2 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[2], prevLstmState[3])
            else:
                state2 = lstm2.zero_state(batch_size, dtype=tf.float32)
            lstm2_inputs = tf.concat([fc6_reshape, lstm1_outputs], 2)
            lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm2' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])
            # (BxT)xC
            outputs_reshape = tf_util.remove_axis(lstm2_outputs, 1)

        # Final FC layer.
        with tf.variable_scope('fc_output'):
            fc_output_out = tf_util.fc_layer(outputs_reshape, 4, activation=None)

    if prevLstmState is not None:
        return fc_output_out, state1, state2
    else:
        return fc_output_out
