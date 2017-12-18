"""Create the Tensorflow RNN Model."""

import tensorflow as tf

import params


def inference(input_x, chars_size):
    """Create Tensorflow RNN Model.

    Args:
        input_x (Tensor): input tensor.
        chars_size (int): The number of unique characters in the text.

    Returns:
        Tensor: logits.

    """
    # RNN output node weights and biases
    weights = tf.Variable(
        tf.random_normal([params.N_HIDDEN, chars_size], name='Weight'))
    biases = tf.Variable(tf.random_normal([chars_size], name='Bias'))

    # Split batch of tensor x into sub tensors along axis 1
    x_in = tf.split(
        input_x,
        num_or_size_splits=params.N_INPUT, axis=1, name='Split_IP_Into_Chars')

    # Single layer LSTM RNN - option to uncomment and replace multilayer LSTM
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(params.N_HIDDEN)
    # Multilayer LSTM RNN with dropout
    rnn_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(params.N_HIDDEN),
            output_keep_prob=0.5
        )
        for _ in range(3)
    ])

    y_out, _ = tf.nn.static_rnn(rnn_cell, x_in, dtype=tf.float32)

    logits = tf.add(
        tf.matmul(y_out[-1], weights), biases, name='Logits')

    # Divide logits by Temperature value between 0 and 1
    # to reduce diversity and number of mistakes
    logits = tf.div(logits, params.TEMPERATURE, name='Temp_Adjusted_Logits')

    return logits


def cost(logits, labels):
    """Calculate cross entropy cost/loss.

    Args:
        logits (Tensor): logits.
        labels (Tensor): Truth one hot tensors corresponding to the input.

    Returns:
        Tensor: cross entropy cost/loss.

    """
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='Batch_Cross_Entropies'),
        name='Mean_Cost_Entropy')

    tf.summary.scalar('cross_entropy', loss)

    return loss


def optimizer(loss):
    """Create the TensorFlow optimizer.

    Args:
        loss (Tensor): The cross entropy cost.

    Returns:
        Operation: Model optimizer.

    """
    opt = tf.train.AdamOptimizer(    #Can also use RMSPropOptimizer here.
        learning_rate=params.LEARNING_RATE).minimize(loss)

    return opt


def accuracy(logits, labels):
    """Calculate accuracy.

    Args:
        logits (Tensor): logits.
        labels (Tensor): Truth one hot tensors corresponding to the input.

    Returns:
        Tensor: Accuracy.

    """
    correct_pred = tf.equal(
        tf.argmax(logits, 1, name='Prediction'),
        tf.argmax(labels, 1, name='Truth'), name='Correct_Predicts')

    acc = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32, name='Cast_Correct_Predicts'),
        name='Accuracy')

    tf.summary.scalar('accuracy', acc)

    return acc
