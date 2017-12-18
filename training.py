#!/usr/bin/env python
"""RNN Text Training."""

import tensorflow as tf

import evaluation
import model
import params
import utils


def main():             # pylint: disable=too-many-locals, too-many-statements
    """Create the RNN model and train it, outputting the text results.

    Periodically:
        (1) the training/evaluation set cost and accuracies are printed, and
        (2) the RNN is given a random input feed to output its own
            self-generated output text for our amusement.

    """
    text = utils.retrieve_text(params.TEXT_FILE)
    chars = set(text)
    chars_size = len(chars)
    dictionary, reverse_dictionary = utils.build_dataset(chars)
    train_one_hots, eval_one_hots = utils.create_one_hots(text, dictionary)

    x = tf.placeholder(tf.float32, [None, params.N_INPUT * chars_size])
    labels = tf.placeholder(tf.float32, [None, chars_size])
    logits = model.inference(x, chars_size)
    cost = model.cost(logits, labels)
    optimizer = model.optimizer(cost)
    accuracy = model.accuracy(logits, labels)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            params.SUMMARIES_DIR + '/train', sess.graph)
        eval_writer = tf.summary.FileWriter(params.SUMMARIES_DIR + '/eval')
        loss_total = 0
        accuracy_total = 0

        for epoch in range(params.EPOCHS):

            for index in range(0,
                               len(train_one_hots) -
                               params.N_INPUT - params.BATCH_SIZE,
                               params.BATCH_SIZE):

                input_x, output_y = utils.create_training_io(
                    train_one_hots, index, params.BATCH_SIZE, chars_size)

                _, acc, loss, summary = sess.run(
                    [optimizer, accuracy, cost, merged],
                    feed_dict={x: input_x, labels: output_y})

                step = epoch * (len(train_one_hots) - params.N_INPUT) + index
                train_writer.add_summary(summary, step)
                loss_total += loss
                accuracy_total += acc

                if index % params.TRAINING_DISPLAY_STEP == 0 and index:
                    print(
                        'Epoch: {}  Training Step: {}\n'
                        'Training Set: Loss: {:.3f}  '
                        'Accuracy: {:.3f}'.format(
                            epoch,
                            index,
                            loss_total *
                            params.BATCH_SIZE / params.TRAINING_DISPLAY_STEP,
                            accuracy_total *
                            params.BATCH_SIZE / params.TRAINING_DISPLAY_STEP,
                        )
                    )
                    loss_total = accuracy_total = 0
                    evaluation.evaluation(sess, step, eval_one_hots,
                                          x, labels, accuracy, cost,
                                          eval_writer, chars_size, merged)
                    utils.create_example_text(sess, x, logits, chars,
                                              dictionary, reverse_dictionary)


if __name__ == '__main__':
    if tf.gfile.Exists(params.SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(params.SUMMARIES_DIR)
    main()
