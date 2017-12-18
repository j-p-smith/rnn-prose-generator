"""Calculate predictions performance on evaluation text."""

import params
import utils


def evaluation(sess,  # pylint: disable=too-many-arguments, too-many-locals
               step, text_eval_one_hot, x, labels,
               accuracy, cost, eval_writer, chars_size, merged):
    """Evaluate and report RNN performance on evaluation set."""
    eval_batch_size = len(text_eval_one_hot) - params.N_INPUT
    input_x, output_y = utils.create_training_io(
        text_eval_one_hot, 0, eval_batch_size, chars_size)

    acc, loss, summary = sess.run(
        [accuracy, cost, merged], feed_dict={x: input_x, labels: output_y})

    print('Evaluation Set ({} chars): '
          'Loss: {:.3f}  Accuracy: {:.3f}'.format(eval_batch_size, loss, acc))

    eval_writer.add_summary(summary, step)
