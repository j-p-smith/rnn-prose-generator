"""RNN Model Constants."""


# Name of text file to use
# Shakespeare complete text from
# https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
TEXT_FILE = 'test-shakespeare.txt'
# TEXT_FILE = 'test-text.txt'
# TensorBoard directory
SUMMARIES_DIR = '/tmp/tensorboard/rnn-text'
# Factor that is evaluation set vs training set (0 - 1)
EVAL_SPLIT = 0.01
# Learning rate
LEARNING_RATE = 0.001
# Number of hidden units in RNN cell
N_HIDDEN = 512
# Number of character inputs to RNN
N_INPUT = 70
# Number of epochs to train on, each being one iteration of the text.
EPOCHS = 30
# Batch size
BATCH_SIZE = 128
# Step frequency to show example text output; MUST be multiple of batch size
TRAINING_DISPLAY_STEP = 40960
# Test example text output length
TEST_TEXT_LENGTH = 1200
# Temperature of LSTM
TEMPERATURE = 1
