"Various supporting utility functions."

import random
import urllib3

import numpy as np

import params


def download_text(
        address='http://www.umich.edu/~umfandsf/other/ebooks/alice30.txt'):
    """Download test text and store in file test-text.txt.

    Alice in Wonderland is downloaded by default.

    Returns:
        str: Text of Alice in Wonderland.

    """
    http = urllib3.PoolManager()
    response = http.request('GET', address)
    text = response.data.decode('utf-8')
    with open('test-text.txt', 'w') as file:
        file.write(text)
    return text


def retrieve_text(filename):
    """Retrieve the text string from file test-text.txt.

    Args:
        filename (str): Name of file, stored in project directory, to open.

    Returns:
        str: Text from file.

    """
    with open(filename, 'r') as file:
        text = file.read()
    return text


def create_one_hots(text, dictionary):
    """Create list of one hot Numpy arrays from text.

    Args:
        text (str): The text being used.
        dictionary (dict): A dictionary of
            characters to unique character number.

    Returns:
        tuple: tuple[0]: a list of training one-hot numpy arrays representing
            the training text; tuple[1]: the same but for the evaluation text.

    """
    text_one_hots = []
    for char in list(text):
        one_hot = dictionary[char]
        text_one_hots.append(one_hot)

    split = int(params.EVAL_SPLIT * len(text_one_hots))
    train_one_hots = text_one_hots[split:]
    eval_one_hots = text_one_hots[:split]

    return train_one_hots, eval_one_hots


def output_one_hot(chars_size, text_value):
    """Create one-hot numpy array from a character's number.

    Args:
        chars_size (int): Number of set of unique characters in our text.
        text_value (int): Character number.

    Returns:
        numpy.ndarray: One_hot of
            dimension chars_size representing text character.

    """
    onehot = np.zeros([chars_size], dtype=float)
    onehot[text_value] = 1.0
    onehot = np.reshape(onehot, [1, -1])
    return onehot


def build_dataset(chars):
    """Create dict and reverse dict of text chars <=> one_hot / char number.

    Args:
        chars (set): Set of characters (str) in the text.

    Returns:
        tuple: tuple[0] is dict of schema
                    {<char (str)>: <char one_hot (numpy)>...}
               tuple[1] is dict of schema
                    {<char number (int)>: <char (str)>...}

    """
    chars_size = len(chars)
    dictionary = {}
    reverse_dictionary = {}
    chars = sorted(chars)

    for char in chars:
        char_number = len(dictionary)
        one_hot = output_one_hot(chars_size, char_number)
        dictionary[char] = one_hot
        reverse_dictionary[char_number] = char

    return dictionary, reverse_dictionary


def create_training_io(text_one_hots, index, batch, chars_size):
    """Create Numpy input and output arrays for a batch.

    Args:
        text_one_hots (List): List of one hot
            Numpy arrays representing the text's characters.
        index (int): The starting index in text_one_hots
            from which to create the input/output arrays.
        batch (int): Size of a batch.
        chars_size (int): The number of unique characters in the text.

    Returns:
        tuple: Input and output numpy arrays.

    """
    input_x = np.zeros((batch, params.N_INPUT * chars_size), dtype='float32')
    output_y = np.zeros((batch, 1 * chars_size), dtype='float32')

    for item in range(0, batch):

        input_vectors = text_one_hots[index: index + params.N_INPUT]
        input_vectors = np.concatenate(input_vectors, axis=1)
        input_vectors = np.reshape(input_vectors, [-1, input_vectors.shape[1]])

        output_hot = text_one_hots[index + params.N_INPUT]
        output_hot = np.reshape(output_hot, [-1, chars_size])

        input_x[item] = input_vectors[0]
        output_y[item] = output_hot[0]

        index += 1

    return input_x, output_y


def one_hot_to_char(logits, reverse_dictionary):
    """Determine the highest probability character from the RNN logits output.

    Args:
        logits (numpy.ndarray): Logits
        reverse_dictionary (dict): Dict of
            character number (int) to character (str).

    Returns:
        str: Single character string.

    """
    char_no = np.argmax(logits)
    char = reverse_dictionary[char_no]
    return char


def create_example_text(                # pylint: disable=too-many-arguments
        sess, x, logits, chars, dictionary, reverse_dictionary):
    """Iteratively create text based on a random input to the trained RNN.

    Args:
        sess (Session): Tensorflow session.
        x (Tensor): Tensorflow model placeholder input to RNN.
        logits (Operation): Tensorflow logits operation.
        chars (set): Set of characters in the training/evaluation text.
        dictionary (dict):  Dict of character (str)
            to character one_hot numpy representation.
        reverse_dictionary (dict): Dict of
            character number (int) to character (str).

    """
    chars_list = list(chars)
    chars_size = len(chars)
    test_text = [chars_list[random.randint(0, chars_size - 1)]
                 for _ in range(params.N_INPUT)]
    characters = ''
    while len(characters) < params.TEST_TEXT_LENGTH:
        test_input = []
        # Create one hot inputs to RNN to get the next character out.
        for character in test_text:
            one_hot = dictionary[character]
            test_input.append(one_hot)
        test_input = np.concatenate(test_input, axis=1)
        # Get the RNN output logits.
        result = sess.run(logits, feed_dict={x: test_input})
        # Determine which character has the highest logits value.
        character = one_hot_to_char(result, reverse_dictionary)
        # Add character to output text.
        characters += character
        # Prepare the input characters for the next RNN input iteration.
        test_text.pop(0)
        test_text.append(character)
    print('\n', characters, '\n')
