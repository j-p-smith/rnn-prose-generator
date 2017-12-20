RNN Prose Generator
===================

This TensorFlow training project was for a bit of fun to play with a Tensorflow
RNN. It is an implementation of Andrej Karpathy's text generator idea. See
[here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for his blog
where he describes it.

The RNN is trained from any text - say, the complete works of Shakespeare. The
RNN is trained by feeding in a number of sequential characters at a time from
the text - say 70 characters. The RNN is trained to predict the next character
as its single output. Training is repeated by cycling through multiple epochs
of the entire text.

As it trains, a random set of 70 characters are fed into the RNN for the
next character to be predicted. This character can be added to the end of the
sequence of 70 input characters and the first character dropped. This is then
used as the input to find the next output character, and so on. The output
characters make a very interesting bit of artificially generated prose!

For example, after a day's training on a GTX 1080 Ti GPU, we get automated
pseudo Shakespeare being spat out each display cycle, such as:

```
SCENE II.
Athena. Antony. The DUKE'S palace

Enter a MENENIUS and CORIOLANUS and COMINIUS and CORIOLANUS

                           ENOBARDO 
                                                                 Exit 
                                                              Exit 
  COMINIUS. What shall you see the court of this thing?
    The more of the court of the counters,
    And the consulting of the country of the state,
    And the which they are so for the poor seas
    That what the state was so to be a man
    That hath a stranger to the world. I have seen
    The suit of the street of the seasons of the state
    That have the common strencth of the son of the streams
    That hath been some of thee.
  CORIOLANUS. Why then, the court of this in this way
    We have a man and some of the common private
    That he hath spoke to him.
  SICINIUS. What shall I see you to the sea?
  CORIOLANUS. I have the good thing that I have sent them
    That he hath seen the state.
  COUNIEL. I will be so.
```

In fact, after only just a few minutes of training it already starts to output
interesting text, but a few hours of training certainly makes it richer.

Notice how it has learnt Shakespearean play writing syntax, introducing the
scene, giving a location, who enters, then alternating who speaks. Quite an 
example of the characteristics of LSTM!

Similar effects can be seen on other texts. Have a look at the Kaparthy blog 
above to see some examples.

Operation
---------

To run this RNN prose generator:
1. Create the desired text file, placing it in project's root directory and
naming the file in `params.py` (see there for ref. to the location of the full 
works of Shakespeare. A text with several million characters is desirable. 
1. Install the project dependencies from `requirements.txt`.
1. Run the RNN with `python training.py`. Every number of steps (set in 
`params.py`), the accuracy and loss of the training and evaluation set are 
displayed, along with an example of prose text generation from the RNN. As the 
model trains, the prose will become coherent in the style of the input text.
1.  Run TensorBoard to see the model Graph and the loss and accuracy progress:
`tensorboard --logdir='/tmp/tensorboard/rnn-text/train' --port=6006`
1.  Run another instance of TensorBoard to see the evaluation set loss and 
accuracy progress: 
`tensorboard --logdir='/tmp/tensorboard/rnn-text/train' --port=6007`
1. Leave it to run for a few hours and enjoy watching the model learn to create
it's own 'twaddle-speak' output!
1. See the `params.py` file for adjusting the opearting parameters of the RNN,
if you wish to experiment.

Directory `sample_output` houses an example of terminal text output after
around 13 epochs / 80 million training steps after approximately 1.5 days 
training on a 4-core i5 CPU and GTX 1080 Ti GPU, although readable improvements
after a few of hours of training become marginal.
