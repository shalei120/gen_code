from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import cPickle,os,sys
import numpy as np
import tensorflow as tf
import preprocessing

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '/home/shalei/gen_data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '/home/shalei/gen_data/',
                    "Model output directory.")

FLAGS = flags.FLAGS


def data_type():
    return  tf.float32


def load_data():
    datafile = open(FLAGS.data_path + 'gen_'+sys.argv[1]+'_data.dat','rb')
    train_x = cPickle.load(datafile)
    train_y= cPickle.load(datafile)
    dev_x = cPickle.load(datafile)
    dev_y= cPickle.load(datafile)
    test_x = cPickle.load(datafile)
    test_y = cPickle.load(datafile)
    return train_x, train_y, dev_x, dev_y, test_x, test_y

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class S2TInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.item_num = 10
        self.input_table_title = tf.placeholder('int32', [None, self.item_num])
        self.input_table_items = tf.placeholder('int32', [None, self.item_num, None])
        self.input_item_length = tf.placeholder('int32', [None, self.item_num])
        self.targets = tf.placeholder('int32', [None, None])


class Structure2TextModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        #_input: table
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.Variable(index2vec)
            table_title = tf.nn.embedding_lookup(embedding, input_.input_table_title)
            table_items = tf.nn.embedding_lookup(embedding, input_.input_table_items)
            targets = tf.nn.embedding_lookup(embedding, input_.targets)

        batch_size = tf.shape(table_title)[0]
        slot_num = tf.shape(table_title)[1]
        max_content_length = tf.shape(table_title)[2]
        embedding_size = sys.argv[1]


        # embedding the content of the table
        with tf.variable_scope("RNN1"):
            table_items_reshaped = tf.reshape(table_items, [batch_size*slot_num, max_content_length, embedding_size])
            table_item_length_reshaped = tf.reshape(table_items, [batch_size*slot_num])
            table_encode_cell = lstm_cell()
            table_item_vector_reshaped = tf.nn.dynamic_rnn(table_encode_cell, table_items_reshaped, sequence_length = table_item_length_reshaped)
            cellout_size = tf.shape(table_item_vector_reshaped)[2]
            table_item_vector = tf.reshape(table_item_vector_reshaped, [batch_size, slot_num, cellout_size])

        #if is_training and config.keep_prob < 1:
       #     inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        cell_output = tf.Variable(tf.truncated_normal([batch_size, config.hidden_size*config.num_layers]))
        W = weight_variable([embedding_size, embedding_size])
        W_out2emb = weight_variable([config.hidden_size*config.num_layers, embedding_size])
        b_out2emb = weight_variable([ embedding_size])
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                #cell_output_tile = tf.tile(cell_output, [1,input_.item_num])
               # cell_out_1 = tf.reshape(cell_output_tile,[batch_size, input_.item_num, -1])
               # cell_out_1 * table_title
                attention = tf.matmul(tf.expand_dims(tf.matmul(cell_output, W), 1), tf.transpose(table_title, prem = [0,2,1])) [:, 0,:]  # the weight of table items


                if time_step == 0:
                    attention = tf.Variable(np.asarray([[0.1] * input_.item_num] * batch_size))

                attention = tf.nn.softmax(attention)

                tableinput = tf.matmul(tf.expand_dims(attention, 1) , table_item_vector)[:,0,:]   # batch 1 embed_size
                nextinput = tf.concat([cell_output, tableinput], 1)
                (cell_output, state) = cell(nextinput, state)
                out_wordemb = tf.matmul(cell_output, W_out2emb) + b_out2emb   # batch * embsize
                vocab_score = tf.matmul(out_wordemb, tf.transpose(embedding))   # batch * vocabsize
                chosen_word_indexes = tf.argmax(vocab_score)   # batch
                chosen_words = tf.nn.embedding_lookup(embedding, chosen_word_indexes)  # batch * embsize
                cell_output = chosen_words

                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = len(index2word)


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = len(index2word)


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = len(index2word)


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = len(index2word)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = load_data()
    train_x, train_y, valid_x, valid_y, test_x, test_y = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = S2TInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Structure2TextModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = S2TInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Structure2TextModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = S2TInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Structure2TextModel(is_training=False, config=eval_config,
                                 input_=test_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
