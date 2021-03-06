from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import cPickle,os,sys
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PythonROUGE import PythonROUGE

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
    datafile = open(FLAGS.data_path + 'gen_'+sys.argv[1]+'_data1.dat','rb')

    index2vec = cPickle.load(datafile)
    index2word = cPickle.load(datafile)
    word2index = cPickle.load(datafile)
    print(index2word[10030])
    #print(len(index2vec))
    title_set = cPickle.load(datafile)

    train_data = cPickle.load(datafile)
    dev_data = cPickle.load(datafile)
    test_data = cPickle.load(datafile)
    return train_data, dev_data, test_data, index2vec,index2word, word2index,title_set

train_data, dev_data, test_data, index2vec,index2word, word2index, title_set = load_data()


def prepare_data(data, start, end):
    data0 = data[0][start:end]
    data1 = data[1][start:end]
    data2 = data[2][start:end]
    data3 = data[3][start:end]
    data4 = data[4][start:end]
    batch_size = len(data0)
    conf = get_config()
    res = dict()

    title_nums = [len(t) for t in data0]
    title_word_num = [[len(title)  for title in tlist ]for tlist in data0]
    max_title_word_num = max([max(l) for l in title_word_num])
    #max_title_num = max(title_nums)
    tmp_input_table_title = np.zeros([batch_size, conf.item_num, max_title_word_num], dtype='int32')
    tmp_input_table_title_mask = np.zeros([batch_size, conf.item_num, max_title_word_num], dtype='int32')
    for ii in range(batch_size):
        if title_nums[ii] > conf.item_num:
            data0[ii] = data0[ii][:conf.item_num]
        for j in range(min(conf.item_num, title_nums[ii])):
            arr = np.asarray(data0[ii][j], dtype='int32')
            tmp_input_table_title[ii, j,:title_word_num[ii][j]] = arr
            tmp_input_table_title_mask[ii, j,:title_word_num[ii][j]] = 1

        # print (title_nums[ii], arr.shape)
    tmp_title_word_num = np.zeros([batch_size, conf.item_num],dtype = 'int32')
    for ii in  range(batch_size):
        if title_nums[ii] > conf.item_num:
            title_word_num[ii]= title_word_num[ii][:conf.item_num]
        arr = np.asarray(title_word_num[ii], dtype = 'int32')
        tmp_title_word_num[ii, : title_nums[ii]] = arr

    each_len = data2 # batch itemnum
    #max_item_len = max([max(l) for l in each_len])
    tmp_table_items = np.zeros([batch_size, conf.item_num, conf.content_len], dtype='int32')
    tmp_table_items_mask = np.zeros([batch_size, conf.item_num, conf.content_len], dtype='int32')

    for ii in range(batch_size):
        for j in range(min(title_nums[ii], conf.item_num)):
            tmp_table_items[ii, j, :min(each_len[ii][j], conf.content_len)] = np.asarray(
                data1[ii][j], dtype='int32')[:conf.content_len]
            tmp_table_items_mask[ii, j, :min(each_len[ii][j], conf.content_len)] = 1

    # print(tmp_table_items.shape)

    tmp_input_item_length = np.zeros([batch_size,conf.item_num], dtype='int32')
    for ii in range(batch_size):
        if title_nums[ii] > conf.item_num:
            data2[ii] = data2[ii][:conf.item_num]
        arr = np.asarray(data2[ii], dtype='int32')
        tmp_input_item_length[ii, :title_nums[ii]] = arr

    each_target_len = data4
    max_target_len = max(each_target_len)
    tmp_targets = np.zeros([batch_size, max_target_len+1], dtype='int32')
    tmp_targets_mask = np.zeros([batch_size, max_target_len+1], dtype='int32')

    for ii in range(batch_size):
        tmp_targets[ii, 0] = word2index['START_TOKEN']
        tmp_targets[ii, 1:(each_target_len[ii]+1)] = np.asarray(data3[ii], dtype='int32')
        tmp_targets_mask[ii, :(each_target_len[ii] + 1)] = 1

    res['input_table_title'] = tmp_input_table_title
    res['input_table_items'] = tmp_table_items
    res['input_item_length'] = tmp_input_item_length
    res['targets'] = tmp_targets
    res['target_numsteps'] = np.asarray(data4, dtype='int32')
    res['input_table_title_word_num'] = tmp_title_word_num
    res['max_target_len'] = max_target_len
    res['input_table_title_mask'] = tmp_input_table_title_mask
    res['input_table_items_mask'] = tmp_table_items_mask
    res['targets_mask'] = tmp_targets_mask
    return res

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
        self.hidden_size = config.hidden_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = len(data[0]) // batch_size -1
        self.item_num = config.item_num
        self.input_table_title = tf.placeholder('int32', [None, self.item_num, None])
        self.input_table_title_mask = tf.placeholder('int32', [None, self.item_num, None])
        self.input_table_title_word_num =  tf.placeholder('int32', [None, self.item_num])

        self.input_table_items = tf.placeholder('int32', [None, self.item_num, None])
        self.input_table_items_mask = tf.placeholder('int32', [None, self.item_num, None])
        self.input_item_length = tf.placeholder('int32', [None, self.item_num])
        #for i in xrange(buckets[-1][1] + 1):
        #    self.targets.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        self.targets = tf.placeholder('int32', [None, None])
        self.targets_mask = tf.placeholder('int32', [None, None])  # need one more in dim 1
        self.target_numsteps = tf.placeholder('int32',[None])
        self.embedding_size = int(sys.argv[1])

        self.initial_cell = tf.placeholder('float32', [None, config.hidden_size ])

        self.content_len = config.content_len




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



        with tf.device("/cpu:0"):
            embedding = tf.Variable(index2vec)
            table_title =  tf.nn.embedding_lookup(embedding, input_.input_table_title) * tf.cast(tf.expand_dims(input_.input_table_title_mask, 3), tf.float32)
            table_title = tf.reduce_sum(table_title, axis = 2)
            table_title = table_title / (tf.cast(tf.expand_dims(input_.input_table_title_word_num,2), tf.float32) + 0.0001)


            table_items =  tf.nn.embedding_lookup(embedding, input_.input_table_items) * tf.cast(tf.expand_dims(input_.input_table_items_mask, 3), tf.float32)
            table_item_lengths = input_.input_item_length
            #sta = tf.Variable(tf.zeros([batch_size,], tf.int32) + word2index['START_TOKEN'], trainable = False)
            targets =  tf.nn.embedding_lookup(embedding, input_.targets) * tf.cast(tf.expand_dims(input_.targets_mask, 2), tf.float32)
            # add start_op

        batch_size = tf.shape(table_title)[0]
        slot_num = tf.shape(table_title)[1]
        max_content_length = tf.shape(table_items)[2]
        embedding_size = int(sys.argv[1])
        num_steps = tf.shape(targets)[1] - 1
        target_real_num_steps = input_.target_numsteps

        self._initial_state = cell.zero_state(batch_size, data_type())
        # embedding the content of the table
        with tf.variable_scope("RNN1"):
            table_items_reshaped = tf.reshape(table_items, [batch_size*slot_num, max_content_length, embedding_size])
            table_item_length_reshaped = tf.reshape(table_item_lengths, [batch_size*slot_num])
            table_encode_cell = lstm_cell()
            table_item_vector_reshaped, final_state = tf.nn.dynamic_rnn(table_encode_cell, table_items_reshaped, sequence_length = table_item_length_reshaped, dtype = data_type())
            table_item_vector = tf.reshape(table_item_vector_reshaped[:,-1,:], [batch_size, slot_num, config.hidden_size])

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
        cell_output = input_.initial_cell#tf.Variable(tf.truncated_normal([batch_size, config.hidden_size*config.num_layers]))
        W = weight_variable([embedding_size, embedding_size])

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        state = self._initial_state

        output = self.decoder_last(state, [cell, W, input_, batch_size, table_title,table_item_vector, embedding, softmax_w, softmax_b,num_steps, targets, embedding_size, target_real_num_steps])
        pred_output = self.decoder_gen(state, [cell, W, input_, batch_size, table_title,table_item_vector, embedding, softmax_w, softmax_b, embedding_size, target_real_num_steps])

        logits = tf.reshape(output, [batch_size * num_steps, vocab_size])
        pred_logits = tf.reshape(pred_output, [batch_size * num_steps, vocab_size])

        self.sup_tokens = tf.reshape(tf.argmax(logits, axis = 1), [batch_size, num_steps])
        self.pred_tokens = tf.reshape(tf.argmax(pred_logits, axis = 1), [batch_size, num_steps])

        self.tt = [targets]
        #vocab_score = tf.matmul(out_wordemb, tf.transpose(embedding))  # batch * vocabsize
        #chosen_word_indexes = tf.argmax(vocab_score, axis=1)  # batch
        #chosen_words = tf.nn.embedding_lookup(embedding, chosen_word_indexes)  # batch * embsize
        #out_wordemb = tf.matmul(cell_output_nt, W_out2emb) + b_out2emb   # batch * embsize


        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [pred_logits],
            [tf.reshape(input_.targets[:,1:], [-1])],
            [tf.cast(tf.reshape(input_.targets_mask[:,1:], [batch_size * num_steps]), tf.float32)])

        self._cost = cost = tf.reduce_sum(loss)/ tf.cast(batch_size, tf.float32)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.001, trainable=False)
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

    def decoder_last(self,state, params):
        cell, W, input_, batch_size, table_title,table_item_vector, embedding, softmax_w, softmax_b, num_steps, targets, embedding_size, target_real_num_steps = params

        with tf.variable_scope("RNN"):
            # for time_step in range(input_.fuck):
            emit_vs = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
            time = tf.constant(0, dtype=tf.int32)
            f0 = tf.zeros([batch_size], dtype=tf.bool)
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=num_steps+1)
            inputs_ta = inputs_ta.unstack(tf.transpose(targets, [1, 0,2]))


            def loop_fn(t, state, emit_vs, finished):

                #cell_output_tile = tf.tile(cell_output, [1,input_.item_num])
               # cell_out_1 = tf.reshape(cell_output_tile,[batch_size, input_.item_num, -1])
               # cell_out_1 * table_title
                x_nt = tf.cond(tf.reduce_all(finished),
                               lambda: tf.zeros([batch_size, embedding_size], dtype=tf.float32),
                               lambda: inputs_ta.read(t))

                attention = tf.cond(tf.equal(t, 0),
                            lambda: tf.ones([batch_size,input_.item_num], dtype=tf.float32),
                            lambda: tf.matmul(tf.expand_dims(tf.matmul(x_nt, W), 1), tf.transpose(table_title, perm = [0,2,1])) [:, 0,:])  # the weight of table items

                finished = tf.greater_equal(t + 1, target_real_num_steps)

                attention = tf.nn.softmax(attention)

                title_content = tf.concat([table_title, table_item_vector], axis = 2 )

                tableinput = tf.matmul(tf.expand_dims(attention, 1) , title_content)[:,0,:]   # batch 1 embed_size


                nextinput = tf.concat([x_nt, tableinput], 1)
                (cell_output_nt, state_nt) = cell(nextinput, state)
                out_word_quasi_prob = tf.matmul(cell_output_nt, softmax_w) + softmax_b
                emit_vs = emit_vs.write(t,out_word_quasi_prob)
               # outputs.append(vocab_score)
                return t+1, state_nt, emit_vs, finished

            _, _, emit_vs,_ = tf.while_loop(
               cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
               body=loop_fn,
               loop_vars=(time, state, emit_vs, f0))

            output = tf.transpose(emit_vs.stack(), [1, 0, 2])
            return output


    def decoder_gen(self,state, params):
        cell, W, input_, batch_size,table_title ,table_item_vector, embedding,softmax_w, softmax_b, embedding_size, target_real_num_steps = params
        with tf.variable_scope("RNNGen"):
            # for time_step in range(input_.fuck):
            emit_vs = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
            time = tf.constant(0, dtype=tf.int32)
            f0 = tf.zeros([batch_size], dtype=tf.bool)
            cell_output = tf.tile(tf.nn.embedding_lookup(embedding, word2index['START_TOKEN']), tf.expand_dims(batch_size,0))
            cell_output = tf.reshape(cell_output, [batch_size, embedding_size])

            def loop_fn(t, cell_output, state, emit_vs, finished):

                #cell_output_tile = tf.tile(cell_output, [1,input_.item_num])
               # cell_out_1 = tf.reshape(cell_output_tile,[batch_size, input_.item_num, -1])
               # cell_out_1 * table_title

                attention = tf.cond(tf.equal(t, 0),
                            lambda: tf.ones([batch_size,input_.item_num], dtype=tf.float32),
                            lambda: tf.matmul(tf.expand_dims(tf.matmul(cell_output, W), 1), tf.transpose(table_title, perm = [0,2,1])) [:, 0,:])  # the weight of table items

                finished = tf.greater_equal(t + 1, target_real_num_steps)


                attention = tf.nn.softmax(attention)

                tableinput = tf.matmul(tf.expand_dims(attention, 1) , table_item_vector)[:,0,:]   # batch 1 embed_size

                x_nt = cell_output
                nextinput = tf.concat([x_nt, tableinput], 1)
                (cell_output_nt, state_nt) = cell(nextinput, state)
                out_word_quasi_prob = tf.matmul(cell_output_nt, softmax_w) + softmax_b
                out_word_index = tf.argmax(out_word_quasi_prob, axis = 1)
                out_word_emd = tf.nn.embedding_lookup(embedding, out_word_index)

                emit_vs = emit_vs.write(t,out_word_quasi_prob)
                return t+1, out_word_emd, state_nt, emit_vs, finished

            _, _, _, emit_vs,_ = tf.while_loop(
               cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
               body=loop_fn,
               loop_vars=(time, cell_output, state, emit_vs, f0))

            output = tf.transpose(emit_vs.stack(), [1, 0, 2])
            return output

    def generate(self, data_dict, sess):
        rng = np.random.RandomState(1234)
        feed_dict = dict()
        feed_dict[self._input.input_table_title] = data_dict['input_table_title']
        feed_dict[self._input.input_table_items] = data_dict['input_table_items']
        feed_dict[self._input.input_item_length] = data_dict['input_item_length']
        feed_dict[self._input.targets] = data_dict['targets']
        feed_dict[self._input.target_numsteps] = data_dict['target_numsteps']
        feed_dict[self._input.initial_cell] = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (self._input.batch_size +self._input.embedding_size)),
            high=np.sqrt(6. / (self._input.batch_size + self._input.embedding_size)),
            size=(self._input.batch_size, self._input.hidden_size)), dtype='float32')
        feed_dict[self._input.input_table_title_mask] = data_dict['input_table_title_mask']
        feed_dict[self._input.input_table_title_word_num] = data_dict['input_table_title_word_num']
        feed_dict[self._input.input_table_items_mask] = data_dict['input_table_items_mask']
        feed_dict[self._input.targets_mask] = data_dict['targets_mask']

        predictions = sess.run(self.pred_tokens, feed_dict)
        return predictions
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
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 120
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 64
    vocab_size = len(index2vec)
    item_num = 25
    content_len = 15
    embedding_size = int(sys.argv[1])



def run_epoch(session, model, data, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = session.run(model.initial_state)


    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    print(model.input.epoch_size)

    rng = np.random.RandomState(1234)

    for i in range(model.input.epoch_size):
        data_dict = prepare_data(data, i * model.input.batch_size, (i+1) * model.input.batch_size)
        feed_dict = dict()
        feed_dict[model.input.input_table_title] = data_dict['input_table_title']
        feed_dict[model.input.input_table_items] = data_dict['input_table_items']
        feed_dict[model.input.input_item_length] = data_dict['input_item_length']
        feed_dict[model.input.targets] = data_dict['targets']
        feed_dict[model.input.target_numsteps] = data_dict['target_numsteps']
        feed_dict[model.input.initial_cell] = np.asarray(rng.uniform(
                                                low=-np.sqrt(6. / (model.input.batch_size + model.input.embedding_size)),
                                                high=np.sqrt(6. / (model.input.batch_size + model.input.embedding_size)),
                                                size=(model.input.batch_size, model.input.hidden_size)), dtype='float32')
        feed_dict[model.input.input_table_title_mask] =data_dict['input_table_title_mask']
        feed_dict[model.input.input_table_title_word_num] = data_dict['input_table_title_word_num']
        feed_dict[model.input.input_table_items_mask] = data_dict['input_table_items_mask']
        feed_dict[model.input.targets_mask] = data_dict['targets_mask']

        #print(session.run( model.tt, feed_dict))
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += data_dict['max_target_len']

        #print('iter', i)
       # print(model.input.epoch_size)
        if verbose and i % 1000==0:
           # print(costs, iters)
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (i * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
        return SmallConfig()

last_best = 0

gold_file = dict()
def evaluate(sess, model, data, type):
    global last_best
    pred_path = FLAGS.data_path+ 'evaluate/' + type + '/'
    gold_path = FLAGS.data_path+ 'evaluate/' + type + '_gold/'

    pred_list =  []
    batch_num = len(data[0]) // model._input.batch_size -1

    pred_set = []
    gold_set = []
    gold_list = []

    for i in range(batch_num):
        #print(i * model.input.batch_size, (i + 1) * model.input.batch_size, len(data[0]))
        np_datum = prepare_data(data, i * model.input.batch_size, (i + 1) * model.input.batch_size)
        predictions = model.generate(np_datum, sess)
        for k, (case, gold) in enumerate(zip(predictions, data[3])):
            with open(pred_path + str(k) + '.txt', 'w') as sw:
                summary = list(case)
                if word2index['END_TOKEN'] in summary:
                    summary = summary[:summary.index(word2index['END_TOKEN'])] if summary[0] != word2index['END_TOKEN'] else [word2index['END_TOKEN']]
                sw.write(" ".join([str(x) for x in summary]) + '\n')
                pred_list.append([str(x) for x in summary])
                pred_set.append(pred_path + str(k) + '.txt')


            if type not in gold_file :
                with open(gold_path + str(k) + '.txt', 'w') as g:
                    summary = list(gold)
                    #if 2 in summary:
                     #   summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                    g.write(" ".join([str(x) for x in summary]) + '\n')
                    gold_list.append([str(x) for x in summary])
                    gold_set.append([gold_path + str(k)+ '.txt'])

    if type not in gold_file:
        gold_file[type] = (gold_set, gold_list)
    else:
        (gold_set, gold_list) = gold_file[type]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    nist = corpus_bleu(gold_list, pred_list, smoothing_function=SmoothingFunction().method3)
    result = "F_measure: %s Recall: %s Precision: %s BLEU: %s, NIST: %s\n" % (str(F_measure), str(recall), str(precision), str(bleu), str(nist))
    if float(F_measure[-1]) + float(bleu) > last_best:
        last_best = float(F_measure[-1]) + float(bleu)
        to_word(pred_list, FLAGS.data_path)
    return result

def to_word(pred_list, save_dir):
    ss = open(save_dir + "test_summary.txt", "w")
    for item in pred_list:
        ss.write(" ".join([index2word[int(id)] for id in item]) + '\n')

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    #raw_data = load_data()
    #train_data, dev_data, test_data , index2vec, title_set= raw_data

    config = get_config()
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
            valid_input = S2TInput(config=config, data=dev_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Structure2TextModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = S2TInput(config=config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Structure2TextModel(is_training=False, config=config,
                                 input_=test_input)

        #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True

        with tf.Session(config = config_gpu) as session:
            init = tf.global_variables_initializer()
            session.run(init)

            #print(evaluate(session, mvalid, dev_data, 'dev'))
            for i in range(config.max_max_epoch):
                #lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
               # m.assign_lr(session, config.learning_rate * lr_decay)
               # print ('lrlrlr:',config.learning_rate * lr_decay)

                restrain = evaluate(session, m, [train_data[0][:500],train_data[1][:500], train_data[2][:500],train_data[3][:500],train_data[4][:500]], 'train')
                print (restrain)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, eval_op=m.train_op, verbose=True )
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, dev_data)
                resdev = evaluate(session, mvalid, dev_data, 'dev')
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity),resdev)

                test_perplexity = run_epoch(session, mtest, test_data)
                restest = evaluate(session, mtest, test_data, 'test')
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, test_perplexity), restest)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                saver = tf.train.Saver()
                saver.save(session, FLAGS.save_path)


if __name__ == "__main__":
    tf.app.run()
