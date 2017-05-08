# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 16:02:06 2016

@author: shalei
"""
import nltk, os, re, sys
import numpy, cPickle, random
from itertools import groupby
import time

reload(sys)
sys.setdefaultencoding('utf8')
tic = time.time()
print tic


# sys.argv.append('100')
def read_word2vec():
    f = open('/home/shalei/glove.6B.' + sys.argv[1] + 'd.txt')
    # f = open('D:/Grade4_1/AnswerSelection/glove.6B.'+sys.argv[1]+'d.txt')
    lines = f.readlines()
    g = lambda x: numpy.array(x, dtype=numpy.float32)
    dic = {line.strip().split(' ')[0]: g(line.strip().split(' ')[1:]) for line in lines}
    dic['</s>'] = numpy.random.normal(size=[int(sys.argv[1])]).astype('float32')
    dic['-1'] = numpy.zeros([int(sys.argv[1])]).astype('float32')
    print 'Dictionary Got!'
    return dic


word2vec = read_word2vec()
print time.time(), 'Wemb'
index2word = []
print time.time(), 'Wembkey'
word2index = {}
for i, a in enumerate(word2vec):
    index2word.append(a)
    word2index[a] = i

index2vec = numpy.asarray([word2vec[a] for a in index2word])
print time.time(), 'Wembid'

title_set = dict()

def TurnWordID(words):
    res = []
    for w in words:
        if w in index2word:
            res.append(word2index[w])
        else:
            res.append(word2index['</s>'])
    return res


def KnuthSampling(total_num, sampling_num):
    n = total_num
    m = sampling_num
    res = []
    for i in range(n):
        if random.random() < 1.0 * m / n:
            res.append(i)
            m -= 1
            n -= 1
        else:
            n -= 1

        if m == 0:
            break

    return res


def DataPreprocessing():
    print 'wikiqa', time.time()
    mainfolder = '/home/shalei/wikibio/wikipedia-biography-dataset/'
    datafolder = '/home/shalei/gen_data/'



    def transform_infobox(info):
        items = info.split()
        items = [(t.split(':')[0], t.split(':')[1]) for t in items]
        dealed_items = []
        for item in items:
            if item[0].split('_')[-1].isdigit():
                num = int(item[0].split('_')[-1])
                if num == 1:
                    title = ' '.join(item[0].split('_')[:-1])
                    if title not in title_set:
                        title_set[title] = len(title_set)
                    ditem = (title , item[1])
                    dealed_items.append(ditem)
                else:
                    dealed_items[-1][1] += ' ' + item[1]
            else:
                title = item[0]
                if title not in title_set:
                    title_set[title] = len(title_set)

                if item[1] != '<none>':
                    dealed_items.append(item)

        res = []
        for item in dealed_items:
            resitem = []
            resitem.append(title_set[item[0]])
            content_words = item[1].split()
            content_word_ids = []
            for word in content_words:
                if word in index2word:
                    content_word_ids.append(word2index[word])
                else:
                    word2index[word] = len(index2word)
                    index2word.append(word)
                    word2vec[word] = numpy.random.normal(size=[int(sys.argv[1])]).astype('float32')
                    index2vec = numpy.append(index2vec, word2vec[word], axis = 1)

                    content_word_ids.append(word2index[word])

            resitem.append(content_word_ids)
            res.append(resitem)

        return res






    def deal_set(setname):
        folder = mainfolder + setname + '/'
        infoboxfile = open(folder + setname + '.box', 'r')
        sentencenumfile = open(folder + setname + '.nb', 'r')
        sentencefile = open(folder + setname + '.sent', 'r')

        lines = infoboxfile.readlines()

        sentnbs = sentencenumfile.readlines()
        sentnbs = [int(a) for a in sentnbs]

        boxes = []
        passages = []

        for line, nb in zip(lines, sentnbs):
            box = transform_infobox(line)

            sentences = []

            for i in range(nb):
                sent = sentencefile.readline()
                words = sent.split()
                sentids = TurnWordID(words)
                sentences.append(sentids)

            boxes.append(box)
            passages.append(sentences)


        return boxes, passages



    train_x, train_y = deal_set('train')
    print 'traindata got!', time.time()

    dev_x, dev_y = deal_set('valid')
    print 'dev got!', time.time()

    test_x, test_y = deal_set('test')
    print 'test got!', time.time()

    recordfile = open(datafolder + 'gen_' + sys.argv[1] + '_data1.dat', 'wb')
    cPickle.dump(train_x, recordfile, -1)
    cPickle.dump(train_y, recordfile, -1)
    cPickle.dump(dev_x, recordfile, -1)
    cPickle.dump(dev_y, recordfile, -1)
    cPickle.dump(test_x, recordfile, -1)
    cPickle.dump(test_y, recordfile, -1)



if __name__ == '__main__':
    DataPreprocessing()

toc = time.time()
print toc - tic, 's'