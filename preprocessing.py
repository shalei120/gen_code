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
    dic = {line.strip().split(' ')[0]: g(line.strip().split(' ')[1:]) for index,line in enumerate(lines) if index < 20000}
    dic['</s>'] = numpy.random.normal(size=[int(sys.argv[1])]).astype('float32')
    dic['UNK'] = numpy.zeros([int(sys.argv[1])]).astype('float32')
    dic['START_TOKEN'] = numpy.zeros([int(sys.argv[1])]).astype('float32')
    dic['END_TOKEN'] = numpy.zeros([int(sys.argv[1])]).astype('float32')
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

index2word_set = set(index2word)
title_set = dict()

def TurnWordID(words):
    res = []
    for w in words:
        if w in index2word_set:
            res.append(word2index[w])
        else:
            res.append(word2index['UNK'])
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

from collections import Counter
print 'start', time.time()
mainfolder = '/home/shalei/wikibio/wikipedia-biography-dataset/'
datafolder = '/home/shalei/gen_data/'
def transfer_boxes():
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
                        title_set[title] = len(title_set)+1
                    ditem = [title , item[1]]
                    dealed_items.append(ditem)
                else:
                    dealed_items[-1][1] += ' ' + item[1]
            else:
                title = item[0]
                if title not in title_set:
                    title_set[title] = len(title_set)

                if item[1] != '<none>':
                    dealed_items.append(item)

        def legal(str):
            sw = ['image', 'caption','update','source']
            sc = ['--']
           # if ''.join(str[1].split(' ')).isdigit():
           #     return False
            for s in sc:
                if s in str[1]:
                    return False

            for s in sw:
                if s in str[0]:
                    return False

            return True

        dealed_items = [d for d in dealed_items if legal(d)]
        #if len(dealed_items) > 25:
         #   print dealed_items
        return dealed_items

    def deal_set_info(setname):
        folder = mainfolder + setname + '/'
        infoboxfile = open(folder + setname + '.box', 'r')
        lines = infoboxfile.readlines()
        boxes=[]
        for line in lines:
            box = transform_infobox(line.lower())
            boxes.append(box)
        return boxes

    infoboxout = open(datafolder + 'boxout.dat', 'wb')
    train_box  = deal_set_info('train')
    print 'train dealed'
    valid_box = deal_set_info('valid')
    print 'valid dealed'
    test_box = deal_set_info('test')
    print 'test dealed'

    cPickle.dump(title_set, infoboxout, -1)
    cPickle.dump(train_box, infoboxout, -1)
    cPickle.dump(valid_box, infoboxout, -1)
    cPickle.dump(test_box, infoboxout, -1)



def DataPreprocessing():
    infoboxfile = open(datafolder + 'boxout.dat', 'rb')
    title_set = cPickle.load(infoboxfile)



    def deal_set(setname, dealed_table_list):
        folder = mainfolder + setname + '/'
        sentencenumfile = open(folder + setname + '.nb', 'r')
        sentencefile = open(folder + setname + '.sent', 'r')
        inum=0
        total = 0
        boxes = []
        buffer_vec = []
        count = 0
        for dealed_table in dealed_table_list:
            #count+=1
            #if count%10000 == 0:
            #    print count,'completed'
            box = []
            for index,item in enumerate(dealed_table):
                resitem = []
                resitem.append(TurnWordID(item[0].split()))
                content_words = item[1].split()
                if len(content_words) > 15:
                    inum+=1
                total +=1
                content_word_ids = []
                for word in content_words:
                    if word in index2word_set:
                        content_word_ids.append(word2index[word])
                    else:
                        #content_word_ids.append(len(index2word))
                        #word2index[word] = len(index2word)
                        #index2word.append(word)
                        #index2word_set.add(word)
                        #print word
                        #wv = numpy.random.normal(size=[int(sys.argv[1])]).astype('float32')
                        #word2vec[word] = wv

                        #buffer_vec.append(wv)
                        content_word_ids.append(word2index['</s>'])

                resitem.append(content_word_ids)
                box.append(resitem)

            boxes.append(box)

        print 'fuck num', inum, total

        #buffer_np_vec = numpy.random.normal(size=[len(title_set), int(sys.argv[1])]).astype('float32')
        #buffer_np_vec = numpy.asarray(buffer_vec)
        #global index2vec
        #index2vec = numpy.concatenate([index2vec, buffer_np_vec], axis=0)


        sentnbs = sentencenumfile.readlines()
        sentnbs = [int(a) for a in sentnbs]

        passages = []

        count = 0

        print len(dealed_table_list), len(sentnbs)

        for nb in  sentnbs:
            #count += 1
            #if count % 1000 == 0:
            #    print count, ' cases completed!!'

            #print line

            sentences = []

            for i in range(nb):
                sent = sentencefile.readline()
                words =  sent.split()
                sentids = TurnWordID(words)
                sentences += sentids
                #print sent

            passages.append(sentences)
           # print box
           # print nb,sentences
           # break


        assert len(boxes) == len(passages)

        def gene_case(box,p):
            titles = [item[0] for item in box]       #[]
            contents = [item[1] for item in box]     #[[] [] []]
            contents_len = [len(item[1]) for item in box]     #[[] [] []]
            target = p
            target_len = len(p)
            if target_len > 100:
                target_len = 100
                target = target[:100]

            target.append(word2index['END_TOKEN'])
            target_len += 1

            return (titles, contents, contents_len, target, target_len)

        cases = [gene_case(box,p) for box, p in zip(boxes, passages)]
        sorted_cases = sorted(cases, key = lambda x: x[4])

        titles = [case[0] for case in sorted_cases]
        contents = [case[1] for case in sorted_cases]
        content_len = [case[2] for case in sorted_cases]
        target = [case[3] for case in sorted_cases]
        target_len = [case[4] for case in sorted_cases]

        print sorted(Counter(target_len).items())

        return (titles, contents, content_len, target, target_len)

    train_table_list = cPickle.load(infoboxfile)
    train_data = deal_set('train',train_table_list)
    print 'traindata got!', time.time()

    dev_table_list = cPickle.load(infoboxfile)
    dev_data = deal_set('valid',dev_table_list)
    print 'dev got!', time.time()

    test_table_list = cPickle.load(infoboxfile)
    test_data = deal_set('test',test_table_list)
    print 'test got!', time.time()


    recordfile = open(datafolder + 'gen_' + sys.argv[1] + '_data1.dat', 'wb')
    cPickle.dump(index2vec, recordfile, -1)
    cPickle.dump(index2word, recordfile, -1)
    cPickle.dump(word2index, recordfile, -1)
    cPickle.dump(title_set, recordfile, -1)

    cPickle.dump(train_data, recordfile, -1)
    cPickle.dump(dev_data, recordfile, -1)
    cPickle.dump(test_data, recordfile, -1)



if __name__ == '__main__':
    transfer_boxes()
    DataPreprocessing()

toc = time.time()
print toc - tic, 's'