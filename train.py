import argparse

import sys
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
from random import shuffle

ProductDocument = namedtuple('ProductText', 'words tags')
alldocs= []
def train(input_file,doc_limit,n_epochs=1):
    with open(input_file) as fi:
        print ("Reding in: {}".format(input_file))
        for i,line in enumerate(fi):
            line = line.split()
            sku = [line[0]]
            text = line[1:]
            if i%100000==0: print ("Processing line: {}".format(i))
            if i>=doc_limit:
                break
            alldocs.append(ProductDocument(text,sku))
    model = Doc2Vec(size=128, window=5, negative=5, min_count=20, workers=31)
    print ("Building model vocab...")
    model.build_vocab(alldocs)
    alpha, min_alpha, passes = (0.025, 0.001, n_epochs)
    alpha_delta = (alpha - min_alpha) / passes
    print ("Training model...")
    for epoch in range(n_epochs):
        shuffle(all_docs)
        model.alpha -=alpha  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.train(alldocs)
        alpha -= alpha_delta
        print ("Epoch: {}, alpha: {}".format(epoch,alpha))
    return model
if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--doc_limit", type=int,
                            help="Number of lines to include in model, 0 for all")
    
    arg_parser.add_argument("-o", "--model_name",help="Name of file to write out")
    arg_parser.add_argument("-i", "--input_corpus",help="Name of corpus file to read in")
    arg_parser.add_argument("-e", "--epochs",type=int,help="Number of epochs to run")
    args = arg_parser.parse_args()
    if args.doc_limit == 0: # use all lines when doc_limit == 0
        args.doc_limit = np.inf
    model = train(args.input_corpus,args.doc_limit,n_epochs=args.epochs)
    model.save(args.model_name)
