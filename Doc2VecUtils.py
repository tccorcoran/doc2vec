import argparse

import sys
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np

ProductDocument = namedtuple('ProductText', 'words tags')
alldocs= []
def train(doc_limit,n_epochs=5):
    with open('corpus_lemmas_by_doc.txt') as fi:    
        for i,line in enumerate(fi):
            line = line.split()
            sku = [line[0]]
            text = line[1:]
            if i>=doc_limit:
                break
            alldocs.append(ProductDocument(text,sku))
    model = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=20, workers=31)
    model.build_vocab(alldocs)
    alpha, min_alpha, passes = (0.025, 0.001, n_epochs)
    alpha_delta = (alpha - min_alpha) / passes
    for epoch in range(n_epochs):
        model.alpha -=alpha  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.train(alldocs)
        alpha -= alpha_delta
    return model
if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--doc_limit", type=int,
                            help="Number of lines to include in model, 0 for all")
    
    arg_parser.add_argument("-o", "--model_name",help="Name of file to write out")
    arg_parser.add_argument("-e", "--epochs",type=int,help="Number of epochs to run")
    args = arg_parser.parse_args()
    if args.doc_limit == 0: # use all lines when doc_limit == 0
        args.doc_limit = np.inf
    model = train(args.doc_limit,n_epochs=args.epochs)
    model.save(args.model_name)
