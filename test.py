from gensim.models import Doc2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
import numpy as np
import csv
import ast
import urllib
import sys
from PIL import Image as IMG
from IPython.display import Image, display
import itertools
import os

dir = '/mnt/Data/product_data/doc2vec/'

model2 = Doc2Vec.load(dir + 'models/5cbf35b/dbow.dv') #'doc2vec2_model.d2v'

doc_vec = model2.infer_vector("Nike shoes")
result = model2.docvecs.most_similar([doc_vec])
result_line = []
for i in range(len(result)):
    result_line.append(result[i][0])
    
listcsv = []
with open(dir + 'models/5cbf35b/out_put.csv','r') as fi:
        csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
        for line in csvreader:
            listcsv.append(line)
temp = []
for result in result_line:
    for line2 in listcsv:
        for line in line2:
            if result in line:
                temp.append(line2)

file  = open("output.txt",'w')
for item in temp:
  file.write("%s\n" % item)
