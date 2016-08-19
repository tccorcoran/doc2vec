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
import pandas as pd

dir = '/mnt/Data/product_data/doc2vec/'
csv.field_size_limit(sys.maxsize)

with open(dir + sys.argv[1]) as f:  #'query_words.txt'
    queries = f.readlines()

print('queries = f.readlines')
model2 = Doc2Vec.load(dir + sys.argv[2]) #'doc2vec2_model.d2v'

def similar_results(queries):
    for query in queries:
        doc_vec = model2.infer_vector(query)
        yield model2.docvecs.most_similar([doc_vec])

print('yield model2.docvecs.most_similar([doc_vec])')

results = []
results = [line for line in similar_results(queries)]

print('results = [line for line in similar_results(queries)')

df = pd.DataFrame.from_csv('/mnt/Data/product_data/' + sys.argv[3], index_col=False)
df["url"] = (df[" gender_stle"].map(str) )
df.drop(df.columns[[0,1,3,4,5,6,7,8,9,10]], axis=1, inplace=True)
urls = []
url_sku_database = []
for i, line in enumerate((df['url'])):
    try:
        temp = (ast.literal_eval(line))
    except:
        temp = 'missing'
    temps = (df[' sitedetails.sku'][i],temp[0])
    print('    url_sku_database.append(temps)')
    url_sku_database.append(temps)

print('url_sku_database.append(temps)')

similar_result_skus = []
for line in (results):
    result_line = []
    for i in range(len(line)):
        result_line.append(line[i][0])
    similar_result_skus.append(result_line)

print('matching query result and database skus and URL')
query_url = []
for single_query_results in similar_result_skus:
    for search_result_sku in single_query_results:
        i=0
        while url_sku_database[i][0] != search_result_sku:
            i +=1
            pass
        else:
            temp2 = (search_result_sku,url_sku_database[i][1])
            query_url.append(temp2)

print('loading model')
            
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(query_url)
