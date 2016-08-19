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

print('loading model')
model2 = Doc2Vec.load(dir + sys.argv[2]) #'doc2vec2_model.d2v'

def similar_results(queries):
    for query in queries:
        doc_vec = model2.infer_vector(query)
        yield model2.docvecs.most_similar([doc_vec])
        
results = []
results = [line for line in similar_results(queries)]

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
    url_sku_database.append(temps)

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

try:
    os.stat(dir + sys.argv[4])
except:
    os.mkdir(dir + sys.argv[4])       

for i, line in enumerate(query_url):
    list_im = []
    min_shape = (800,600)

    if line[1] != 'missing':
        
        current_img = dir + str(sys.argv[4]) + str(i)+ '.jpg'
        try:
            a =urllib.request.urlretrieve(line[1], current_img)
        except:
            continue
        try:
            a = IMG.open(current_img)
            a.resize(min_shape)
        except:
            os.remove(current_img)
    else:
        pass

listoflists = []
counter = 0
for i in range(66):
    list = []
    for _ in range(10):
        img_name = dir + str(sys.argv[4]) + str(counter) +".jpg"
        list.append(img_name)
        counter +=1
    listoflists.append(list)   

for i, line in enumerate(listoflists):
    imgs = line
    concat_img = []
    for img in imgs:
        try:
            concat_img.append(IMG.open(img))
        except:
            continue
            
    min_shape = (800,600)
    imgs_comb = []
    try:
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in concat_img ) )
        imgs_comb = IMG.fromarray(imgs_comb)
        display(imgs_comb)
    except:
        continue
        
    imgs_comb.save(dir + sys.argv[5] + queries[i]+'.jpg')
    print(imgs)
    print(queries[i])
