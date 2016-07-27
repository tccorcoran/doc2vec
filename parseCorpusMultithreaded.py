import spacy.en
import os
import csv 
import itertools
import sys

nlp = spacy.load('en')

def iter_docs(fname):
    with open(fname,'r') as fi:
        csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
        for line in csvreader:
            text = ''
            for i in (0,1,3,5,10,11): #title,description,class_level
                try:
                    t = line[i]
                    if t and t != 'Not Defined':
                        text += line[i]
                        if t[-1] not in ('.','?','!'): text += ". "
                        else: text += ' '
                except IndexError:
                    continue
            sku = line[2]
            if text:
                yield text,sku

def clean(text):
    text = text.replace('-','')
    text = text.replace('|', ' ')
    return text
def parseTexts(fname):
    gen1,gen2 = itertools.tee(iter_docs(fname))
    skus = (sku for (doc,sku) in gen1)
    texts = (clean(doc) for (doc,sku) in gen2)
    docs = nlp.pipe(texts,n_threads=31, batch_size=1000)
    for sku,doc in zip(skus,docs):
        yield sku,[tok.lower_ for tok in doc if not tok.is_stop and tok.is_alpha and not tok.is_punct]

if __name__ == "__main__":
    inp = sys.argv[1]
    outp = os.path.splitext(inp)[0]
    with open(outp+'_.sku_doc','w') as fo:
        for sku,doc in parseTexts(inp):
            fo.write("{} {}\n".format(sku,' '.join(doc)))

