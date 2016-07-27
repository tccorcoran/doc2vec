import multiprocessing
import os
import csv
import sys
from gensim.corpora.textcorpus import TextCorpus
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
#from gensim.models.doc2vec import LabeledSentence 
import spacy

csv.field_size_limit(sys.maxsize)

parseThis = spacy.load('en')
def process_article(args):
    """
    Parse description, returning its content as a list of tokens
    (utf8-encoded strings).
    """
    text, chunk_nouns, sku  = args
    text = parseThis(text)
    if chunk_nouns:
        result = [x.lemma_ for x in text.noun_chunks]
    else:
        result = [tok.lower_ for tok in text if not tok.is_stop and tok.is_alpha and not tok.is_punct]        
    return result,sku

class GoFindCorpus(TextCorpus):
    def __init__(self,fname,processes=None,chunk_nouns=False, dictionary=None,metadata=None):
        self.fname = fname
        self.chunk_nouns = chunk_nouns
        
        self.metadata = metadata
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        if dictionary is None:
            self.dictionary = Dictionary()
            self.dictionary.add_documents(self.get_texts())
        else:
            self.dictionary = dictionary
    def iter_sents(self):
        with open(self.fname,'r') as fi:
            csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
            for line in csvreader:
                try:
                    title = line[0]
                    desc = line[1]
                    text = "{}. {}".format(title,desc)
                except IndexError:
                    continue
                text = parseThis(text)
                for sentence in text.sents:
                    yield [tok.lemma_ for tok in sentence]
    def extract_data(self,fname):
        with open(fname,'r') as fi:
            csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
            for line in csvreader:
                try:
                    title = line[0]
                    desc = line[1]
                    text = "{}. {}".format(title,desc)
                    sku = line[2]
                except IndexError:
                    continue
                yield text, sku

 
    def get_texts(self):
        """ Process desciptions in parallel
        """
        texts = ((text, self.chunk_nouns, sku) for  text, sku in self.extract_data(self.fname))
        pool = multiprocessing.Pool(self.processes)
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, sku in pool.imap(process_article, group):
                if self.metadata:
                    yield (tokens, sku)
                else:
                    yield tokens
        pool.terminate()

class GoFindSentences(GoFindCorpus):
    def __iter__(self):
        with open(self.fname,'r') as fi:
            csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
            for line in csvreader:
                try:
                    title = line[0]
                    desc = line[1]
                    text = "{}. {}".format(title,desc)
                except IndexError:
                    continue
                text = parseThis(text)
                for sentence in text.sents:
                    yield [tok.lemma_ for tok in sentence]
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        with open(self.filename,'r') as fi:
            for line in fi:
                yield LabeledSentence(words=line.split()[1:], labels=['%s' % line[0]])


def iter_sents(self):
    with open(self.fname,'r') as fi:
        csvreader = csv.reader(fi,delimiter = ",", quotechar = "\"")
        for line in csvreader:
            try:
                title = line[0]
                desc = line[1]
                text = "{}. {}".format(title,desc)
                sku = line[2]
            except IndexError:
                continue
            args = (text,False,sku)
            processed_docs,sku = process_article(args)

if __name__ == '__main__':
    inp = sys.argv[1]
    outp = os.path.splitext(inp)[0]
    g = GoFindCorpus(inp)
    g.metadata=True
    with open(outp+'_.sku_doc','wb') as fo:
        for doc in g:
            print(doc[1])
#            fo.write("{} {}\n".format(doc[1],' '.join(doc[0])))
#            fo.flush()
    
#    g.dictionary.save_as_text(outp + '_wordids.txt.bz2')
#    MmCorpus.serialize(outp + '_bow.mm',g)
