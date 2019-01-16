import logging
import codecs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint



stop_word_path=''
with codecs.open(stop_word_path, 'r') as f:
    stop_words=[word.strip() for word in f]
stop_words=set(stop_words+[u'\ue40c', '．', '%'])

segged_text_path=text_path=''
with codecs.open(segged_text_path, 'r') as f:
    texts=[[word for word in line.split() if word not in stop_words] for line in f]

frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
texts=[[token for token in text if frequency[token]>1] for text in texts]

dictionary=corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]

num_topics=300
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
# lsi.print_topics(num_topics)

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
corpus_lda = lda[corpus]

for doc in corpus_lda:
    print(doc)

print('--------------------')

hdp = models.HdpModel(corpus, id2word=dictionary)
corpus_hdp = hdp[corpus]

for doc in corpus_hdp:
    print(doc)