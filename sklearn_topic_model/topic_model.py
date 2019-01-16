#!/usr/bin/python
# -*- coding: utf-8 -*-  
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from gensim.sklearn_api import HdpTransformer

import pyLDAvis
import pyLDAvis.sklearn
import codecs


def word_segment(text):
    return ' '.join(jieba.cut(text))

def print_top_words(model, feature_names, n_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic #{}'.format(topic_index))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def read_data_from_csv():
    df=pd.read_csv('/home/hadoop/dataset/datascience.csv', encoding='gb18030')
    df['content_cutted']=df.content.apply(word_segment)
    # print(df.content_cutted[2])
    return df.content_cutted

def read_data_from_file():
    path='/data/hou_data_donot_del/sogou_and_sohu_news/SogouCA/segmented_domain_texts_IT'
    with codecs.open(path, 'r') as f:
        return [line for line in f]

if __name__ == '__main__':
    # splitted_lines=read_data_from_csv()
    splitted_lines=read_data_from_file()
    tf_vectorizer=CountVectorizer(strip_accents='unicode', max_features=1000, stop_words='english',
                    max_df=0.5, min_df=10)
    tf=tf_vectorizer.fit_transform(splitted_lines)
    n_topics=5
    lda=LatentDirichletAllocation(n_topics, max_iter=50, learning_method='online', learning_offset=50, random_state=0)
    lda.fit(tf)
    tf_feature_names=tf_vectorizer.get_feature_names()
    n_top_words=20
    print_top_words(lda, tf_feature_names, n_top_words=n_top_words)
    pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
