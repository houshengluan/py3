#!/usr/bin/python
# -*- coding: utf-8 -*-  
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn


def word_segment(text):
    return ' '.join(jieba.cut(text))

def print_top_words(model, feature_names, n_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic #{}'.format(topic_index))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    df=pd.read_csv('/home/hadoop/dataset/datascience.csv', encoding='gb18030')
    df['content_cutted']=df.content.apply(word_segment)
    print(df.content_cutted[2])
    tf_vectorizer=CountVectorizer(strip_accents='unicode', max_features=1000, stop_words='english',
                    max_df=0.5, min_df=10)
    tf=tf_vectorizer.fit_transform(df.content_cutted)
    lda=LatentDirichletAllocation(5, max_iter=50, learning_method='online', learning_offset=50, random_state=0)
    lda.fit(tf)
    tf_feature_names=tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words=20)
    pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
