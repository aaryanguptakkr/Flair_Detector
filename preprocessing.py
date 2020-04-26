#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 


# In[2]:


def no_num(text):
    text = re.sub('[0-9]+', '', text)
    return text

def no_punc(text):
    #nopuncmsg = ''.join([c for c in text if c not in string.punctuation])
    text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(' +', ' ', text)
    return text


# In[3]:


tokenized = []
def tokenizer(text):
    tokenized = word_tokenize(text)
    return tokenized
remov_stop_words = []
def remov_stopwords(text):
    remov_stop_words  =' '.join([c for c in text.split(' ') if c not in stopwords.words('english')])
    return remov_stop_words
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def func_lemmatizer(text):
    lemmatized = []
    lemmatizer = WordNetLemmatizer()
    lemmatized = ' '.join( [ lemmatizer.lemmatize(word) for word in text] )
    return lemmatized


# In[4]:


def preprocessor(text):
    text = text.str.lower()
    text = text.apply(no_num)
    text = text.apply(no_punc)
    text = text.apply(remov_stopwords)
    text = text.apply(deEmojify)
    text = text.apply(tokenizer)
    text = text.apply(func_lemmatizer)
    return text


# In[ ]:




