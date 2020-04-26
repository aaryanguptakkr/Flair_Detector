#!/usr/bin/env python
# coding: utf-8

# In[5]:


import praw
import pandas as pd
import string
import re
from preprocessing import preprocessor
import pickle
import numpy as np


# In[6]:


reddit = praw.Reddit(  client_id = 'j3S-mjVI054xXw',
                        client_secret = 'cGW2OwC5mw23pFKTTDPIGyAbspI',
                        username = 'praw_scraper',
                        password = 'kurukshetra',
                        user_agent = 'prawscraper1')


# In[22]:


# THE MODEL WHICH IS REQUIRED TO BE IMPORTED  (cv)
Pkl_Filename1 = "Pickle_CV_Model.pkl"
Pkl_Filename2 = "Pickle_TF_Model.pkl"
Pkl_Filename3 = "Pickle_NB_Model.pkl"
Pkl_Filename4 = "Pickle_SVM_Model.pkl"
Pkl_Filename5 = "Pickle_RFC_Model.pkl"
with open(Pkl_Filename1, 'rb') as file:  
    pickled_CV_model = pickle.load(file)
with open(Pkl_Filename2, 'rb') as file:  
    pickled_TF_model = pickle.load(file)
with open(Pkl_Filename3, 'rb') as file:  
    pickled_NB_model = pickle.load(file)
with open(Pkl_Filename4, 'rb') as file:  
    pickled_SVM_model = pickle.load(file)
with open(Pkl_Filename5, 'rb') as file:  
    pickled_RFC_model = pickle.load(file)
    

def flair_predictor(input_url):
    submission = reddit.submission(url = input_url)
    input = []
    input.append([submission.title , submission.link_flair_text])
    df = pd.DataFrame(input , columns = ['title','flair'])
    df['title'] = preprocessor(df['title'])
    text = df['title']
    topic = df['flair']

    input_tf = pickled_CV_model.transform(text)
    input_tfidf = pickled_TF_model.transform(input_tf)
    
    output_SVM = pickled_SVM_model.predict(input_tfidf)
    output_NB = pickled_NB_model.predict(input_tfidf)
    output_RFC = pickled_RFC_model.predict(input_tf)
    
    score_svm = np.max(pickled_SVM_model.decision_function(input_tfidf))
    score_nb = np.max(pickled_NB_model.predict_proba(input_tfidf))
    score_rfc = np.max(pickled_RFC_model.predict_proba(input_tf))
    
    if score_nb > score_svm and score_nb > score_rfc:
        print("predicted flair is " , output_NB )
        output = output_NB
    elif score_svm > score_rfc :
        print("predicted flair is " , output_SVM )
        output = output_SVM
    else :
        print("predicted flair is " ,output_RFC )
        output = output_RFC
    return output

