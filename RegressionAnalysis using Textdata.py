# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:55:24 2018

@author: mouli uddanti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_excel('sample.xlsx', sheet_name = 'DataSet')

dataset.columns

#dataset = dataset[dataset['TrainingData'] == 1]


dataset.head()

dataset.info()


dataset.describe()

dataset['fitment_score'].unique()

dataset['fitment_score'].nunique()

dataset['title'].isnull().sum()

dataset['description'].isnull().sum()

dataset['text'] = dataset['title'] + dataset['description']



from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
    #stemmer = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if not word in stop_words]
    #stemmed_words = [stemmer.stem(word) for word in tokens]
    return tokens
    
    
    
dataset['text'] = [preprocess_text(text) for text in dataset['text']]    

features = dataset['text'].astype(str).values

labels = dataset['fitment_score'].values

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.3, random_state = 0)

#term frequency inverse frequency vectorizer,which converts text data into sparse matrix

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', sublinear_tf = True, strip_accents = 'unicode',lowercase = False,
                                  analyzer = 'word', ngram_range = (1,3), max_features = 2000,token_pattern = r'\w{2,}')

word_vector_train = tfidf_vectorizer.fit_transform(x_train)

word_vector_test = tfidf_vectorizer.transform(x_test)


tfidf_vectorizer.idf_
tfidf_vectorizer.vocabulary_

#implementing lightgbm for regression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
import pickle

lightgbm_model = lgb.LGBMRegressor(boosting_type = 'gbdt',objective = 'regression',num_leaves = 31, learning_rate = 0.01, n_estimators = 40,njobs = -1)

lightgbm_model.fit(word_vector_train, y_train, eval_set = [(word_vector_test, y_test)], eval_metric = 'RMSE', early_stopping_rounds = 10)

y_predict = lightgbm_model.predict(word_vector_test, num_iteration = lightgbm_model.best_iteration_)


rmse = np.sqrt(mean_squared_error(y_predict, y_test))

param_grid = {
    'learning_rate': [0.01, 0.1, 0.5,0.8,1],
    'n_estimators': [20, 40, 50]
}


gbm_model = GridSearchCV(lightgbm_model, param_grid, cv = 5)

gbm_model.fit(word_vector_train, y_train)

gbm_model.best_params_

#the best parament is learning rate 0.01 and n_estimators 50

#save model in pickle format
regression_model = 'lightgbm_model.sav'
with open(regression_model, mode = 'wb') as model_file:
    pickle.dump(lightgbm_model, model_file)

#load model from pickle format

with open(regression_model , mode = 'rb') as model_file:
    model = pickle.load(model_file)
    
#for testing model on unseen test data
    
#preprocessing for test dataset

dataset = pd.read_excel('microland.xlsx', sheet_name = 'DataSet', nrows = 1000)

test_dataset = dataset[dataset['TrainingData'] == 0]


test_dataset.columns

test_dataset.info

test_dataset.describe()

test_dataset['fitment_score'].unique()

test_dataset['fitment_score'].nunique()

test_dataset['title'].isnull().sum()

test_dataset['description'].isnull().sum()

test_dataset['text'] = test_dataset['title'] + test_dataset['description']

test_dataset['text'] = [preprocess_text(text) for text in test_dataset['text']]


test_features = test_dataset['text'].astype(str).values

test_labels = test_dataset['fitment_score'].values

word_vector = tfidf_vectorizer.fit_transform(test_features)

predict_vector = model.predict(word_vector)

rmse = np.sqrt(mean_squared_error(test_labels,predict_vector))

#the rmse score for testdata is around 0.55

##############################################################################
##############################################################################
#implement regression model using GBM

from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators': 100,'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}


GBR_model = GradientBoostingRegressor(**params)

GBR_model.fit(word_vector_train, y_train)

y_predict = GBR_model.predict(word_vector_test)

rmse = (mean_squared_error(y_test, y_predict) ** 0.5)
#rmse = np.sqrt(mean_squared_error(y_test,y_predict))


GBR_model.get_params().keys()

n_jobs=4

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              'max_features': [1.0, 0.3, 0.1]
}


GBR_model = GridSearchCV(GBR_model, param_grid, cv =5)

GBR_model.fit(word_vector_train, y_train)

GBR_model.best_params_

#implementing model using SGDRegressor

from sklearn.linear_model import SGDRegressor

SGD_model = SGDRegressor(alpha = 0.0010, loss = 'squared_loss', penalty = 'l2')

SGD_model.fit(word_vector_train, y_train)

y_predict = SGD_model.predict(word_vector_test)

rmse = np.sqrt(mean_squared_error(y_predict,y_test))

param_grid = {
   'alpha':[0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010],
    'loss': ['squared_loss'],
    'penalty': ['l2', 'l1'],
    'learning_rate': ['constant']
}

sgd = GridSearchCV(SGD_model, param_grid)

sgd.fit(word_vector_train, y_train)

sgd.best_score_

#implementing model using Ridge and lasso regressions

from sklearn.linear_model import Ridge

ridge_model = Ridge(normalize = True)

ridge_model.fit(word_vector_train,y_train)

y_predict = ridge_model.predict(word_vector_test)

rmse = np.sqrt(mean_squared_error(y_predict,y_test))

param_grid = {
           'alpha':[0.0005, 0.0006, 0.0007]
}

model = GridSearchCV(ridge_model, param_grid)

model.fit(word_vector_train, y_train)

model.best_params_

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(word_vector_train, y_train)

y_predict = lasso_model.predict(word_vector_test)

rmse = np.sqrt(mean_squared_error(y_test,y_predict))

param_grid = {
           'alpha':[0.0005, 0.0006, 0.0007]
}

model = GridSearchCV(lasso_model, param_grid)

model.fit(word_vector_train, y_train)

model.best_params_

#implementing model using LinearRegression

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(word_vector_train, y_train)

y_predict = linear_model.predict(word_vector_test)

linear_model.fit(word_vector_train, y_train)

y_predict = linear_model.predict(word_vector_test)

rmse = np.sqrt(mean_squared_error(y_test,y_predict))

