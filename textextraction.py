# -*- coding: utf-8 -*-
"""

Created on Tue Feb 25 02:50:55 2019

@author: uddanti mouli
"""
text = 'sample.txt'

import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk,pos_tag
from nltk import conlltags2tree, tree2conlltags
from string import punctuation
stop = set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

text = text.lower()

#parts of speech tagging
pos_tagger = ne_chunk(pos_tag(word_tokenize(str(text))))

print(pos_tagger)
   
grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)

result = cp.parse(pos_tagger)

result = result.flatten()

#print(result)

iob_tags = tree2conlltags(pos_tagger)

tree = conlltags2tree(iob_tags)
#print(tree)

''' Information retreival using spacy '''

import spacy
import en_core_web_sm
from collections import Counter
from spacy import displacy
from collections import defaultdict
from tabulate import tabulate
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'ABILIFY is indicated for the treatment of schizophrenia in adults and in adolescents aged 15 years and older without any history of myocardial infractions.')

displacy.serve(doc, style='dep')
#spacy.displacy.render(doc, style='dep', options={'distance' : 140}, jupyter=True)

for token in doc.noun_chunks:
    print(token)
    
token_dependencies = ((token.text, token.dep_,token.head.text,token.ent_type_,token.tag_) for token in doc)
print(tabulate(token_dependencies, headers=['Token', 'Dependency Relation', 'Parent Token' 'Ent', 'Tags']))

print(token.text, token.dep_, token.head.text, token.head.pos_,
      [child for child in token.children])     
            

print(u'parsed_children:{0}'.format([(token.text,token.pos_,token.dep_,[(token.text,token.dep_) for token in list(token.children)]) for token in doc]))
print(u'Parsed_ancestors:{0}'.format([(token.text,token.pos_,token.dep_,[(token.text,token.dep_) for token in list(token.ancestors)]) for token in doc]))


drug_disease_pair = []


for token in doc:
    if token.dep_ in ['nsubjpass', 'pobj', 'amod']:
        #print(token.text, token.tag_,token.head)
        #print(token.head.lefts)
        #print(token.head.rights)
        #print(token.head)
        for each_token in token.head.lefts:
            if each_token.dep_ == 'nsubjpass':
                drug_disease_pair.append(each_token.text)        
        for each_token in token.head.rights:
            #print(each_token.text, each_token.tag_, each_token.dep_)
            #print(each_token.head.text,each_token.head.tag_,each_token.head.dep_)
            #print([x for x in each_token.children])
            if each_token.tag_== 'NN' and each_token.head.dep_ == 'prep':
                child  = [x for x in each_token.children]
                if not child:
                    drug_disease_pair.append(each_token.text)


doc2 = nlp('Pantoprazole sodium delayed-release tablets are indicated in adults and pediatric patients five years of age and older for the short-term treatment (up to 8 weeks) in the healing and symptomatic relief of erosive esophagitis (EE).')


for token in doc2.noun_chunks:
    print(token)

token_dependencies = ((token.text, token.dep_,token.head.text,token.ent_type_,token.tag_) for token in doc)
print(tabulate(token_dependencies, headers=['Token', 'Dependency Relation', 'Parent Token' 'Ent', 'Tags']))


drug_disease_pair2 = []

for token in doc2:
    if token.dep_ in ['amod', 'pobj']:
        #print(token.text, token.tag_,token.head)
        #print(token.head.lefts)
        #print(token.head.rights)
        #print(token.head)
        for each_token in token.head.lefts:
            if each_token.dep_ == 'amod' and each_token.tag_ == 'NN':
                drug_disease_pair2.append(each_token.text)        
        for each_token in token.head.rights:
            # print(each_token.text, each_token.tag_, each_token.dep_)
            # print(each_token.head.text,each_token.head.tag_,each_token.head.dep_)
            #print([x for x in each_token.children])
            if each_token.tag_== 'NN' and each_token.head.tag_ == 'IN':
                #print(each_token)
                child  = [x for x in each_token.children if x.dep_ == 'appos']
                if child:
                    drug_disease_pair2.append(each_token.text)
                    

'''Ner model training using spaCy'''

from __future__ import unicode_literals, print_function

import plac
import random
import spacy
from spacy.util import minibatch,compounding
import en_core_web_sm

labels = ['DRUG', 'DISEASE']

    
TRAIN_DATA = [
    (
        'ABILIFY is indicated for the treatment of schizophrenia in adults and in adolescents aged 15 years and older without any history of myocardial infractions.', 
        {'entities': [(0, 6, 'DRUG'), (42, 54, 'DISEASE')]}
    ),
    
    (
        'Pantoprazole sodium delayed-release tablets are indicated in adults and pediatric patients five years of age and older for the short-term treatment (up to 8 weeks) in the healing and symptomatic relief of erosive esophagitis (EE).', 
        {'entities': [(0, 11, 'DRUG'), (205, 223, 'DISEASE')]}
    )
]



model = None
model_name = 'Drug_disease'
output_dir = None
n_iter = 10

nlp = spacy.load('en_core_web_sm')

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

nlp.vocab.vectors.name = 'spacy_pretrained_vectors'

if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()
    

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
#         optimizer = nlp.begin_training() # made changes above for the same
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                       losses=losses)
        print(losses)   
    

test_text = 'sample.txt'
output_dir = 'C:\\Users\\file'
if output_dir is not None:
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


    # test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = nlp2('ABILIFY is a drug to treat schizophrenia disease.')
for ent in doc2.ents:
    print(ent)
    print(ent.label_, ent.text)

# for use of entities from spacy we have to train it much data  
# Iam using entities generated by nltk pretrained library
    
    
''' Model building using bi-directinal lstm'''   

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import model_from_json
import pandas as pd
import numpy as np

with open('sample_drug_disease.txt',encoding = 'utf-8') as f:
   data = f.readlines(1000)
   
text = []
for each in data:
    text.append(each.split())

features = []

for each_list in text:
    try:
        if len(each_list) == 2:
            features.append(each_list[0])
    except:
        continue

labels = []

for each_list in text:
    try:
        if len(each_list) == 2:
            labels.append(each_list[1])
    except:
        continue
   
features = np.array(features)
labels = np.array(labels)

from keras.preprocessing.text import Tokenizer

embedding_size  = 100            
num_cells = 64

tokenizer = Tokenizer()

tokenizer.fit_on_texts(features)

word_index = tokenizer.word_index

vocabulary_size = len(word_index) +1

print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_index)
print(tokenizer.word_docs)

encode_features = tokenizer.texts_to_matrix(features, mode = 'count')

print(encode_features)

le = LabelEncoder()
encode_labels = le.fit_transform(labels)
encode_labels = to_categorical(encode_labels, dtype = 'float64')

train_x,test_x,train_y,test_y = train_test_split(encode_features,encode_labels,test_size = 0.3,random_state = 42)

embeddings_index = {}

f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) +1, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


from keras import backend as K 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
 
model = Sequential()
model.add(Embedding(len(word_index) + 1,embedding_size,trainable = True, mask_zero = True))
model.add(Bidirectional(LSTM(num_cells, dropout = 0.2, recurrent_dropout = 0.2)))
model.add(Dense(5, activation = 'sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])

model.summary()

train_y = train_y.squeeze()
model.fit(train_x, train_y,batch_size = 32,validation_split=0.3, epochs=3)

Y_test_pred = model.predict_classes(test_x)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

print('saved model!')

