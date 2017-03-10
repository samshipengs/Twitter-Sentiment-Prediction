#!/usr/bin/python
# This script loads data, convert them to vectors and save them to disk.

# load libraries
import pandas as pd
import numpy as np
from gensim.models import word2vec
import logging
import nltk
from collections import Counter
import itertools
from nltk.corpus import stopwords
import os.path


# FILE_PATH = '/home/sam/Hhd/twitter_sentiment/'
FILE_PATH = '/home/sam/Data/twitter_sentiment/'

class Data:
	def __init__(self, file_name, file_path):
		self.file_name = file_name
		self.FILE_PATH = file_path
		self.data_path = file_path+'data/'

	# def function to load data from json to dataframe
	def json_df(self, json_fields):
	    print "Loading json: " + self.file_name + " ..."
	    # data_path = FILE_PATH + 'data/'
	    data_df = pd.read_json(self.data_path + self.file_name, lines=True)
	    # we only take the 'text' column
	    drop_columns = list(data_df.columns)
	    # drop_columns.remove('text')
	    for i in json_fields:
	    	drop_columns.remove(i)
	    data_df.drop(drop_columns, axis = 1, inplace = True)
	    data_df.dropna(axis=0, inplace=True) # drop na rows
	    print "Done loading json file to dataframe."
	    return data_df

	def csv_df(self, csv_fields):
		print "Loading csv: " + self.file_name + " ..."
		data_df = pd.read_csv(self.data_path + self.file_name)
		data_df =  data_df[csv_fields]
		data_df.dropna(axis=0, inplace=True) # drop na rows
		return data_df

	# pre-processing text
	def pre_process(self, df):
		print("Note: pre_process changes the dataframe inplace.")
		# remove new line char
		df['text'].replace(regex=True,inplace=True,to_replace=r'\\n',value=r'')
		# remove https links
		df['text'].replace(regex=True,inplace=True,to_replace=r'(http|https):\/\/[^(\s|\b)]+',value=r'')
		# remove user name
		df['text'].replace(regex=True,inplace=True,to_replace=r'@\w+',value=r'')
		# remove non-alphabet, this includes number and punctuation
		df['text'].replace(regex=True,inplace=True,to_replace=r'[^a-zA-Z\s]',value=r'')
		# tokenize each tweets to form sentences.
		df['tokenized'] = df['text'].apply(lambda row: nltk.word_tokenize(row.lower()))
		# remove stop words
		stop_words = stopwords.words('english')
		add_stop_words = ['amp', 'rt']
		stop_words += add_stop_words
		#     print "sample stopping words: ", stop_words[:5]
		df['tokenized'] = df['tokenized'].apply(lambda x: [item for item in x if item not in stop_words])

		# now let us bring in the wordvec trained using text8 dataset
	def build_wordvec(self, model_name='tweets.model.bin', size = 200):
		self.vec_size = size
		sentences = word2vec.Text8Corpus(self.FILE_PATH + 'data/text8') # use text 8
		model_path = self.FILE_PATH + 'wordvec/' + model_name
		if os.path.isfile(model_path):
			print "Loading existing model {} ...".format(model_name)
			model = word2vec.Word2Vec.load(model_path)
		else:
			logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
			print "Training for {} ...".format(model_name)
			model = word2vec.Word2Vec(sentences, size=self.vec_size, sg=1, workers=4)
			model.save(model_path)
			# If you finished training a model (no more updates, only querying), you can do
			model.init_sims(replace=True)
		print "Done building."
		return model

	# transform our tweets using vector representation
	# first find the max length since that decides the padding
	def max_len(self, df):
		df['size'] = df['tokenized'].apply(lambda x: len(x))
		print "max sentence length is: ", df['size'].max()
		return df['size'].max() 

	# initialize empty arry to fill with vector repsentation
	def convert2vec(self, df, max_length, model):
		tweet_tokens = df['tokenized']
		n = tweet_tokens.shape[0]
		m = max_length
		n_absent = 0
		tweet_vecs = np.zeros((n,m,self.vec_size))
		vocabs = model.wv.vocab.keys()
		for i in range(n):
			token_i = [x for x in tweet_tokens[i] if x in vocabs]
			m_i = len(token_i)
			if m_i == 0:
			    n_absent += 1
			else:
				diff_i = abs(m_i - m)
				vecs_i = model[token_i]
				tweet_vecs[i] = np.lib.pad(vecs_i, ((0,diff_i),(0,0)), 'constant', constant_values=0)
		print "Done converting tweets to vec!"
		print "Total {} not in vocab.".format(n_absent)
		return tweet_vecs

	# save tweet_vecs to disk in npy
	def save_vec(self, tweet_vecs, name):
		file_name = self.FILE_PATH + name
		if os.path.isfile(file_name + '.npy') and os.path.isfile(file_name + '.npz'):
			print "npy already exists."
		else:
			np.save(file_name, tweet_vecs)
			np.savez(file_name, tweet_vecs)
			print "Saved {} to disk.".format(name)
