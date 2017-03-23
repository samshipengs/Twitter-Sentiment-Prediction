# This file makes predictions on incoming data and perform basic analysis.
import pandas as pd
import lasagne
import theano
import theano.tensor as T

import nltk
from load_data import Data
from utils import *

import sys
import os.path

# def feed forward pass ftns until dense layer to generate features for svm
def generate_features(net):
    dense_layer = layers.get_output(net.layers_['dense'], deterministic=True)
    output_layer = layers.get_output(net.layers_['output'], deterministic=True)
    input_var = net.layers_['input'].input_var

    f_output = theano.function([input_var], output_layer)
    f_dense = theano.function([input_var], dense_layer)
    
    return f_dense, f_output

# convert
def extract_features(net, input_data):
    print "Extracting ... "
    # input_data: n0, 1, n2, n3
    n = input_data.shape
    f_dense = generate_features(net)[0]
    return [f_dense(i.reshape(1,1,n[2],n[3])).flatten() for i in input_data]


class Prediction:
	def __init__(self, file_path, file_name, max_len_train):
		self.file_path = file_path
		self.file_name = file_name
		self.max_len_train = max_len_train # length of sentence from training

	def prepare_data(self, wv_size=600):
		test_data = Data(self.file_name, self.file_path)
		test_df = test_data.csv_df(['text'])
		# make a copy of the original tweets for later use
		original_df = test_df.copy()

		# pre-process data(same as how we trained)
		test_data.pre_process(test_df) 

		# then convert using word2vec
		model = test_data.build_wordvec(size=wv_size, verbose=False)
		# take a look of the max_len of testing. although we still have to use max_len from train
		max_len_test = test_data.max_len(test_df)
		data = test_data.convert2vec(test_df, self.max_len_train, model, name='test_'+self.file_name)
		test_data.save_vec(data, name='test_'+self.file_name)

		self.data = data
		self.test_data = test_data
		self.test_df = test_df
		self.original_df = original_df
		print ">>>Done preparing data.<<<\n"

	def make_prediction(self, data, verbose=True):
		N, M, D = data.shape
		if verbose:
			print "N, M, D:", N, M, D
		data = data.reshape(-1, 1, M, D).astype(theano.config.floatX) # theano needs this way

		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')
		network = self.cnn(M, D, input_var)

		# now load model and do predictions
		saved_params = load_network(self.file_path, "cnn.npz")
		lasagne.layers.set_all_param_values(network, saved_params)

		# define prediction function
		test_prediction = lasagne.layers.get_output(network, deterministic=True)
		predict_label = T.argmax(test_prediction,axis=1)
		test_fn = theano.function([input_var], predict_label)

		test_pred = test_fn(data) + 1

		self.test_pred = test_pred
		return test_pred

	def get_result(self, n_preview=10, n_top = 20, name = 'default', verbose=True):

		# get predictions
		test_predictions = self.make_prediction(self.data, verbose)
		### Take a look at the predictions with raw tweets
		self.test_df['prediction'] = test_predictions
		# lets take a look of the 
		if verbose:
			print self.test_df['prediction'].value_counts()

		# write to original dataframe
		self.original_df['prediction'] = test_predictions
		# convert numeric prediction to categorical
		class_label = {1:'positive', 2: 'neutral', 3: 'negative'}
		self.original_df = self.test_data.num2cat(self.original_df, 'prediction', class_label)
		if verbose:
			# take a quick look at the prediction and its corresponding tweet
			for i in range(n_preview):
				print self.original_df.values[i,]

		# save to csv
		self.original_df.to_csv(name+'.csv')

		# look at most frequent words in different groups
		print "===Positive==="
		print most_freq(self.test_df, 1, top=n_top)
		print "===Neutral==="
		print most_freq(self.test_df, 2)
		print "===Negative==="
		print most_freq(self.test_df, 3, top=n_top)

	def check(self, word, sentiment, n_view=10):
		# we can take a look of the tweets where the frequent word is mentioned, e.g. lookup 'gate' in negative tweets
		look_up(self.original_df, self.test_df, word, sentiment, look=n_view)