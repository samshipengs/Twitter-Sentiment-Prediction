require 'torch'
require 'paths'
npy4th = require 'npy4th'


-- read a .npy file into a torch tensor
FILE_PATH = '/home/sam/Data/twitter_sentiment/'
array = npy4th.loadnpy(FILE_PATH .. 'tweet_vecs.npy')
data_table = npy4th.loadnpz(FILE_PATH .. 'tweet_vecs.npz')