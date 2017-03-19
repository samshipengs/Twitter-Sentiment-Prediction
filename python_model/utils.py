# Provide save model and load model and make predictions function
import numpy as np
import lasagne


def save_network(filepath, filename, network):
	print "Saving {}.npz to disk ...".format(filename)
	np.savez(filepath + 'model/' + filename, *lasagne.layers.get_all_param_values(network))
	print "Done saving."


def load_network(filepath, filename):
	with np.load(filepath + 'model/' + filename) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	return param_values