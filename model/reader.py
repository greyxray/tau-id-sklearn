import json
import numpy as np
from root_numpy import root2array, root2rec


class PlainReader(object):

	with open('config/variables.v.1.json') as f:
		trainVars = json.load(f)['variables']

	def __init__(self):
		super(PlainReader, self).__init__()

	@classmethod
	def reads_data(klass, startS = None, stopS=None, startB = None, stopB=None,
			filenameroot = 'data/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt1aLTDB_photonPtSumOutsideSignalConePtGt0p5',
			tree = 'reweightedTauIdMVATrainingNtuple'
		):
		print 'Reading files...'

		# Read from files
		sdata, sweights = klass._data_weights('_signal.root', startS, stopS, filenameroot, tree)
		bdata, bweights = klass._data_weights('_background.root', startB, stopB, filenameroot, tree)

		# Join the data
		data = np.concatenate((sdata, bdata)) # deprecated but faster would be hstack
		# Need a matrix-like array instead of a 1-D array of lists for sklearn
		data = (np.asarray([data[var] for var in klass.trainVars])).transpose()

		# Join weights
		weights = np.concatenate((sweights, bweights))
		weights = weights['evtWeight']

		print "data.shape:", data.shape
		return data, weights, klass._targets(sweights ,bweights)

		
	@staticmethod
	def _targets(sweights, bweights):
		nS, nB  = len(sweights), len(bweights)
		targets = np.concatenate((np.ones(nS), np.zeros(nB)))
		return targets


	@classmethod
	def _data_weights(klass, cltype, start, stop, filenameroot, tree):
		if stop and start == None: 
			start = 0
		# see: http://scikit-hep.org/root_numpy/reference/generated/root_numpy.root2rec.html
		weights = root2array(filenames = filenameroot + cltype, treename = tree , branches = ['evtWeight'], start = start, stop = stop).view()
		data = root2array(filenames = filenameroot + cltype, treename = tree , branches = klass.trainVars, start = start, stop = stop)
		return data, weights
