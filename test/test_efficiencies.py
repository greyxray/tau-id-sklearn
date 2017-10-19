import unittest
from model.sk_train import *

class TestEfficiency(unittest.TestCase):

	def test_eff_res(self):
		classifier = 'GBRT' # 'Ada' #'GBRT'
		doTrain = True

		# print 'Read training and test files...'
		# training, weights, targets = reads_data(0, 10000, 0, 100000)

		# print 'Sizes'
		# print training.nbytes, weights.nbytes, targets.nbytes

		# if doTrain:
		# 	print 'Start training'

		# 	if classifier == 'GBRT':
		# 		clf = trainGBRT(training, targets, weights)
		# 	elif classifier == 'Ada':
		# 		clf = trainAdaBoost(training, targets, weights)
		# 	elif classifier == 'RF':
		# 		clf = trainRandomForest(training, targets, weights)
		# 	else:
		# 		print 'ERROR: no valid classifier', classifier
		print "Finished"
