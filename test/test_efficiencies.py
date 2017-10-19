import pprint
import unittest

from model.sk_train import *
from model.reader import PlainReader as rd

class TestEfficiency(unittest.TestCase):

	def test_eff_res(self):
		classifier = 'GBRT' # 'Ada' #'GBRT'
		doTrain = True

		# print 'Read training and test files...'
		# training, weights, targets = rd.reads_data(0, 10000, 0, 100000)

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


	def test_gives_the_same_efficiency(self):
		print "TEST : test_eff_res"
		# TODO: Move these values to the config/nominal_eff.json
		# TODO: remove all extra code from here
		efficiency_nominal = [0.00223910435826, 0.00437824870052, 0.00791683326669, 0.0127548980408, 0.0210115953619, 0.03168732507, 0.0534986005598, 0.104118352659]
		classifier = 'GBRT' # 'Ada' #'GBRT'
		doTrain = True

		print 'Read training and test files...'
		training, weights, targets = rd.reads_data(0, 10000, 0, 100000)

		print 'Sizes'
		print training.nbytes, weights.nbytes, targets.nbytes

		if doTrain:
			print 'Start training'

			if classifier == 'GBRT':
				result = trainGBRT(training, targets, weights)
			elif classifier == 'Ada':
				result = trainAdaBoost(training, targets, weights)
			elif classifier == 'RF':
				result = trainRandomForest(training, targets, weights)
			else:
				print 'ERROR: no valid classifier', classifier
		if len(result.efficiency_res) > 0:
			if len([i for i, j in zip(efficiency_nominal, result.efficiency_res[0]) if i - j > 0.000001]) == 0:
				print "\n\n", "="*10,"\nThe first instance is the same as nominal"
			else:
				print "\n\n", "="*10,"\nNot identical. \n\tefficiency_nominal :\n", efficiency_nominal, "\n\tresult.efficiency_res :"
				pp = pprint.PrettyPrinter(indent=4)
				pp.pprint(result.efficiency_res)
		else:
			print "\n\n", "="*10,"\nno resulting efficiencies found"
