import pprint
import unittest

from model.sk_train import *
from model.reader import PlainReader as rd

class TestEfficiency(unittest.TestCase):

	with open('config/nominal_eff.json') as f:
		efficiency_nominal = json.load(f)['efficiency_nominal']

	def test_eff_res(self):
		classifier = 'GBRT' # 'Ada' #'GBRT'
		doTrain = True
		print "Finished"

	def test_gives_the_same_efficiency(self):
		print "TEST : test_eff_res"

		classifier = 'GBRT' # 'Ada' #'GBRT'

		print 'Read training and test files...'
		training, weights, targets = rd.reads_data(0, 100, 0, 1000)

		print 'Sizes'
		print training.nbytes, weights.nbytes, targets.nbytes

		print 'Start training'
		if classifier == 'GBRT':
			result = trainGBRT(training, targets, weights)
		elif classifier == 'Ada':
			result = trainAdaBoost(training, targets, weights)
		elif classifier == 'RF':
			result = trainRandomForest(training, targets, weights)
		else:
			print 'ERROR: no valid classifier', classifier

		try:
			for i, j in zip(efficiency_nominal, result.efficiency_res[0]):
				self.assertEqual(round(i, 7), round(j, 7))
		except AssertionError:
			print result
			raise
		except:
			raise