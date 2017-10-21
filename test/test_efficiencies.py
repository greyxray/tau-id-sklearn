import pprint
import unittest
import json

from model.sk_train import *

class TestEfficiency(unittest.TestCase):

	with open('config/nominal_eff.json') as f:
		efficiency_nominal = json.load(f)['efficiency_nominal']

	def test_gives_the_same_efficiency(self):
		print "TEST : test_gives_the_same_efficiency"

		# I don't know why above is not working anymore
		with open('config/nominal_eff.json') as f:
			efficiency_nominal = json.load(f)['efficiency_nominal']

		doTrain = True
		doTest = False

		classifier = Training('GradientBoosting') # 'Ada' #'GBRT'
		classifier.readData(0, 100, 0, 1000)

		if doTrain:
			result = classifier.getTraining()
			try:
				for i, j in zip(efficiency_nominal, result.efficiency_res[0]):
					self.assertEqual(round(i, 7), round(j, 7))
			except AssertionError:
				print result
				raise
			except:
				raise

		if doTest: clf = classifier.getJobLib()