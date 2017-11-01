import pprint
import unittest
import json

# Style: because `python -c "import this" | grep -i "explicit "`
from model.sk_train import Training

class TestEfficiency(unittest.TestCase):

	def setUp(self):
		# This is called before all the tests
		with open('config/nominal_eff.json') as f:
			self.efficiency_nominal = json.load(f)['efficiency_nominal']

	def test_gives_the_same_efficiency(self):
		# NB: Avoid printing any messages here 

		classifier = Training('GradientBoosting') # 'Ada' #'GBRT'
		classifier.readData(0, 100, 0, 1000)

		result = classifier.getTraining()

		# NB: The idea of assertions here is to tell to the testrunner/testsuit
		#     that the condition has failed. This will result in 'F' (fatal) output message
		#

		for i, j in zip(self.efficiency_nominal, result.efficiency_res[0]):
			self.assertEqual(round(i, 7), round(j, 7))


	def tearDown(self):
		# NB: This method is called after all the tests
		#     if you need to delete something or clean memory
		#     you have to do it here. This method can be deleted later,
		#     look here https://docs.python.org/2/library/unittest.html
		pass
		