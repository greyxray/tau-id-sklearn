import unittest

from model.reader import PlainReader as rd

class TestData(unittest.TestCase):

	def test_reads_data(self):
		sig_bkgr, weights, targets = rd.reads_data(startS = 0, stopS=100, startB = 0, stopB=100)
		
		self.assertEqual(sig_bkgr.size, 4600)
		self.assertEqual(weights.size, 200)
		self.assertEqual(targets.size, 200)

