import unittest
from model.reader import reads_data

class TestData(unittest.TestCase):

	def test_reads_data(self):
		reads_data(startS = 0, stopS=100, startB = 0, stopB=100)
		print 'Hello data!'
