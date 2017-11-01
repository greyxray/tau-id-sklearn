## Ad-hoc tau ID training with sklearn using ROOT trees as input
# Requires root_numpy https://github.com/rootpy/root_numpy
# Jan Steggemann 27 Aug 2015
import json
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn.cross_validation import train_test_split #cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve

# For model I/O
from sklearn.externals import joblib
from reader import PlainReader as rd 


class TrainingResult(object):

	def __init__(self, clf, efficiency_res):
		self.clf = clf
		self.efficiency_res = efficiency_res

	def __str__(self):
		return "TrainingResult:\n\tclf: {}\n\tefficiency_res:".format(self.clf is not None) + "{}".format(self.efficiency_res)

class Training(object):

	def __init__(self, name=None, training=None, weights=None, targets=None):
		self.name = name
		self.training = training
		self.weights = weights
		self.targets = targets
		self.algorithms = {
			'GradientBoosting': self.trainGBRT,
			'AdaBoost': self.trainAdaBoost,
			'RandomForest': self.trainRandomForest
		}

	@classmethod
	def trainRandomForest(klass, training_data, target, weights):
		clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=True, n_jobs=1, random_state=1, verbose=1, min_density=None, compute_importances=None)
		return klass.train(clf, training_data, target, weights)

	@classmethod
	def trainAdaBoost(klass, training_data, target, weights):
		clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=6, max_features='auto', min_density=None, min_samples_leaf=2, min_samples_split=2, random_state=1, splitter='best'), n_estimators=5000, learning_rate=0.0025, algorithm='SAMME.R')
		return klass.train(clf, training_data, target, weights)

	@classmethod
	def trainGBRT(klass, training_data, target, weights, learning_rate=0.01, max_depth=6, n_estimators=1000, subSample=0.5):
		clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=1, loss='deviance', verbose=1, subsample=subSample, max_features=0.4) #loss='exponential'/'deviance'
		 # loss='deviance', verbose=1, subsample=subSample)
		return klass.train(clf, training_data, target, weights)

	@staticmethod
	def train(clf, training_data, target, weights):
		print clf
		efficiency_res = []
		sumWeightsSignal = np.sum(weights * target)
		sumWeightsBackground = sum(weights * (1 - target))

		print 'Sum weights signal', sumWeightsSignal
		print 'Sum weights background', sumWeightsBackground

		aveWeightSignal = sumWeightsSignal / np.sum(target)
		print 'Average weight signal', aveWeightSignal

		aveWeightBG = sumWeightsSignal / np.sum(1 - target)
		print 'Average weight background', aveWeightBG

		nCrossVal = 2
		kf = KFold(len(training_data), nCrossVal, shuffle=True, random_state=1)

		print 'Cross-validation:', nCrossVal, 'folds'
		print "len(kf):", len(kf)
		for trainIndices, testIndices in kf:
			print 'Starting fold'
			efficiency_res.append([])
			d_train = training_data[trainIndices]
			d_test = training_data[testIndices]

			t_train = target[trainIndices]
			t_test = target[testIndices]

			w_train = weights[trainIndices]
			w_test = weights[testIndices]

			del training_data, target, weights, trainIndices, testIndices, kf

			# import pdb; pdb.set_trace()

			clf.fit(d_train, t_train, w_train)

			print 'Produce scores'
			scores = clf.decision_function(d_test)

			print 'Roc-curves'
			fpr, tpr, tresholds = roc_curve(t_test, scores, sample_weight=w_test)

			print 'dump'
			joblib.dump((fpr, tpr, tresholds), 'roc_vals.pkl')

			effs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

			for eff in effs:
				print 'Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)]
				efficiency_res[-1].append(fpr[np.argmax(tpr>eff)])
			print "break"
			break

		# Can save with different features if necessary
		print "Can save with different features if necessary"
		joblib.dump(clf, 'train/{name}_clf.pkl'.format(name=clf.__class__.__name__), compress=9)

		# if doCrossVal:
		print 'Feature importances:'
		print clf.feature_importances_

		varList = rd.trainVars
		for i, imp in enumerate(clf.feature_importances_):
			print imp, varList[i] if i<len(varList) else 'N/A'

		return TrainingResult(clf, efficiency_res)#(clf, efficiency_res)

	def readData(self, startS=None, stopS=None, startB=None, stopB=None):
		self.training, self.weights, self.targets = rd.reads_data(startS, stopS, startB, stopB)

	def getJobLib(self, name=None):
		print 'Loading classifier'
		if name is None: 
			name = self.name

		return joblib.load('train/' + name + 'Classifier_clf.pkl')

	def getTraining(self, name=None):
		print 'Start training'
		if name is None: 
			name = self.name

		algorithm = self.algorithms.get(name)

		if not algorithm:
			raise KeyError("getTraining: No such algorithm " + name)

		return algorithm(self.training, self.targets, self.weights)


	def __str__(self):
		return "Training:\n\name:" + name + " \n\tlength training: {}".format(len(self.training)) + " \n\tlength targets: {}".format(len(self.targets)) + " \n\tlength weights: {}".format(len(self.weights))

