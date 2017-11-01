from model.sk_train import Training

if __name__ == '__main__':
	train, test = True, False

	classifier = Training('GradientBoosting') # 'Ada' #'GBRT'
	classifier.readData(0, 100, 0, 1000)

	if train:
		result = classifier.getTraining()

	if test:
		clf = classifier.getJobLib()