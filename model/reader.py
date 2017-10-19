import numpy as np
from root_numpy import root2array, root2rec

def trainVars():
	return [
		'TMath::Log(TMath::Max(1., recTauPt))',
		'TMath::Abs(recTauEta)',
		'TMath::Log(TMath::Max(1.e-2, chargedIsoPtSum))',
		'TMath::Log(TMath::Max(1.e-2, neutralIsoPtSum))',
		'TMath::Log(TMath::Max(1.e-2, puCorrPtSum))',
		'TMath::Log(TMath::Max(1.e-2, photonPtSumOutsideSignalCone))',
		'recTauDecayMode',
		'TMath::Min(30., recTauNphoton)',
		'TMath::Min(0.5, recTauPtWeightedDetaStrip)',
		'TMath::Min(0.5, recTauPtWeightedDphiStrip)',
		'TMath::Min(0.5, recTauPtWeightedDrSignal)',
		'TMath::Min(0.5, recTauPtWeightedDrIsolation)',
		'TMath::Min(100., recTauLeadingTrackChi2)',
		'TMath::Min(1., recTauEratio)',
		'TMath::Sign(+1., recImpactParam)',
		'TMath::Sqrt(TMath::Abs(TMath::Min(1., TMath::Abs(recImpactParam))))',
		'TMath::Min(10., TMath::Abs(recImpactParamSign))',
		'TMath::Sign(+1., recImpactParam3D)',
		'TMath::Sqrt(TMath::Abs(TMath::Min(1., TMath::Abs(recImpactParam3D))))',
		'TMath::Min(10., TMath::Abs(recImpactParamSign3D))',
		'hasRecDecayVertex',
		'TMath::Sqrt(recDecayDistMag)',
		'TMath::Min(10., recDecayDistSign)'
	]

def reads_data(startS = None, stopS=None, startB = None, stopB=None):
	print 'Reading files...'
	if stopS and startS == None: startS = 0
	if stopB and startB == None: startB = 0

	# see: http://scikit-hep.org/root_numpy/reference/generated/root_numpy.root2rec.html
	weightsS = root2array(filenames = '/nfs/dust/cms/user/glusheno/TauIDMVATraining2016/Summer16_25ns_V1_allPhotonsCut/tauId_v3_0/trainfilesfinal_v1/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt1aLTDB_photonPtSumOutsideSignalConePtGt0p5_signal.root', treename = 'reweightedTauIdMVATrainingNtuple', branches = ['evtWeight'], start = startS, stop = stopS).view()
	weightsB = root2array(filenames = '/nfs/dust/cms/user/glusheno/TauIDMVATraining2016/Summer16_25ns_V1_allPhotonsCut/tauId_v3_0/trainfilesfinal_v1/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt1aLTDB_photonPtSumOutsideSignalConePtGt0p5_background.root', treename = 'reweightedTauIdMVATrainingNtuple', branches = ['evtWeight'], start = startB, stop = stopB).view()
	fullWeight = np.concatenate((weightsS, weightsB))
	fullWeight = fullWeight['evtWeight']

	nS = len(weightsS)
	nB = len(weightsB)
	del weightsS, weightsB

	arS = root2array(filenames = '/nfs/dust/cms/user/glusheno/TauIDMVATraining2016/Summer16_25ns_V1_allPhotonsCut/tauId_v3_0/trainfilesfinal_v1/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt1aLTDB_photonPtSumOutsideSignalConePtGt0p5_signal.root', treename = 'reweightedTauIdMVATrainingNtuple', branches = trainVars(), start = startS, stop = stopS)
	arB = root2array(filenames = '/nfs/dust/cms/user/glusheno/TauIDMVATraining2016/Summer16_25ns_V1_allPhotonsCut/tauId_v3_0/trainfilesfinal_v1/reweightTreeTauIdMVA_mvaIsolation3HitsDeltaR05opt1aLTDB_photonPtSumOutsideSignalConePtGt0p5_background.root', treename = 'reweightedTauIdMVATrainingNtuple', branches = trainVars(), start = startB, stop = stopB)
	arrSB = np.concatenate((arS, arB)) # deprecated but faster would be hstack
	del arS, arB

	# Need a matrix-like array instead of a 1-D array of lists for sklearn
	arrSB = (np.asarray([arrSB[var] for var in trainVars()])).transpose()
	print "arrSB.shape:", arrSB.shape

	targets = np.concatenate((np.ones(nS), np.zeros(nB)))
	return arrSB, fullWeight, targets
