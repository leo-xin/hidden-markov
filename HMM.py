import numpy as np

class HMM(object):
	
	def __init__(self,initProb=None,transMatrix=None,genMatrix=None):
		if genMatrix!=None:
			genMatrix=np.array(genMatrix)
			if len(genMatrix.shape)!=2:
				raise ValueError("demension not 2")
			rowSum=genMatrix.sum(axis=1)
			if all(rowSum!=1):
				raise ValueError(" ")
		if transMatrix!=None:
			transMatrix=np.array(transMatrix)
			if len(transMatrix.shape)!=2:
				raise ValueError("demension not 2")
			rowSum=transMatrix.sum(axis=1)
			if all(rowSum!=1):
				raise ValueError(" ")
		if initProb!=None:
			initProb=np.array(initProb)
			if len(initProb.shape)>1:
				raise ValueError(" ")
		self.initProb=initProb
		self.transMatrix=transMatrix
		self.genMatrix=genMatrix
		
	def forward_calc_prob(self,observe):
		T=len(observe)
		alphaList=self.initProb*self.genMatrix[:,observe[0]]
		for k in range(1,T):
			alphaList=alphaList.dot(self.transMatrix)*self.genMatrix[:,observe[k]]
		return alphaList.sum()
	
	def backward_calc_prob(self,observe):
		T=len(observe)
		betaList=np.ones(T)
		for k in range(T-1,0,-1):
			betaList=self.transMatrix.dot(self.genMatrix[:,observe[k]]*betaList)
		return (betaList*self.genMatrix[:,observe[0]]*self.initProb).sum()
		
	def viterbi_decoding(self,observe):
		if self.initProb is None or self.transMatrix is None or self.genMatrix is None:
			raise ValueError("uncomplete input parameter")
		T=len(observe)
		N=self.transMatrix.shape[0] 
		epsilon=(self.initProb*self.genMatrix[:,observe[0]]).reshape((N,1))
		pathTracking=np.zeros((N,1))
		for t in range(1,T):
			eps=self.transMatrix*epsilon
			epsilon=(np.amax(eps,axis=0)*self.genMatrix[:,observe[t]]).reshape((N,1))
			path=np.argmax(eps,axis=0).reshape((N,1))
			pathTracking=np.hstack((pathTracking,path))
		maxProb=np.amax(epsilon,axis=0)
		maxProbInd=np.argmax(epsilon,axis=0)[0]
		maxProbPath=pathTracking[maxProbInd,:][1:].tolist()
		maxProbPath.append(maxProbInd)
		return maxProbPath