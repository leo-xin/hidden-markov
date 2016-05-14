from HMM import HMM

initProb=[0.2,0.4,0.4]
transMatrix=[[0.5,0.2,0.3],
			 [0.3,0.5,0.2],
			 [0.2,0.3,0.5]]
genMatrix=[[0.5,0.5],
		   [0.4,0.6],
		   [0.7,0.3]]
hmm=HMM(initProb,transMatrix,genMatrix)
observe=[0,1,0]
prob1=hmm.forward_calc_prob(observe)
prob2=hmm.backward_calc_prob(observe)
path=hmm.viterbi_decoding(observe)
print(prob1,prob2)
print(path)