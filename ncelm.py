#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:12:41 2018

@author: arnav
"""

from __future__ import division
import pickle, random, string
import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as L
from collections import Counter

sequenceLen = 50
batchSz = 32
maxIter = 1000
neuralNetworkSz = 512
dropOutProbability = 0.1

numWords = 10
vocabularySize = 10000

K = 10
Z = pow(np.e, 9)

gradientNormClip = True
maxGradientNorm = 15
summaryFreq = 10
valueFreq = 50
loadTrainingModel = False
saveTrainingModel = True
trainingModelPath = 'models/RNN_training_model.pkl'


class NCE(L.DenseLayer):
    
    def __init__(self, inputConnections, num_units, Z, W = lasagne.init.GlorotUniform(), b = lasagne.init.Constant(0.), **kwargs):
        super(L.DenseLayer, self).__init__(inputConnections, **kwargs)
        self.num_units = num_units
        numInputs = int(np.prod(self.input_shape[1:]))
        self.W = self.add_param(W, (numInputs, num_units), name = "W")
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units, ), name = "b", regularizable = False)
            
            
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
    
    
    def get_output_for(self, input, **kwargs):
        #if more than 2 dimensions, flatten it out
        if input.ndim > 2:
            input = input.flatten(2)
            
        activateVal = T.dot(input, self.W)
        
        if self.b is not None:
            activateVal = activateVal + self.b.dimshuffle('x', 0)
        
        return T.exp(activateVal)/Z
    
    
class RandomNoiseDistribution:
    
    def __init__(self, freq, vocabulary):
		self.dist = {}
		
		Sum = np.sum([i[1] for i in freq])
		for i in range(len(freq)):
			self.dist[vocabulary[freq[i][0]]] = freq[i][1]/Sum
			
		self.npDistance = np.array(self.dist.values())
		self.npDistance = T.reshape(self.npDistance, (vocabularySize, ))

	
    #gives the random words in the sample with the probability of their occurence
    def sample(self, k):
		arr = np.random.choice(self.dist.keys(), k, p = self.dist.values())
		return ([self.dist[wd] for wd in arr], arr)		
		

#gives sequential parts of the corpus, starts at random points, wraps around data ending
def produceDataBatch(corpus, sz = batchSz):
    beginIndex = np.random.randint(0, len(corpus)-sequenceLen-1, size = sz)
    while True:
		items = np.array([corpus[i:i+sequenceLen+1] for i in beginIndex])
		beginIndex = (beginIndex+sequenceLen)%(len(corpus)-sequenceLen-1)
		yield items
		

#one-hot encoding after sampling and make target sequence shifting one character
def prepareDataBatch(batch, vocabulary, vocabularySize, sequenceLen):
	sequenceX = np.zeros((len(batch), sequenceLen), dtype = 'int32')
	sequenceY = np.zeros((len(batch), sequenceLen), dtype = 'int32')

	for i, item in enumerate(batch):
		for j in range(sequenceLen):
			if item[j] in vocabulary.keys():
				sequenceX[i, j] = vocabulary[item[j]]
			else:
				sequenceX[i, j] = vocabulary['UNK']
				
			if item[j+1] in vocabulary.keys():
				sequenceY[i, j] = vocabulary[item[j+1]]
			else:
				sequenceY[i, j] = vocabulary['UNK']
				
	return sequenceX, sequenceY


def makeRNN(xInputRNN, hiddenInitRNN, hidden2InitRNN, sequenceLen, vocabularySize, neuralNetworkSz):

	input_Layer = L.InputLayer(input_var = xInputRNN, shape = (None, sequenceLen))
	hidden_Layer = L.InputLayer(input_var = hiddenInitRNN, shape = (None, neuralNetworkSz))
	hidden_Layer2 = L.InputLayer(input_var = hidden2InitRNN, shape = (None, neuralNetworkSz))
	input_Layer = L.EmbeddingLayer(input_Layer, input_size = vocabularySize, output_size = neuralNetworkSz)

	RNN_Layer = L.LSTMLayer(input_Layer, num_units = neuralNetworkSz, hid_init = hidden_Layer)
	h = L.DropoutLayer(RNN_Layer, p = dropOutProbability)
	RNN_Layer2 = L.LSTMLayer(h, num_units = neuralNetworkSz, hid_init = hidden_Layer2)
	h = L.DropoutLayer(RNN_Layer2, p = dropOutProbability)

	layerShape = L.ReshapeLayer(h, (-1, neuralNetworkSz))
	
	predictions = NCE(layerShape, num_units = vocabularySize, Z = Z)
	predictions = L.ReshapeLayer(predictions, (-1, sequenceLen, vocabularySize))
	return RNN_Layer, RNN_Layer2, predictions


def preProcess(corpus):
	dataSet = corpus.split(' ') #corpus is a single text string, should be split into words 
	dataSet = [''.join(j for j in i if j.isalpha() or j in string.whitespace) for i in dataSet]
	dataSet = [i.rstrip() for i in dataSet] #remove '\t' and '\n'
	dataSet = [i.lower() for i in dataSet]
	dataSet = [i.replace('\n','') for i in dataSet]
	dataSet = [i for i in dataSet if i != ''] #remove strings with 0 length
	return dataSet


def makeIndex(n, m):
	idx = np.arange(n, dtype = 'int32')
	idx = np.stack([idx for i in range(m)], axis = 1)
	idx = np.ndarray.flatten(idx)
	return idx


def main():
	print "Dataset is being loaded..."
	corpus = open('google_reddit_chat.csv', 'r').read()
	print "Dataset is being processed..."
	corpus = preProcess(corpus)
	
	if loadTrainingModel:
		print "Vocabulary is being loaded..."
		Arr = pickle.load(open(trainingModelPath, 'r'))
		vocabulary = Arr['vocabulary']
	else:
		print "Vocabulary is being made..."
		freq = Counter(corpus)
		print "Total number of words: ", len(freq)
		freq = freq.most_common(vocabularySize-1) #vocabulary size is reduced by 1 to accomodate UNK token
        vocabulary = {}
        idx = 0
        
        for wd,_ in freq:
            vocabulary[wd] = idx
            idx += 1
            
        vocabulary['UNK'] = vocabularySize - 1
        freq.append(('UNK',20))
		
	noiseDistribution = RandomNoiseDistribution(freq, vocabulary)
		
	inv_vocabulary = {v:k for k, v in vocabulary.items()}

	trainingSet = corpus[:(len(corpus) * 9 // 10)]
	testingSet = corpus[(len(corpus) * 9 // 10):]
			
	xInputRNN = T.imatrix()
	yInputRNN = T.imatrix()
	hiddenInitRNN = T.matrix()
	hidden2InitRNN = T.matrix()
	initRNN = T.scalar()
	noiseWordIdx = T.ivector()
	batchIdx = T.ivector()
	
	print "Model is being built..."
	RNN_Layer, RNN_Layer2, outLayer = makeRNN(xInputRNN, hiddenInitRNN, hidden2InitRNN, sequenceLen, vocabularySize, neuralNetworkSz)
	
	#get hidden state of each layer of RNN, because only that is required at the last time step
	outHidden, outHidden2, outProbability = L.get_output([RNN_Layer, RNN_Layer2, outLayer])

	outHidden = outHidden[:, -1]
	outHidden2 = outHidden2[:, -1]

	batchIdx = makeIndex(batchSz,sequenceLen)
	sequenceIdx = makeIndex(sequenceLen, batchSz)	

	initRNN = outProbability[batchIdx, sequenceIdx, T.flatten(yInputRNN)] #(batchSz)
	initRNN = T.reshape(initRNN, (batchSz, sequenceLen))

	Pn = noiseDistribution.npDistance[T.flatten(yInputRNN)]
	Pn = T.reshape(Pn,(batchSz,sequenceLen))
	Pc_RNN = initRNN/(initRNN + K*Pn) #(batchSz)
	
	batchIdx = makeIndex(batchSz, sequenceLen*K)
	sequenceIdx = makeIndex(sequenceLen, batchSz*K)
	
	noiseSamples = noiseDistribution.sample(batchSz*sequenceLen*K)
	Pn_wd_i_j = np.array(noiseSamples[0], dtype = 'float32') #(batchSz*K)
	Pn_wd_i_j = T.reshape(Pn_wd_i_j, (batchSz, sequenceLen, K))
	noiseWordIdx = np.array(noiseSamples[1], dtype = 'int32') #(batchSz*K)
	Pn_wd_i_j *= K
	
	Pnce_wd_i_j = outProbability[batchIdx, sequenceIdx, noiseWordIdx]
	Pnce_wd_i_j = T.reshape(Pnce_wd_i_j,(batchSz, sequenceLen, K))

	Pcn_arrayList = Pn_wd_i_j/(Pnce_wd_i_j + Pn_wd_i_j) #(batchSz, K)
	
	totalLoss = -(T.log(Pc_RNN) + T.sum(T.log(Pcn_arrayList), axis = (2)))
	totalLoss = T.mean(totalLoss)

	parameters = L.get_all_params(outLayer, trainable = True)
	gradients = T.grad(totalLoss, parameters)
	
	if gradientNormClip:
		gradients = [T.clip(i, -5, 5) for i in gradients]
		gradients, norm = lasagne.updates.total_norm_constraint(gradients, maxGradientNorm, return_norm = True)

	upd = lasagne.updates.adam(gradients, parameters)

	trainMethod = theano.function([xInputRNN, yInputRNN, hiddenInitRNN, hidden2InitRNN], [totalLoss, outHidden, outHidden2], updates = upd, on_unused_input = 'warn')

	testMethod = theano.function([xInputRNN, yInputRNN, hiddenInitRNN, hidden2InitRNN], [totalLoss, outHidden, outHidden2], on_unused_input = 'warn')

	hidden = np.zeros((batchSz, neuralNetworkSz), dtype = 'float32')
	hidden2 = np.zeros((batchSz, neuralNetworkSz), dtype = 'float32')

	#every iter = rand subSeq with (number of words = sequenceLen)
	trainingBatch = produceDataBatch(trainingSet)
	testingBatch = produceDataBatch(testingSet)
	
	if loadTrainingModel:
		print "Model is being loaded..."
		arr = pickle.load(open(trainingModelPath, 'r'))
		L.set_all_param_values(outLayer, arr['param values'])

	trainingLoss = []
	for i in range(maxIter):
		x, y = prepareDataBatch(next(trainingBatch), vocabulary, vocabularySize, sequenceLen)
		trainLoss, _, _ = trainMethod(x, y, hidden, hidden2) #hidden states getting updated
		trainingLoss.append(trainLoss)

		if i % summaryFreq == 0:
			print 'Iter: {}\tTrain Error: {}'.format(i, np.mean(trainingLoss))
			trainingLoss = []
			
		if i % valueFreq == 0 and i > 0:
			x, y = prepareDataBatch(next(testingBatch), vocabulary, vocabularySize, sequenceLen)
			testLoss, _, _ = testMethod(x, y, hidden, hidden2)
			print 'Test Error: {}'.format(testLoss)

			param_values = L.get_all_param_values(outLayer)
			Brr = {'param values': param_values, 'vocabulary': vocabulary, }
			
			if saveTrainingModel:
				path = "models/RNN_training_model.pkl"
				pickle.dump(Brr, open(path, 'w'), protocol = pickle.HIGHEST_PROTOCOL)

	predictionMethod = theano.function([xInputRNN, hiddenInitRNN, hidden2InitRNN], [outProbability, outHidden, outHidden2])

	hidden = np.zeros((batchSz, neuralNetworkSz), dtype = 'float32')
	hidden2 = np.zeros((batchSz, neuralNetworkSz), dtype = 'float32')

	#RNN is built with sequenceLen = 1 for making the process of sampling faster
	RNN_Layer, RNN_Layer2, outLayer = makeRNN(xInputRNN, hiddenInitRNN, hidden2InitRNN, 1, vocabularySize, neuralNetworkSz)
												
	outHidden, outHidden2, outProbability = L.get_output([RNN_Layer, RNN_Layer2, outLayer])
	
	outHidden = outHidden[:, -1]
	outHidden2 = outHidden2[:, -1]
	outProbability = outProbability[0, -1]

	L.set_all_param_values(outLayer, Brr['param values'])

	predictionMethod = theano.function([xInputRNN, hiddenInitRNN, hidden2InitRNN], [outProbability, outHidden, outHidden2])

	#one char at a time given to the RNN for primimg. o/p prob distribution is sampled at every timestep to get a sample str. give the selected char to the RNN and terminate at 1st break-of-line.
	hidden = np.zeros((1, neuralNetworkSz), dtype = 'float32')
	hidden2 = np.zeros((1, neuralNetworkSz), dtype = 'float32')
	x = np.zeros((1, 1), dtype = 'int32')

	#random strings from testing set
	start = random.randint(0, len(testingSet))
	primer = testingSet[start: min(len(testingSet)-1, start+numWords)]

	#giving the primer as input to the RNN
	for i in primer:
		prob, hidden, hidden2 = predictionMethod(x, hidden, hidden2)
		if i in vocabulary.keys():
			x[0, 0] = vocabulary[i]
		
	#create a new str of a fixed size
	str = ''
	for _ in range(50):
		prob, hidden, hidden2 = predictionMethod(x, hidden, hidden2)
		prob = prob/(1 + 1e-6)
		
		prob /= np.sum(prob) #Normalization
		
		st = np.random.multinomial(1,prob)
		str += inv_vocabulary[st.argmax(-1)] + ' '
		x[0, 0] = st.argmax(-1)
			
	print 'Primer: ' + ' '.join(primer)
	print 'Generated: ' + str


if __name__ == "__main__":
	main()