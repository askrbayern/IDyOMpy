from idyom import data
from idyom import markovChain
from idyom import markovChainOrder0

import numpy as np
import pickle
from tqdm import tqdm
import math

VERBOSE = False	

class longTermModel():
	"""
	Module implementing the Long Term Model from IDyOM, this model contains several Markov Chains of different orders weighted by their respective shanon entropy.

	:param viewPoint: viewPoint to use, cf. data.getViewPoints()
	:param maxOrder: maximal order of the models
	:param alphabetSize(optional): size of the alphabet, number of viewPoints value to take account in

	:type viewPoint: string
	:type maxOrder: int
	:type alphabetSize(optional): int
	"""

	def __init__(self, viewPoint, maxOrder=None, STM=False, init=None, evolutive=False, use_original_PPM=False):

		# ViewPoint to use
		self.viewPoint = viewPoint

		# maximum order if given
		self.maxOrder = maxOrder

		# to track if is LTM or STM
		self.STM = STM

		# whether it's an evolutive model
		self.evolutive = evolutive

		# whether to use the original PPM
		self.use_original_PPM = use_original_PPM

		# in order to compute model entropy directly from MC entropies
		self.entropies = {}

		if init is not None:

			maxOrder = len(init)

			if self.maxOrder is None: 
				maxOrder = maxOrder // 2 
			else:
				maxOrder = self.maxOrder

			self.maxOrder = maxOrder

			if VERBOSE:
				print("The maximal order is:", self.maxOrder)

		# list contening different order markov chains
		self.models = []
		for order in range(1, self.maxOrder+1):
			self.models.append(markovChain.markovChain(order, STM=self.STM, evolutive=evolutive))

		self.modelOrder0 = markovChainOrder0.markovChainOrder0(STM=self.STM, evolutive=evolutive)

		self.benchmark = [0, 0, 0]

	def getObservations(self):
		ret = 0
		for model in self.models:
			ret += model.getObservationsSum()
		return ret

	def train(self, data, shortTerm=False, preComputeEntropies=False):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: list of np.array or list of list of int
		"""

		if shortTerm is True:
			# training all the models
			self.modelOrder0.train([data[0][-1:]])
			for i in range(len(self.models)):
				self.models[i].train([data[0][-self.models[i].order-1:]])
				if self.models[i].usedScores == 0:
					if VERBOSE:
						print("The order is too high for these data, we stop the training here.")
					break
			return

		if VERBOSE:
			print("The maximal order is:", self.maxOrder)
		import time
		# training all the models
		self.modelOrder0.train(data)
		for i in range(len(self.models)):
			self.models[i].train(data, preComputeEntropies=preComputeEntropies)
			if self.models[i].usedScores == 0:
				if VERBOSE:
					print("The order is too high for these data, we stop the training here.")
				break

	def getPrediction(self, sequence):
		"""
		Returns the probability distribution from a given state
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)

		:type sequence: np.array(length)

		:return: dictionary | dico[z] = P(z|sequence) (float)
		"""

		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		dico = {}

		for z in alphabet:
			dico[str(z)] = self.getLikelihood(sequence, z)

		return dico

	def getEntropyMax(self, state):
		"""
		Return the maximum entropy for an alphabet. This is the case where all element is equiprobable.

		:param state: state to compute from
		:type state: list or str(list)

		:return: maxEntropy (float)	
		"""

		alphabetSize = np.count_nonzero(list(self.getPrediction(state).values()))

		maxEntropy = 0

		for i in range(alphabetSize):
			maxEntropy -= (1/alphabetSize) * math.log(1/alphabetSize, 2)

		return maxEntropy

	def getAlphabet(self):
		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		return list(set(alphabet))

	def getEntropy(self, state, genuine_entropies=False):
		"""
		Return shanon entropy of the distribution of the model from a given state

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)
		"""

		if not genuine_entropies:
			num_entropies = len(self.entropies[str(state)])
			weights = []
			for i in range(num_entropies):
				weights.append(self.getRelativeEntropy(state))

			approximated_entropies = self.entropies[str(state)]
			return self.mergeProbas(approximated_entropies, weights)

		P = self.getPrediction(state).values()

		if None in P:
			print("It's not possible to compute this entropy, we kill the execution.")
			print("state:",state)
			print("probabilities:", P)
			quit()

		entropy = 0

		for p in P:
			if p != 0:
				entropy -= p * math.log(p, 2)

		return entropy

	def getRelativeEntropy(self, state, genuine_entropies=False):
		"""
		Return the relative entropy H(m)/Hmax(m). It is used for weighting the merging of models without bein affected by the alphabet size.

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)		
		"""

		maxEntropy = self.getEntropyMax(state)

		if maxEntropy > 0:
			return self.getEntropy(state, genuine_entropies=genuine_entropies)/maxEntropy
		else:
			return 1



	def getLikelihood(self, state, note):
		"""
		Returns the likelihood of a note given a state
		
		:param state: a sequence of viewPoint data, cf. data.getData(viewPoint)
		:param note: the integer or name of the note

		:type state: np.array(length)
		:type note:	int or string

		:return: float value of the likelihood
		"""
		probas = []
		weights = []
		entropies = []
		#observations = []

		if self.use_original_PPM:
			return self.mergeProbasPPM(state, note)

		k = -1
		for model in self.models:
			k += 1
			# we don't want to take in account a model that is not capable of prediction
			if model.order <= len(state) and model.getLikelihood(str(list(state[-model.order:])), note) is not None:
				if model.getObservations(state[-model.order:]) is not None:
					probas.append(model.getLikelihood(state[-model.order:], note))
					weights.append(model.getRelativeEntropy(state[-model.order:]))
					entropies.append(model.getEntropy(state[-model.order:]))

		# Order 0
		probas.append(self.modelOrder0.getLikelihood(int(note)))
		weights.append(self.modelOrder0.getRelativeEntropy())
		entropies.append(self.modelOrder0.getEntropy())
		
		if probas == []:
			return None


		self.entropies[str(state)] = np.array(entropies)

		return self.mergeProbas(probas, np.array(weights))

	def mergeProbas(self, probas, weights, b=1):
		"""
		Merging probabilities from different orders, for now we use arithmetic mean

		:param probas: probabilities to merge
		:param weights: weights for the mean, should be get from normalized entropy

		:type probas: list or numpy array
		:type weights: list or numpy array

		:return: merged probabilities (float)
		"""

		# we inverse the entropies
		weights = (weights.astype(float)+np.finfo(float).eps)**(-b)
		
		# Doomy normalization
		for w in weights:
			if w < 0:
				weights += abs(min(weights))
				break
		if np.sum(weights) == 0:
			weights = np.ones(len(weights))

		weights = weights/np.sum(weights)

		ret = 0
		for i in range(len(probas)):
			ret += probas[i]*weights[i]

		return ret


	def mergeProbasPPM(self, state, note):
		"""
		type state: list of int
		type note: str or int, both are possible
		"""
		# initialize probability and escape probability
		probability = 0.0
		escape_probability = 1.0
		# convert note to a string
		note = str(note)

		max_order = min(self.maxOrder, len(state)) if self.maxOrder is not None else len(state)

		for order in range(max_order, -2, -1):
			if order == -1:
				# Order -1 model, uniform distribution
				alphabet_size = len(self.getAlphabet())
				excluded_count = 1 # not activate now
				denominator = alphabet_size + 1 - excluded_count
				if denominator > 0:
					probability += escape_probability * (1.0 / float(denominator))
				break
			if order == 0:
				# Order 0 model, no context
				model = self.modelOrder0
				count_total = model.getTotalCount()
				count_note = model.getCount(note)
				unique_symbols = model.getUniqueSymbolCount()
				context = "Order 0"
			else:
				model = self.models[order - 1]
				# the context is the string version of a list of (int?)
				context = str(list(state[-order:]))
				count_total = model.getTotalCount(context)
				count_note = model.getCount(context, note)
				unique_symbols = model.getUniqueSymbolCount(context)

			denominator = count_total + unique_symbols

			if denominator > 0:
				weight = count_total / denominator
				escape = unique_symbols / denominator
				if count_total > 0:
					alpha = count_note / count_total
					probability += escape_probability * weight * alpha
				escape_probability *= escape
			else:
				escape_probability *= 1.0  # or escape_probability remains the same

			if escape_probability < 1e-10:
				break
			
		return probability


	def sample(self, state):
		"""
		Return a element sampled from the model given the sequence S

		:param S: sequence to sample from

		:type S: list of integers

		:return: sampled element (int)
		"""


		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		distribution = []
		# We reconstruct the distribution according to the sorting of the alphabet
		for elem in alphabet:
			distribution.append(self.getLikelihood(state, elem))

		ret = int(np.random.choice(alphabet, p=distribution))

		return ret

	def generate(self, length):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence (in elements, not beat so it depends on the quantization)
		:type length: int

		:return: sequence (np.array()) 
		"""

		S = []
		# We uniformly choose the first element
		S.append(int(np.random.choice(self.models[0].alphabet)))

		while len(S) < length and str([S[-1]]) in self.models[0].stateAlphabet :

			S.append(self.sample(S))

		return S

	def save(self, file):
		"""
		Save a trained model
		
		:param file: path to the file
		:type file: string
		"""

		f = open(file, 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()

	def load(self, path):
		"""
		Load a trained model

		:param path: path to the file
		:type path: string
		"""

		f = open(path, 'rb')
		tmp_dict = pickle.load(f)
		f.close()          

		self.__dict__.update(tmp_dict) 

