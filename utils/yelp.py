class YelpDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = 0
	OppStyle = {POS:NEG,NEG:POS}
	def __init__(self): #, wordDictFile): #, labeled=True, needLabel=True):
		super(YelpDataset, self).__init__()
		trainfile = './data/sentiment.train'
		devfile = './data/sentiment.dev'

		self.traindata,self.trainlabel = self.readData(trainfile)
		self.devdata,self.devlabel = self.readData(devfile)

		with open('./emb/wordDict',"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.sos_id = self.wordDict['@@START@@']
		self.eos_id = self.wordDict['@@END@@']

	def load_data(self):
		return (self.traindata+self.devdata,self.trainlabel+self.devlabel)
	def get_word_index(self):
		return self.wordDict

	def isValidSentence(self,sentence):
		if(sentence == [] or 
			sentence == 'Positive' or 
			sentence == 'Negative'):
			return False
		return True

	def readData(self,datafile):
		data = [] #{self.POS:[], self.NEG:[]}
		label = []
		# proc .0 file (negative)
		def subread(postfix,style):
			with open(datafile+postfix,'r') as f:
				line = f.readline()
				# i = 0
				while line:
					sentence = line.split(' ')[:-1]
					if self.isValidSentence(sentence):
						data.append(self.word2index([sentence])[0])
						label.append(style)
					line = f.readline()
					# i += 1
		subread('.0',self.NEG)
		subread('.1',self.POS)
		return data

	def word2index(self, sList, sos=False):
		resList = []
		for sentence in sList:
			indArr = []
			if sos:
				indArr.append(self.sos_id)
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
			indArr.append(self.eos_id) 
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList