import os,pickle,sys,numpy,nn,copy,scipy,scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
seed  = 3453
print(sys.argv)
split = int(sys.argv[1]) # test split

mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

# --------------------------------------------
# Load data
# --------------------------------------------
numpy.random.seed(seed)
if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('/home/edabk/cuong/test_project/qm7.mat') # a dict, 'X', 'R', 'Z', 'T','P', each contained an array
# 'X' is (7165,23,23)
# 'R' is (7165,23,3)
# 'Z' is (7165,23)
# 'T' is (1,7165)
# 'P' is (5,1433)


# --------------------------------------------
# Extract training data
# --------------------------------------------
# P = dataset['P'][range(0,split)+range(split+1,5)].flatten() # extract specific elements from the P array 
															# intentionally excluding the elements at the index indicated by the split variable, but this is the wrong code
# Here is the fixed code:
indices = [i for i in range(0, split)] + [i for i in range(split + 1, 5)] # if we choose split = 4 , the indices are [0,1,2,3]
P = dataset['P'][indices].flatten() # (5732,) which is 4*1433, since we disregard the split 4, only use 0,1,2,3
X = dataset['X'][P] # (7165,23,23) -> (5732,23,23) , only choose the sample chosen by P
T = dataset['T'][0,P] # (1,7165) ->   (5732,)

# X is the joint-node features : 23*23 matrix joint features between 23 joints
# T is the label               : 1 scalar number from -404.88 -> -2192.0 represent the energy of each atom sample

# --------------------------------------------
# Create a neural network
# --------------------------------------------
I,O = nn.Input(X),nn.Output(T)
# So the input is batch_size=5732, 23 , 23
#   the output is batch_size=5732, 
nnsgd = nn.Sequential([I,nn.Linear(I.nbout,400),nn.Sigmoid(),nn.Linear(400,100),nn.Sigmoid(),nn.Linear(100,O.nbinp),O])
nnsgd.modules[-2].W *= 0
nnavg = copy.deepcopy(nnsgd)

# --------------------------------------------
# Train the neural network
# --------------------------------------------
for i in range(1,1000001):

	if i > 0:     lr = 0.001  # learning rate
	if i > 500:   lr = 0.0025
	if i > 2500:  lr = 0.005
	if i > 12500: lr = 0.01

	r = numpy.random.randint(0,len(X),[mb])
	Y = nnsgd.forward(X[r])
	nnsgd.backward(Y-T[r])
	nnsgd.update(lr)
	nnavg.average(nnsgd,(1/hist)/((1/hist)+i))
	nnavg.nbiter = i

	if i % 100 == 0: pickle.dump(nnavg,open('nn-%d.pkl'%split,'wb'),pickle.HIGHEST_PROTOCOL)

