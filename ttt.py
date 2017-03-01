import tensorflow as tf
import numpy
import os

####Testing parameters###############
learning_rates = [0.001]
learning_rate_decays = [1.0] #may not be necessary/not yet implemented
pretraining_conditions = [False]
description_etas = [0.001]
num_runs_per = 20

#####data parameters###########################
n = 3 #size of board
k = 3 #number in row to win, 
#n = 3, k = 3 is tic tac toe
#NOTE: code does not actually implement arbitrary k/n at the moment.

#####network/learning parameters###############
nhidden = 100
nhiddendescriptor = 100

vocab = ["first","second","third","row","column","diagonal","my","opponent's","piece","empty","square","what's","in","END","PAD","GO"]
control_vocab = ["how","many","are","there","zero","one","two","three","four","five","six","seven","eight","nine","END","PAD","GO"]
sentence_length = 10 #All sentences will be padded to this length.
embedding_size = 20 
discount_factor = 1.0 #for Q-learning
epsilon = 0.1 #epsilon greedy
nepochs = 20
games_per_epoch = 20
description_pretraining_states = 20000


######Vocab setup##############################
def onehot(i,n): #vector of length n with ith entry 1 and rest zero
    vec = numpy.zeros(n,dtype=numpy.int32)
    vec[i] = 1
    return vec

vocab_size = len(vocab)
vocab_dict = {vocab[i]: i for i in xrange(vocab_size)}
control_vocab_size = len(control_vocab)
control_vocab_dict = {control_vocab[i]: i for i in xrange(control_vocab_size)}

def words_to_indices(word_list,this_vocab_dict):
    return numpy.array([this_vocab_dict[word] for word in word_list])

def words_to_onehot(word_list,this_vocab_dict,this_vocab_size):
    return numpy.array([onehot(this_vocab_dict[word],this_vocab_size) for word in word_list])


######Question setup##########################
what_questions = [["what's","in",x,y,"END","PAD","PAD","PAD","PAD","PAD"] for x in ["first","second","third"] for y in ["row","column"]] +  [["what's","in",x,"diagonal","END","PAD","PAD","PAD","PAD","PAD"] for x in ["first","second"]] 
questions = what_questions #can expand later
num_questions = len(questions)

def pad_to_length(sentence,n):
    if len(sentence) > n:
	raise ValueError("The sentence provided is longer than the padding length")
    return sentence + ["PAD"]*(n-len(sentence))

control_questions = [pad_to_length(["how","many","are","there","END"],sentence_length)]

###############################################

def threeinrow(state): #helper, expects state to be in square shape, only looks for positive 3 in row
    return numpy.any(numpy.sum(state,axis=0) == 3) or numpy.any(numpy.sum(state,axis=1) == 3) or numpy.sum(numpy.diagonal(state)) == 3 or  numpy.sum(numpy.diagonal(numpy.fliplr(state))) == 3 

def unblockedopptwo(state): #helper, expects state to be in square shape, only looks for negative 2 in row without positive one in remaining spot.
    return numpy.any(numpy.sum(state,axis=0) == -2) or numpy.any(numpy.sum(state,axis=1) == -2) or numpy.sum(numpy.diagonal(state)) == -2 or numpy.sum(numpy.diagonal(numpy.fliplr(state))) == -2 

def oppfork(state): #helper, expects state to be in square shape, looks for fork for -1 player (i.e. two unblockedopptwo in different directions) 
    return numpy.sum(numpy.sum(state,axis=0) == -2) + numpy.sum(numpy.sum(state,axis=1) == -2) + (numpy.sum(numpy.diagonal(state)) == -2) + (numpy.sum(numpy.diagonal(numpy.fliplr(state))) == -2) >= 2
    
def catsgame(state):  #helper, expects state to be in square shape, checks whether state is a cats game 
    return numpy.sum(numpy.abs(state)) >= 8 and not (threeinrow(state) or unblockedopptwo(state) or threeinrow(-state) or unblockedopptwo(-state))

def reward(state):
    state = state.reshape((3,3))
    if threeinrow(state):
	return 1.
    elif unblockedopptwo(state): 
	return -1.
    return 0.

def get_description_target(state,question): #helper, generates description target utterance for a given state and question
    answer = []
    #"What" questions
    if question[:2] == ["what's","in"]:
	if question[3] == "row":
	    r_index = ["first","second","third"].index(question[2])
	    for square in state[r_index,:]:
		if square == 1:
		    answer.extend(["my","piece"])
		elif square == -1:
		    answer.extend(["opponent's","piece"])
		else:
		    answer.extend(["empty","square"])
	elif question[3] == "column":
	    c_index = ["first","second","third"].index(question[2])
	    for square in state[:,c_index]:
		if square == 1:
		    answer.extend(["my","piece"])
		elif square == -1:
		    answer.extend(["opponent's","piece"])
		else:
		    answer.extend(["empty","square"])
	elif question[3] == "diagonal":
	    if question[2] == "first":
		for square in numpy.diag(state):
		    if square == 1:
			answer.extend(["my","piece"])
		    elif square == -1:
			answer.extend(["opponent's","piece"])
		    else:
			answer.extend(["empty","square"])
	    else:
		for square in numpy.diag(numpy.fliplr(state)):
		    if square == 1:
			answer.extend(["my","piece"])
		    elif square == -1:
			answer.extend(["opponent's","piece"])
		    else:
			answer.extend(["empty","square"])
    else:
	answer = ["I","don't","know"]
    answer.append("END")
    
    return pad_to_length(answer,sentence_length)

def get_control_description_target(state,question): #helper, generates description target utterance for a given state and question
    answer = []
    #"What" questions
    if question[:4] == ["how","many","are","there"]:
	count = (["zero","one","two","three","four","five","six","seven","eight","nine"])[numpy.sum(numpy.abs(state),dtype=numpy.int32)]	
	answer = ["there","are",count]
    else:
	answer = ["I","don't","know"]
    answer.append("END")
    
    return pad_to_length(answer,sentence_length)

def nbyn_input_to_2bynbyn_input(state):
    """Converts inputs from n x n -1/0/+1 representation to two n x n arrays, each containing the piece locations for one player"""
    return numpy.concatenate((numpy.ndarray.flatten(1*(state == -1)),numpy.ndarray.flatten(1*(state == 1))))


##########Opponents##############

def random_opponent(state):
    newstate = numpy.copy(state)
    selection = numpy.random.randint(0,9)
    if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	selection = numpy.unravel_index(selection,(n,n))
    while newstate[selection] != 0:
	selection = numpy.random.randint(0,9)
	if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	    selection = numpy.unravel_index(selection,(n,n))
    newstate[selection] = -1
    return newstate 

def single_move_foresight_opponent(state):
    newstate = numpy.copy(state)
    newstate = newstate.reshape((n,n))
    if unblockedopptwo(-state) or unblockedopptwo(state):
	colsum = numpy.sum(state,axis=0)
	rowsum = numpy.sum(state,axis=1) 
	d1sum = numpy.sum(numpy.diag(state))
	d2sum = numpy.sum(numpy.diag(numpy.fliplr(state)))
	if numpy.any(rowsum == -2):
	    selection = numpy.outer(rowsum == -2,state[rowsum == -2,:] == 0) 
	elif numpy.any(colsum == -2):
	    selection = numpy.outer(state[:,colsum == -2] == 0,colsum == -2) 
	elif d1sum == -2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == -2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 	
	elif numpy.any(rowsum == 2):
	    selection = numpy.outer(rowsum == 2,state[rowsum == 2,:] == 0) 
	elif numpy.any(colsum == 2):
	    selection = numpy.outer(state[:,colsum == 2] == 0,colsum == 2) 
	elif d1sum == 2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == 2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 
	else:
	    print "Error! Unhandled position for single_move_foresight_opponent"
	    exit(1)
    else:
	selection = numpy.random.randint(0,9)
	selection = numpy.unravel_index(selection,(n,n))
	while newstate[selection] != 0:
	    selection = numpy.random.randint(0,9)
	    selection = numpy.unravel_index(selection,(n,n))
    newstate[selection] = -1
    return newstate.reshape(numpy.shape(state)) #return in original shape 

def optimal_opponent(state):
    newstate = numpy.copy(state)
    colsum = numpy.sum(state,axis=0)
    rowsum = numpy.sum(state,axis=1) 
    d1sum = numpy.sum(numpy.diag(state))
    d2sum = numpy.sum(numpy.diag(numpy.fliplr(state)))
    if numpy.sum(numpy.abs(state)) == 0: #If first play
	selection = numpy.unravel_index(4,(n,n))
    elif numpy.sum(numpy.abs(state)) == 1: #If second play
	if newstate[1,1] == 0:
	    selection = (1,1)
	else:
	    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
    elif numpy.sum(numpy.abs(state)) == 2: #If third play
	if newstate[0,0] == 1:
	    selection =  [(0,2),(2,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	elif newstate[0,2] == 1: 
	    selection =  [(0,0),(2,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	elif newstate[2,0] == 1:
	    selection =  [(0,0),(2,2),(0,2)][numpy.random.randint(3)] #play in random other corner
	elif newstate[2,2] == 1:
	    selection =  [(0,0),(0,2),(2,0)][numpy.random.randint(3)] #play in random other corner
	else:
	    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
    elif unblockedopptwo(-state) or unblockedopptwo(state):
	if numpy.any(rowsum == -2):
	    selection = numpy.outer(rowsum == -2,state[rowsum == -2,:] == 0) 
	elif numpy.any(colsum == -2):
	    selection = numpy.outer(state[:,colsum == -2] == 0,colsum == -2) 
	elif d1sum == -2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == -2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 	
	elif numpy.any(rowsum == 2):
	    selection = numpy.outer(rowsum == 2,state[rowsum == 2,:] == 0) 
	elif numpy.any(colsum == 2):
	    selection = numpy.outer(state[:,colsum == 2] == 0,colsum == 2) 
	elif d1sum == 2:
	    dis0 = numpy.diag(state) == 0
	    selection = numpy.outer(dis0,dis0) 
	elif d2sum == 2:
	    dis0 = numpy.diag(numpy.fliplr(state)) == 0
	    selection = numpy.fliplr(numpy.outer(dis0,dis0)) 
	else:
	    print "Error! Unhandled two in row position for optimal_opponent"
	    exit(1)
    elif numpy.sum(numpy.abs(state)) == 3: #If fourth play and nothing to block or win
	if newstate[1,1] == -1: #If we hold center
	    if numpy.sum(rowsum == 1) == 1 and numpy.sum(colsum == 1) == 1: #If both plays are on center edge, adjacent 
		selection = numpy.outer(rowsum == 1,colsum == 1) 
	    elif numpy.sum(rowsum == -1) == 1 and numpy.sum(colsum == -1) == 1: #If both plays are on corners, and must be opposite or would have been caught above 
		selection = [(0,1),(1,0),(2,1),(1,2)][numpy.random.randint(4)]
	    else: #opposite center edges or one center one corner  
		if (rowsum[0] == 0 and rowsum[2] == 0) or (colsum[0] == 0 and colsum[2] == 0): #opposite center edges
		    selection = [(0,0),(0,2),(2,2),(2,0)][numpy.random.randint(4)] #play in random corner
		else: #one center one corner: play between
		    selection = numpy.outer(rowsum == 1,colsum == 1)*(state == 0) 
	else: #Opponent holds center, and must have played in opposite corner from us, or we would have blocked. Take another corner and then plays will be forced from there
		selection = [x for x in [(0,0),(0,2),(2,0),(2,2)] if state[x] == 0][numpy.random.randint(2)]
    else: #If fifth play or above and nothing to block or win
	possible_plays = [x for x in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)] if state[x] == 0]
	for possible_play in possible_plays:
	    temp_new_state = numpy.copy(newstate)
	    temp_new_state[possible_play] = -1
	    if oppfork(state): #Making a threat is best we can hope for
		selection = possible_play
		break
	else:
	    for possible_play in possible_plays:
		temp_new_state = numpy.copy(newstate)
		temp_new_state[possible_play] = -1
		if unblockedopptwo(state): #Making a threat is best we can hope for
		    selection = possible_play
		    break
	    else:
		selection = possible_plays[numpy.random.randint(len(possible_plays))]	
    newstate[selection] = -1
    return newstate.reshape(numpy.shape(state)) #return in original shape 


#################################


def update_state(state,selection):
    newstate = numpy.copy(state)
    if numpy.shape(newstate) == (n,n): #handle non-flattened arrays
	selection = numpy.unravel_index(selection,(n,n))
    if newstate[selection] == 0:
	newstate[selection] = 1
	return newstate 
    else:
	return [] #illegal move, array is easier to check for than -1 because python is silly sometimes




#############Q-approx network####################
class Q_approx(object):
    def __init__(self):
	self.input_ph = tf.placeholder(tf.float32, shape=[n*n,1])
	self.target_ph = tf.placeholder(tf.float32, shape=[n*n,1])
	self.W1 = tf.Variable(tf.random_normal([nhidden,n*n],0,0.1)) 
	self.b1 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W2 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	self.b2 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W3 = tf.Variable(tf.random_normal([nhidden,nhidden],0,0.1))
	self.b3 = tf.Variable(tf.random_normal([nhidden,1],0,0.1))
	self.W4 = tf.Variable(tf.random_normal([n*n,nhidden],0,0.1))
	self.b4 = tf.Variable(tf.random_normal([n*n,1],0,0.1))
	self.keep_prob = tf.placeholder(tf.float32) 
	self.representation = tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W2,tf.nn.dropout(tf.nn.tanh(tf.matmul(self.W1,self.input_ph)+self.b1),keep_prob=self.keep_prob))+self.b2),keep_prob=self.keep_prob)
	self.output = tf.nn.tanh(tf.matmul(self.W4,tf.nn.tanh(tf.matmul(self.W3,self.representation)+self.b3))+self.b4)
	self.error = tf.square(self.output-self.target_ph)
	self.eta = tf.placeholder(tf.float32) 
	self.optimizer = tf.train.GradientDescentOptimizer(self.eta)
	self.train = self.optimizer.minimize(tf.reduce_sum(self.error))
	self.epsilon = epsilon #epsilon greedy
	self.curr_eta = 0.0 #default 
	self.sess = tf.Session() 

    def init_all(self):
	self.sess.run(tf.global_variables_initializer())

    def Q(self,state,keep_prob=1.0): #Outputs estimated Q-value for each move in this state
	return self.sess.run(self.output,feed_dict={self.input_ph: (state).reshape((9,1)),self.keep_prob: keep_prob})  

    def train_Q(self,state):
	curr = self.Q(state,keep_prob=0.5)	
	if numpy.random.rand() > self.epsilon:
	    curr_legal = numpy.copy(curr) 
	    curr_legal[numpy.reshape(state,(9,1)) != 0] = -numpy.inf #Filter out illegal moves
	    selection = numpy.argmax(curr_legal) #only selects legal moves
	else:
	    selection = numpy.random.randint(0,9)
	    while state[numpy.unravel_index(selection,(n,n))] != 0: #illegal move
		selection = numpy.random.randint(0,9)
	new_state = update_state(state,selection)
	this_reward = reward(new_state)
	if this_reward in [1,-1]: #if won or lost
	    curr[selection] = this_reward 
	else:
	    curr[selection] = this_reward+discount_factor*max(self.Q(new_state))
	self.sess.run(self.train,feed_dict={self.input_ph: (state).reshape((9,1)),self.target_ph: curr,self.keep_prob: 0.5,self.eta: self.curr_eta}) 
	return new_state

    def Q_move(self,state,train=False): #Executes a move and returns the new state. Replaces illegal moves with random legal moves
	if train:
	    results = self.train_Q(state)		
	else:
	    curr = self.Q(state,keep_prob = 1.0)	
	    curr_legal = numpy.copy(curr) 
	    curr_legal[numpy.reshape(state,(9,1)) != 0] = -numpy.inf #Filter out illegal moves
	    selection = numpy.argmax(curr_legal) #only selects legal moves
	    results = update_state(state,selection)
	return results

    def play_game(self,opponent,train=False,display=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		state = self.Q_move(state,train=train)  
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	return reward

    def train_on_games(self,opponents,numgames=1000):
	score = 0
	num_opponents = len(opponents)
	for game in xrange(numgames):
	    score += self.play_game(opponents[game % num_opponents],train=True)
	return  (float(score)/numgames) 

    def test_on_games(self,opponent,numgames=1000):
	wins = 0
	draws = 0
	losses = 0
	for game in xrange(numgames):
	    this_score = self.play_game(opponent,train=False)
	    if this_score == 1:
		wins += 1
	    elif this_score == -1:
		losses += 1
	    else:
		draws += 1 
	    
	return  (float(wins)/numgames,float(draws)/numgames,float(losses)/numgames) 

class Q_approx_and_descriptor(Q_approx):
    def __init__(self,vocab,vocab_dict,questions,description_target_generator):
	super(Q_approx_and_descriptor,self).__init__()
	self.vocab = vocab
	self.vocab_dict = vocab_dict
	self.vocab_size = len(vocab) 
	self.questions = questions
	self.get_description_target = description_target_generator
	self.curr_description_eta = 1.0
	
	self.description_targets_ph = tf.placeholder(tf.int32, shape=[sentence_length,self.vocab_size])
	self.description_inputs_ph = tf.placeholder(tf.int32, shape=[sentence_length,1])
	self.description_inputs_reversed = tf.reverse(self.description_inputs_ph,[True,False]) #Reversing input may help
	self.word_embeddings = embeddings = tf.Variable(tf.random_uniform([self.vocab_size,embedding_size],-1,1))
	self.description_inputs = tf.nn.embedding_lookup(self.word_embeddings,self.description_inputs_reversed) 
	__,self.encoded_descr_input = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicRNNCell(nhiddendescriptor),self.description_inputs,dtype=tf.float32,time_major=True)
	self.W3d = tf.Variable(tf.random_normal([nhiddendescriptor,nhidden+nhiddendescriptor],0,0.1))
	self.b3d = tf.Variable(tf.random_normal([nhiddendescriptor,1],0,0.1))
	self.W4d = tf.Variable(tf.random_normal([embedding_size,nhiddendescriptor],0,0.1))
	self.b4d = tf.Variable(tf.random_normal([embedding_size,1],0,0.1))
	self.descr_hidden = tf.nn.tanh(tf.matmul(self.W3d,tf.concat(0,[tf.reshape(self.encoded_descr_input,[nhiddendescriptor,1]),self.representation]))+self.b3d)
	with tf.variable_scope('output') as scope:
	    output_cell = tf.nn.rnn_cell.BasicRNNCell(nhiddendescriptor)
	    curr_decoder_state = tf.transpose(self.descr_hidden)
	    curr_decoder_input = tf.reshape(tf.nn.embedding_lookup(self.word_embeddings,vocab_dict["GO"]),[1,embedding_size])
	    outputs = []
	    for i in xrange(sentence_length):
		if i > 0:
		   scope.reuse_variables() 
		output , curr_decoder_state = output_cell(curr_decoder_input, curr_decoder_state)
		output = tf.nn.tanh(tf.matmul(self.W4d,tf.transpose(output))+self.b4d)
		outputs.append(output)	
		curr_decoder_input = tf.transpose(output)
	
	self.output_logits = [tf.matmul(self.word_embeddings,output) for output in outputs]	
	self.descr_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(tf.concat(1,self.output_logits)),self.description_targets_ph)) 

	self.descr_optimizer = tf.train.AdamOptimizer(self.eta)
	self.descr_train = self.descr_optimizer.minimize(self.descr_loss) 	

    def describe_state(self,state):
	"""Prints questions and the network's answers for the given state"""
	print state
	for question in self.questions:
	    print "Q: "
	    print question
	    print "A: "
	    print map(lambda x: self.vocab[numpy.argmax(x)],self.sess.run(self.output_logits,feed_dict={self.input_ph: state.reshape((9,1)),self.description_inputs_ph: words_to_indices(question,vocab_dict).reshape(-1,1),self.keep_prob: 1.0})) 
	    print "Desired:"
	    print get_description_target(state,question)
	    print
	
    def train_description(self,state,nquestions=1):
	"""Trains descriptions by asking nquestions random questions to the network and training correct response"""
	num_total_questions = len(self.questions) 
	for i in xrange(nquestions):
	    question = self.questions[numpy.random.randint(num_total_questions)]
	    self.sess.run(self.descr_train,feed_dict={self.input_ph: state.reshape((9,1)),self.description_inputs_ph: words_to_indices(question,self.vocab_dict).reshape(-1,1),self.description_targets_ph: words_to_onehot(self.get_description_target(state,question),self.vocab_dict,self.vocab_size),self.keep_prob: 0.5,self.eta: self.curr_description_eta})

    def play_game_train_descriptions(self,opponent,display=False,ask_every=1,questions_per_ask=1):
	ask_offset = numpy.random.randint(0,ask_every) #So we don't only learn to describe states that occur at a certain step
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		state = self.Q_move(state,train=True)  
	    if i % questions_per_ask == ask_offset:
		self.train_description(state,nquestions=questions_per_ask)
	    i += 1
	    if display:
		print state
		print
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	return reward

    def train_on_games_with_descriptions(self,opponents,numgames=1000,ask_every=1,questions_per_ask=1):
	score = 0
	num_opponents = len(opponents)
	for game in xrange(numgames):
	    score += self.play_game_train_descriptions(opponents[game % num_opponents],ask_every=ask_every,questions_per_ask=questions_per_ask)
	return  (float(score)/numgames) 

    def test_descriptions(self,state):
	loss = 0.
	for question in self.questions:
	    loss += self.sess.run(self.descr_loss,feed_dict={self.input_ph: state.reshape((9,1)),self.description_inputs_ph: words_to_indices(question,self.vocab_dict).reshape(-1,1),self.description_targets_ph: words_to_onehot(self.get_description_target(state,question),self.vocab_dict,self.vocab_size),self.keep_prob: 1.0})

    def play_game(self,opponent,train=False,display=False,test_descriptions=False):
	gofirst = numpy.random.randint(0,2)
	state = numpy.zeros((3,3))
	i = 0
	descr_loss = 0.
	if display:
	    print "starting game..."
	while not (catsgame(state) or threeinrow(state) or threeinrow(-state)):
	    if i % 2 == gofirst:
		state = opponent(state)
	    else: 
		state = self.Q_move(state,train=train)  
	    i += 1
	    if display:
		print state
		print
	    if test_descriptions:

		for question in self.questions:
		    descr_loss += self.sess.run(self.descr_loss,feed_dict={self.input_ph: state.reshape((9,1)),self.description_inputs_ph: words_to_indices(question,self.vocab_dict).reshape(-1,1),self.description_targets_ph: words_to_onehot(self.get_description_target(state,question),self.vocab_dict,self.vocab_size),self.keep_prob: 1.0})
		
		
	reward = 0
	if threeinrow(state):
	    reward = 1
	elif threeinrow(-state) or unblockedopptwo(state):
	    reward = -1
	
	if test_descriptions:
	    return reward,descr_loss/(i+1)
	else:
	    return reward

    def test_on_games_with_descriptions(self,opponent,numgames=1000,ask_every=1,questions_per_ask=1):
	wins = 0
	draws = 0
	losses = 0
	descr_loss = 0.
	for game in xrange(numgames):
	    this_score,this_descr_loss = self.play_game(opponent,train=False,test_descriptions=True)
	    if this_score == 1:
		wins += 1
	    elif this_score == -1:
		losses += 1
	    else:
		draws += 1 
	    descr_loss += this_descr_loss
	    
	return  (float(wins)/numgames,float(draws)/numgames,float(losses)/numgames,float(descr_loss)/numgames) 
########################################







##########Running#################


#descr_net = Q_approx_and_descriptor()
#descr_net.init_all()
#test_state = numpy.array([[-1,0,1],[0,-1,0],[0,0,-1]])
#descr_net.describe_state(test_state)
#descr_net.train_on_games_with_descriptions([random_opponent],numgames=1000)
#descr_net.describe_state(test_state)

for description_eta in description_etas:
    for learning_rate in learning_rates:
	for lr_decay in learning_rate_decays:
	    for pretrain in pretraining_conditions:
		for run in xrange(num_runs_per):
		    if os.path.exists('descr_net_track_pretrain-%s_learning_rate-%f_description_learning_rate-%f_lr_decay-%f_run-%i.csv'%(str(pretrain),learning_rate,description_eta,lr_decay,run)):
			print "skipping completed run" 
			continue
		    print "run %i" %run
		    basic_track = []
		    descr_track = []
		    control_track = []
		    
		    for currently_training in ["basic","descr","control"]: #,"basic"
			#init
			numpy.random.seed(run)
			tf.set_random_seed(run)	
			
			if currently_training == "descr":
			    net = Q_approx_and_descriptor(vocab,vocab_dict,questions,get_description_target)
			elif currently_training == "control":
			    net = Q_approx_and_descriptor(control_vocab,control_vocab_dict,control_questions,get_control_description_target)
			elif currently_training == "basic":
			    net = Q_approx()

			net.curr_eta = learning_rate
			net.curr_description_eta = description_eta 
			net.init_all()

			# pre-train
			if pretrain:
			    if currently_training == "descr":
				    for i in xrange(description_pretraining_states):
					this_state = numpy.random.randint(-1,2,(3,3))	
		
					net.train_description(this_state,8)
			    elif currently_training == "control":
				    for i in xrange(description_pretraining_states):
					this_state = numpy.random.randint(-1,2,(3,3))	
		
					net.train_description(this_state,8)

		       
			# pre-test
			if currently_training == "descr":
			    descr_track.append(net.test_on_games_with_descriptions(opponent=optimal_opponent,numgames=1000,ask_every=1,questions_per_ask=1)) 
			    print "training descr, initial, results %s" %(str(descr_track[-1]))
			elif currently_training == "control":
			    control_track.append(net.test_on_games_with_descriptions(opponent=optimal_opponent,numgames=1000,ask_every=1,questions_per_ask=1)) 
			    print "training control, initial, results %s" %(str(control_track[-1]))
			elif currently_training == "basic":
			    basic_track.append(net.test_on_games(opponent=optimal_opponent,numgames=1000)) 
			    print "training basic, initial, results %s" %(str(basic_track[-1]))

			
	

			for epoch in xrange(nepochs):
			    # train
			    if currently_training == "descr":
				net.train_on_games_with_descriptions([optimal_opponent],numgames=games_per_epoch,ask_every=1,questions_per_ask=1)
				net.train_on_games([optimal_opponent],numgames=games_per_epoch)

			    elif currently_training == "control":
				net.train_on_games_with_descriptions([optimal_opponent],numgames=games_per_epoch,ask_every=1,questions_per_ask=1)

			    elif currently_training == "basic":
				net.train_on_games([optimal_opponent],numgames=games_per_epoch)
	


			    # test 
			    if currently_training == "descr":
				descr_track.append(net.test_on_games_with_descriptions(opponent=optimal_opponent,numgames=1000,ask_every=1,questions_per_ask=1)) 
				print "training descr, epoch %i, results %s" %(epoch,str(descr_track[-1]))
			    elif currently_training == "control":
				control_track.append(net.test_on_games_with_descriptions(opponent=optimal_opponent,numgames=1000,ask_every=1,questions_per_ask=1)) 
				print "training control, epoch %i, results %s" %(epoch,str(control_track[-1]))
			    elif currently_training == "basic":
				basic_track.append(net.test_on_games(opponent=optimal_opponent,numgames=1000)) 
				print "training basic, epoch %i, results %s" %(epoch,str(basic_track[-1]))
			#Reset
			net.sess.close() 
			tf.reset_default_graph()


	    
		    # output
		    numpy.savetxt('descr_net_track_pretrain-%s_learning_rate-%f_description_learning_rate-%f_lr_decay-%f_run-%i.csv'%(str(pretrain),learning_rate,description_eta,lr_decay,run),descr_track,delimiter=',')
		    numpy.savetxt('control_net_track_pretrain-%s_learning_rate-%f_description_learning_rate-%f_lr_decay-%f_run-%i.csv'%(str(pretrain),learning_rate,description_eta,lr_decay,run),control_track,delimiter=',')
		    numpy.savetxt('basic_net_track_pretrain-%s_learning_rate-%f_description_learning_rate-%f_lr_decay-%f_run-%i.csv'%(str(pretrain),learning_rate,description_eta,lr_decay,run),basic_track,delimiter=',')


