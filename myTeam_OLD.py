# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'AlphaBetaAgent',
							 second = 'AlphaBetaAgent'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.  isRed is True if the red team is being created, +and
	will be False if the blue team is being created.

	As a potentially helpful development aid, this function can take
	additional string-valued keyword arguments ("first" and "second" are
	such arguments in the case of this function), which will come from
	the --redOpts and --blueOpts command-line arguments to capture.py.
	For the nightly contest, however, your team will be created without
	any extra arguments, so you should make sure that the default
	behavior is what you want for the nightly contest.
	"""

	# The following line is an example only; feel free to change it.
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

#############
# Functions #
#############

def inMyHome(state, myIndex):
	"""
	Boolean function that determines if an agent is in their base and 
	able to deposit pellets
	"""
	isRed = state.isOnRedTeam(myIndex)
	boundary = (state.data.food.width / 2) - 1
	myX, myY = state.getAgentState(myIndex).getPosition()
	if isRed:
		return (myX < boundary)
	else:
		return (myX > boundary)

def heuristic(state, myIndex):
	"""
	Heuristic function for a star search - goal is to return to home
	"""
	# return difference of my x and home base's x
	isRed = state.isOnRedTeam(myIndex)
	boundary = (state.data.food.width / 2) - 1
	myX, myY = state.getAgentState(myIndex).getPosition()
	if isRed:
		return abs(myX - boundary - 2)
	else:
		return abs(myX - boundary + 2)

def aStarSearch(state, myIndex):
	"""
	Search the node that has the lowest combined cost and heuristic first
	Goal for agent is to return home
	"""
	fringe = util.PriorityQueue()
	currPath = []
	fringe.push((state, currPath, 0, 0), 0)
	expandedList = []
	while not fringe.isEmpty():
		# pop next state from fringe
		currState, currPath, gCost, currCost = fringe.pop()
		# if already expanded, skip this one
		if currState in expandedList:
			continue
		# else
		expandedList.append(currState)
		# if i made it home, return list of actions
		if inMyHome(currState,  myIndex):
			return currPath
		# get successors generated for all legal actions
		actions = currState.getLegalActions(myIndex)
		for a in actions:
			successor = currState.generateSuccessor(myIndex, a)
			cost = 1
			h = heuristic(successor, myIndex)
			fringe.push((successor, currPath + [a], gCost + cost, gCost + cost + h), gCost + cost + h)
	return -1 # error

#########
# Agent #
#########

# combined offense and defense into single class so offensive agent can
#	accurately predict moves of defensive teammate, and vice versa

class AlphaBetaAgent(CaptureAgent):
	"""
	An agent that uses minimax search with alpha beta pruning
	"""

	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)
		self.start = gameState.getAgentPosition(self.index)

	def chooseAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""

		# call minimax recursive minimax function with initial values assigned to variables
		num_of_agents = gameState.getNumAgents()
		start = time.time()
		value, action = self.minimax(gameState, '', num_of_agents+2, float('-inf'),
			float('inf'), self.index, True)
		# 	note: depth value must be even to correctly evaluate state

		# if carrying more than x pellets, return to base
		if gameState.getAgentState(self.index).numCarrying > 2:
			pathHome = aStarSearch(gameState, self.index)
			action = pathHome[0]

		return action

	def getSuccessor(self, gameState, action):
		"""
		Finds the next successor which is a grid position (location tuple).
		Only finds successor of self, not other agents
		"""
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			# Only half a grid position was covered
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def evaluate(self, state, action, myIndex):
		"""
		Uses linear combination of features and weights
		"""
		if myIndex-2 < 0:
			# evaluate defense agent (max of one on my team)
			features = self.getDefFeatures(state, action, myIndex)
			weights = self.getDefWeights()
			return features * weights
		else:
			# evaluate offense agent (max of team size minus one)
			features = self.getOffFeatures(state, action, myIndex)
			weights = self.getOffWeights(state, action, myIndex, features)
			return features * weights

	def getDefFeatures(self, state, action, myIndex):
		features = util.Counter()
		#successor = self.getSuccessor(state, action)
		successor = state.generateSuccessor(myIndex, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# computes whether agent is on defense(1) or offense(0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

		# computes distance to invaders visible
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition != None]
		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		# use getFoodYouAreDefending()??

		# needs to avoid pacman with power pellet

		return features

	def getDefWeights(self):
		return {'numInvaders': -1000, 'onDefense': 500, 'invaderDistance': -10, 'stop': -500, 'reverse': -2}

	def getOffFeatures(self, state, action, myIndex):
		# # extract raw features
		# features = util.Counter()
		# successor = state.generateSuccessor(myIndex, action)
		# foodList = self.getFood(successor).asList()
		# myState = successor.getAgentState(myIndex)
		# myPos = myState.getPosition()

		# # don't stop
		# features['stop'] = 0
		# if action == Directions.STOP: features['stop'] = 1

		# # reward for depositing food and increasing score
		# features['successorScore'] = abs(self.getScore(successor))

		# # reward picking up pellets
		# features['foodRemaining'] = len(foodList)

		# # move towards pellets
		# minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
		# features['distanceToFood'] = minDistance

		# # avoid ghosts that are not scared
		# enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		# visible = filter(lambda ghost: not ghost.isPacman # if enemy is ghost
		# 	and not ghost.getPosition() == None # and within sight
		# 	and ghost.scaredTimer <= 0, enemies) # and is not scared
		# features['distToEnemies'] = 0
		# features['distToHome'] = 0
		# features['trapped'] = 0
		# if len(visible) > 0:
		# 	# enemies visible, avoid them
		# 	for enemy in visible:
		# 		distToEnemy = self.getMazeDistance(myPos, enemy.getPosition())
		# 		# enemy close - danger
		# 		if distToEnemy < 5:
		# 			features['distToEnemies'] += distToEnemy #maximize
		# 			features['distToHome'] = heuristic(successor, myIndex) #minimize
		# 			features['distanceToFood'] = 0 #irrelevant
		# 			avoid getting trapped
		# 			legalActions = successor.getLegalActions(myIndex)
		# 			if len(legalActions) < 3: # only stop and one other direction
		# 				features['trapped'] = 1
		# 			# if carrying no food, don't care about returning safely
		# 			if successor.getAgentState(myIndex).numCarrying == 0:
		# 				features['distToHome'] = 0

		# extract raw features
		features = util.Counter()
		successor = state.generateSuccessor(myIndex, action)
		foodList = self.getFood(successor).asList()
		myState = successor.getAgentState(myIndex)
		myPos = myState.getPosition()
		SAFE_RADIUS = 4

		# determine if we're within certain radius of enemies
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		visible = filter(lambda ghost: not ghost.isPacman # if enemy is ghost
			and not ghost.getPosition() == None # and within sight
			and ghost.scaredTimer <= 0, enemies) # and is not scared
		minDistToEnemy = 50
		if len(visible) > 0 and not inMyHome(successor, myIndex):
			minDistToEnemy = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in visible])
			features['enemyClose'] = 1 if (minDistToEnemy <= SAFE_RADIUS) else 0
		else:
			features['enemyClose'] = 0

		# get distance to nearest food
		minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
		features['distToFood'] = minDistance

		# get distance to home territory
		features['distToHome'] = heuristic(successor, myIndex)

		# determine if we should retreat
		if not inMyHome(successor, myIndex):
			features['retreat'] = 1 if (minDistToEnemy <= SAFE_RADIUS) else 0
		else:
			features['retreat'] = 0

		# determine if agent is in its home
		features['safeInBase'] = 1 if inMyHome(successor, myIndex) else 0

		# get length of remaining food list
		features['foodRemaining'] = len(foodList)

		# # print details
		# print('successor being evaluated:')
		# print(successor)
		# print('enemies visible:')
		# print(visible)
		# print('distance to closest enemy:')
		# print(minDistToEnemy)
		# print('features for this state:')
		# print(features)
		# weights = self.getOffWeights(state, action, myIndex, features)
		# print('weights for this state:')
		# print(weights)
		# print('evaluation for action:', action)
		# print(features*weights)

		return features

	def getOffWeights(self, state, action, myIndex, features):
		# return {'successorScore': 1000, 'distanceToFood': -1, 'distToEnemies': 20,
		# 	'stop': -1000, 'trapped': -50, 'distToHome': -20, 'foodRemaining': -100}
		"""
		Determine strategy and adjust weights accordingly
		"""
		weights = util.Counter()
		successor = state.generateSuccessor(myIndex, action)
		foodList = self.getFood(successor).asList()
		myState = successor.getAgentState(myIndex)
		myPos = myState.getPosition()

		# at all times, avoid certain radius of enemies and reward picking up food
		weights['enemyClose'] = -1000
		weights['foodRemaining'] = -100

		# if in my home, attempt to penetrate enemy territory
		if inMyHome(successor, myIndex):
			weights['distToFood'] = -10
			return weights

		# else in enemy territory
		else:
			# if enemy is close, retreat
			if features['retreat']:
				weights['safeInBase'] = 1000
				weights['distToHome'] = -10
				return weights
			# no enemies around, safe to attack
			else:
				weights['distToFood'] = -10
				return weights

	# recursive minimax function
	def minimax(self, state, action, depth, alpha, beta, agentsTurn, myTeamsTurn):
		"""
		A recursive minimax function that uses alpha-beta pruning
		"""
		# if game is over or if leaf reached
		if depth == 0 or state.isOver():
			return None, None#self.evaluate(state, action), action
			# return and evaluate current successor

		# if my team's turn
		if myTeamsTurn:
			maxEval = float('-inf')
			# for each legal action, evaluate
			actions = state.getLegalActions(agentsTurn)
			bestAction = actions[0]
			for a in actions:
				# generate state that would result if this action occurs
				successor = state.generateSuccessor(agentsTurn, a)
				numAgents = state.getNumAgents()
				val, act = self.minimax(successor, a, depth-1, alpha, beta, (agentsTurn+1)%numAgents, False)
				# leaf reached, evaluate successor
				val = self.evaluate(state, a, agentsTurn)
				if val > maxEval:
					maxEval = val
					bestAction = a
				alpha = max(alpha, val)
				if beta <= alpha:
					break
			return maxEval, bestAction

		# if opponent team's turn
		else:
			minEval = float('inf')
			# for each legal action, evaluate
			actions = state.getLegalActions(agentsTurn)
			bestAction = actions[0]
			for a in actions:
				# generate state that would result if this action occurs
				successor = state.generateSuccessor(agentsTurn, a)
				numAgents = state.getNumAgents()
				val, act = self.minimax(successor, a, depth-1, alpha, beta, (agentsTurn+1)%numAgents, True)
				# should never be min at leaf, no evaluation required
				#val = self.evaluate(state, a, agentsTurn)
				if val < minEval:
					minEval = val
					bestAction = a
				beta = min(beta, val)
				if beta <= alpha:
					break
			return minEval, bestAction
