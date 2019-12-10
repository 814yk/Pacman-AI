# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        
        ghostDist = [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates]
        gotScore = successorGameState.getScore() - currentGameState.getScore()


        pos = currentGameState.getPacmanPosition()
        foods = currentGameState.getFood().asList()
        foodDist = [manhattanDistance(pos, food) for food in foods]


        newFoods = newFood.asList()
        newFoodsDist = [manhattanDistance(newPos, food) for food in foods]
        newNearestFoodDist = 0 if not newFoodsDist else min(newFoodsDist)

        isNearer = min(foodDist) - newNearestFoodDist


        direction = currentGameState.getPacmanState().getDirection()

        # Reflex formula

        if min(ghostDist) <= 1 or action == Directions.STOP:
            return 0
        if gotScore > 0:
            return 8
        elif isNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxV(state, depth):
            action = state.getLegalActions(0)
            if not action or depth == self.depth:
                return self.evaluationFunction(state)
            return max(minV(state.generateSuccessor(0, x), 0 + 1, depth + 1) for x in action)
        
        def minV(state, i, depth):
            action = state.getLegalActions(i)
            if not action:
                return self.evaluationFunction(state)
            if i == state.getNumAgents() - 1:
                return min(maxV(state.generateSuccessor(i, x), depth) for x in action)
            else:
                return min(minV(state.generateSuccessor(i, x), i + 1, depth) for x in action)
            
            
        action = max(gameState.getLegalActions(0),
                         key=lambda x: minV(gameState.generateSuccessor(0, x), 1, 1))
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf=float('inf')
        def minValue(state, agentIndex, depth, a, b):
            action = state.getLegalActions(agentIndex)
            if not action:
                return self.evaluationFunction(state)

            V = inf
            for x in action:
                newState = state.generateSuccessor(agentIndex, x)

                # Is it the last ghost?
                if agentIndex == state.getNumAgents() - 1:
                    newV = maxValue(newState, depth, a, b)
                else:
                    newV = minValue(newState, agentIndex + 1, depth, a, b)

                V = min(V, newV)
                if V < a:
                    return V
                b = min(b, V)
            return V

        def maxValue(state, depth, a, b):
            action = state.getLegalActions(0)
            if not action or depth == self.depth:
                return self.evaluationFunction(state)

            V = -inf
            # For enable second ply pruning
            if depth == 0:
                bestAction = action[0]
            for x in action:
                newState = state.generateSuccessor(0, x)
                newV = minValue(newState, 0 + 1, depth + 1, a, b)
                if newV > V:
                    V = newV
                    if depth == 0:
                        bestaction = x
                if V > b:
                    return V
                a = max(a, V)

            if depth == 0:
                return bestaction
            return V

        bestAction = maxValue(gameState, 0, -inf, inf)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expect_max(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:  
                return max(expect_max(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else: 
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expect_max(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        """Performing maximizing task for the root node i.e. pacman"""
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = expect_max(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    NearestFoodDis = min(manhattanDistance(pos, food) for food in foods) if foods else 0.5
    score = currentGameState.getScore()

    '''
      Sometimes pacman will stay put even when there's a dot right besides, because
      stop action has the same priority with other actions, so might be chosen when
      multiple actions have the same evaluation, upon which we can improve maybe.
    '''
    evaluation = 1.0 / NearestFoodDis + score
    return evaluation

# Abbreviation
better = betterEvaluationFunction

