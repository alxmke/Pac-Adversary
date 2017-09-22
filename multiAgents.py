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
import random, util, operator, math #added operator, numpy

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

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float("inf")

        if successorGameState.isLose():
            return float("-inf")

        if action == "Stop":
            return float("-inf")

        too_spooky = any([True if util.manhattanDistance(ghost.getPosition(), newPos) <= 1 and ghost.scaredTimer == 0 else False for ghost in newGhostStates])
        if too_spooky:
            return float("-inf")

        food_dists = [util.manhattanDistance(newPos, food_pos) for food_pos in newFood.asList()]

        if not food_dists:
            food_dists = [1]

        return ( 1.0/min(food_dists)
               + 2.0*(currentGameState.getNumFood() - successorGameState.getNumFood())) #2 prio-factor improves greatly


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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        efn = self.evaluationFunction
        def lmin(gameState, depth, agent_index, n_min_agents):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            if agent_index == n_min_agents:
                return min([lmax(gameState.generateSuccessor(agent_index, action), depth-1, n_min_agents) for action in gameState.getLegalActions(agent_index)] + [float("inf")])
            else:
                return min([lmin(gameState.generateSuccessor(agent_index, action), depth, agent_index+1, n_min_agents) for action in gameState.getLegalActions(agent_index)] + [float("inf")])

        def lmax(gameState, depth, n_min_agents):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            return max([lmin(gameState.generateSuccessor(0, action), depth, 1, n_min_agents) for action in gameState.getLegalActions(0)] + [float("-inf")])

        n_min_agents = gameState.getNumAgents()-1
        return max([(lmin(gameState.generateSuccessor(0, action), self.depth, 1, n_min_agents), action) for action in gameState.getLegalActions(0)] + [(float("-inf"), Directions.STOP)], key=operator.itemgetter(0))[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        efn = self.evaluationFunction
        def lmin(gameState, depth, agent_index, n_min_agents, alpha, beta):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            if agent_index == n_min_agents:
                v = float("inf")
                for action in gameState.getLegalActions(agent_index):
                    v = min(v, lmax(gameState.generateSuccessor(agent_index, action), depth-1, n_min_agents, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v)
                return v
            else:
                v = float("inf")
                for action in gameState.getLegalActions(agent_index):
                    v = min(v, lmin(gameState.generateSuccessor(agent_index, action), depth, agent_index+1, n_min_agents, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v)
                return v

        def lmax(gameState, depth, n_min_agents, alpha, beta):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            v = float("-inf")
            for action in gameState.getLegalActions(0):
                v = max(v, lmin(gameState.generateSuccessor(0, action), depth, 1, n_min_agents, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v

        n_min_agents = gameState.getNumAgents()-1
        alpha = float("-inf")
        beta = float("inf")
        aba_action = Directions.STOP
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            act_v = lmin(gameState.generateSuccessor(0, action), self.depth, 1, n_min_agents, alpha, beta)
            if(v < act_v):
                aba_action = action
                v = act_v
            alpha = max(alpha, v)
        return aba_action

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
        efn = self.evaluationFunction
        def lexpecti(gameState, depth, agent_index, n_min_agents):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            
            legal_actions = gameState.getLegalActions(agent_index) # no numpy in this version of python apparently, so I've got to do this goofy, self-implemented average-fn shit
            if agent_index == n_min_agents:
                return sum([lmax(gameState.generateSuccessor(agent_index, action) , depth-1, n_min_agents) for action in legal_actions])/len(legal_actions)
            else:
                return sum([lexpecti(gameState.generateSuccessor(agent_index, action) , depth, agent_index+1, n_min_agents) for action in legal_actions])/len(legal_actions)

        def lmax(gameState, depth, n_min_agents):
            if(depth == 0 or gameState.isWin() or gameState.isLose()): return efn(gameState)
            return max([lexpecti(gameState.generateSuccessor(0, action), depth, 1, n_min_agents) for action in gameState.getLegalActions(0)] + [float("-inf")])

        n_min_agents = gameState.getNumAgents()-1
        return max([(lexpecti(gameState.generateSuccessor(0, action), self.depth, 1, n_min_agents), action) for action in gameState.getLegalActions(0)] + [(float("-inf"), Directions.STOP)], key=operator.itemgetter(0))[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    #need to make standing still a disaster
    #dont get too spooked if there's a ghost behind a wall
    #dont go into deadends if you're getting tailed by a ghost, if possible to implement

    # better if pacman gets spooked at 2 in this iteration, thus far
    too_spooky = any([True if util.manhattanDistance(ghost.getPosition(), pos) <= 3 and ghost.scaredTimer == 0 else False for ghost in ghostStates])
    if too_spooky:
        return float("-inf")

    food_dists = [util.manhattanDistance(pos, food_pos) for food_pos in food.asList()]

    if not food_dists:
        food_dists = [1]

    capd = [util.manhattanDistance(pos, cap_pos) for cap_pos in currentGameState.getCapsules()]
    if not capd:
        capd = 0
    else:
        capd = min(capd)

    return ( 1.0/min(food_dists)
           - 2.0*currentGameState.getNumFood()
           #+ 1.0/(1+capd)
           + 0.5/(1+len(currentGameState.getCapsules()))
           + currentGameState.getScore()
           + 0)

# Abbreviation
better = betterEvaluationFunction

