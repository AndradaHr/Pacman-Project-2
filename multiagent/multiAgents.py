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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = successorGameState.getScore()
        closestFoodDist = float('inf')
        # iteram prin fiecare celula a grilei newFood
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:  # verifica daca exista mancare in celula
                    # calculeaza distanta de la Pacman la mancarea din celula
                    dist = manhattanDistance(newPos, (x, y))
                    # cautam distanta minima
                    if dist < closestFoodDist:
                        closestFoodDist = dist
        # daca s-a gasit mancare, ajusteaza scorul
        if closestFoodDist != float('inf'):
            score += 1.0 / closestFoodDist

        # parcurgem lista de stări ale fantomelor
        i = 0
        while i < len(newGhostStates):
            ghostState = newGhostStates[i] # starea curentă a fantomei
            ghostPos = ghostState.getPosition() # pozitia fantomei
            distanceToGhost = manhattanDistance(newPos, ghostPos) # distanta de la Pacman la fantoma
            # verif daca fantoma este speriata sau nu
            if ghostState.scaredTimer != 0:
                # daca e speriata si e aproape (mai putin de 2 unit)
                # primim 50 de pct
                if distanceToGhost < 2:
                    score += 50
            else:
                # daca e nue speriata si e aproape
                # ni se scad 50 de puncte
                if distanceToGhost < 2:
                    score -= 50
            i += 1

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        result = self.get_value(gameState, 0, 0)
        return result[1] #returnam actiunea

    def get_value(self, game_state, index, depth):
        #gameState: state-ul curent al jocului
        #index: index-ul agentului curent
        #depth: adancimea curenta in arborele de cautare
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return [game_state.getScore(), ""]
        #Pacman -> index = 0 => maximizam
        if index == 0:
            return self.max_value(game_state, index, depth)
        else: #index = 1 => fantoma => minimizare
            return self.min_value(game_state, index, depth)

    def max_value(self, game_state, index, depth):
        maximum = float('-inf')
        max_action = ""
        for action in game_state.getLegalActions(index):
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            #daca agentul e Pacman, actualizam indexul agentului si adancimea
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]
            if current_value > maximum:
                maximum = current_value
                max_action = action
        return maximum, max_action

    def min_value(self, game_state, index, depth):
        minimum = float('inf')
        min_action = ""
        for action in game_state.getLegalActions(index):
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            #daca agentul e Pacman, actualizam indexul agentului si adancimea
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.get_value(successor, successor_index, successor_depth)[0]
            if current_value < minimum:
                minimum = current_value
                min_action = action
        return minimum, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        result = self.get_value(gameState, 0, 0, float('-inf'), float('inf'))
        return result[1]  #returnam actiunea

    def get_value(self, game_state, index, depth, alpha, beta):
        #suntem in stare terminala
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return [game_state.getScore(), ""]

        if index == 0:  # e Pacman => maximizam
            return self.max_value(game_state, index, depth, alpha, beta)
        else:  # e fantoma => minimizam
            return self.min_value(game_state, index, depth, alpha, beta)

    def max_value(self, game_state, index, depth, alpha, beta):
        maximum = float('-inf')
        max_action = ""
        for action in game_state.getLegalActions(index):
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            #luam recursiv valoarea succesorului
            current_value = self.get_value(successor, successor_index, successor_depth, alpha, beta)[0]
            if current_value > maximum:
                maximum = current_value
                max_action = action

            #updatam valoarea alpha la maximul dintre alpha si valoarea maxima curenta
            alpha = max(alpha, maximum)

            #daca maximul e mai mare decat beta, facem "prune" la search
            if maximum > beta:
                return [maximum, max_action]

        #returnam valoarea maxima si actiunea care ii corespunde
        return [maximum, max_action]

    def min_value(self, game_state, index, depth, alpha, beta):
        minimum = float('inf')
        min_action = ""
        for action in game_state.getLegalActions(index):
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            #calculam scorul-actiune al succesorului
            current_value = self.get_value(successor, successor_index, successor_depth, alpha, beta)[0]
            if current_value < minimum:
                minimum = current_value
                min_action = action

            #updatam beta la minimul dintre beta si valoarea curenta a minimului
            beta = min(beta, minimum)

            #daca minimul e mai mic sau egal cu alpha, facem "prune" la search
            if minimum < alpha:
                return [minimum, min_action]
        #returnam minimul si actiunea care ii corespunde
        return [minimum, min_action]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax_value(state, agentIndex, depth):
            # verif daca starea curenta este o stare finala sau daca s-a atins adancimea maxima
            if state.isWin() or state.isLose() or depth == self.depth:
                return (None, self.evaluationFunction(state))

            next_agent = (agentIndex + 1) % state.getNumAgents() # det urmatorul agent
            next_depth = depth + 1 if next_agent == 0 else depth  # adancimea se incrementeaza doar cand revenim la Pacman

            # Nod de sansa (pentru fantome)
            if agentIndex != 0:
                expected_value = 0
                actions = state.getLegalActions(agentIndex)
                probability = 1 / len(actions)  # Calculeaza probabilitatea pentru fiecare actiune
                for action in actions:
                    next_state = state.generateSuccessor(agentIndex, action)
                    action_unused, value = expectimax_value(next_state, next_agent, next_depth)  # calculeaza valoarea expectimax pentru starea urmatoare
                    expected_value += probability * value
                return None, expected_value

            # Nod maximizator (pentru Pacman)
            else:
                max_value = float("-inf")
                best_action = None
                # Parcurgem toate actiunile legale ale lui Pacman
                for action in state.getLegalActions(agentIndex):
                    next_state = state.generateSuccessor(agentIndex, action)
                    action_unused, value = expectimax_value(next_state, next_agent, next_depth)
                    # Daca valoarea este mai mare decat valoarea maxima curenta actualizam max si cea mai buna actiune
                    if value > max_value:
                        max_value = value
                        best_action = action
                return best_action, max_value

        # Pornim cautarea expectimax de la Pacman la adancimea 0
        best_action, action_value = expectimax_value(gameState, 0, 0)
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    # aflam starile curente ale fantomelor
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # evaluam distanta pana la cel mai apropiat punct de mancare
    if len(foodList) > 0:
        closestFoodDist = min([manhattanDistance(pacmanPosition, food) for food in foodList])
    else:
        closestFoodDist = 0

    # evaluam numarul de puncte de mancare ramase
    remainingFood = len(foodList)

    # evaluam distanta pana la cea mai apropiata fantoma
    closestGhostDist = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])

    # verificam daca vreuna din fantome e speriata; daca da, consideram distanta lor si adaugam un bonus
    scaredGhosts = [ghost for ghost, scaredTime in zip(ghostStates, scaredTimes) if scaredTime > 0]
    scaredGhostBonus = 0
    if scaredGhosts:
        scaredGhostDist = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in scaredGhosts])
        scaredGhostBonus = 1.0 / (scaredGhostDist + 1)

    # combinam factorii pentru a afla scorul
    score = currentGameState.getScore() + 1.0 / (
                closestFoodDist + 1) + scaredGhostBonus - 5.0 * closestGhostDist - 2.0 * remainingFood

    return score

# Abbreviation
better = betterEvaluationFunction
