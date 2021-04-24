# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def generalSearch(problem, prirotyFunction, debug=False):
    frontier = util.PriorityQueue()
    frontier.push(item=(problem.getStartState(), [], 0), priority=0)
    explored = set()

    while not frontier.isEmpty():
        current_node, actions_to_curr_node, cost_to_curr = frontier.pop()

        if problem.isGoalState(current_node):
            return actions_to_curr_node

        if debug:
            print('----------------------------------------------------------------')
            print('\033[95m')  # Magenta
            print('| Current Node\t| Cost to Current\t|')
            print('| ------------\t| ---------------\t|')
            print('\033[92m')  # Green
            print(f'| {current_node}\t| {cost_to_curr}\t|')
            print()
            print('\033[96m')  # cyan
            print('| Successor\t| Cost to Next\t| Actions to Next\t|')
            print('| ---------\t| ------------\t| ---------------\t|')
            print('\033[0m')

        for successor in problem.getSuccessors(current_node):
            successor_node, action_to_neighbour, _ = successor
            if successor_node not in explored:
                cost_to_neighbour = prirotyFunction(current_node, successor, cost_to_curr)

                for priority, _, heap_item in frontier.heap:
                    if successor_node == heap_item[0]:
                        if priority <= cost_to_neighbour:
                            break

                        frontier.update(
                            item=(
                            successor_node, actions_to_curr_node.copy() + [action_to_neighbour], cost_to_neighbour),
                            priority=cost_to_neighbour
                        )
                        break
                else:
                    frontier.push(
                        item=(successor_node, actions_to_curr_node.copy() + [action_to_neighbour], cost_to_neighbour),
                        priority=cost_to_neighbour
                    )

                if debug:
                    print('\033[93m')  # yellow
                    print(f'| {successor_node}\t| {cost_to_neighbour}\t| {action_to_neighbour}\t|')
                    print('\033[0m')

        explored.add(current_node)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    def dfsPriorityFunction(_node, _successor_state, cost_to_node):
        return cost_to_node - 1

    return generalSearch(problem=problem, prirotyFunction=dfsPriorityFunction)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    def bfsPriorityFunction(_node, _successor_state, cost_to_node):
        return cost_to_node + 1

    return generalSearch(problem=problem, prirotyFunction=bfsPriorityFunction)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    def ucsPriorityFunction(_node, successor_state, cost_to_node):
        _, _, step_cost = successor_state
        # print(f'---\t Node: {_node}, Successor: {successor_state[0]}')
        # print(f'---\t Cost: {cost_to_node}, Step Cost: {step_cost}')

        return cost_to_node + step_cost

    return generalSearch(problem=problem, prirotyFunction=ucsPriorityFunction)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def aStarPriorityFunction(node, successor_state, cost_to_node):
        prev_heuristic = heuristic(node, problem)
        successor_node, _, step_cost = successor_state
        new_heuristic = heuristic(successor_node, problem)

        return cost_to_node - prev_heuristic + (step_cost + new_heuristic)

    return generalSearch(problem=problem, prirotyFunction=aStarPriorityFunction)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
