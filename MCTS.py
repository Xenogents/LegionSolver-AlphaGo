import numpy as np
import tensorflow as tf

# Monte Carlo Tree Search in this code is modified such that it can play any specified number of games in tandem.
# Code becomes more complex to understand but saves significant time on computation

# Root node of tree search is defined differently in order to add Dirichlet noise and because it has no parents
class RootNode:
    def __init__(self, priors, legal_actions):
        self.parent = None
        self.children = None
        self.children_priors = None
        self.children_values = None
        self.children_visits = None
        self.children_legal_actions = None
        self.visits = 0
        self.priors = priors
        self.noise = []

    # Model assumes all actions are possible at all points in time, must be normalized before using
    def get_normalized_priors(self):
        priors = self.priors[self.children_legal_actions]
        priors = priors / np.sum(priors)
        if len(self.noise) == 0: self.noise = np.random.dirichlet([0.03]*len(priors))
        if len(priors) > 1: priors = priors = 0.75*priors + 0.25*self.noise
        return priors

# Information in tree is stored in a somewhat roundabout way, in order to access certain pieces of information about
# your current node, you have to go to its parent and then that parent has information about all its children
# Unsure if efficient but the code I was referencing from had this node structure so I left it as is
class Node:
    def __init__(self, idx, parent, action):
        self.children = None
        self.children_priors = None
        self.children_values = None
        self.children_visits = None
        self.children_legal_actions = None
        self.idx = idx
        self.parent = parent
        self.action = action

    def get_normalized_priors(self):
        priors = self.priors[self.children_legal_actions]
        return priors / np.sum(priors)
    @property
    def priors(self):
        return self.parent.children_priors[self.idx]
    @priors.setter
    def priors(self, x):
        self.parent.children_priors[self.idx] = x
    @property
    def visits(self):
        return self.parent.children_visits[self.idx]
    @visits.setter
    def visits(self, x):
        self.parent.children_visits[self.idx] = x
    @property
    def value(self):
        return self.parent.children_values[self.idx]
    @value.setter
    def value(self, x):
        self.parent.children_values[self.idx] = x

# Choosing function for search tree, exploration constant c can be modified freely
def get_ucb_scores(node, c=0.1):
    priors = node.get_normalized_priors()
    return node.children_values + c*priors*node.visits**0.5 / (node.children_visits+1)

# Traverse the tree until you reach an unexpanded node
def select(root, game):
    current = root
    while current.children is not None:
        ucb_scores = get_ucb_scores(current)
        current = current.children[np.argmax(ucb_scores)]
        game.step(current.action)
    return current

def expand(games, leaves, model, children_actions):
    states = []
    pieces = []
    # Record all the states for a single action taken in each direction, this way a call to the model to perform a prediction
    # only needs to be performed a single time.
    for i, game in enumerate(games):
        leaves[i].children_legal_actions = [action[0] for action in children_actions[i]]
        for action in children_actions[i]:
            position = game.position
            game.step(action)
            states.append(game.get_board())
            pieces.append(game.get_available_pieces())
            game.undo((action[0], position))
    states = tf.convert_to_tensor(np.array(states), dtype="float32")
    pieces = tf.convert_to_tensor(np.array(pieces), dtype="float32")

    children_priors, children_values = model(states, pieces)
    children_values = children_values[:,0]
    # Cannot embed multiple dimensions into prediction batch, so we must seperate the dimensions manually
    num_branches = np.cumsum([0]+[len(children_actions[i]) for i in range(len(games))])
    for i, leaf in enumerate(leaves):
        leaf.children = [Node(idx, leaf, action) for idx, action in enumerate(children_actions[i])]
        leaf.children_priors = children_priors.numpy()[num_branches[i]:num_branches[i+1]]
        leaf.children_values = children_values.numpy()[num_branches[i]:num_branches[i+1]]
        leaf.children_visits = np.zeros(len(children_actions[i]))

def backpropagate(leaf, value):
    current = leaf
    while current.parent is not None:
        current.value = (current.value * current.visits + value) / (current.visits + 1)  # incremental mean update
        current.visits += 1
        current = current.parent
    current.visits += 1

# Monte Carlo Tree search
def search(games, model, iterations):
    states = tf.convert_to_tensor(np.array([game.get_board() for game in games]), dtype="float32")
    pieces = tf.convert_to_tensor(np.array([game.get_available_pieces() for game in games]), dtype="float32")
    priors = model(states, pieces)[0].numpy()
    roots = [RootNode(priors[i], games[i].get_legal_actions()) for i in range(len(games))]

    positions = [game.position for game in games]
    states = [game.state.copy() for game in games]
    available_pieces = [game.available_pieces.copy() for game in games]
    for _ in range(iterations):
        leaves = np.array([select(roots[i], games[i]) for i in range(len(games))])
        children_actions = [games[i].get_legal_actions() for i in range(len(games))]
        while (unfinished_games:=[i for i in range(len(games)) if children_actions[i] != []]) != []:
            expand(games[unfinished_games], leaves[unfinished_games], model,
                            [actions for actions in children_actions if actions != []])
            children_actions = [[] for _ in range(len(games))]
            for game_idx in unfinished_games:
                leaves[game_idx] = leaves[game_idx].children[np.argmax(leaves[game_idx].children_values)]
                games[game_idx].step(leaves[game_idx].action)
                children_actions[game_idx] = games[game_idx].get_legal_actions()
        for i, game in enumerate(games):
            backpropagate(leaves[i], game.value())
            game.position = positions[i]
            game.state = states[i].copy()
            game.available_pieces = available_pieces[i].copy()
    return roots

def add(a,b):
    return a+b