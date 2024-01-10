import numpy as np

# Very roughly coded ruleset for the game with manually defined pieces, not super important to code this bit cleanly

# Finding legal actions takes significant computing power, this dictionary will spit back any previously computed
# legal actions. Will only overflow when given larger cases.
class Polyomino():
    def __init__(self, h, w, idx):
        # 1d Encoding of board makes logic much simpler
        self.idx = idx
        self.h = h
        self.w = w
        self.state = np.zeros(h*w)
        self.position = 0

        # Representation not unique for pieces longer than the width of the board, does not scale but works for our cases
        # Currently commented out this set of tetrominoes, used when testing easier cases
        # self.pieces = np.array([
        #     [0,1,2,w+2], [0,2*w-1,w,2*w], [0,w,w+1,w+2], [0,w,2*w,1], [0,w,1,2], [0,1,w+1,2*w+1], [0,w-2,w-1,w], [0,w,2*w,2*w+1], # L piece
        #     [0,1,w+1,w+2], [0,2*w-1,w-1,w], [0,w-1,w,1], [0,w,w+1,2*w+1], # Z piece
        #     [0,1,w+1,2], [0,w-1,w,2*w], [0,w-1,w,w+1], [0,w,2*w,w+1], # T piece
        #     [0,1,2,3], [0,w,2*w,3*w], [0,1,w,w+1]]) # I and square piece
        # self.max_pieces = [2,2,2,2,2]
        # self.available_pieces = self.max_pieces.copy()
        # self.kernel = [(1,0),(2,0),(3,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),(-1,2),(0,2),(1,2),(0,3)]
        # self.flat_kernel = [grid[1]*self.w + grid[0] for grid in self.kernel]

        # Sets of pentominoes and all of their rotations/reflections. Small optimization can be made by requiring
        # that these representations all start with 0. Other useful precomputed values are just manually coded.
        self.pieces = np.array([
            [0,1,2,w+2,w+3], [0,w,2*w-1,2*w,3*w-1], [0,1,w+1,w+2,w+3], [0,w-1,2*w-1,3*w-1,w], # Z piece
            [0,1,w-2,w-1,w], [0,w,2*w,2*w+1,3*w+1], [0,1,2,w-1,w], [0,w,w+1,2*w+1,3*w+1], # Z piece reflected
            [0,1,w+1,2*w+1,2*w+2], [0,w-2,2*w-2,w-1,w], [0,2*w-1,w,2*w,1], [0,w,w+1,w+2,2*w+2], # S piece
            [0,w,2*w,w+1,w+2], [0,1,w+1,2*w+1,2], [0,w-2,w-1,w,2*w], [0,2*w-1,w,2*w,2*w+1], # T piece
            [0,1,2,3,4], [0,w,2*w,3*w,4*w], [0,w-1,w,2*w,w+1]]) # I and cross piece
        self.max_pieces = [2,2,2,6,4]
        self.available_pieces = self.max_pieces.copy()
        self.kernel = [(1,0),(2,0),(3,0),(4,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),(3,1),
                        (-2,2),(-1,2),(0,2),(1,2),(2,2),(-1,3),(0,3),(1,3),(0,4)]
        self.flat_kernel = [grid[1]*self.w + grid[0] for grid in self.kernel]

        # Retrieve correct piece for a given action and vice versa
        self.piece_divider = {0:0, 1:8, 2:12, 3:16, 4:18, 5:19}
        self.piece_from_action = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1,11:1,12:2,13:2,14:2,15:2,16:3,17:3,18:4}

    # Get all possible actions from a given state
    def get_legal_actions(self):
        actions = []
        available_pieces = []
        prev_position = self.position
        for i, num_pieces in enumerate(self.available_pieces):
            if num_pieces != 0:
                available_pieces.extend(self.pieces[self.piece_divider[i] : self.piece_divider[i+1]])
        available_pieces = set(tuple(piece) for piece in available_pieces)
        legal_grids = self.generate_legal_grids()
        while (self.position < self.h*self.w) and (self.state[self.position] == 1 or
                    (actions := [(i, self.position) for i, piece in enumerate(self.pieces)
                        if tuple(piece) in available_pieces and all(x in legal_grids for x in piece[1:])]) == []):
            self.position += 1
            legal_grids = self.generate_legal_grids()

        self.position = prev_position
        return actions

    # Checks the general area around where the piece is to be placed and returns which grids are free
    def generate_legal_grids(self):
        x = self.position % self.w
        y = self.position // self.w
        kernel = [self.flat_kernel[i] for i, grid in enumerate(self.kernel)
                  if 0 <= grid[0]+x < self.w and grid[1]+y < self.h and self.state[self.position + self.flat_kernel[i]] == 0]
        return kernel

    # Place a piece
    def step(self, action):
        self.position = action[1]
        self.state[self.pieces[action[0]] + self.position] = 1
        self.available_pieces[self.piece_from_action[action[0]]] -= 1

    # Unplace a piece
    def undo(self, action):
        self.state[self.pieces[action[0]] + self.position] = 0
        self.position = action[1]
        self.available_pieces[self.piece_from_action[action[0]]] += 1

    # Resulting value of the game, only called once it has been determined that there are no legal actions remaining
    def value(self):
        value = np.sum(self.state)
        while (self.position < self.h*self.w) and (self.state[self.position] == 1):
            self.position += 1
        # Place a remaining piece allowing for overlaps or protruding edges to get a more continuous value function
        if self.position < self.h*self.w:
            piece = np.argmax(np.array(self.available_pieces))
            max_grids_covered = 0
            for action in self.pieces[self.piece_divider[piece] : self.piece_divider[piece+1]]:
                grids_covered = 0
                left_x = 0
                for i, grid in enumerate(action):
                    if grid + self.position < self.h*self.w:
                        offset = self.position % self.w
                        if i == 0 and grid > 0:
                            left_x = grid % self.w - self.w
                            if left_x + offset >= 0 and self.state[grid + self.position] == 0:
                                grids_covered += 1
                        else:
                            if grid % self.w + offset < self.w and self.state[grid + self.position] == 0:
                                grids_covered += 1
                if grids_covered > max_grids_covered:
                    max_grids_covered = grids_covered
            value += max_grids_covered
        return value / (self.h*self.w)

    # Method to reset the game to avoid creating too many game instances
    def reset_game(self):
        self.state = np.zeros(self.h*self.w)
        self.position = 0
        self.available_pieces = self.max_pieces.copy()

    def get_board(self):
        return np.reshape(self.state.copy(), (self.h, self.w,1))

    def get_available_pieces(self):
        return self.available_pieces.copy()

def add(a,b):
    return a+b