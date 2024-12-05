import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class TicTacToeEnvironment:
    def __init__(self):
        self.board = [''] * 9
        self.current_player = 'X'

    def get_state(self):
        return self.board
    
    def get_valid_actions(self):
        return list(filter(lambda x: self.board[x] == "", range(len(self.board))))
    def isDraw(self):
        for i in self.board:
            if i == '':
                return False
        return True
    def isWinner(self):
        winning = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        for i in range(len(winning)):
            A, B, C = winning[i]
            if (self.board[A] and self.board[A] == self.board[B] and self.board[A] == self.board[C]):
                return self.board[A]
        return False

    def step(self, action):
        self.board[action] = self.current_player
        ended = False
        if self.isWinner() == self.current_player:
            reward = 1
            ended = True
        elif self.isWinner() and self.isWinner() != self.current_player:
            reward = -1
            ended = True
        elif self.isDraw():
            reward = 0.1
            ended = True
        else:
            reward = 0
            ended = False
        return self.one_hot_encode_board(self.board), reward, ended
    def one_hot_encode_board(self, board_state):
        encoded_board = []
        for cell in board_state:
            if cell == 'X':
                encoded_board.extend([0, 1, 0]) 
            elif cell == 'O':
                encoded_board.extend([0, 0, 1]) 
            else:
                encoded_board.extend([1, 0, 0]) 
        # X_train_one_hot = [np.array(encoded_board)]

        # X_train_one_hot = np.array(X_train)
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(encoded_board)

        # X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
        return np.array(encoded_board, dtype=np.float32)

    def reset(self):
        self.board = [''] * 9
        self.current_player = 'X'
        return self.one_hot_encode_board(self.board)




env = TicTacToeEnvironment()
state = env.get_state()
print(state)
valid_actions = env.get_valid_actions()
print(valid_actions)
action = valid_actions[0]
print(action)
next_state, reward, done = env.step(action)
print(next_state, reward, done)


