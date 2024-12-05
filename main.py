import random
import time

fp = open("tictactoemoves.csv", "a")

def play(board):
    
    empty_spaces = list(filter(lambda x: board[x] == "", range(len(board))))
    return random.choice(empty_spaces)

def checkWinner(board):
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
        if (board[A] and board[A] == board[B] and board[A] == board[C]):
            return board[A]
    return False



def isDraw(board):
    for i in range(len(board)):
        if not (board[i]):
            return False
    return True

def getBoardState(board):
    sentence = ""
    for i in range(len(board)):
        if board[i] == "":
            sentence += "1,0,0,"
        elif board[i] == "X":
            sentence += "0,1,0,"
        else:
            sentence += "0,0,1,"
    return sentence



def game():
    board = [''] * 9
    current_player = 'X'
    player1 = "X"
    player2 = "O"

    while True:
        time.sleep(5)
        print(board[:3])
        print(board[3:6])
        print(board[6:9])
        print('')
        print('----------------------------------------')
        print('')
        print('')

        winner = checkWinner(board)
        if winner:

            break
        draw = isDraw(board)
        if draw and not winner:

            break
        if current_player == "X":
            choice = play(board)
            board[choice] = player1
            sentence= getBoardState(board) + str(choice) +"\n"
            fp.write(sentence)

            current_player = "O"

        else:
            choice = play(board)
            board[choice] = player2
            sentence= getBoardState(board) + str(choice) +"\n"
            fp.write(sentence)
            current_player = "X" 
i = 0
game()






