import numpy as np
import random
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X เริ่มก่อน

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def make_move(self, action):
        r, c = divmod(action, 3)
        if self.board[r, c] == 0:
            self.board[r, c] = self.current_player
            self.current_player *= -1
            return True
        return False

    def check_winner(self):
        b = self.board
        lines = list(b) + list(b.T) + [b.diagonal(), np.fliplr(b).diagonal()]
        for line in lines:
            if abs(sum(line)) == 3:
                return np.sign(sum(line))
        if 0 not in b:
            return 0  # เสมอ
        return None

    def print_board(self):
        symbols = {0: None, 1: "X", -1: "O"}
        print("\nกระดาน:")
        for i in range(3):
            row = []
            for j in range(3):
                val = self.board[i, j]
                if val == 0:
                    row.append(str(i * 3 + j + 1))  # แสดงเลขช่องว่าง
                else:
                    row.append(symbols[val])
            print(" | ".join(row))
            if i < 2:
                print("--+---+--")

class QLearningAI:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def state_key(self, board):
        return str(board.flatten().tolist())

    def choose_action(self, board, training=True):
        state = self.state_key(board)
        available = [i for i in range(9) if board.flatten()[i] == 0]
        if training and random.random() < self.epsilon:
            return random.choice(available)
        q_values = self.q_table[state]
        return max(available, key=lambda a: q_values[a])

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state]) if next_state else 0
        self.q_table[state][action] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)

def train(episodes=1000):
    ai = QLearningAI()
    game = TicTacToe()

    for ep in range(episodes):
        state = ai.state_key(game.reset())
        while True:
            action = ai.choose_action(game.board, training=True)
            game.make_move(action)
            winner = game.check_winner()
            next_state = ai.state_key(game.board)

            if winner is not None:
                reward = 1 if winner == 1 else -1 if winner == -1 else 0
                ai.update(state, action, reward, None)
                break
            else:
                ai.update(state, action, 0, next_state)
                state = next_state
    return ai

def play_against_ai(ai):
    game = TicTacToe()
    game.reset()
    history = []  # เก็บ state, action ของ AI

    print("\n=== เริ่มเกมใหม่ ===")
    game.print_board()

    while True:
        state = ai.state_key(game.board)

        if game.current_player == -1:  # คนเล่น O
            move = int(input("เลือกตำแหน่ง (1-9): ")) - 1
            if not game.make_move(move):
                print("❌ ช่องนี้ใช้แล้ว!")
                continue
        else:  # AI เล่น X
            move = ai.choose_action(game.board, training=False)
            game.make_move(move)
            history.append((state, move))  # เก็บไว้เพื่อเรียนรู้ตอนหลัง
            print(f"🤖 AI เลือก {move+1}")

        game.print_board()
        winner = game.check_winner()
        if winner is not None:
            # แจก reward ให้ AI จากผลเกม
            for state, action in history:
                if winner == 1:      # AI ชนะ
                    reward = 1
                elif winner == -1:   # AI แพ้
                    reward = -1
                else:                # เสมอ
                    reward = 0
                ai.update(state, action, reward, None)
            if winner == 1:
                print("🤖 AI ชนะ!")
            elif winner == -1:
                print("🎉 คุณชนะ!")
            else:
                print("🤝 เสมอ!")
            break

if __name__ == "__main__":
    trained_ai = train()
    while True:
        play_against_ai(trained_ai)
