import numpy as np
import pickle
import random
import os
from collections import defaultdict

class TicTacToeQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, data_file="q_table.pkl"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.data_file = data_file

        # โหลด Q-table หากมีไฟล์อยู่แล้ว
        self.load_q_table()

    def save_q_table(self):
        """บันทึก Q-table ลงไฟล์"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"บันทึก Q-table แล้ว ({len(self.q_table)} states)")

    def load_q_table(self):
        """โหลด Q-table จากไฟล์"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                loaded_data = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), loaded_data)
            print(f"โหลด Q-table แล้ว ({len(self.q_table)} states)")
        else:
            print("ไม่พบไฟล์ Q-table สร้างใหม่")

    def state_to_string(self, board):
        """แปลงสถานะกระดานเป็น string เพื่อใช้เป็น key"""
        flat_board = board.flatten()
        converted = [str(2 if x == -1 else x) for x in flat_board]
        return ''.join(converted)

    def get_available_actions(self, board):
        """หาตำแหน่งที่ยังว่างบนกระดาน"""
        return [i for i in range(9) if board.flatten()[i] == 0]

    def check_win_condition(self, board, player):
        """ตรวจสอบว่าผู้เล่นชนะหรือไม่"""
        # ตรวจแถว
        for row in board:
            if np.sum(row == player) == 3:
                return True

        # ตรวจคอลัมน์
        for col in range(3):
            if np.sum(board[:, col] == player) == 3:
                return True

        # ตรวจเส้นทแยง
        if np.sum([board[i, i] == player for i in range(3)]) == 3:
            return True
        if np.sum([board[i, 2 - i] == player for i in range(3)]) == 3:
            return True

        return False

    def get_action_reasoning(self, board, action, q_values, training=False):
        """อธิบายเหตุผลในการเลือก action"""
        reasons = []

        # แสดง Q-value
        state = self.state_to_string(board)
        q_val = self.q_table[state][action - 1]
        if q_val > 0:
            reasons.append(f"📈 Q-value: {q_val:.3f} (ประสบการณ์เชิงบวก)")
        elif q_val < 0:
            reasons.append(f"📉 Q-value: {q_val:.3f} (ประสบการณ์เชิงลบ)")
        else:
            reasons.append("🆕 Q-value: 0.000 (ยังไม่เคยลองตำแหน่งนี้)")

        # แสดงสถิติการเรียนรู้
        if training and random.random() < self.epsilon:
            reasons.append(f"🎲 การเลือกแบบสุ่มเพื่อสำรวจ (ε={self.epsilon})")

        return reasons

    def choose_action(self, board, training=True, show_thinking=False):
        """เลือก action โดยใช้ epsilon-greedy strategy"""
        state = self.state_to_string(board)
        available_actions = self.get_available_actions(board)

        if not available_actions:
            return None

        # สร้าง Q-values สำหรับทุก action ที่เป็นไปได้
        q_values = {}
        for action in available_actions:
            q_values[action] = self.q_table[state][action]

        if show_thinking:
            print("\n🧠 AI กำลังคิด...")
            print(f"🔍 สถานการณ์ปัจจุบัน: {state}")

            # ตรวจสอบว่าเคยเจอสถานการณ์นี้มาก่อนหรือไม่
            if state in self.q_table and any(self.q_table[state].values()):
                print("📚 AI เคยเจอสถานการณ์นี้มาก่อน!")
            else:
                print("🆕 สถานการณ์ใหม่ที่ AI ยังไม่เคยเจอ")

            print("📊 Q-values สำหรับแต่ละตำแหน่ง:")

            # แสดง Q-values ทั้งหมด
            for action in available_actions:
                q_val = q_values[action]
                if q_val > 0.001:
                    print(f"   ช่อง {action + 1}: {q_val:.3f} 📈")
                elif q_val < -0.001:
                    print(f"   ช่อง {action + 1}: {q_val:.3f} 📉")
                else:
                    print(f"   ช่อง {action + 1}: {q_val:.3f} ⚪")

        # ในระหว่างการเทรน ใช้ epsilon-greedy
        if training and random.random() < self.epsilon:
            action = random.choice(available_actions)
            selected_action = action + 1

            if show_thinking:
                print(f"\n🎲 เลือกแบบสุ่มเพื่อสำรวจ: ช่อง {selected_action}")

            return selected_action

        # เลือก action ที่มี Q-value สูงสุด
        best_action = None
        best_value = -float('inf')

        for action in available_actions:
            q_value = q_values[action]
            if q_value > best_value:
                best_value = q_value
                best_action = action

        # หากไม่มี Q-value ให้เลือกแบบสุ่ม
        if best_action is None:
            best_action = random.choice(available_actions)

        selected_action = best_action + 1

        if show_thinking:
            reasons = self.get_action_reasoning(board, selected_action, q_values, training)
            print(f"\n✅ ตัดสินใจเลือก: ช่อง {selected_action}")
            print("💭 เหตุผล:")
            for reason in reasons:
                print(f"   {reason}")

        return selected_action

    def update_q_value(self, state, action, reward, next_state, show_learning=False):
        """อัพเดท Q-value ด้วย Q-learning formula"""
        old_q = self.q_table[state][action]

        # หา max Q-value ของ state ถัดไป
        next_board_flat = [int(x) if x != '2' else -1 for x in next_state]
        next_board = np.array(next_board_flat).reshape(3, 3)
        next_available_actions = self.get_available_actions(next_board)

        if next_available_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_available_actions])
        else:
            max_next_q = 0

        # Q-learning formula
        new_q = old_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - old_q
        )

        self.q_table[state][action] = new_q

        if show_learning and abs(new_q - old_q) > 0.001:
            print(f"📝 การเรียนรู้: ช่อง {action + 1} | {old_q:.3f} → {new_q:.3f} (Δ{new_q - old_q:+.3f})")
            if reward > 0:
                print(f"   ✅ ได้รับรางวัล +{reward} จากการเดินนี้")
            elif reward < 0:
                print(f"   ❌ โดนลงโทษ {reward} จากการเดินนี้")
            else:
                print(f"   ⚖️6 ไม่ได้รับรางวัลหรือบทลงโทษจากการเดินนี้")

    def get_learning_stats(self):
        """แสดงสถิติการเรียนรู้"""
        total_states = len(self.q_table)
        total_actions = sum(len(actions) for actions in self.q_table.values())

        positive_q = 0
        negative_q = 0
        zero_q = 0

        for state_actions in self.q_table.values():
            for q_val in state_actions.values():
                if q_val > 0:
                    positive_q += 1
                elif q_val < 0:
                    negative_q += 1
                else:
                    zero_q += 1

        return {
            'total_states': total_states,
            'total_actions': total_actions,
            'positive_q': positive_q,
            'negative_q': negative_q,
            'zero_q': zero_q
        }


class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        """รีเซ็ตเกม"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def make_move(self, position, player):
        """ทำการเดิน (position: 1-9)"""
        if position is None or position < 1 or position > 9:
            return False

        position = position - 1

        if self.board.flatten()[position] != 0:
            return False

        row, col = position // 3, position % 3
        self.board[row, col] = player
        return True

    def check_winner(self):
        """ตรวจสอบผู้ชนะ"""
        # ตรวจแถว
        for row in self.board:
            if abs(sum(row)) == 3:
                return row[0]

        # ตรวจคอลัมน์
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]

        # ตรวจเส้นทแยง
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return self.board[0, 0]

        if abs(sum([self.board[i, 2 - i] for i in range(3)])) == 3:
            return self.board[0, 2]

        # ตรวจเสมอ
        if 0 not in self.board.flatten():
            return 0

        return None

    def print_board(self):
        """แสดงกระดาน"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        RED = '\033[31m'
        GREEN = '\033[32m'
        BLUE = '\033[34m'
        RESET = '\033[0m'

        print("  " + "=" * 13)
        for i in range(3):
            row_str = "  |"
            for j in range(3):
                if self.board[i, j] == 0:
                    pos = i * 3 + j + 1
                    row_str += f" {GREEN}{pos}{RESET} |"
                elif self.board[i, j] == 1:
                    row_str += f" {RED}{symbols[self.board[i, j]]}{RESET} |"
                else:
                    row_str += f" {BLUE}{symbols[self.board[i, j]]}{RESET} |"
            print(row_str)
            if i < 2:
                print("  " + "-" * 13)
        print("  " + "=" * 13)