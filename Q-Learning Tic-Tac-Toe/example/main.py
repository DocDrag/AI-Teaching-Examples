import numpy as np
import json
import os
from algorithm import TicTacToeQLearning
from algorithm import TicTacToeGame

def load_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' not found. Returning empty dictionary.")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Data successfully loaded from '{file_path}'.")
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. File might be corrupted or empty.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return {}

def update_json_file(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully updated to '{file_path}'.")
        return True
    except Exception as e:
        print(f"Error: Could not update JSON file '{file_path}': {e}")
        return False

def rule_based_ai_move(board):
    """AI ฝั่ง O ตัดสินใจแบบ if-else"""
    def check_win(b, p):
        for i in range(3):
            if np.sum(b[i, :] == p) == 2 and 0 in b[i, :]:
                return i, list(b[i, :]).index(0)
            if np.sum(b[:, i] == p) == 2 and 0 in b[:, i]:
                return list(b[:, i]).index(0), i
        if np.sum([b[i, i] == p for i in range(3)]) == 2:
            for i in range(3):
                if b[i, i] == 0:
                    return i, i
        if np.sum([b[i, 2 - i] == p for i in range(3)]) == 2:
            for i in range(3):
                if b[i, 2 - i] == 0:
                    return i, 2 - i
        return None

    def is_fork_move(b, p):
        """เช็คว่าฝ่าย p มีโอกาสจะ fork หรือไม่"""
        count = 0
        for i in range(3):
            for j in range(3):
                if b[i, j] == 0:
                    b[i, j] = p
                    if check_win(b, p):
                        count += 1
                    b[i, j] = 0
        return count >= 2

    # 1. ชนะได้ให้ชนะ
    pos = check_win(board, -1)
    if pos: return pos[0] * 3 + pos[1] + 1

    # 2. กันไม่ให้ X ชนะ
    pos = check_win(board, 1)
    if pos: return pos[0] * 3 + pos[1] + 1

    # 3. เลือกตรงกลางถ้าว่าง
    if board[1, 1] == 0: return 5

    # 4. เลือกมุมว่างก่อน
    for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        if board[i, j] == 0:
            return i * 3 + j + 1

    # 5. ป้องกัน Fork ของ X
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = 1
                if is_fork_move(board, 1):
                    board[i, j] = 0
                    return i * 3 + j + 1
                board[i, j] = 0

    # 6. สร้าง Fork ให้ O
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = -1
                if is_fork_move(board, -1):
                    board[i, j] = 0
                    return i * 3 + j + 1
                board[i, j] = 0

    # 7. ไม่งั้นเลือกช่องว่างแรก
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                return i * 3 + j + 1

    return None

def train_ai(ai, game_data, episodes=1000):
    """เทรน AI โดยให้เล่นกับตัวเอง"""
    print(f"เริ่มเทรน AI {episodes} ตา...")

    wins = {'ai': 0, 'opponent': 0, 'draw': 0}

    for episode in range(episodes):
        game = TicTacToeGame()
        game_history = []

        while True:
            current_state = game.board.copy()

            if game.current_player == 1:
                action = ai.choose_action(current_state, training=True)
            else:
                a = rule_based_ai_move(current_state)
                if a is not None:
                    action = a
                else:
                    action = ai.choose_action(current_state, training=True)

            if action is None:
                break

            game_history.append([
                ai.state_to_string(current_state),
                action - 1,
                game.current_player
            ])

            game.make_move(action, game.current_player)

            winner = game.check_winner()
            if winner is not None:
                if winner == 1:
                    wins['ai'] += 1
                elif winner == -1:
                    wins['opponent'] += 1
                else:
                    wins['draw'] += 1

                for i, (state, act, player) in enumerate(game_history):
                    if winner == 1:
                        reward = 2 if player == 1 else -2
                    elif winner == -1:
                        reward = -2 if player == 1 else 2
                    else:
                        reward = -1 if player == 1 else 1

                    if i < len(game_history) - 1:
                        next_state = game_history[i + 1][0]
                    else:
                        next_state = ai.state_to_string(game.board)

                    if player == 1:
                        ai.update_q_value(state, act, reward, next_state)

                break

            game.current_player *= -1

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} - "
                  f"AI ชนะ: {wins['ai']}, แพ้: {wins['opponent']}, เสมอ: {wins['draw']}")

    print(f"\nการเทรนเสร็จสิ้น!")
    print(f"ผลรวม - AI ชนะ: {wins['ai']}, แพ้: {wins['opponent']}, เสมอ: {wins['draw']}")
    game_data['win'] += wins['ai']
    game_data['lose'] += wins['opponent']
    game_data['draw'] += wins['draw']

    # แสดงสถิติการเรียนรู้
    stats = ai.get_learning_stats()
    print(f"\n📊 สถิติการเรียนรู้:")
    print(f"   สถานการณ์ที่เรียนรู้: {stats['total_states']}")
    print(f"   การกระทำทั้งหมด: {stats['total_actions']}")
    print(f"   ประสบการณ์เชิงบวก: {stats['positive_q']}")
    print(f"   ประสบการณ์เชิงลบ: {stats['negative_q']}")

    ai.epsilon = 0.05
    ai.save_q_table()


def play_with_human(ai, game_data):
    """เล่นกับผู้เล่นจริง"""
    print("\n=== เริ่มเล่นกับ AI ===")
    print("คุณเป็น O (สีน้ำเงิน), AI เป็น X (สีแดง)")
    print("ใส่ตำแหน่งเป็นตัวเลข 1-9")

    show_ai_thinking = True
    show_learning = True

    game_count = 0
    wins = {'human': 0, 'ai': 0, 'draw': 0}

    while True:
        game = TicTacToeGame()
        game_history = []
        game_count += 1

        print(f"\n{'=' * 50}")
        print(f"เกมที่ {game_count}")
        print('=' * 50)

        # แสดงสถิติการเรียนรู้ปัจจุบัน
        if game_count == 1:
            stats = ai.get_learning_stats()
            print(f"📚 AI ได้เรียนรู้จาก {stats['total_states']} สถานการณ์แล้ว")

        game.print_board()

        while True:
            current_state = game.board.copy()

            if game.current_player == 1:  # AI เล่น
                action = ai.choose_action(current_state, training=False, show_thinking=show_ai_thinking)

                if action is None:
                    break

                game_history.append([
                    ai.state_to_string(current_state),
                    action - 1,
                    game.current_player
                ])

                game.make_move(action, game.current_player)
                print(f"\n🤖 AI เลือกตำแหน่ง {action}")

            else:  # Human เล่น
                game.print_board()
                try:
                    human_input = input(
                        "\nใส่ตำแหน่งของคุณ (1-9), 't' เพื่อเปิด/ปิดการแสดงความคิด AI, 'l' เพื่อเปิด/ปิดการแสดงการเรียนรู้, หรือ 'q' เพื่อออก: ")

                    if human_input.lower() == 'q':
                        return
                    elif human_input.lower() == 't':
                        show_ai_thinking = not show_ai_thinking
                        status = "เปิด" if show_ai_thinking else "ปิด"
                        print(f"การแสดงความคิดของ AI: {status}")
                        continue
                    elif human_input.lower() == 'l':
                        show_learning = not show_learning
                        status = "เปิด" if show_learning else "ปิด"
                        print(f"การแสดงการเรียนรู้: {status}")
                        continue

                    action = int(human_input)
                    if action < 1 or action > 9:
                        print("❌ ตำแหน่งต้องอยู่ระหว่าง 1-9")
                        continue

                    if not game.make_move(action, game.current_player):
                        print("❌ ตำแหน่งนี้ถูกใช้แล้ว!")
                        continue

                    game_history.append([
                        ai.state_to_string(current_state),
                        action - 1,
                        game.current_player
                    ])

                except ValueError:
                    print("❌ กรุณาใส่ตัวเลข!")
                    continue

            winner = game.check_winner()
            if winner is not None:
                print("\n" + "=" * 50)
                game.print_board()

                if winner == 1:
                    print("🤖 AI ชนะ!")
                    wins['ai'] += 1
                    game_data["win"] += 1
                elif winner == -1:
                    print("🎉 คุณชนะ!")
                    wins['human'] += 1
                    game_data["lose"] += 1
                else:
                    print("🤝 เสมอ!")
                    wins['draw'] += 1
                    game_data["draw"] += 1

                # แสดงการเรียนรู้ของ AI
                if show_learning:
                    print(f"\n🧠 AI กำลังเรียนรู้จากเกมนี้...")

                # AI เรียนรู้จากการเล่นกับผู้เล่น
                learned_something = False
                for i, (state, act, player) in enumerate(game_history):
                    if player == 1:  # เฉพาะการเดินของ AI
                        old_q = ai.q_table[state][act]

                        if winner == 1:
                            reward = 1
                        elif winner == -1:
                            reward = -1
                        else:
                            reward = 0

                        if i < len(game_history) - 1:
                            next_state = game_history[i + 1][0]
                        else:
                            next_state = ai.state_to_string(game.board)

                        ai.update_q_value(state, act, reward, next_state, show_learning=show_learning)

                        # ตรวจสอบว่ามีการเรียนรู้หรือไม่
                        new_q = ai.q_table[state][act]
                        if abs(new_q - old_q) > 0.001:
                            learned_something = True

                print(f"\n📊 สถิติ - คุณ: {wins['human']}, AI: {wins['ai']}, เสมอ: {wins['draw']}")
                game_data['all_game'] = game_data['win'] + game_data['lose'] + game_data['draw']
                print(f"\n📊 สถิติ AI - จำนวนการเล่น: {game_data['all_game']}, ชนะ: {game_data['win']}, แพ้: {game_data['lose']}, เสมอ: {game_data['draw']}")

                # แสดงการเรียนรู้ของ AI
                if learned_something:
                    if winner == -1:  # หาก AI แพ้
                        print("🧠 AI ได้เรียนรู้จากความผิดพลาดนี้แล้ว (Q-values ถูกปรับลด)")
                    elif winner == 1:  # หาก AI ชนะ
                        print("🧠 AI ได้เสริมสร้างกลยุทธ์ที่ใช้ในเกมนี้ (Q-values ถูกปรับเพิ่ม)")
                    else:
                        print("🧠 AI ได้เรียนรู้จากเกมเสมอนี้")
                else:
                    print("📝 AI ไม่ได้เรียนรู้อะไรใหม่จากเกมนี้ (สถานการณ์คุ้นเคย)")

                break

            game.current_player *= -1

        if winner is not None and input("\nเล่นอีกไหม? (y/n): ").lower() == 'n':
            break

    ai.save_q_table()
    print("💾 บันทึกการเรียนรู้แล้ว!")


def main():
    print("🎮 === Q-Learning Tic-Tac-Toe AI ===")
    print("AI ที่เรียนรู้และแสดงกระบวนการคิด")

    json_file = 'game_stats.json'
    if not os.path.exists(json_file):
        initial_data = {
            "all_game": 0,
            "win": 0,
            "lose": 0,
            "draw": 0
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=4, ensure_ascii=False)
        print(f"Created initial file '{json_file}' with default data.")
    game_data = load_json_data(json_file)

    ai = TicTacToeQLearning()

    if not os.path.exists(ai.data_file) or len(ai.q_table) == 0:
        print("ไม่พบข้อมูลการเทรน กำลังเทรน AI ใหม่...")
        train_ai(ai, game_data, 1000)
    else:
        stats = ai.get_learning_stats()
        print(f"พบข้อมูลการเทรนแล้ว ({stats['total_states']} สถานการณ์)")
        print("พร้อมเล่น!")

    play_with_human(ai, game_data)

    if update_json_file(json_file, game_data):
        print("Updated game data:", game_data)


if __name__ == "__main__":
    main()