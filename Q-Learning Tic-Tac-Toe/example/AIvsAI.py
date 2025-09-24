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

def train_ai(ai, ai2, game_data, episodes=1000):
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
                action = ai2.choose_action(current_state, training=True)

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
                        reward = 1 if player == 1 else -1
                    elif winner == -1:
                        reward = -1 if player == 1 else 1
                    else:
                        reward = 0

                    if i < len(game_history) - 1:
                        next_state = game_history[i + 1][0]
                    else:
                        next_state = ai.state_to_string(game.board)
                        next_state = ai2.state_to_string(game.board)

                    if player == 1:
                        ai.update_q_value(state, act, reward, next_state)
                    else:
                        ai2.update_q_value(state, act, reward, next_state)

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
    game_data['all_game'] = game_data['win'] + game_data['lose'] + game_data['draw']

    # แสดงสถิติการเรียนรู้
    stats = ai.get_learning_stats()
    print(f"\n📊 สถิติการเรียนรู้:")
    print(f"   สถานการณ์ที่เรียนรู้: {stats['total_states']}")
    print(f"   การกระทำทั้งหมด: {stats['total_actions']}")
    print(f"   ประสบการณ์เชิงบวก: {stats['positive_q']}")
    print(f"   ประสบการณ์เชิงลบ: {stats['negative_q']}")

    ai.epsilon = 0.05
    ai.save_q_table()
    ai2.save_q_table()

    if update_json_file('game_stats.json', game_data):
        print("Updated game data:", game_data)


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
    ai2 = TicTacToeQLearning(data_file="o_q_table.pkl")

    train_ai(ai, ai2, game_data, 1000)


if __name__ == "__main__":
    main()