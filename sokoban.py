import pygame
import numpy as np
import cv2

class SokobanMap:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = None
        self.player_pos = None
        self.stuck_msg = ""

    def create_grid(self):
        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        inner_walls = [(3, 3), (3, 4), (5, 6), (5, 7), (7, 4)]
        for r, c in inner_walls:
            grid[r, c] = 1

        goals = [(2, 6), (4, 2), (6, 8)]
        for r, c in goals:
            grid[r, c] = 3

        boxes = [(2, 4), (4, 7), (6, 5)]
        for r, c in boxes:
            grid[r, c] = 2

        grid[6, 3] = 4

        self.grid = grid
        self.player_pos = np.argwhere((grid == 4) | (grid == 6))[0]

    def reset(self):
        self.create_grid()

    def get_state(self, cell_size):
        surf = pygame.Surface((self.cols * cell_size, self.rows * cell_size))
        colors = {
            0: (255, 255, 255),
            1: (0, 0, 0),
            2: (0, 0, 255),
            3: (0, 255, 0),
            4: (255, 0, 0),
            5: (128, 0, 128),
            6: (255, 165, 0)
        }
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                pygame.draw.rect(surf, colors[val], (c*cell_size, r*cell_size, cell_size, cell_size))
                pygame.draw.rect(surf, (0, 0, 0), (c*cell_size, r*cell_size, cell_size, cell_size), 1)
        return surf

    def check_win(self):
        return not np.any(self.grid == 3) and not np.any(self.grid == 2)

    def is_deadlock(self, pos):
        r, c = pos
        if self.grid[r, c] == 5:
            return False
        walls_or_boxes = [1, 2, 5]
        if (self.grid[r-1, c] in walls_or_boxes and self.grid[r, c-1] in walls_or_boxes):
            return True
        if (self.grid[r-1, c] in walls_or_boxes and self.grid[r, c+1] in walls_or_boxes):
            return True
        if (self.grid[r+1, c] in walls_or_boxes and self.grid[r, c-1] in walls_or_boxes):
            return True
        if (self.grid[r+1, c] in walls_or_boxes and self.grid[r, c+1] in walls_or_boxes):
            return True
        return False

    def check_lose(self):
        boxes = np.argwhere((self.grid == 2) | (self.grid == 5))
        for box in boxes:
            if self.is_deadlock(box):
                self.stuck_msg = "Stuck: Box in deadlock!"
                return True
        self.stuck_msg = ""
        return False

    def move_player(self, direction):
        new_pos = self.player_pos + direction
        value = self.grid[new_pos[0], new_pos[1]]
        reward = -0.1

        if value == 1:
            return reward

        if value in [2, 5]:
            box_new_pos = new_pos + direction
            next_value = self.grid[box_new_pos[0], box_new_pos[1]]

            if next_value in [0, 3]:
                if next_value == 3:
                    self.grid[box_new_pos[0], box_new_pos[1]] = 5
                    reward = 1.0
                else:
                    self.grid[box_new_pos[0], box_new_pos[1]] = 2
                    reward = -0.1

                if value == 5:
                    self.grid[new_pos[0], new_pos[1]] = 3
                    reward = -1.0
                else:
                    self.grid[new_pos[0], new_pos[1]] = 0
            else:
                return reward

        current_value = self.grid[self.player_pos[0], self.player_pos[1]]
        self.grid[self.player_pos[0], self.player_pos[1]] = 3 if current_value == 6 else 0
        self.grid[new_pos[0], new_pos[1]] = 6 if value in [3, 5] else 4
        self.player_pos = new_pos
        return reward


class Sokoban:
    def __init__(self, width, height, grid_rows, grid_cols, mode="human"):
        pygame.init()
        self.WIDTH = width
        self.HEIGHT = height
        self.GRID_ROWS = grid_rows
        self.GRID_COLS = grid_cols
        self.GRID_SIZE = width // grid_cols
        self.SCREEN_HEIGHT = height + self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Sokoban - Reward System")

        self.colors = {
            "WHITE": (255, 255, 255),
            "BLACK": (0, 0, 0),
            "BLUE": (0, 0, 255),
            "GREEN": (0, 255, 0),
            "RED": (255, 0, 0),
            "PURPLE": (128, 0, 128),
            "ORANGE": (255, 165, 0),
            "OVERLAY": (0, 0, 0, 180)
        }

        self.mode = mode
        self.render_enabled = True if mode == "human" else False

        self.map = SokobanMap(grid_rows, grid_cols)
        self.reset()

        # Ghi video khi chơi thủ công (human mode)
        self.video_writer = None
        if self.mode == "human":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter("sokoban_playback.avi", fourcc, 10.0, (self.WIDTH, self.SCREEN_HEIGHT))

    def reset(self):
        self.map.reset()
        self.reward = 0.0
        self.total_reward = 0.0
        self.game_over = False
        self.win = False
        self.lose = False
        self.stuck_msg = ""
        return self.get_state()

    def get_state(self):
        surf = self.map.get_state(self.GRID_SIZE)
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2))
        return arr

    def get_action_space(self):
        return 4

    def step(self, action):
        if self.game_over:
            return self.get_state(), 0.0, True, {}

        direction_map = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }
        direction = direction_map[action]
        reward = self.map.move_player(direction)

        self.reward = reward
        self.total_reward += reward

        if self.map.check_win():
            self.win = True
            self.game_over = True
            self.reward = 10.0
            self.total_reward += self.reward
        elif self.map.check_lose():
            self.lose = True
            self.game_over = True
            self.stuck_msg = self.map.stuck_msg

        return self.get_state(), self.reward, self.game_over, {}

    def render(self):
        if not self.render_enabled:
            return
        self.screen.fill(self.colors["WHITE"])

        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                rect = pygame.Rect(col * self.GRID_SIZE, (row + 1) * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                value = self.map.grid[row, col]
                if value == 1:
                    color = self.colors["BLACK"]
                elif value == 2:
                    color = self.colors["BLUE"]
                elif value == 3:
                    color = self.colors["GREEN"]
                elif value == 4:
                    color = self.colors["RED"]
                elif value == 5:
                    color = self.colors["PURPLE"]
                elif value == 6:
                    color = self.colors["ORANGE"]
                else:
                    color = self.colors["WHITE"]
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors["BLACK"], rect, 1)

        font = pygame.font.Font(None, 28)
        self.screen.blit(font.render(f"Reward: {self.reward:.1f}", True, self.colors["BLACK"]), (200, 5))
        self.screen.blit(font.render(f"Total Reward: {self.total_reward:.1f}", True, self.colors["BLACK"]), (400, 5))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.colors["OVERLAY"])
            self.screen.blit(overlay, (0, self.GRID_SIZE))
            big_font = pygame.font.Font(None, 72)
            small_font = pygame.font.Font(None, 36)
            msg = "You Win!" if self.win else "You Lose!"
            self.screen.blit(big_font.render(msg, True, self.colors["WHITE"]), (self.WIDTH // 2 - 150, self.HEIGHT // 2 - 50))
            if self.stuck_msg:
                self.screen.blit(small_font.render(self.stuck_msg, True, self.colors["WHITE"]), (self.WIDTH // 2 - 150, self.HEIGHT // 2 + 10))
            self.screen.blit(small_font.render("Press R to Restart, Q to Quit", True, self.colors["WHITE"]), (self.WIDTH // 2 - 180, self.HEIGHT // 2 + 60))

        pygame.display.flip()

        if self.mode == "human" and self.video_writer is not None:
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def close(self):
        if self.video_writer:
            self.video_writer.release()
        pygame.quit()


def main():
    env = Sokoban(400, 400, 9, 10, mode="human")
    clock = pygame.time.Clock()
    running = True

    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                else:
                    if not env.game_over:
                        action = None
                        if event.key == pygame.K_UP:
                            action = 0
                        elif event.key == pygame.K_DOWN:
                            action = 1
                        elif event.key == pygame.K_LEFT:
                            action = 2
                        elif event.key == pygame.K_RIGHT:
                            action = 3
                        if action is not None:
                            env.step(action)

        clock.tick(10)

    env.close()


if __name__ == "__main__":
    main()
