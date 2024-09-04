import numpy as np
import random
import pygame
from gym import spaces
import sys

class SimpleMazeGrid:
    def __init__(self, n, k, m, render_option=False, random_seed=None):
        self.n = n
        self.k = k
        self.m = m
        self.render_option = render_option
        self.terminated = False
        self.reset(random_seed)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Discrete(n * n)
        
        if self.render_option:
            pygame.init()
            self.screen_width = 500
            self.screen_height = 500
            self.info_width = 200
            self.total_width = self.screen_width + self.info_width
            self.screen = pygame.display.set_mode((self.total_width, self.screen_height))
            pygame.display.set_caption("Simple Maze Grid Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
                            # Draw the Q-values as text in the grid cells
            self.small_font = pygame.font.Font(None, 20)  # Smaller font size for Q-values
                        
        
    def reset(self, random_seed = None):

        self.player_pos = [0, 0]
        self.terminated = False  # Reset termination status

        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize positions
        self.goal_pos = [random.randint(self.k, self.n-1), random.randint(self.k, self.n-1)]
        self.pits = []
        while len(self.pits) < self.m:
            pit = [random.randint(0, self.n-1), random.randint(0, self.n-1)]
            if pit != [0, 0] and pit != self.goal_pos and pit not in self.pits:
                self.pits.append(pit)
        
        self.cumulative_reward = 0
        self.steps = 0
        
        return self._get_state(), {}


    def retry(self):
        self.player_pos = [0, 0]
        self.terminated = False  # Reset termination status
        self.cumulative_reward = 0
        self.steps = 0
        
        return self._get_state(), {}


    def step(self, action, render_option = None):
        if self.terminated:  # Prevent action if game is terminated
            return self._get_state(), 0, True, {}

        

        if action == 0:  # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1:  # Down
            self.player_pos[0] = min(self.n - 1, self.player_pos[0] + 1)
        elif action == 2:  # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 3:  # Right
            self.player_pos[1] = min(self.n - 1, self.player_pos[1] + 1)
        
        reward = -1
        if self.player_pos == self.goal_pos:
            reward = 10
            self.terminated = True
        elif self.player_pos in self.pits:
            reward = -10
            self.terminated = True
            
        self.cumulative_reward += reward
        self.steps += 1
            
        if (render_option is None) or (render_option is False):
            pass
        else:
            self.render()            
        
        return self._get_state(), reward, self.terminated, {}
    
    def _get_state(self):
        state = np.zeros((self.n, self.n))
        state[self.player_pos[0], self.player_pos[1]] = 1
        state[self.goal_pos[0], self.goal_pos[1]] = 2
        for pit in self.pits:
            state[pit[0], pit[1]] = 3
        return state
    
    def render(self, fps = 30):
        self.screen.fill((255, 255, 255))
        cell_size = self.screen_width // self.n
        
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        
        pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        
        # Draw score board
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        score_text = self.font.render(f"Return: {self.cumulative_reward}", True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(score_text, (self.screen_width + 10, 10))
        self.screen.blit(steps_text, (self.screen_width + 10, 50))

        if self.terminated:
            finished_text = self.font.render("FINISHED", True, (0, 0, 0))
            self.screen.blit(finished_text, (self.screen_width // 2 - 50, self.screen_height // 2 - 20))
        pygame.display.flip()
        self.clock.tick(fps)
                

        
         
    def render_q_values(self, q_table, episode_number):

        pygame.event.get()
 

        self.screen.fill((255, 255, 255))
        cell_size = self.screen_width // self.n
        
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                # Get the Q-values for the current cell
                q_values = q_table[i * self.n + j]
                
                SENSITIVITY = 30.0
                # Draw diagonals with color corresponding to Q-values
                for action in range(4):
                    q_a_value = q_values[action]
                    if q_a_value <= 0:  # Q-value is negative
                        red_color_intensity = int(255 + q_a_value * SENSITIVITY)
                        red_color_intensity = min(max(red_color_intensity, 0), 255)  # Clamp to [0, 255]
                        color_code = (255, red_color_intensity, red_color_intensity)
                    else:  # Q-value is positive or zero
                        green_color_intensity = int(255 - q_a_value * SENSITIVITY)
                        green_color_intensity = min(max(green_color_intensity, 0), 255)  # Clamp to [0, 255]
                        color_code = (green_color_intensity, 255, green_color_intensity)



                    if action == 0:  # Up
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size, i * cell_size),
                            ((j + 1) * cell_size, i * cell_size),
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
                        ])
                    elif action == 1:  # Down
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            ((j + 1) * cell_size, (i + 1) * cell_size),
                            (j * cell_size, (i + 1) * cell_size)
                        ])
                    elif action == 2:  # Left
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size, i * cell_size),
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            (j * cell_size, (i + 1) * cell_size)
                        ])
                    elif action == 3:  # Right
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            ((j + 1) * cell_size, i * cell_size),
                            ((j + 1) * cell_size, (i + 1) * cell_size)
                        ])



                # Draw Q-values
                for action in range(4):
                    text_surface = self.small_font.render(f"{q_values[action]:.1f}", True, (0, 0, 0))
                    # Positioning text to avoid overlap and to fit well in the cell
                    if action == 0:  # Up
                        text_x = j * cell_size + 10 +(cell_size // 3)
                        text_y = i * cell_size + 2 
                    elif action == 1:  # Down
                        text_x = j * cell_size + 10 + 1 * (cell_size // 3)
                        text_y = i * cell_size + 2 + 2 * (cell_size // 3)
                    elif action == 2:  # Left
                        text_x = j * cell_size + 10
                        text_y = i * cell_size + 2 + 1 * (cell_size // 3)
                    elif action == 3:  # Right
                        text_x = j * cell_size + 10 + 2 * (cell_size // 3)
                        text_y = i * cell_size + 2 + 1 * (cell_size // 3)

                    self.screen.blit(text_surface, (text_x, text_y))

                # Draw each grid
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        # Draw the goal
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        
        # Draw the pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        
        # Draw the player
        pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        
        # Draw score board
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        episode_number_text = self.font.render(f"Episode #: {episode_number}", True, (0, 0, 0))
        score_text = self.font.render(f"Return: {self.cumulative_reward}", True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(episode_number_text, (self.screen_width + 10, 10))
        self.screen.blit(score_text, (self.screen_width + 10, 50))
        self.screen.blit(steps_text, (self.screen_width + 10, 90))
        
        
        pygame.display.flip()        
        self.clock.tick(30)


    def close(self):
        if self.render_option:
            pygame.quit()
            sys.exit()

    def handle_keyboard_input(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and not self.terminated:                        
                        next_state, reward, terminated, _ = self.step(0)
                    elif event.key == pygame.K_DOWN and not self.terminated:
                        next_state, reward, terminated, _ = self.step(1)
                    elif event.key == pygame.K_LEFT and not self.terminated:
                        next_state, reward, terminated, _ = self.step(2)
                    elif event.key == pygame.K_RIGHT and not self.terminated:
                        next_state, reward, terminated, _ = self.step(3)
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE: # Quit
                        running = False
                    elif event.key == pygame.K_r: # Reset
                        next_state, reward = self.reset()


                    
                    self.render()
                    if self.render_option:
                        print(f"state = \n{next_state}; \nreward = {reward}")
                    
            self.clock.tick(10)
        
        self.close()


    def set_player_pos(self, player_pos):
        self.player_pos = player_pos[:]

    def get_player_pos(self):
        return self.player_pos[:]

    def get_all_states(self):
        all_states = []
        for i in range(self.n):
            for j in range(self.n):
                all_states.append([i, j])
        return all_states

    def simulate_action(self, player_pos, action):
        self.set_player_pos(player_pos)
        next_state, reward, terminated, _ = self.step(action, render_option=False)
        return next_state, reward, terminated

              
    def render_values(self, value_table, policy, iteration_number):
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        cell_size = self.screen_width // self.n
        SENSITIVITY = 10.0
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                value = value_table[i * self.n + j]
                color_intensity = int(255 - (value * SENSITIVITY) if value >= 0 else 255 + (value * SENSITIVITY))
                color_intensity = min(max(color_intensity, 0), 255)
                color = (color_intensity, 255, color_intensity) if value >= 0 else (255, color_intensity, color_intensity)
                
                pygame.draw.rect(self.screen, color, rect)
                text_surface = self.font.render(f"{value:.2f}", True, (0, 0, 0))
                self.screen.blit(text_surface, (j * cell_size + 10, i * cell_size + 10))
                
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                
                
        
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        iteration_text = self.font.render(f"Iteration #: {iteration_number}", True, (0, 0, 0))
        self.screen.blit(iteration_text, (self.screen_width + 10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)

if __name__ == "__main__":
    # Example usage
    env = SimpleMazeGrid(n=5, k=3, m=2, render_option=True, random_seed=42)
    env.reset()
    env.render()
    env.handle_keyboard_input()
