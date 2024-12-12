import gymnasium as gym
import pygame
import sys

def run_cartpole_game():
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    observation, info = env.reset()
    
    while True:
        screen.fill((255, 255, 255))
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0  # Move cart to the left
        elif keys[pygame.K_RIGHT]:
            action = 1  # Move cart to the right
        else:
            action = env.action_space.sample()  # Random action if no key is pressed
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(frame_surface, (600, 400)), (0, 0))
        
        pygame.display.flip()
        clock.tick(60)
        
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    run_cartpole_game()