# Import necessary libraries
import pygame
import sys

# Initialize pygame
pygame.init()

# Set up display dimensions
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Car Game")

# Define colors
GRAY = (100, 100, 100)  # Road color
WHITE = (255, 255, 255)  # Line color
RED = (255, 0, 0)        # Car color

# Car properties
car_width = 50
car_height = 80
car_x = WIDTH // 2 - car_width // 2
car_y = HEIGHT - car_height - 20
car_speed = 5

# Game loop
clock = pygame.time.Clock()
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Handle key presses for car movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and car_x > 0:
        car_x -= car_speed
    if keys[pygame.K_RIGHT] and car_x < WIDTH - car_width:
        car_x += car_speed
    
    # Fill the screen with road color
    screen.fill(GRAY)
    
    # Draw road lines
    for y in range(0, HEIGHT, 100):
        pygame.draw.rect(screen, WHITE, (WIDTH // 2 - 5, y, 10, 50))
    
    # Draw the car
    pygame.draw.rect(screen, RED, (car_x, car_y, car_width, car_height))
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
