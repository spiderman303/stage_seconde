# Import necessary libraries
import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Set up display dimensions
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Game with Scrolling Road")

# Define colors
GRAY = (100, 100, 100)    # Road color
WHITE = (255, 255, 255)   # Line color
RED = (255, 0, 0)         # Car color
GREEN = (50, 205, 50)     # Grass color

# Car properties
car_width = 50
car_height = 80
car_x = WIDTH // 2 - car_width // 2
car_y = HEIGHT - car_height - 20
car_speed = 5

# Road properties
road_width = 300
road_left = (WIDTH - road_width) // 2

# Road line properties
line_height = 50
line_width = 10
line_gap = 40
line_speed = 5
lines_y = []

# Initialize road lines
for y in range(0, HEIGHT, line_height + line_gap):
    lines_y.append(y)

# Function to check if car is on the road
def is_car_on_road():
    car_center_x = car_x + car_width // 2
    return road_left < car_center_x < road_left + road_width

# Game loop
clock = pygame.time.Clock()
running = True
score = 0
on_road_time = 0

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
    
    # Update road lines position (scrolling effect)
    for i in range(len(lines_y)):
        lines_y[i] += line_speed
        if lines_y[i] > HEIGHT:
            lines_y[i] = -line_height
    
    # Fill the screen with grass color (non-road area)
    screen.fill(GREEN)
    
    # Draw the road
    pygame.draw.rect(screen, GRAY, (road_left, 0, road_width, HEIGHT))
    
    # Draw road lines
    for y in lines_y:
        pygame.draw.rect(screen, WHITE, (WIDTH // 2 - line_width // 2, y, line_width, line_height))
    
    # Draw the car
    pygame.draw.rect(screen, RED, (car_x, car_y, car_width, car_height))
    
    # Check if car is on road and update score
    if is_car_on_road():
        on_road_time += 1
        score = on_road_time // 60  # Score increases every second on road
        status_text = "On Road"
        status_color = WHITE
    else:
        status_text = "Off Road!"
        status_color = RED
    
    # Display status and score
    font = pygame.font.SysFont(None, 36)
    status_surface = font.render(status_text, True, status_color)
    score_surface = font.render(f"Score: {score}", True, WHITE)
    screen.blit(status_surface, (10, 10))
    screen.blit(score_surface, (10, 50))
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
