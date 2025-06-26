# Import necessary libraries
import pygame
import sys
import random
import math

# Initialize pygame
pygame.init()

# Set up display dimensions
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Game with Curving Road")

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
road_center_x = WIDTH // 2
road_curve_speed = 0.5
road_curve_change_timer = 0
road_curve_direction = 0
road_segments = []

# Initialize road segments
for y in range(0, HEIGHT + 100, 20):  # Create segments every 20 pixels
    road_segments.append({"y": y, "center_x": road_center_x})

# Function to check if car is on the road
def is_car_on_road():
    # Find the road segment at the car's position
    for segment in road_segments:
        if car_y <= segment["y"] <= car_y + car_height:
            road_left = segment["center_x"] - road_width // 2
            road_right = segment["center_x"] + road_width // 2
            car_center_x = car_x + car_width // 2
            return road_left < car_center_x < road_right
    return False

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
    
    # Update road curve direction
    road_curve_change_timer += 1
    if road_curve_change_timer > 120:  # Change direction roughly every 2 seconds
        road_curve_change_timer = 0
        road_curve_direction = random.uniform(-1, 1)
    
    # Move road segments down and update their positions
    for i in range(len(road_segments)):
        road_segments[i]["y"] += 5  # Move down
        
        # If segment moves off screen, recycle it to the top with new curve
        if road_segments[i]["y"] > HEIGHT:
            road_segments[i]["y"] = -20
            # Get the center_x from the segment that's now at the top
            prev_segment = road_segments[(i - 1) % len(road_segments)]
            # Apply curve to the new segment
            new_center = prev_segment["center_x"] + road_curve_direction * road_curve_speed * 20
            # Keep road within screen bounds
            new_center = max(road_width // 2, min(WIDTH - road_width // 2, new_center))
            road_segments[i]["center_x"] = new_center
    
    # Fill the screen with grass color (non-road area)
    screen.fill(GREEN)
    
    # Draw the road segments
    for i in range(len(road_segments) - 1):
        current = road_segments[i]
        next_seg = road_segments[i + 1]
        
        # Create polygon points for this road segment
        points = [
            (current["center_x"] - road_width // 2, current["y"]),
            (current["center_x"] + road_width // 2, current["y"]),
            (next_seg["center_x"] + road_width // 2, next_seg["y"]),
            (next_seg["center_x"] - road_width // 2, next_seg["y"])
        ]
        pygame.draw.polygon(screen, GRAY, points)
        
        # Draw center line
        if i % 2 == 0:  # Draw dashed line
            center_line_start = (current["center_x"], current["y"])
            center_line_end = (next_seg["center_x"], next_seg["y"])
            pygame.draw.line(screen, WHITE, center_line_start, center_line_end, 4)
    
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
