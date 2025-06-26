# Import necessary libraries
import pygame
import sys
import math

# Initialize pygame
pygame.init()

# Set up display dimensions
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Game with Smooth Curving Road")

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
base_center_x = WIDTH // 2
road_segments = []
num_segments = 30  # More segments for smoother curves
segment_height = 30

# Road curve properties
curve_amplitude = 150  # Maximum curve deviation
curve_length = 1000    # Length of one complete curve cycle
curve_offset = 0       # Current position in the curve

# Initialize road segments
for i in range(num_segments + 1):
    y = HEIGHT - i * segment_height
    road_segments.append({"y": y, "center_x": base_center_x})

# Function to check if car is on the road
def is_car_on_road():
    # Find the road segment at the car's position
    car_center_y = car_y + car_height // 2
    for i in range(len(road_segments) - 1):
        if road_segments[i+1]["y"] <= car_center_y <= road_segments[i]["y"]:
            # Interpolate road center at exact car position
            y_diff = road_segments[i]["y"] - road_segments[i+1]["y"]
            if y_diff == 0:
                ratio = 0
            else:
                ratio = (car_center_y - road_segments[i+1]["y"]) / y_diff
            
            center_x = road_segments[i+1]["center_x"] + ratio * (road_segments[i]["center_x"] - road_segments[i+1]["center_x"])
            road_left = center_x - road_width // 2
            road_right = center_x + road_width // 2
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
    
    # Update curve offset (moves the road)
    curve_offset += 2
    
    # Move road segments up and recycle them
    for i in range(len(road_segments)):
        road_segments[i]["y"] += 5
        
        # If segment moves off screen, recycle it to the top
        if road_segments[i]["y"] > HEIGHT + segment_height:
            # Move to top
            road_segments[i]["y"] = road_segments[i]["y"] - (num_segments + 1) * segment_height
            
            # Calculate new x position based on sine wave
            curve_position = curve_offset + road_segments[i]["y"] / 5
            road_segments[i]["center_x"] = base_center_x + math.sin(curve_position / curve_length * 2 * math.pi) * curve_amplitude
    
    # Sort segments by y position (highest y value first)
    road_segments.sort(key=lambda segment: segment["y"], reverse=True)
    
    # Fill the screen with grass color (non-road area)
    screen.fill(GREEN)
    
    # Draw the road segments
    for i in range(len(road_segments) - 1):
        current = road_segments[i]
        next_seg = road_segments[i + 1]
        
        # Only draw if at least one segment is on screen
        if current["y"] >= 0 or next_seg["y"] <= HEIGHT:
            # Create polygon points for this road segment
            points = [
                (current["center_x"] - road_width // 2, current["y"]),
                (current["center_x"] + road_width // 2, current["y"]),
                (next_seg["center_x"] + road_width // 2, next_seg["y"]),
                (next_seg["center_x"] - road_width // 2, next_seg["y"])
            ]
            pygame.draw.polygon(screen, GRAY, points)
    
    # Draw center lines
    for i in range(0, len(road_segments) - 1, 2):  # Draw dashed line
        current = road_segments[i]
        next_seg = road_segments[i + 1]
        
        # Only draw if the segment is on screen
        if 0 <= current["y"] <= HEIGHT or 0 <= next_seg["y"] <= HEIGHT:
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
