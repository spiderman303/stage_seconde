import pygame
import sys
import math
import random

# Initialisation de Pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Test Route qui Tourne")
clock = pygame.time.Clock()

GRAY = (100, 100, 100)
GREEN = (50, 205, 50)

class RoadSegment:
    def __init__(self, y, center_x):
        self.y = y
        self.center_x = center_x

class TestRoad:
    def __init__(self):
        self.road_width = 300
        self.segment_height = 30
        self.num_segments = 30
        self.road_segments = []
        self.base_center_x = WIDTH // 2

        self.current_direction = 0
        self.target_direction = 0
        self.direction_change_rate = 0.5
        self.max_curve_angle = 45
        self.straight_segments_left = 0

        self.init_road()

    def generate_next_curve_target(self):
        if self.straight_segments_left <= 0:
            turn_direction = random.choice([-1, 1])
            angle = random.uniform(10, self.max_curve_angle)
            self.target_direction = turn_direction * angle
            self.straight_segments_left = random.randint(5, 15)
        else:
            self.straight_segments_left -= 1

    def update_direction(self):
        diff = self.target_direction - self.current_direction
        max_change = self.direction_change_rate
        if abs(diff) > max_change:
            self.current_direction += max_change if diff > 0 else -max_change
        else:
            self.current_direction = self.target_direction

    def init_road(self):
        self.road_segments = []
        center_x = self.base_center_x
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append(RoadSegment(y, center_x))

    def update(self):
        self.generate_next_curve_target()
        self.update_direction()

        for seg in self.road_segments:
            seg.y += 5

        # Remplacer les segments qui sortent
        if self.road_segments[0].y > HEIGHT:
            self.road_segments.pop(0)
            last_seg = self.road_segments[-1]
            direction_rad = math.radians(self.current_direction)
            offset = math.sin(direction_rad) * self.segment_height * 0.4
            new_center = last_seg.center_x + offset + random.uniform(-5, 5)
            new_center = max(120, min(WIDTH - 120, new_center))
            self.road_segments.append(RoadSegment(last_seg.y - self.segment_height, new_center))

    def draw(self, surface):
        for i in range(len(self.road_segments) - 1):
            seg1 = self.road_segments[i]
            seg2 = self.road_segments[i+1]

            points = [
                (seg1.center_x - self.road_width // 2, seg1.y),
                (seg2.center_x - self.road_width // 2, seg2.y),
                (seg2.center_x + self.road_width // 2, seg2.y),
                (seg1.center_x + self.road_width // 2, seg1.y)
            ]
            pygame.draw.polygon(surface, GRAY, points)

# === Boucle principale ===
road = TestRoad()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    road.update()

    screen.fill(GREEN)
    road.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
