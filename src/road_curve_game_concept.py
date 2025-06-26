# Jeu de Voiture Inversé - Le joueur contrôle la route !
import pygame
import sys
import math
import random

# Initialisation de pygame
pygame.init()

# Configuration de l'affichage
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚗 Jeu de Voiture Inversé - Contrôlez la Route !")

# Couleurs
GRAY = (100, 100, 100)    # Route
WHITE = (255, 255, 255)   # Lignes
RED = (255, 0, 0)         # Voiture
GREEN = (50, 205, 50)     # Herbe
BLACK = (0, 0, 0)         # Texte
YELLOW = (255, 255, 0)    # Éléments spéciaux
BLUE = (0, 100, 255)      # UI
ORANGE = (255, 165, 0)    # Obstacles

class RoadSegment:
    def __init__(self, y, center_x):
        self.y = y
        self.center_x = center_x

class Obstacle:
    def __init__(self, x, y, obstacle_type="rock"):
        self.x = x
        self.y = y
        self.type = obstacle_type
        self.size = 20
        self.collected = False

class ReverseCarGame:
    def __init__(self):
        # Propriétés de la voiture (contrôlée par l'IA)
        self.car_width = 40
        self.car_height = 60
        self.car_x = WIDTH // 2 - self.car_width // 2
        self.car_y = HEIGHT - self.car_height - 50
        self.car_speed = 3
        
        # L'IA de la voiture
        self.car_target_x = self.car_x
        self.car_ai_speed = 2
        
        # Propriétés de la route (contrôlée par le joueur)
        self.road_width = 250
        self.segment_height = 25
        self.num_segments = 30
        self.road_segments = []
        
        # Contrôle de la route par le joueur
        self.road_direction = 0  # Direction que le joueur veut donner à la route
        self.road_curve_speed = 0.8  # Vitesse de courbure de la route
        self.max_curve = 3  # Courbure maximum par segment
        
        # Obstacles et collectibles
        self.obstacles = []
        self.collectibles = []
        self.spawn_timer = 0
        
        # Score et objectifs
        self.score = 0
        self.cars_guided = 0
        self.obstacles_avoided = 0
        self.collectibles_gathered = 0
        self.game_time = 0
        
        # Difficulté
        self.difficulty = 1
        self.car_spawn_timer = 0
        self.cars = [{'x': self.car_x, 'y': self.car_y, 'target_x': self.car_x}]
        
        # UI
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 20)
        
        self.init_road()
    
    def init_road(self):
        """Initialise la route droite"""
        self.road_segments = []
        center_x = WIDTH // 2
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append(RoadSegment(y, center_x))
    
    def handle_road_control(self, keys):
        """Le joueur contrôle la direction de la route"""
        # Contrôle direct avec les flèches
        if keys[pygame.K_LEFT]:
            self.road_direction = -self.max_curve
        elif keys[pygame.K_RIGHT]:
            self.road_direction = self.max_curve
        else:
            self.road_direction = 0
        
        # Contrôle plus fin avec A/D
        if keys[pygame.K_a]:
            self.road_direction = max(-self.max_curve, self.road_direction - 0.5)
        if keys[pygame.K_d]:
            self.road_direction = min(self.max_curve, self.road_direction + 0.5)
        
        # Vitesse de défilement
        if keys[pygame.K_UP]:
            self.segment_height = min(40, self.segment_height + 0.5)
        if keys[pygame.K_DOWN]:
            self.segment_height = max(15, self.segment_height - 0.5)
    
    def update_road(self):
        """Met à jour la route selon les contrôles du joueur"""
        # Déplace tous les segments vers le bas
        road_speed = 4 + self.difficulty * 0.5
        for seg in self.road_segments:
            seg.y += road_speed
        
        # Remplace les segments qui sortent
        while self.road_segments and self.road_segments[0].y > HEIGHT:
            self.road_segments.pop(0)
            
            if self.road_segments:
                last_seg = self.road_segments[-1]
                # Applique la direction choisie par le joueur
                new_center = last_seg.center_x + self.road_direction * self.road_curve_speed
                
                # Limites pour garder la route visible
                new_center = max(self.road_width//2 + 50, min(WIDTH - self.road_width//2 - 50, new_center))
                
                new_y = last_seg.y - self.segment_height
                self.road_segments.append(RoadSegment(new_y, new_center))
    
    def get_road_center_at_y(self, y):
        """Calcule le centre de la route à une position Y donnée"""
        for i in range(len(self.road_segments) - 1):
            if self.road_segments[i+1].y <= y <= self.road_segments[i].y:
                # Interpolation
                y_diff = self.road_segments[i].y - self.road_segments[i+1].y
                if y_diff == 0:
                    return self.road_segments[i].center_x
                
                ratio = (y - self.road_segments[i+1].y) / y_diff
                center_x = (self.road_segments[i+1].center_x + 
                          ratio * (self.road_segments[i].center_x - self.road_segments[i+1].center_x))
                return center_x
        return WIDTH // 2
    
    def update_cars_ai(self):
        """Met à jour l'IA des voitures pour qu'elles suivent la route"""
        for car in self.cars:
            # L'IA vise le centre de la route
            road_center = self.get_road_center_at_y(car['y'])
            car['target_x'] = road_center
            
            # Déplacement fluide vers l'objectif
            diff = car['target_x'] - car['x']
            if abs(diff) > self.car_ai_speed:
                car['x'] += self.car_ai_speed if diff > 0 else -self.car_ai_speed
            else:
                car['x'] = car['target_x']
            
            # Vérifie si la voiture est sur la route
            road_left = road_center - self.road_width // 2
            road_right = road_center + self.road_width // 2
            
            if road_left <= car['x'] + self.car_width//2 <= road_right:
                # Voiture bien guidée !
                self.score += 1
            else:
                # Voiture sortie de la route
                self.score -= 10
    
    def spawn_obstacles_and_collectibles(self):
        """Génère des obstacles et objets à collecter"""
        self.spawn_timer += 1
        
        if self.spawn_timer % 60 == 0:  # Toutes les secondes
            # Spawn aléatoire d'obstacles sur la route
            if random.random() < 0.3:
                road_center = self.get_road_center_at_y(50)
                # Place l'obstacle sur le côté de la route
                side = random.choice([-1, 1])
                obs_x = road_center + side * (self.road_width//4)
                self.obstacles.append(Obstacle(obs_x, 0, "rock"))
        
        if self.spawn_timer % 90 == 0:  # Collectibles moins fréquents
            if random.random() < 0.4:
                road_center = self.get_road_center_at_y(50)
                # Place le collectible au centre
                self.collectibles.append(Obstacle(road_center, 0, "coin"))
    
    def update_obstacles(self):
        """Met à jour les obstacles et collectibles"""
        road_speed = 4 + self.difficulty * 0.5
        
        # Bouge les obstacles vers le bas
        for obs in self.obstacles[:]:
            obs.y += road_speed
            if obs.y > HEIGHT:
                self.obstacles.remove(obs)
        
        for col in self.collectibles[:]:
            col.y += road_speed
            if col.y > HEIGHT:
                self.collectibles.remove(col)
        
        # Vérifie les collisions avec les voitures
        for car in self.cars:
            car_rect = pygame.Rect(car['x'], car['y'], self.car_width, self.car_height)
            
            # Collisions avec obstacles
            for obs in self.obstacles[:]:
                obs_rect = pygame.Rect(obs.x - obs.size//2, obs.y - obs.size//2, obs.size, obs.size)
                if car_rect.colliderect(obs_rect):
                    self.obstacles.remove(obs)
                    self.score -= 20  # Pénalité
            
            # Collecte d'objets
            for col in self.collectibles[:]:
                col_rect = pygame.Rect(col.x - col.size//2, col.y - col.size//2, col.size, col.size)
                if car_rect.colliderect(col_rect):
                    self.collectibles.remove(col)
                    self.score += 50  # Bonus !
                    self.collectibles_gathered += 1
    
    def spawn_new_cars(self):
        """Fait apparaître de nouvelles voitures"""
        self.car_spawn_timer += 1
        
        # Plus le jeu avance, plus il y a de voitures
        spawn_rate = max(180 - self.difficulty * 20, 60)
        
        if self.car_spawn_timer % spawn_rate == 0 and len(self.cars) < 5:
            new_x = WIDTH // 2 + random.randint(-50, 50)
            self.cars.append({
                'x': new_x, 
                'y': HEIGHT + 20, 
                'target_x': new_x
            })
    
    def update_difficulty(self):
        """Augmente la difficulté au fil du temps"""
        self.game_time += 1
        
        if self.game_time % 1800 == 0:  # Toutes les 30 secondes
            self.difficulty += 1
            print(f"🔥 Niveau de difficulté: {self.difficulty}")
    
    def draw(self):
        """Dessine tout le jeu"""
        screen.fill(GREEN)
        
        # Dessine la route
        for i in range(len(self.road_segments) - 1):
            seg1 = self.road_segments[i]
            seg2 = self.road_segments[i+1]
            
            points = [
                (seg1.center_x - self.road_width // 2, seg1.y),
                (seg2.center_x - self.road_width // 2, seg2.y),
                (seg2.center_x + self.road_width // 2, seg2.y),
                (seg1.center_x + self.road_width // 2, seg1.y)
            ]
            pygame.draw.polygon(screen, GRAY, points)
        
        # Dessine les lignes centrales
        for i in range(0, len(self.road_segments) - 1, 2):
            if i + 1 < len(self.road_segments):
                seg1 = self.road_segments[i]
                seg2 = self.road_segments[i + 1]
                pygame.draw.line(screen, WHITE, 
                               (seg1.center_x, seg1.y), 
                               (seg2.center_x, seg2.y), 3)
        
        # Dessine les obstacles
        for obs in self.obstacles:
            pygame.draw.circle(screen, BLACK, (int(obs.x), int(obs.y)), obs.size//2)
        
        # Dessine les collectibles
        for col in self.collectibles:
            pygame.draw.circle(screen, YELLOW, (int(col.x), int(col.y)), col.size//2)
            # Petite étoile au centre
            pygame.draw.circle(screen, WHITE, (int(col.x), int(col.y)), col.size//4)
        
        # Dessine les voitures
        for car in self.cars:
            pygame.draw.rect(screen, RED, (car['x'], car['y'], self.car_width, self.car_height))
            # Indique la direction cible
            pygame.draw.line(screen, BLUE, 
                           (car['x'] + self.car_width//2, car['y']), 
                           (car['target_x'], car['y'] - 20), 2)
        
        # Interface utilisateur
        self.draw_ui()
    
    def draw_ui(self):
        """Dessine l'interface utilisateur"""
        # Score principal
        score_text = self.font.render(f"🎯 Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Statistiques
        stats_text = self.small_font.render(f"Voitures: {len(self.cars)} | Collectés: {self.collectibles_gathered} | Niveau: {self.difficulty}", True, WHITE)
        screen.blit(stats_text, (10, 45))
        
        # Contrôles
        controls_title = self.small_font.render("🎮 CONTRÔLES:", True, BLUE)
        screen.blit(controls_title, (10, HEIGHT - 120))
        
        controls = [
            "← → : Diriger la route",
            "A D : Contrôle fin",
            "↑ ↓ : Vitesse de défilement",
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, WHITE)
            screen.blit(text, (10, HEIGHT - 95 + i * 20))
        
        # Objectif
        objective_text = self.small_font.render("🎯 Guidez les voitures et collectez les pièces !", True, YELLOW)
        screen.blit(objective_text, (WIDTH - 350, 10))
        
        # Indicateur de direction de la route
        direction_text = self.small_font.render(f"Direction route: {self.road_direction:.1f}", True, WHITE)
        screen.blit(direction_text, (WIDTH - 200, HEIGHT - 30))
    
    def update(self, keys):
        """Met à jour tout le jeu"""
        self.handle_road_control(keys)
        self.update_road()
        self.update_cars_ai()
        self.spawn_obstacles_and_collectibles()
        self.update_obstacles()
        self.spawn_new_cars()
        self.update_difficulty()

def main():
    """Fonction principale"""
    print("🚗✨ JEU DE VOITURE INVERSÉ ✨🚗")
    print("Vous contrôlez la ROUTE, pas la voiture !")
    print("Guidez les voitures en toute sécurité et collectez des points !")
    print()
    print("🎮 Contrôles:")
    print("• Flèches GAUCHE/DROITE : Diriger la route")
    print("• A/D : Contrôle fin de la direction")
    print("• Flèches HAUT/BAS : Vitesse de défilement")
    print("• ECHAP : Quitter")
    print()
    
    game = ReverseCarGame()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Contrôles
        keys = pygame.key.get_pressed()
        
        # Mise à jour du jeu
        game.update(keys)
        
        # Affichage
        game.draw()
        pygame.display.flip()
        clock.tick(60)
    
    print(f"🏁 Jeu terminé ! Score final: {game.score}")
    print(f"📊 Statistiques:")
    print(f"   • Voitures guidées: {len(game.cars)}")
    print(f"   • Objets collectés: {game.collectibles_gathered}")
    print(f"   • Niveau atteint: {game.difficulty}")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()