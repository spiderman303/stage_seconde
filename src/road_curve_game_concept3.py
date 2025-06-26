# Jeu de Voiture Inversé avec IA - Le joueur contrôle la route, l'IA conduit !
import pygame
import sys
import math
import random
import numpy as np

# Initialisation de pygame
pygame.init()

# Configuration de l'affichage
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚗 Jeu de Voiture Inversé avec IA - Contrôlez la Route !")

# Couleurs
GRAY = (100, 100, 100)    # Route
WHITE = (255, 255, 255)   # Lignes
RED = (255, 0, 0)         # Voiture joueur
BLUE = (0, 0, 255)        # Voiture IA
GREEN = (50, 205, 50)     # Herbe
BLACK = (0, 0, 0)         # Texte
YELLOW = (255, 255, 0)    # Capteurs/Éléments spéciaux
ORANGE = (255, 165, 0)    # Obstacles
PURPLE = (128, 0, 128)    # Voiture IA avancée

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

class QLearningAgent:
    """Agent d'apprentissage par renforcement utilisant Q-Learning"""
    def __init__(self, state_size=243, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Table Q initialisée à zéro
        self.q_table = np.zeros((state_size, action_size))
        
        # Paramètres d'apprentissage
        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration réduite pour le jeu
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        
        # Pré-entrainement basique
        self.pre_train()
    
    def pre_train(self):
        """Pré-entraînement simple pour éviter des comportements complètement aléatoires"""
        # Quelques règles de base hardcodées
        for state in range(self.state_size):
            # Action par défaut : aller tout droit
            self.q_table[state, 1] = 1.0
            
            # Si obstacles détectés à gauche, favoriser droite
            if state % 3 == 0:  # Capteur gauche détecte obstacle
                self.q_table[state, 2] = 1.5
            
            # Si obstacles détectés à droite, favoriser gauche
            if (state // 3) % 3 == 0:  # Capteur droite détecte obstacle
                self.q_table[state, 0] = 1.5
    
    def get_action(self, state):
        """Choisit une action"""
        if np.random.random() <= self.epsilon:
            # Exploration: action aléatoire
            return random.randrange(self.action_size)
        else:
            # Exploitation: utilise sa table Q
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Apprend de l'expérience"""
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AICar:
    """Voiture contrôlée par l'IA avec capteurs et apprentissage"""
    def __init__(self, x, y, game_ref):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 60
        self.speed = 2
        self.game_ref = game_ref
        self.agent = QLearningAgent()
        self.last_state = None
        self.last_action = None
        self.steps_on_road = 0
        self.total_steps = 0
        self.score = 0
        self.show_sensors = False
    
    def get_sensor_readings(self):
        """Capteurs pour détecter l'environnement"""
        car_center_x = self.x + self.width // 2
        car_center_y = self.y + self.height // 2
        
        sensor_distances = []
        sensor_angles = [-60, -30, 0, 30, 60]
        max_sensor_range = 100
        
        for angle in sensor_angles:
            rad = math.radians(angle)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            
            distance = 0
            for step in range(0, max_sensor_range, 5):
                test_x = car_center_x + dx * step
                test_y = car_center_y + dy * step
                
                if test_x < 0 or test_x >= WIDTH or test_y < 0 or test_y >= HEIGHT:
                    distance = step
                    break
                
                if not self.game_ref.is_point_on_road(test_x, test_y):
                    distance = step
                    break
            
            if distance == 0:
                distance = max_sensor_range
            
            sensor_distances.append(distance / max_sensor_range)
        
        return sensor_distances
    
    def get_state(self):
        """Convertit les capteurs en état discret"""
        sensors = self.get_sensor_readings()
        
        discrete_sensors = []
        for sensor in sensors:
            if sensor < 0.3:
                discrete_sensors.append(0)
            elif sensor < 0.7:
                discrete_sensors.append(1)
            else:
                discrete_sensors.append(2)
        
        state = 0
        multiplier = 1
        for sensor_val in discrete_sensors:
            state += sensor_val * multiplier
            multiplier *= 3
        
        return max(0, min(state, 242))
    
    def is_on_road(self):
        """Vérifie si la voiture est sur la route"""
        car_center_x = self.x + self.width // 2
        car_center_y = self.y + self.height // 2
        return self.game_ref.is_point_on_road(car_center_x, car_center_y)
    
    def update(self):
        """Met à jour la voiture IA"""
        current_state = self.get_state()
        
        # Choisit une action
        action = self.agent.get_action(current_state)
        
        # Execute l'action
        if action == 0 and self.x > 0:  # Gauche
            self.x -= self.speed
        elif action == 2 and self.x < WIDTH - self.width:  # Droite
            self.x += self.speed
        # Action 1 = tout droit
        
        # Calcule la récompense
        reward = 0
        if self.is_on_road():
            reward = 1
            self.steps_on_road += 1
            self.score = self.steps_on_road
        else:
            reward = -10
        
        self.total_steps += 1
        
        # Apprend de l'expérience précédente
        if self.last_state is not None and self.last_action is not None:
            self.agent.learn(self.last_state, self.last_action, reward, current_state, False)
        
        self.last_state = current_state
        self.last_action = action
        
        # Remet la voiture sur route si elle sort trop
        if not self.is_on_road():
            road_center = self.game_ref.get_road_center_at_y(self.y)
            if abs(self.x - road_center) > 200:  # Trop loin
                self.x = road_center - self.width // 2
    
    def draw(self, screen):
        """Dessine la voiture et ses capteurs"""
        # Voiture
        pygame.draw.rect(screen, PURPLE, (self.x, self.y, self.width, self.height))
        
        # Capteurs (si activés)
        if self.show_sensors:
            self.draw_sensors(screen)
        
        # Indicateur de performance
        color = GREEN if self.is_on_road() else RED
        pygame.draw.circle(screen, color, (int(self.x + self.width//2), int(self.y - 10)), 5)
    
    def draw_sensors(self, screen):
        """Dessine les capteurs"""
        car_center_x = self.x + self.width // 2
        car_center_y = self.y + self.height // 2
        
        sensor_angles = [-60, -30, 0, 30, 60]
        sensors = self.get_sensor_readings()
        max_range = 100
        
        for i, angle in enumerate(sensor_angles):
            rad = math.radians(angle)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            
            distance = sensors[i] * max_range
            end_x = car_center_x + dx * distance
            end_y = car_center_y + dy * distance
            
            color = YELLOW if sensors[i] > 0.5 else RED
            pygame.draw.line(screen, color, (car_center_x, car_center_y), (end_x, end_y), 2)

class ReverseCarGameWithAI:
    def __init__(self):
        # Propriétés de la route (contrôlée par le joueur)
        self.road_width = 250
        self.segment_height = 25
        self.num_segments = 30
        self.road_segments = []
        
        # Contrôle de la route par le joueur
        self.road_direction = 0
        self.road_curve_speed = 0,8
        self.max_curve = 7
        
        # Voitures IA
        self.ai_cars = []
        self.spawn_timer = 0
        self.max_cars = 3
        
        # Obstacles et collectibles
        self.obstacles = []
        self.collectibles = []
        self.obstacle_timer = 0
        
        # Score et statistiques
        self.score = 0
        self.cars_guided = 0
        self.game_time = 0
        self.difficulty = 1
        
        # UI
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 20)
        self.show_ai_sensors = False
        
        self.init_road()
        self.spawn_initial_cars()
    
    def init_road(self):
        """Initialise la route droite"""
        self.road_segments = []
        center_x = WIDTH // 2
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append(RoadSegment(y, center_x))
    
    def spawn_initial_cars(self):
        """Fait apparaître les premières voitures IA"""
        for i in range(2):
            x = WIDTH // 2 + random.randint(-50, 50)
            y = HEIGHT - 100 - i * 100
            self.ai_cars.append(AICar(x, y, self))
    
    def handle_road_control(self, keys):
        """Le joueur contrôle la direction de la route"""
        if keys[pygame.K_LEFT]:
            self.road_direction = -self.max_curve
        elif keys[pygame.K_RIGHT]:
            self.road_direction = self.max_curve
        else:
            self.road_direction = 0
        
        if keys[pygame.K_a]:
            self.road_direction = max(-self.max_curve, self.road_direction - 0.5)
        if keys[pygame.K_d]:
            self.road_direction = min(self.max_curve, self.road_direction + 0.5)
        
        if keys[pygame.K_UP]:
            self.segment_height = min(40, self.segment_height + 0.5)
        if keys[pygame.K_DOWN]:
            self.segment_height = max(15, self.segment_height - 0.5)
    
    def update_road(self):
        """Met à jour la route selon les contrôles du joueur"""
        road_speed = 4 + self.difficulty * 0.5
        for seg in self.road_segments:
            seg.y += road_speed
        
        while self.road_segments and self.road_segments[0].y > HEIGHT:
            self.road_segments.pop(0)
            
            if self.road_segments:
                last_seg = self.road_segments[-1]
                new_center = last_seg.center_x + self.road_direction * self.road_curve_speed
                new_center = max(self.road_width//2 + 50, min(WIDTH - self.road_width//2 - 50, new_center))
                new_y = last_seg.y - self.segment_height
                self.road_segments.append(RoadSegment(new_y, new_center))
    
    def get_road_center_at_y(self, y):
        """Calcule le centre de la route à une position Y donnée"""
        for i in range(len(self.road_segments) - 1):
            if self.road_segments[i+1].y <= y <= self.road_segments[i].y:
                y_diff = self.road_segments[i].y - self.road_segments[i+1].y
                if y_diff == 0:
                    return self.road_segments[i].center_x
                
                ratio = (y - self.road_segments[i+1].y) / y_diff
                center_x = (self.road_segments[i+1].center_x + 
                          ratio * (self.road_segments[i].center_x - self.road_segments[i+1].center_x))
                return center_x
        return WIDTH // 2
    
    def is_point_on_road(self, x, y):
        """Vérifie si un point est sur la route"""
        road_center = self.get_road_center_at_y(y)
        road_left = road_center - self.road_width // 2
        road_right = road_center + self.road_width // 2
        return road_left <= x <= road_right
    
    def spawn_new_cars(self):
        """Fait apparaître de nouvelles voitures IA"""
        self.spawn_timer += 1
        
        if self.spawn_timer % 300 == 0 and len(self.ai_cars) < self.max_cars:
            x = WIDTH // 2 + random.randint(-30, 30)
            y = HEIGHT + 50
            new_car = AICar(x, y, self)
            self.ai_cars.append(new_car)
    
    def update_ai_cars(self):
        """Met à jour toutes les voitures IA"""
        for car in self.ai_cars[:]:
            car.update()
            
            # Supprime les voitures qui sortent de l'écran
            if car.y < -100:
                self.ai_cars.remove(car)
                if car.is_on_road():
                    self.score += car.score
                    self.cars_guided += 1
    
    def spawn_obstacles(self):
        """Génère des obstacles"""
        self.obstacle_timer += 1
        
        if self.obstacle_timer % 120 == 0:
            if random.random() < 0.3:
                road_center = self.get_road_center_at_y(50)
                side = random.choice([-1, 1])
                obs_x = road_center + side * (self.road_width//4)
                self.obstacles.append(Obstacle(obs_x, 0, "rock"))
    
    def update_obstacles(self):
        """Met à jour les obstacles"""
        road_speed = 4 + self.difficulty * 0.5
        
        for obs in self.obstacles[:]:
            obs.y += road_speed
            if obs.y > HEIGHT:
                self.obstacles.remove(obs)
    
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
        
        # Dessine les voitures IA
        for car in self.ai_cars:
            car.draw(screen)
        
        # Interface utilisateur
        self.draw_ui()
    
    def draw_ui(self):
        """Dessine l'interface utilisateur"""
        # Score principal
        score_text = self.font.render(f"🎯 Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Statistiques
        stats_text = self.small_font.render(f"Voitures IA: {len(self.ai_cars)} | Guidées: {self.cars_guided} | Niveau: {self.difficulty}", True, WHITE)
        screen.blit(stats_text, (10, 45))
        
        # Contrôles
        controls_title = self.small_font.render("🎮 CONTRÔLES:", True, BLUE)
        screen.blit(controls_title, (10, HEIGHT - 140))
        
        controls = [
            "← → : Diriger la route",
            "A D : Contrôle fin",
            "↑ ↓ : Vitesse",
            "S : Capteurs IA",
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, WHITE)
            screen.blit(text, (10, HEIGHT - 115 + i * 20))
        
        # Objectif
        objective_text = self.small_font.render("🎯 Guidez les voitures IA avec votre route !", True, YELLOW)
        screen.blit(objective_text, (WIDTH - 350, 10))
        
        # Performances des voitures IA
        if self.ai_cars:
            ai_stats = self.small_font.render("🤖 Performances IA:", True, PURPLE)
            screen.blit(ai_stats, (WIDTH - 200, 40))
            
            for i, car in enumerate(self.ai_cars[:3]):
                on_road = "✓" if car.is_on_road() else "✗"
                perf_text = self.small_font.render(f"Voiture {i+1}: {on_road} ({car.score})", True, WHITE)
                screen.blit(perf_text, (WIDTH - 200, 60 + i * 20))
        
        # Indicateur capteurs
        if self.show_ai_sensors:
            sensor_text = self.small_font.render("👁️ Capteurs IA activés", True, YELLOW)
            screen.blit(sensor_text, (WIDTH - 200, HEIGHT - 40))
    
    def update(self, keys):
        """Met à jour tout le jeu"""
        self.handle_road_control(keys)
        self.update_road()
        self.update_ai_cars()
        self.spawn_new_cars()
        self.spawn_obstacles()
        self.update_obstacles()
        
        self.game_time += 1
        if self.game_time % 1800 == 0:
            self.difficulty += 1
        
        # Toggle capteurs IA
        for car in self.ai_cars:
            car.show_sensors = self.show_ai_sensors

def main():
    """Fonction principale"""
    print("🚗🤖 JEU DE VOITURE INVERSÉ AVEC IA 🤖🚗")
    print("Vous contrôlez la ROUTE, l'IA conduit les voitures !")
    print("Guidez les voitures IA en modelant la route avec vos contrôles !")
    print()
    print("🎮 Contrôles:")
    print("• Flèches GAUCHE/DROITE : Diriger la route")
    print("• A/D : Contrôle fin de la direction")
    print("• Flèches HAUT/BAS : Vitesse de défilement")
    print("• S : Afficher/masquer les capteurs IA")
    print("• ECHAP : Quitter")
    print()
    print("🎯 Objectif: Aidez les voitures IA à naviguer sur votre route !")
    print()
    
    game = ReverseCarGameWithAI()
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
                elif event.key == pygame.K_s:
                    game.show_ai_sensors = not game.show_ai_sensors
                    print(f"Capteurs IA: {'ON' if game.show_ai_sensors else 'OFF'}")
        
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
    print(f"   • Voitures guidées: {game.cars_guided}")
    print(f"   • Niveau atteint: {game.difficulty}")
    print(f"   • Temps de jeu: {game.game_time // 60} secondes")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()