# Import des bibliothèques nécessaires
import pygame
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialisation de pygame
pygame.init()

# Configuration de l'affichage
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IA Voiture - Génération Procédurale Avancée")

# Couleurs
GRAY = (100, 100, 100)    # Route
WHITE = (255, 255, 255)   # Lignes
RED = (255, 0, 0)         # Voiture joueur
BLUE = (0, 0, 255)        # Voiture IA
GREEN = (50, 205, 50)     # Herbe
BLACK = (0, 0, 0)         # Texte
YELLOW = (255, 255, 0)    # Capteurs
ORANGE = (255, 165, 0)    # UI Controls

class RoadSegment:
    """Classe pour représenter un segment de route"""
    def __init__(self, y, center_x):
        self.y = y
        self.center_x = center_x

class CarGame:
    def __init__(self):
        # Propriétés de la voiture
        self.car_width = 50
        self.car_height = 80
        self.car_speed = 5
        self.reset_car()
        
        # Propriétés de la route - utilise maintenant le système procédural
        self.road_width = 300
        self.segment_height = 30
        self.num_segments = 30
        self.road_segments = []
        self.base_center_x = WIDTH // 2

        # Système de génération procédurale - NOUVEAUX PARAMÈTRES CONTRÔLABLES
        self.current_direction = 0
        self.target_direction = 0
        self.direction_change_rate = 0.5
        
        # PARAMÈTRES CONTRÔLABLES PAR LE JOUEUR
        self.curve_frequency = 0.3      # Fréquence des virages (0.1 = rare, 1.0 = très fréquent)
        self.curve_intensity = 0.5      # Intensité des virages (0.1 = doux, 1.0 = très serrés)
        self.min_curve_angle = 5        # Angle minimum des virages
        self.max_curve_angle = 45       # Angle maximum des virages
        self.min_straight_segments = 5  # Minimum de segments droits
        self.max_straight_segments = 25 # Maximum de segments droits
        
        self.straight_segments_left = 0
        
        # Score et temps
        self.score = 0
        self.on_road_time = 0
        self.total_time = 0
        self.steps_on_road = 0
        
        # UI pour les contrôles
        self.show_controls = True
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 20)
        
        self.init_road()
        self.update_curve_parameters()
    
    def update_curve_parameters(self):
        """Met à jour les paramètres de génération basés sur les contrôles"""
        # Ajuste les angles basés sur l'intensité
        self.max_curve_angle = self.min_curve_angle + (60 - self.min_curve_angle) * self.curve_intensity
        
        # Ajuste la fréquence des segments droits
        range_segments = self.max_straight_segments - self.min_straight_segments
        # Plus la fréquence est haute, moins il y a de segments droits
        self.current_min_straight = self.min_straight_segments + int(range_segments * (1 - self.curve_frequency))
        self.current_max_straight = self.max_straight_segments - int(range_segments * self.curve_frequency * 0.5)
        
        # S'assure que min < max
        if self.current_min_straight >= self.current_max_straight:
            self.current_max_straight = self.current_min_straight + 1
    
    def handle_controls(self, keys):
        """Gère les contrôles pour ajuster la génération procédurale"""
        # Contrôles pour la fréquence des virages
        if keys[pygame.K_q]:  # Q pour diminuer la fréquence
            self.curve_frequency = max(0.1, self.curve_frequency - 0.01)
            self.update_curve_parameters()
        if keys[pygame.K_w]:  # W pour augmenter la fréquence
            self.curve_frequency = min(1.0, self.curve_frequency + 0.01)
            self.update_curve_parameters()
        
        # Contrôles pour l'intensité des virages
        if keys[pygame.K_a]:  # A pour diminuer l'intensité
            self.curve_intensity = max(0.1, self.curve_intensity - 0.01)
            self.update_curve_parameters()
        if keys[pygame.K_s]:  # S pour augmenter l'intensité
            self.curve_intensity = min(1.0, self.curve_intensity + 0.01)
            self.update_curve_parameters()
        
        # Toggle des contrôles
        if keys[pygame.K_h]:  # H pour montrer/cacher les contrôles
            self.show_controls = not self.show_controls
    
    def reset_car(self):
        """Remet la voiture à sa position de départ"""
        self.car_x = WIDTH // 2 - self.car_width // 2
        self.car_y = HEIGHT - self.car_height - 20
    
    def generate_next_curve_target(self):
        """Génère la prochaine direction cible pour la route (système procédural amélioré)"""
        if self.straight_segments_left <= 0:
            # Décide s'il faut faire un virage basé sur la fréquence
            if random.random() < self.curve_frequency:
                turn_direction = random.choice([-1, 1])
                # Utilise l'intensité pour déterminer l'angle
                angle = random.uniform(self.min_curve_angle, self.max_curve_angle)
                self.target_direction = turn_direction * angle
            else:
                # Reste droit
                self.target_direction = 0
            
            # Détermine la longueur du prochain segment droit
            self.straight_segments_left = random.randint(self.current_min_straight, self.current_max_straight)
        else:
            self.straight_segments_left -= 1

    def update_direction(self):
        """Met à jour la direction courante vers la direction cible"""
        diff = self.target_direction - self.current_direction
        max_change = self.direction_change_rate
        if abs(diff) > max_change:
            self.current_direction += max_change if diff > 0 else -max_change
        else:
            self.current_direction = self.target_direction
    
    def init_road(self):
        """Initialise les segments de route"""
        self.road_segments = []
        center_x = self.base_center_x
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append(RoadSegment(y, center_x))
    
    def get_sensor_readings(self):
        """
        L'IA utilise des 'capteurs' pour voir l'environnement
        """
        car_center_x = self.car_x + self.car_width // 2
        car_center_y = self.car_y + self.car_height // 2
        
        # 5 capteurs : gauche, centre-gauche, centre, centre-droite, droite
        sensor_distances = []
        sensor_angles = [-60, -30, 0, 30, 60]  # Angles en degrés
        max_sensor_range = 150
        
        for angle in sensor_angles:
            # Convertit l'angle en radians
            rad = math.radians(angle)
            
            # Calcule la direction du capteur
            dx = math.sin(rad)
            dy = -math.cos(rad)  # Négatif car Y augmente vers le bas
            
            # Lance le "rayon" du capteur
            distance = 0
            for step in range(0, max_sensor_range, 5):
                test_x = car_center_x + dx * step
                test_y = car_center_y + dy * step
                
                # Vérifie si on touche le bord de l'écran
                if test_x < 0 or test_x >= WIDTH or test_y < 0 or test_y >= HEIGHT:
                    distance = step
                    break
                
                # Vérifie si on touche l'herbe (hors de la route)
                if not self.is_point_on_road(test_x, test_y):
                    distance = step
                    break
            
            # Si aucun obstacle trouvé, distance maximale
            if distance == 0:
                distance = max_sensor_range
            
            # Normalise la distance (0-1)
            sensor_distances.append(distance / max_sensor_range)
        
        return sensor_distances
    
    def is_point_on_road(self, x, y):
        """Vérifie si un point est sur la route"""
        road_center = self.get_road_center_at_y(y)
        road_left = road_center - self.road_width // 2
        road_right = road_center + self.road_width // 2
        return road_left <= x <= road_right
    
    def get_simple_state(self):
        """
        État simple basé sur les capteurs
        """
        sensors = self.get_sensor_readings()
        
        # Discrétise chaque capteur en 3 zones : proche (0), moyen (1), loin (2)
        discrete_sensors = []
        for sensor in sensors:
            if sensor < 0.3:
                discrete_sensors.append(0)  # Obstacle proche
            elif sensor < 0.7:
                discrete_sensors.append(1)  # Obstacle moyen
            else:
                discrete_sensors.append(2)  # Obstacle loin
        
        # Combine tous les capteurs en un seul état
        state = 0
        multiplier = 1
        for sensor_val in discrete_sensors:
            state += sensor_val * multiplier
            multiplier *= 3
        
        # S'assure que l'état est dans les limites (0 à 242)
        return max(0, min(state, 242))
    
    def get_road_center_at_y(self, y):
        """Calcule le centre de la route à une position Y donnée avec interpolation"""
        for i in range(len(self.road_segments) - 1):
            if self.road_segments[i+1].y <= y <= self.road_segments[i].y:
                y_diff = self.road_segments[i].y - self.road_segments[i+1].y
                if y_diff == 0:
                    ratio = 0
                else:
                    ratio = (y - self.road_segments[i+1].y) / y_diff
                
                center_x = self.road_segments[i+1].center_x + ratio * (self.road_segments[i].center_x - self.road_segments[i+1].center_x)
                return center_x
        return self.base_center_x
    
    def is_car_on_road(self):
        """Vérifie si la voiture est sur la route"""
        car_center_y = self.car_y + self.car_height // 2
        car_center_x = self.car_x + self.car_width // 2
        return self.is_point_on_road(car_center_x, car_center_y)
    
    def step(self, action):
        """
        Execute une action et retourne le nouvel état, la récompense, et si le jeu est fini
        Actions: 0=gauche, 1=tout droit, 2=droite
        """
        # Mouvement basé sur l'action
        if action == 0 and self.car_x > 0:  # Gauche
            self.car_x -= self.car_speed
        elif action == 2 and self.car_x < WIDTH - self.car_width:  # Droite
            self.car_x += self.car_speed
        # Action 1 = tout droit (pas de mouvement)
        
        # Mettre à jour la route avec génération procédurale
        self.update_road()
        self.total_time += 1
        
        # Calculer la récompense
        reward = 0
        done = False
        
        if self.is_car_on_road():
            # Petite récompense pour rester sur la route
            reward = 0.1
            self.steps_on_road += 1
            self.score = self.steps_on_road
            
            # Bonus si on reste longtemps sur la route
            if self.steps_on_road % 100 == 0:
                reward += 5
        else:
            # GROSSE pénalité pour sortir de la route
            reward = -50
            done = True
        
        # Fin du jeu si trop de temps passé sans progrès
        if self.total_time > 10000:
            done = True
            # Bonus pour avoir survécu longtemps
            reward += self.score / 5
        
        new_state = self.get_simple_state()
        
        return new_state, reward, done
    
    def update_road(self):
        """Met à jour la position de la route avec génération procédurale"""
        # Utilise le système de génération procédurale
        self.generate_next_curve_target()
        self.update_direction()

        # Déplace tous les segments vers le bas
        for seg in self.road_segments:
            seg.y += 5

        # Remplacer les segments qui sortent (système procédural intégré)
        if self.road_segments[0].y > HEIGHT:
            self.road_segments.pop(0)
            last_seg = self.road_segments[-1]
            direction_rad = math.radians(self.current_direction)
            offset = math.sin(direction_rad) * self.segment_height * 0.4
            new_center = last_seg.center_x + offset + random.uniform(-5, 5)
            new_center = max(120, min(WIDTH - 120, new_center))
            self.road_segments.append(RoadSegment(last_seg.y - self.segment_height, new_center))
    
    def draw_controls_ui(self):
        """Dessine l'interface des contrôles"""
        if not self.show_controls:
            return
        
        # Fond semi-transparent pour l'UI
        ui_surface = pygame.Surface((300, 200))
        ui_surface.set_alpha(200)
        ui_surface.fill((0, 0, 0))
        screen.blit(ui_surface, (WIDTH - 310, 10))
        
        # Titre
        title_text = self.font.render("CONTRÔLES DE GÉNÉRATION", True, ORANGE)
        screen.blit(title_text, (WIDTH - 300, 20))
        
        # Fréquence des virages
        freq_text = self.small_font.render(f"Fréquence virages: {self.curve_frequency:.2f}", True, WHITE)
        screen.blit(freq_text, (WIDTH - 300, 50))
        control_text = self.small_font.render("Q/W pour ajuster", True, GRAY)
        screen.blit(control_text, (WIDTH - 300, 70))
        
        # Intensité des virages
        intensity_text = self.small_font.render(f"Intensité virages: {self.curve_intensity:.2f}", True, WHITE)
        screen.blit(intensity_text, (WIDTH - 300, 100))
        control_text2 = self.small_font.render("A/S pour ajuster", True, GRAY)
        screen.blit(control_text2, (WIDTH - 300, 120))
        
        # Paramètres calculés
        angle_text = self.small_font.render(f"Angles: {self.min_curve_angle:.0f}°-{self.max_curve_angle:.0f}°", True, WHITE)
        screen.blit(angle_text, (WIDTH - 300, 150))
        
        segments_text = self.small_font.render(f"Segments droits: {self.current_min_straight}-{self.current_max_straight}", True, WHITE)
        screen.blit(segments_text, (WIDTH - 300, 170))
        
        # Instructions
        help_text = self.small_font.render("H pour masquer", True, ORANGE)
        screen.blit(help_text, (WIDTH - 300, 190))
    
    def draw(self, ai_mode=False, show_sensors=False):
        """Dessine le jeu avec le rendu procédural"""
        screen.fill(GREEN)
        
        # Dessiner la route (système procédural intégré)
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
        
        # Dessiner les lignes centrales
        for i in range(0, len(self.road_segments) - 1, 3):
            seg1 = self.road_segments[i]
            seg2 = self.road_segments[i + 1] if i + 1 < len(self.road_segments) else seg1
            
            if 0 <= seg1.y <= HEIGHT or 0 <= seg2.y <= HEIGHT:
                pygame.draw.line(screen, WHITE, 
                               (seg1.center_x, seg1.y), 
                               (seg2.center_x, seg2.y), 4)
        
        # Dessiner les capteurs de l'IA (si demandé)
        if show_sensors and ai_mode:
            self.draw_sensors()
        
        # Dessiner la voiture
        car_color = BLUE if ai_mode else RED
        pygame.draw.rect(screen, car_color, (self.car_x, self.car_y, self.car_width, self.car_height))
        
        # Afficher les infos
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)
        
        mode_text = "IA - Génération Procédurale" if ai_mode else "Joueur - Génération Procédurale"
        score_text = f"Score: {self.score}"
        time_text = f"Temps: {self.total_time}"
        direction_text = f"Direction: {self.current_direction:.1f}°"
        
        mode_surface = font.render(mode_text, True, WHITE)
        score_surface = font.render(score_text, True, WHITE)
        time_surface = small_font.render(time_text, True, WHITE)
        direction_surface = small_font.render(direction_text, True, WHITE)
        
        screen.blit(mode_surface, (10, 10))
        screen.blit(score_surface, (10, 50))
        screen.blit(time_surface, (10, 170))
        screen.blit(direction_surface, (10, 190))
        
        if self.is_car_on_road():
            status_surface = font.render("Sur la route", True, WHITE)
        else:
            status_surface = font.render("CRASH!", True, RED)
        screen.blit(status_surface, (10, 90))
        
        # Dessiner l'interface des contrôles
        self.draw_controls_ui()
    
    def draw_sensors(self):
        """Dessine les capteurs de l'IA (lignes jaunes)"""
        car_center_x = self.car_x + self.car_width // 2
        car_center_y = self.car_y + self.car_height // 2
        
        sensor_angles = [-60, -30, 0, 30, 60]
        sensors = self.get_sensor_readings()
        max_range = 150
        
        for i, angle in enumerate(sensor_angles):
            rad = math.radians(angle)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            
            distance = sensors[i] * max_range
            end_x = car_center_x + dx * distance
            end_y = car_center_y + dy * distance
            
            pygame.draw.line(screen, YELLOW, (car_center_x, car_center_y), (end_x, end_y), 2)
    
    def reset(self):
        """Remet le jeu à zéro"""
        self.reset_car()
        self.current_direction = 0
        self.target_direction = 0
        self.straight_segments_left = random.randint(5, 15)
        self.init_road()
        self.score = 0
        self.on_road_time = 0
        self.total_time = 0
        self.steps_on_road = 0
        return self.get_simple_state()

class QLearningAgent:
    """
    Agent d'apprentissage par renforcement utilisant Q-Learning
    """
    def __init__(self, state_size=243, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Table Q initialisée à zéro
        self.q_table = np.zeros((state_size, action_size))
        
        # Paramètres d'apprentissage
        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        
        # Statistiques
        self.scores = []
        self.episode_count = 0
        self.total_crashes = 0
        self.best_score = 0
    
    def get_action(self, state):
        """Choisit une action"""
        if np.random.random() <= self.epsilon:
            # Exploration: action complètement aléatoire
            return random.randrange(self.action_size)
        else:
            # Exploitation: utilise sa table Q
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Apprend de l'expérience"""
        # Valeur Q actuelle
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Met à jour la connaissance de l'IA
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Diminue l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_episode_stats(self, score, crashed):
        """Sauvegarde les statistiques"""
        self.scores.append(score)
        self.episode_count += 1
        if crashed:
            self.total_crashes += 1
        if score > self.best_score:
            self.best_score = score
    
    def get_stats(self):
        """Retourne les statistiques actuelles"""
        if len(self.scores) == 0:
            return "Aucune partie jouée"
        
        recent_scores = self.scores[-20:] if len(self.scores) >= 20 else self.scores
        avg_score = np.mean(recent_scores)
        crash_rate = (self.total_crashes / self.episode_count) * 100
        
        return f"Moy. 20 dernières: {avg_score:.1f} | Meilleur: {self.best_score} | Crashes: {crash_rate:.1f}% | Exploration: {self.epsilon:.1%}"

def train_ai_slowly():
    """Entraîne l'IA en montrant chaque étape"""
    game = CarGame()
    agent = QLearningAgent()
    
    clock = pygame.time.Clock()
    episodes = 2000
    
    print("🤖 ENTRAÎNEMENT IA - GÉNÉRATION PROCÉDURALE AVANCÉE")
    print("Route avec génération procédurale contrôlable !")
    print("Pendant l'entraînement, vous pouvez ajuster :")
    print("- Q/W : Fréquence des virages")
    print("- A/S : Intensité des virages")
    print("- H : Afficher/masquer les contrôles")
    print("- ESPACE : Accélérer, ECHAP : Quitter")
    
    episode = 0
    show_visual = True
    speed_multiplier = 1
    
    while episode < episodes:
        state = game.reset()
        crashed = False
        
        while True:
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return agent
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return agent
                    elif event.key == pygame.K_SPACE:
                        speed_multiplier = 5 if speed_multiplier == 1 else 1
                    elif event.key == pygame.K_v:
                        show_visual = not show_visual
            
            # Gestion des contrôles de génération
            keys = pygame.key.get_pressed()
            game.handle_controls(keys)
            
            # L'IA choisit une action
            action = agent.get_action(state)
            
            # Execute l'action
            next_state, reward, done = game.step(action)
            
            # L'IA apprend
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            
            # Affichage
            if show_visual:
                game.draw(ai_mode=True, show_sensors=True)
                
                # Affiche les statistiques
                font = pygame.font.SysFont(None, 24)
                stats_text = agent.get_stats()
                episode_text = f"Episode: {episode+1}/{episodes}"
                action_text = f"Action: {['Gauche', 'Tout droit', 'Droite'][action]}"
                state_text = f"État: {state}"
                
                stats_surface = font.render(stats_text, True, WHITE)
                episode_surface = font.render(episode_text, True, WHITE)
                action_surface = font.render(action_text, True, WHITE)
                state_surface = font.render(state_text, True, WHITE)
                
                screen.blit(stats_surface, (10, 130))
                screen.blit(episode_surface, (10, 150))
                screen.blit(action_surface, (10, 210))
                screen.blit(state_surface, (10, 230))
                
                pygame.display.flip()
                clock.tick(60 * speed_multiplier)
            
            if done:
                if not game.is_car_on_road():
                    crashed = True
                break
        
        # Sauvegarde les stats de cet épisode
        agent.save_episode_stats(game.score, crashed)
        episode += 1
        
        # Affiche les progrès dans la console
        if episode % 100 == 0:
            print(f"Episode {episode}: {agent.get_stats()}")
    
    print("🎉 Entraînement terminé !")
    print(f"Score final moyen: {np.mean(agent.scores[-100:]):.1f}")
    print(f"Meilleur score: {agent.best_score}")
    
    return agent

def main():
    """Fonction principale"""
    print("=== IA VOITURE - GÉNÉRATION PROCÉDURALE AVANCÉE ===")
    print("Route générée de façon procédurale avec contrôles dynamiques !")
    print()
    print("1. Entraîner l'IA (avec visualisation et contrôles)")
    print("2. Mode joueur humain (avec contrôles)")
    
    choice = input("Choisissez (1/2): ")
    
    if choice == "1":
        trained_agent = train_ai_slowly()
        
        if trained_agent and len(trained_agent.scores) > 0:
            print("\n📊 RÉSULTATS FINAUX:")
            print(f"• Parties jouées: {trained_agent.episode_count}")
            print(f"• Meilleur score: {trained_agent.best_score}")
            print(f"• Score moyen (100 dernières): {np.mean(trained_agent.scores[-100:]):.1f}")
            print(f"• Taux de crash final: {(trained_agent.total_crashes/trained_agent.episode_count)*100:.1f}%")
    
    elif choice == "2":
        # Mode joueur humain
        game = CarGame()
        clock = pygame.time.Clock()
        running = True
        
        print("🎮 Mode joueur - Contrôles disponibles :")
        print("• Flèches gauche/droite : Déplacer la voiture")
        print("• Q/W : Ajuster la fréquence des virages")
        print("• A/S : Ajuster l'intensité des virages")
        print("• H : Afficher/masquer l'interface des contrôles")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            
            # Gestion des contrôles de génération procédurale
            game.handle_controls(keys)
            
            # Mouvement de la voiture
            if keys[pygame.K_LEFT] and game.car_x > 0:
                game.car_x -= game.car_speed
            if keys[pygame.K_RIGHT] and game.car_x < WIDTH - game.car_width:
                game.car_x += game.car_speed
            
            # Mise à jour du jeu
            game.update_road()
            game.total_time += 1
            
            if game.is_car_on_road():
                game.steps_on_road += 1
                game.score = game.steps_on_road
            else:
                # Collision détectée
                print(f"🔥 CRASH ! Score final: {game.score}")
                game.reset()
            
            # Affichage
            game.draw(ai_mode=False, show_sensors=False)
            pygame.display.flip()
            clock.tick(60)
        
        print(f"Jeu terminé ! Score final: {game.score}")
    
    else:
        print("Choix invalide !")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()