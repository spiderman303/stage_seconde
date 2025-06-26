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
pygame.display.set_caption("VRAIE IA qui apprend de zéro")

# Couleurs
GRAY = (100, 100, 100)    # Route
WHITE = (255, 255, 255)   # Lignes
RED = (255, 0, 0)         # Voiture joueur
BLUE = (0, 0, 255)        # Voiture IA
GREEN = (50, 205, 50)     # Herbe
BLACK = (0, 0, 0)         # Texte
YELLOW = (255, 255, 0)    # Capteurs

class CarGame:
    def __init__(self):
        # Propriétés de la voiture
        self.car_width = 50
        self.car_height = 80
        self.car_speed = 5
        self.reset_car()
        
        # Propriétés de la route
        self.road_width = 300
        self.base_center_x = WIDTH // 2
        self.road_segments = []
        self.num_segments = 30
        self.segment_height = 30
        
        # Propriétés des courbes
        self.curve_amplitude = 150
        self.curve_length = 1000
        self.curve_offset = 0
        
        # Score et temps
        self.score = 0
        self.on_road_time = 0
        self.total_time = 0
        self.steps_on_road = 0
        
        self.init_road()
    
    def reset_car(self):
        """Remet la voiture à sa position de départ"""
        self.car_x = WIDTH // 2 - self.car_width // 2
        self.car_y = HEIGHT - self.car_height - 20
    
    def init_road(self):
        """Initialise les segments de route"""
        self.road_segments = []
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append({"y": y, "center_x": self.base_center_x})
    
    def get_sensor_readings(self):
        """
        L'IA utilise des 'capteurs' pour voir l'environnement
        Comme des radars qui détectent les bords de la route
        Elle ne connaît PAS directement le centre de la route !
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
        L'IA ne voit que ce que ses capteurs détectent !
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
        
        # Combine tous les capteurs en un seul état de façon plus simple
        # On encode les 5 capteurs comme un nombre en base 3
        state = 0
        multiplier = 1
        for sensor_val in discrete_sensors:
            state += sensor_val * multiplier
            multiplier *= 3
        
        # S'assure que l'état est dans les limites (0 à 242)
        return max(0, min(state, 242))
    
    def get_road_center_at_y(self, y):
        """Calcule le centre de la route à une position Y donnée"""
        for i in range(len(self.road_segments) - 1):
            if self.road_segments[i+1]["y"] <= y <= self.road_segments[i]["y"]:
                y_diff = self.road_segments[i]["y"] - self.road_segments[i+1]["y"]
                if y_diff == 0:
                    ratio = 0
                else:
                    ratio = (y - self.road_segments[i+1]["y"]) / y_diff
                
                center_x = self.road_segments[i+1]["center_x"] + ratio * (self.road_segments[i]["center_x"] - self.road_segments[i+1]["center_x"])
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
        
        # Mettre à jour la route
        self.update_road()
        self.total_time += 1
        
        # Calculer la récompense - Plus réaliste maintenant !
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
        if self.total_time > 2000:  # Beaucoup plus long maintenant !
            done = True
            # Bonus pour avoir survécu longtemps
            reward += self.score / 5
        
        new_state = self.get_simple_state()
        
        return new_state, reward, done
    
    def update_road(self):
        """Met à jour la position de la route"""
        self.curve_offset += 2
        
        for i in range(len(self.road_segments)):
            self.road_segments[i]["y"] += 5
            
            if self.road_segments[i]["y"] > HEIGHT + self.segment_height:
                self.road_segments[i]["y"] = self.road_segments[i]["y"] - (self.num_segments + 1) * self.segment_height
                curve_position = self.curve_offset + self.road_segments[i]["y"] / 5
                self.road_segments[i]["center_x"] = self.base_center_x + math.sin(curve_position / self.curve_length * 2 * math.pi) * self.curve_amplitude
        
        self.road_segments.sort(key=lambda segment: segment["y"], reverse=True)
    
    def draw(self, ai_mode=False, show_sensors=False):
        """Dessine le jeu"""
        screen.fill(GREEN)
        
        # Dessiner la route
        for i in range(len(self.road_segments) - 1):
            current = self.road_segments[i]
            next_seg = self.road_segments[i + 1]
            
            if current["y"] >= 0 or next_seg["y"] <= HEIGHT:
                points = [
                    (current["center_x"] - self.road_width // 2, current["y"]),
                    (current["center_x"] + self.road_width // 2, current["y"]),
                    (next_seg["center_x"] + self.road_width // 2, next_seg["y"]),
                    (next_seg["center_x"] - self.road_width // 2, next_seg["y"])
                ]
                pygame.draw.polygon(screen, GRAY, points)
        
        # Dessiner les lignes centrales
        for i in range(0, len(self.road_segments) - 1, 2):
            current = self.road_segments[i]
            next_seg = self.road_segments[i + 1]
            
            if 0 <= current["y"] <= HEIGHT or 0 <= next_seg["y"] <= HEIGHT:
                center_line_start = (current["center_x"], current["y"])
                center_line_end = (next_seg["center_x"], next_seg["y"])
                pygame.draw.line(screen, WHITE, center_line_start, center_line_end, 4)
        
        # Dessiner les capteurs de l'IA (si demandé)
        if show_sensors and ai_mode:
            self.draw_sensors()
        
        # Dessiner la voiture
        car_color = BLUE if ai_mode else RED
        pygame.draw.rect(screen, car_color, (self.car_x, self.car_y, self.car_width, self.car_height))
        
        # Afficher les infos avec plus de détails pour debug
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)
        
        mode_text = "IA qui apprend" if ai_mode else "Mode Joueur"
        score_text = f"Score: {self.score}"
        time_text = f"Temps: {self.total_time}"
        
        mode_surface = font.render(mode_text, True, WHITE)
        score_surface = font.render(score_text, True, WHITE)
        time_surface = small_font.render(time_text, True, WHITE)
        
        screen.blit(mode_surface, (10, 10))
        screen.blit(score_surface, (10, 50))
        screen.blit(time_surface, (10, 170))
        
        if self.is_car_on_road():
            status_surface = font.render("Sur la route", True, WHITE)
        else:
            status_surface = font.render("CRASH!", True, RED)
        screen.blit(status_surface, (10, 90))
    
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
        self.init_road()
        self.score = 0
        self.on_road_time = 0
        self.total_time = 0
        self.steps_on_road = 0
        self.curve_offset = 0
        return self.get_simple_state()

class QLearningAgent:
    """
    Agent d'apprentissage par renforcement utilisant Q-Learning
    Cette fois, il commence vraiment de zéro !
    """
    def __init__(self, state_size=243, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Table Q initialisée à zéro - l'IA ne sait RIEN au début !
        self.q_table = np.zeros((state_size, action_size))
        
        # Paramètres d'apprentissage plus agressifs
        self.learning_rate = 0.3     # Apprend plus vite
        self.discount_factor = 0.9   # Se concentre sur le présent
        self.epsilon = 1.0           # 100% d'exploration au début
        self.epsilon_decay = 0.999   # Diminue très lentement
        self.epsilon_min = 0.05      # Garde toujours un peu d'exploration
        
        # Statistiques
        self.scores = []
        self.episode_count = 0
        self.total_crashes = 0
        self.best_score = 0
    
    def get_action(self, state):
        """
        Choisit une action
        Au début : 100% aléatoire (l'IA ne sait rien !)
        Progressivement : utilise ce qu'elle apprend
        """
        if np.random.random() <= self.epsilon:
            # Exploration: action complètement aléatoire
            return random.randrange(self.action_size)
        else:
            # Exploitation: utilise sa table Q
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Apprend de l'expérience - c'est ici que la magie opère !
        """
        # Valeur Q actuelle (ce que l'IA pensait avant)
        current_q = self.q_table[state, action]
        
        if done:
            # Si c'est fini, pas de récompense future
            target_q = reward
        else:
            # Sinon, ajoute la meilleure récompense future possible
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Met à jour la connaissance de l'IA
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Diminue l'exploration très lentement
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
    """
    Entraîne l'IA en montrant chaque étape
    Tu peux voir l'IA apprendre en temps réel !
    """
    game = CarGame()
    agent = QLearningAgent()
    
    clock = pygame.time.Clock()
    episodes = 2000  # Plus d'épisodes car l'IA est vraiment nulle au début
    
    print("🤖 ENTRAÎNEMENT D'UNE VRAIE IA DE ZÉRO")
    print("Au début, elle va être TRÈS mauvaise !")
    print("Regarde-la s'améliorer petit à petit...")
    print("Appuyez sur ESPACE pour accélérer, ECHAP pour quitter")
    
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
                    elif event.key == pygame.K_s:
                        show_visual = not show_visual
            
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
                
                # Affiche les statistiques avec plus d'infos pour debug
                font = pygame.font.SysFont(None, 24)
                stats_text = agent.get_stats()
                episode_text = f"Episode: {episode+1}/{episodes}"
                action_text = f"Dernière action: {['Gauche', 'Tout droit', 'Droite'][action]}"
                state_text = f"État actuel: {state}"
                
                stats_surface = font.render(stats_text, True, WHITE)
                episode_surface = font.render(episode_text, True, WHITE)
                action_surface = font.render(action_text, True, WHITE)
                state_surface = font.render(state_text, True, WHITE)
                
                screen.blit(stats_surface, (10, 130))
                screen.blit(episode_surface, (10, 150))
                screen.blit(action_surface, (10, 190))
                screen.blit(state_surface, (10, 210))
                
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
    """
    Fonction principale
    """
    print("=== VRAIE IA QUI APPREND DE ZÉRO ===")
    print("Cette fois, l'IA commence vraiment nulle !")
    print()
    print("1. Entraîner l'IA (avec visualisation)")
    print("2. Mode joueur humain")
    
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
        
        print("🎮 Mode joueur - Utilisez les flèches gauche/droite")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and game.car_x > 0:
                game.car_x -= game.car_speed
            if keys[pygame.K_RIGHT] and game.car_x < WIDTH - game.car_width:
                game.car_x += game.car_speed
            
            game.update_road()
            game.total_time += 1
            
            if game.is_car_on_road():
                game.steps_on_road += 1
                game.score = game.steps_on_road
            else:
                print(f"Game Over ! Score final: {game.score}")
                game.reset()
            
            game.draw(ai_mode=False)
            pygame.display.flip()
            clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()