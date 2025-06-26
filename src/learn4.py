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
pygame.display.set_caption("VRAIE IA qui apprend de zéro - Route Aléatoire")

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
        
        # Propriétés de la route améliorées
        self.road_width = 300
        self.base_center_x = WIDTH // 2
        self.road_segments = []
        self.num_segments = 30
        self.segment_height = 30
        
        # Nouveau système de génération de route aléatoire
        self.current_direction = 0  # Direction actuelle en degrés
        self.target_direction = 0   # Direction cible
        self.direction_change_rate = 5.5  # Vitesse de changement de direction
        self.max_curve_angle = 45   # Angle maximum de virage (en degrés)
        self.curve_smoothness = 8   # Plus c'est élevé, plus les courbes sont lisses
        
        # Variables pour contrôler la complexité
        self.straight_segments_left = random.randint(5, 15)  # Segments droits avant le prochain virage
        self.curve_intensity = 0.3  # Intensité des courbes (0-1)
        
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
    
    def generate_next_curve_target(self):
        """Génère la prochaine direction cible pour la route"""
        # Décide s'il faut aller tout droit ou tourner
        if self.straight_segments_left <= 0:
            # Temps de tourner !
            turn_options = []
            
            # Évite de sortir de l'écran
            if self.base_center_x > WIDTH * 0.3:
                turn_options.extend([-1, -1, 0])  # Favorise la gauche
            if self.base_center_x < WIDTH * 0.7:
                turn_options.extend([1, 1, 0])    # Favorise la droite
            if WIDTH * 0.3 <= self.base_center_x <= WIDTH * 0.7:
                turn_options.extend([-1, 0, 1])  # Toutes les directions
            
            turn_direction = random.choice(turn_options)
            
            if turn_direction == 0:
                # Ligne droite
                self.target_direction = 0
                self.straight_segments_left = random.randint(3, 12)
            else:
                # Virage
                angle_variation = random.uniform(15, self.max_curve_angle)
                self.target_direction = turn_direction * angle_variation
                self.straight_segments_left = random.randint(8, 20)  # Longueur du virage
        else:
            self.straight_segments_left -= 1
    
    def update_road_direction(self):
        """Met à jour la direction de la route de manière fluide"""
        # Transition douce vers la direction cible
        direction_diff = self.target_direction - self.current_direction
        
        # Limite le changement de direction pour éviter les virages trop brusques
        max_change = self.direction_change_rate
        if abs(direction_diff) > max_change:
            if direction_diff > 0:
                self.current_direction += max_change
            else:
                self.current_direction -= max_change
        else:
            self.current_direction = self.target_direction
    
    def init_road(self):
        """Initialise les segments de route avec génération aléatoire"""
        self.road_segments = []
        current_center = self.base_center_x
        
        for i in range(self.num_segments + 1):
            y = HEIGHT - i * self.segment_height
            self.road_segments.append({
                "y": y, 
                "center_x": current_center,
                "direction": 0
            })
            
            # Génère la position du prochain segment
            if i > 0:
                # Ajoute une petite variation aléatoire
                noise = random.uniform(-2, 2)
                direction_rad = math.radians(self.current_direction + noise)
                offset = math.sin(direction_rad) * self.segment_height * 0.3
                current_center += offset
                
                # Garde la route dans les limites de l'écran
                current_center = max(150, min(WIDTH - 150, current_center))
                
                self.road_segments[i]["center_x"] = current_center
                self.road_segments[i]["direction"] = self.current_direction
    
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
        """Calcule le centre de la route à une position Y donnée avec interpolation"""
        for i in range(len(self.road_segments) - 1):
            if self.road_segments[i+1]["y"] <= y <= self.road_segments[i]["y"]:
                y_diff = self.road_segments[i]["y"] - self.road_segments[i+1]["y"]
                if y_diff == 0:
                    ratio = 0
                else:
                    ratio = (y - self.road_segments[i+1]["y"]) / y_diff
                
                # Interpolation plus fluide avec une courbe
                ratio = 0.5 * (1 - math.cos(ratio * math.pi))  # Interpolation cosinus pour plus de fluidité
                
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
        if self.total_time > 10000:  # Beaucoup plus long maintenant !
            done = True
            # Bonus pour avoir survécu longtemps
            reward += self.score / 5
        
        new_state = self.get_simple_state()
        
        return new_state, reward, done
    
    def update_road(self):
        """Met à jour la position de la route avec génération procédurale"""
        # Met à jour la direction de la route
        self.update_road_direction()
        
        # Décide si il faut changer de direction
        if random.random() < 0.08:  # 2% de chance à chaque frame
            self.generate_next_curve_target()
        
        # Déplace tous les segments vers le bas
        for i in range(len(self.road_segments)):
            self.road_segments[i]["y"] += 5
            
            # Quand un segment sort de l'écran, on en crée un nouveau en haut
            if self.road_segments[i]["y"] > HEIGHT + self.segment_height:
                # Position du nouveau segment
                new_y = self.road_segments[i]["y"] - (self.num_segments + 1) * self.segment_height
                
                # Trouve le segment précédent pour calculer la nouvelle position
                prev_segment = None
                for j in range(len(self.road_segments)):
                    if self.road_segments[j]["y"] < new_y and (prev_segment is None or self.road_segments[j]["y"] > prev_segment["y"]):
                        prev_segment = self.road_segments[j]
                
                if prev_segment:
                    # Calcule la nouvelle position basée sur la direction actuelle
                    direction_rad = math.radians(self.current_direction)
                    base_offset = math.sin(direction_rad) * self.segment_height * 0.4
                    
                    # Ajoute une variation aléatoire pour plus de naturel
                    random_variation = random.uniform(-8, 8)
                    new_center_x = prev_segment["center_x"] + base_offset + random_variation
                    
                    # Garde la route dans les limites
                    new_center_x = max(120, min(WIDTH - 120, new_center_x))
                    
                    # Met à jour le segment
                    self.road_segments[i]["y"] = new_y
                    self.road_segments[i]["center_x"] = new_center_x
                    self.road_segments[i]["direction"] = self.current_direction
                else:
                    # Fallback si on ne trouve pas de segment précédent
                    self.road_segments[i]["y"] = new_y
                    self.road_segments[i]["center_x"] = self.base_center_x
        
        # Trie les segments par position Y pour maintenir l'ordre
        self.road_segments.sort(key=lambda segment: segment["y"], reverse=True)
    
    def draw(self, ai_mode=False, show_sensors=False):
        """Dessine le jeu"""
        screen.fill(GREEN)
        
        # Dessiner la route avec des segments plus fluides
        for i in range(len(self.road_segments) - 1):
            current = self.road_segments[i]
            next_seg = self.road_segments[i + 1]
            
            if current["y"] >= -self.segment_height or next_seg["y"] <= HEIGHT + self.segment_height:
                # Crée des points pour un polygone plus fluide
                points = []
                
                # Points de gauche (de haut en bas)
                points.append((current["center_x"] - self.road_width // 2, current["y"]))
                points.append((next_seg["center_x"] - self.road_width // 2, next_seg["y"]))
                
                # Points de droite (de bas en haut)
                points.append((next_seg["center_x"] + self.road_width // 2, next_seg["y"]))
                points.append((current["center_x"] + self.road_width // 2, current["y"]))
                
                # Dessine le segment de route
                if len(points) >= 3:
                    pygame.draw.polygon(screen, GRAY, points)
        
        # Dessiner les lignes centrales avec espacement variable
        for i in range(0, len(self.road_segments) - 1, 3):  # Plus d'espacement entre les lignes
            current = self.road_segments[i]
            next_seg = self.road_segments[i + 1] if i + 1 < len(self.road_segments) else current
            
            if 0 <= current["y"] <= HEIGHT or 0 <= next_seg["y"] <= HEIGHT:
                # Ligne centrale pointillée
                line_length = 40
                gap_length = 20
                
                start_x, start_y = current["center_x"], current["y"]
                end_x, end_y = next_seg["center_x"], next_seg["y"]
                
                # Calcule la distance totale
                total_distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                
                if total_distance > 0:
                    # Nombre de segments de ligne
                    num_segments = max(1, int(total_distance / (line_length + gap_length)))
                    
                    for seg in range(num_segments):
                        ratio_start = seg * (line_length + gap_length) / total_distance
                        ratio_end = min(1.0, (seg * (line_length + gap_length) + line_length) / total_distance)
                        
                        if ratio_start < 1.0:
                            line_start_x = start_x + (end_x - start_x) * ratio_start
                            line_start_y = start_y + (end_y - start_y) * ratio_start
                            line_end_x = start_x + (end_x - start_x) * ratio_end
                            line_end_y = start_y + (end_y - start_y) * ratio_end
                            
                            pygame.draw.line(screen, WHITE, 
                                           (line_start_x, line_start_y), 
                                           (line_end_x, line_end_y), 4)
        
        # Dessiner les capteurs de l'IA (si demandé)
        if show_sensors and ai_mode:
            self.draw_sensors()
        
        # Dessiner la voiture
        car_color = BLUE if ai_mode else RED
        pygame.draw.rect(screen, car_color, (self.car_x, self.car_y, self.car_width, self.car_height))
        
        # Afficher les infos avec plus de détails pour debug
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)
        
        mode_text = "IA qui apprend - Route Aléatoire" if ai_mode else "Mode Joueur - Route Aléatoire"
        score_text = f"Score: {self.score}"
        time_text = f"Temps: {self.total_time}"
        direction_text = f"Direction route: {self.current_direction:.1f}°"
        
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
    
    print("🤖 ENTRAÎNEMENT D'UNE VRAIE IA DE ZÉRO - ROUTE ALÉATOIRE")
    print("Au début, elle va être TRÈS mauvaise !")
    print("La route est maintenant imprévisible avec des virages variés !")
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
    """
    Fonction principale
    """
    print("=== VRAIE IA QUI APPREND DE ZÉRO - ROUTE ALÉATOIRE ===")
    print("Cette fois, l'IA commence vraiment nulle !")
    print("Et la route est imprévisible avec des courbes aléatoires !")
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
        print("La route est maintenant aléatoire avec des virages surprises !")
        
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