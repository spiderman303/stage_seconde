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
pygame.display.set_caption("IA apprend à conduire - Q-Learning")

# Couleurs
GRAY = (100, 100, 100)    # Route
WHITE = (255, 255, 255)   # Lignes
RED = (255, 0, 0)         # Voiture joueur
BLUE = (0, 0, 255)        # Voiture IA
GREEN = (50, 205, 50)     # Herbe
BLACK = (0, 0, 0)         # Texte

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
    
    def get_state(self):
        """
        Récupère l'état actuel pour l'IA
        L'état contient des informations sur la position de la voiture par rapport à la route
        """
        # Position de la voiture
        car_center_x = self.car_x + self.car_width // 2
        car_center_y = self.car_y + self.car_height // 2
        
        # Trouve le centre de la route à la position de la voiture
        road_center = self.get_road_center_at_y(car_center_y)
        
        # Calcule la distance par rapport au centre (normalisée)
        distance_from_center = car_center_x - road_center
        
        # Discrétise la position (divise en zones)
        # Zone 0: très à gauche, Zone 4: centre, Zone 8: très à droite
        position_zone = min(8, max(0, int((distance_from_center + self.road_width//2) / (self.road_width/9))))
        
        # Regarde la direction de la courbe à venir (simple approximation)
        future_road_center = self.get_road_center_at_y(car_center_y - 100)
        curve_direction = 0  # 0: tout droit, 1: gauche, 2: droite
        
        if future_road_center < road_center - 20:
            curve_direction = 1  # Courbe vers la gauche
        elif future_road_center > road_center + 20:
            curve_direction = 2  # Courbe vers la droite
        
        return position_zone * 3 + curve_direction  # État unique combiné
    
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
        
        road_center = self.get_road_center_at_y(car_center_y)
        road_left = road_center - self.road_width // 2
        road_right = road_center + self.road_width // 2
        
        return road_left < car_center_x < road_right
    
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
        
        # Calculer la récompense
        reward = 0
        done = False
        
        if self.is_car_on_road():
            reward = 1  # Récompense pour rester sur la route
            self.on_road_time += 1
            self.score = self.on_road_time
        else:
            reward = -10  # Pénalité pour sortir de la route
            done = True  # Fin du jeu si on sort de la route
        
        # Fin du jeu si trop de temps passé
        if self.total_time > 3000:  # Limite de temps
            done = True
        
        new_state = self.get_state()
        
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
    
    def draw(self, ai_mode=False):
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
        
        # Dessiner la voiture (bleue pour l'IA, rouge pour le joueur)
        car_color = BLUE if ai_mode else RED
        pygame.draw.rect(screen, car_color, (self.car_x, self.car_y, self.car_width, self.car_height))
        
        # Afficher les infos
        font = pygame.font.SysFont(None, 36)
        mode_text = "IA en apprentissage" if ai_mode else "Mode Joueur"
        score_text = f"Score: {self.score}"
        
        mode_surface = font.render(mode_text, True, WHITE)
        score_surface = font.render(score_text, True, WHITE)
        
        screen.blit(mode_surface, (10, 10))
        screen.blit(score_surface, (10, 50))
        
        if self.is_car_on_road():
            status_surface = font.render("Sur la route", True, WHITE)
        else:
            status_surface = font.render("Hors route!", True, RED)
        screen.blit(status_surface, (10, 90))
    
    def reset(self):
        """Remet le jeu à zéro"""
        self.reset_car()
        self.init_road()
        self.score = 0
        self.on_road_time = 0
        self.total_time = 0
        self.curve_offset = 0
        return self.get_state()

class QLearningAgent:
    """
    Agent d'apprentissage par renforcement utilisant Q-Learning
    C'est le "cerveau" de l'IA qui apprend à jouer
    """
    def __init__(self, state_size=27, action_size=3):
        self.state_size = state_size  # Nombre d'états possibles
        self.action_size = action_size  # Nombre d'actions possibles (gauche, droit, tout droit)
        
        # Table Q : contient les "notes" pour chaque combinaison état-action
        self.q_table = np.zeros((state_size, action_size))
        
        # Paramètres d'apprentissage
        self.learning_rate = 0.1    # Vitesse d'apprentissage
        self.discount_factor = 0.95  # Importance du futur vs présent
        self.epsilon = 1.0          # Exploration vs exploitation
        self.epsilon_decay = 0.995  # Diminution de l'exploration
        self.epsilon_min = 0.01     # Exploration minimum
        
        # Statistiques
        self.scores = []
        self.episode_count = 0
    
    def get_action(self, state):
        """
        Choisit une action basée sur l'état actuel
        Exploration: essaie des actions au hasard
        Exploitation: utilise ce qu'elle a appris
        """
        if np.random.random() <= self.epsilon:
            # Exploration: action aléatoire
            return random.randrange(self.action_size)
        else:
            # Exploitation: meilleure action connue
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Met à jour la table Q basée sur l'expérience
        C'est ici que l'IA "apprend" !
        """
        # Valeur Q actuelle
        current_q = self.q_table[state, action]
        
        if done:
            # Si le jeu est fini, pas de récompense future
            target_q = reward
        else:
            # Sinon, ajoute la meilleure récompense future possible
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Met à jour la table Q
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Diminue l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_score(self, score):
        """Sauvegarde le score pour les statistiques"""
        self.scores.append(score)
        self.episode_count += 1

def train_ai():
    """
    Entraîne l'IA sur plusieurs épisodes
    """
    game = CarGame()
    agent = QLearningAgent()
    
    episodes = 1000  # Nombre de parties d'entraînement
    display_every = 50  # Affiche tous les 50 épisodes
    
    print("🚗 Début de l'entraînement de l'IA...")
    print("L'IA va jouer", episodes, "parties pour apprendre !")
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        
        while True:
            # L'IA choisit une action
            action = agent.get_action(state)
            
            # Execute l'action dans le jeu
            next_state, reward, done = game.step(action)
            
            # L'IA apprend de cette expérience
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Affichage pendant l'entraînement (optionnel)
            if episode % display_every == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return agent
                
                game.draw(ai_mode=True)
                pygame.display.flip()
                pygame.time.delay(10)
            
            if done:
                break
        
        agent.save_score(game.score)
        
        # Affiche les progrès
        if episode % 100 == 0:
            avg_score = np.mean(agent.scores[-100:]) if len(agent.scores) >= 100 else np.mean(agent.scores)
            print(f"Episode {episode}, Score moyen: {avg_score:.1f}, Exploration: {agent.epsilon:.3f}")
    
    print("🎉 Entraînement terminé !")
    return agent

def play_with_trained_ai(agent):
    """
    Montre l'IA entraînée en action
    """
    game = CarGame()
    agent.epsilon = 0  # Plus d'exploration, utilise seulement ce qu'elle a appris
    
    clock = pygame.time.Clock()
    running = True
    
    print("🤖 L'IA entraînée joue maintenant !")
    print("Appuyez sur ECHAP pour quitter")
    
    state = game.reset()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # L'IA choisit la meilleure action
        action = agent.get_action(state)
        
        # Execute l'action
        next_state, reward, done = game.step(action)
        
        if done:
            print(f"Partie terminée ! Score final: {game.score}")
            state = game.reset()
        else:
            state = next_state
        
        # Dessine le jeu
        game.draw(ai_mode=True)
        pygame.display.flip()
        clock.tick(60)

def main():
    """
    Fonction principale
    """
    print("=== JEU DE VOITURE AVEC IA ===")
    print("1. Entraînement de l'IA")
    print("2. IA entraînée joue")
    print("3. Mode joueur humain")
    
    choice = input("Choisissez (1/2/3): ")
    
    if choice == "1":
        # Entraîne l'IA
        trained_agent = train_ai()
        
        # Montre les statistiques
        if len(trained_agent.scores) > 0:
            print(f"Score maximum atteint: {max(trained_agent.scores)}")
            print(f"Score moyen sur les 100 dernières parties: {np.mean(trained_agent.scores[-100:]):.1f}")
        
        # Propose de voir l'IA jouer
        if input("Voulez-vous voir l'IA jouer ? (o/n): ").lower() == 'o':
            play_with_trained_ai(trained_agent)
    
    elif choice == "2":
        print("⚠️  Aucune IA pré-entraînée disponible.")
        print("Veuillez d'abord choisir l'option 1 pour entraîner l'IA.")
    
    elif choice == "3":
        # Mode joueur humain (code original)
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
                game.on_road_time += 1
                game.score = game.on_road_time
            
            game.draw(ai_mode=False)
            pygame.display.flip()
            clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()