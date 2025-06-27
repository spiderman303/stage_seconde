# Projets IA Python pour débutant

Bienvenue sur ce dépôt contenant les projets développés lors de mon stage de seconde chez Orange. Ce dépôt regroupe deux projets principaux axés sur l'Intelligence Artificielle et la programmation Python.

## Projets Inclus

1.  **IA de Reconnaissance de Chiffres Manuscrites** : Une intelligence artificielle basée sur **PyTorch** capable de reconnaître des chiffres manuscrits (de 0 à 9). Ce projet intègre un système d'entraînement incrémental et une interface utilisateur pour tester l'IA. Il utilise le jeu de données MNIST (c'est un test standard en IA qui regroupe 60000 images d'apprentissage et 10000 images de test)
2.  **Jeu de Voiture pour l'Entraînement d'une IA (Q-Learning)** : Un jeu développé avec **Pygame** servant d'environnement pour entraîner une IA à la conduite autonome sur une route aléatoire et sinueuse, en utilisant l'algorithme d'apprentissage par renforcement du **Q-learning**.

## Contenu du dépôt

* Le répertoire ```.\src``` contient les scripts python.
* Le répertoire ```.\doc\raw``` contient les notes que j'ai prises pendant le stage et un exemple de transcription d'une conversation avec ChatGPT et Claude.
* Le fichier ```.\doc\paper.md``` est un compte-rendu du stage que j'ai fait faire par Google NotebookLM.
* Le fichier ```.\doc\rapport.md``` est le rapport que j'ai écrit pour mon Lycée.

## Prérequis

Assurez-vous d'avoir **Python** (version compatible avec Python 3.13.2 recommandée pour certains scripts) installé sur votre machine. Il est **fortement recommandé** d'utiliser un environnement virtuel pour gérer les dépendances de chaque projet. Voici les commandes sur Windows ; c'est trés facile de les traduire si vous êtes sur Linux ou MacOS.

1. **Recopiez le dépôt :**
* Si git est installé sur votre machine ; tapez :
    ```powershell
    git clone https://github.com/spiderman303/stage_seconde.git
    ```
* Si vous ne savez pas vous servir de git, vous pouvez télécharger le projet sous forme d'un fichier .zip en cliquant sur le bouton vert intitulé "<> Code".
* ouvrez un terminal PowerShell et naviguez avec la commande ```cd``` jusqu'à la racine du projet (à la fin, vous devez taper ```cd stage_seconde```)

2.  **Créer un environnement virtuel (optionnel mais recommandé) :**
    ```powershell
    python -m venv .venv
    ```
3.  **Activer l'environnement virtuel sur Windows :**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
    Si vous rencontrez des problèmes d'exécution de script, vous devrez peut-être temporairement contourner la politique d'exécution :
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ```
4.  **Installer les dépendances :**
    Un fichier `requirements.txt` est déjà présent à la racine du dépôt. Une fois votre environnement virtuel activé, installez toutes les bibliothèques nécessaires avec la commande suivante :
    ```powershell
    pip install -r requirements.txt
    ```

## Comment Lancer les Projets

### 1. IA de Reconnaissance de Chiffres Manuscrites

Ce projet permet d'entraîner et de tester une IA de reconnaissance de chiffres manuscrits.

*   **Entraîner l'IA :**
    Le script `IA_nombres_perso3.py` gère l'entraînement incrémental de l'IA.
    **Pour démarrer ou reprendre l'entraînement, exécutez :**
    ```powershell
    py .\src\IA_nombres_perso3.py
    ```
    Le script vous demandera combien d'époques vous souhaitez ajouter à l'entraînement. Les progrès (précision, perte) sont automatiquement sauvegardés après chaque époque.

*   **Tester l'IA via l'interface :**
    Le script `IA_nombres_perso_interface2.py` fournit une interface interactive pour utiliser l'IA entraînée. Il charge automatiquement le modèle sauvegardé s'il existe.
    **Pour lancer l'interface interactive, exécutez :**
    ```powershell
    py .\src\IA_nombres_perso_interface2.py
    ```
    Un menu s'affichera, vous proposant plusieurs options :
    *   **1. 🖼️ Tester une de mes images** : Permet de soumettre une image personnalisée pour que l'IA la prédise.
    *   **2. 📁 Tester toutes les images d'un dossier** : Analyse un dossier complet d'images.
    *   **3. 🎨 Créer une image de test** : Génère une image simple d'un chiffre.
    *   **4. 📊 Voir l'historique complet d'entraînement** : Affiche les statistiques et graphiques de toutes les sessions d'entraînement passées.
    *   **5. 🔧 Informations sur le modèle** : Affiche des détails techniques sur le modèle chargé.
    *   **`reinitialiser_modele()`** : (Fonctionnalité accessible directement dans le script d'entraînement ou en explorant le code) Permet de supprimer toutes les sauvegardes du modèle et de l'historique pour recommencer l'entraînement à zéro.

### 2. Jeu de Voiture pour l'Entraînement d'une IA (Q-Learning)

Ce projet propose un environnement de jeu pour entraîner une IA à la conduite autonome sur une route aléatoire.

*   **Lancer le jeu / Entraîner l'IA :**
    Le script `learn4.py` contient la logique du jeu et l'agent d'apprentissage par renforcement (Q-Learning).
    **Pour lancer le projet, exécutez :**
    ```powershell
    py .\src\learn4.py
    ```
    Vous aurez alors deux options au démarrage :
    *   **1. Entraîner l'IA (avec visualisation)** : Lance le processus d'apprentissage de l'IA à partir de zéro. Vous pourrez observer l'IA s'améliorer progressivement.
        *   Pendant l'entraînement : appuyez sur **ESPACE** pour accélérer la simulation et sur **S** pour afficher/masquer les capteurs de l'IA.
    *   **2. Mode joueur humain** : Vous permet de contrôler manuellement la voiture avec les flèches gauche/droite. La route est aléatoire avec des virages surprises.

## Commandes PowerShell Essentielles (pour la gestion du dépôt)

Ces commandes sont utiles pour naviguer dans le dépôt, gérer les fichiers et exécuter les scripts Python :

*   **Navigation & Fichiers :**
    *   **`ls`** : Liste le contenu du dossier actuel.
    *   **`cd <dossier>`** : Navigue vers le dossier spécifié (ex: `cd mon_dossier`).
    *   **`cd ..`** : Remonte d'un niveau dans l'arborescence des dossiers.
    *   **`cd ~`** : Retourne au répertoire de départ (utilisateur).
    *   **`pwd`** : Affiche le chemin d'accès complet du dossier actuel.
    *   **`Mv <chemin_source> <chemin_destination>`** : Déplace ou renomme un fichier ou un dossier.
    *   **`history`** : Affiche l'historique des commandes PowerShell.

*   **Exécution Python & Environnement Virtuel :**
    *   **`py .\src\<nom_du_script>.py`** : Lance l'exécution d'un script Python.
    *   **`py .\src\-m pip install <nom_du_paquet>`** : Installe une bibliothèque Python spécifique.
    *   **`.\.venv\Scripts\Activate.ps1`** : Active l'environnement virtuel Python (nécessaire avant d'installer les dépendances ou de lancer les scripts si vous utilisez un environnement virtuel).
    *   **`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`** : Active temporairement l'exécution de scripts PowerShell pour la session en cours. Cela peut être nécessaire si votre système bloque l'exécution de l'environnement virtuel ou d'autres scripts.
