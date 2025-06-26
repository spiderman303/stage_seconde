# Script pour tester tes images avec l'IA déjà entraînée
# Lance ce script APRÈS avoir entraîné ton modèle

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Vérifier si GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Utilisation de : {device}")

# Définir la même architecture de réseau
class ReseauChiffres(nn.Module):
    def __init__(self):
        super(ReseauChiffres, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Charger le modèle entraîné
print("📁 Chargement du modèle entraîné...")
model = ReseauChiffres().to(device)

try:
    model.load_state_dict(torch.load('modele_chiffres_pytorch.pth', map_location=device))
    model.eval()
    print("✅ Modèle chargé avec succès !")
except FileNotFoundError:
    print("❌ Fichier 'modele_chiffres_pytorch.pth' non trouvé !")
    print("   Lance d'abord le script d'entraînement principal.")
    exit()

def predire_image_perso(chemin_image):
    """
    Prédit le chiffre dans une image que tu fournis
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(chemin_image):
            print(f"❌ Fichier non trouvé : {chemin_image}")
            return None, 0
        
        print(f"🔍 Analyse de l'image : {chemin_image}")
        
        # Charger l'image
        image_originale = Image.open(chemin_image)
        
        # Préparation de l'image pour l'IA
        if image_originale.mode != 'L':
            image_gris = image_originale.convert('L')
        else:
            image_gris = image_originale
        
        # Redimensionner à 28x28 pixels
        image_redimensionnee = image_gris.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir en array numpy
        image_array = np.array(image_redimensionnee)
        
        # Inverser les couleurs si nécessaire (fond blanc -> fond noir)
        if np.mean(image_array) > 127:
            image_array = 255 - image_array
        
        # Normaliser comme pour MNIST
        image_array = image_array.astype(np.float32) / 255.0
        image_array = (image_array - 0.1307) / 0.3081
        
        # Convertir en tensor PyTorch
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Prédiction
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
            confidence = torch.exp(output.max()).item() * 100
        
        # Affichage des résultats
        plt.figure(figsize=(12, 4))
        
        # Image originale
        plt.subplot(1, 3, 1)
        plt.imshow(image_originale, cmap='gray' if image_originale.mode == 'L' else None)
        plt.title('Image originale')
        plt.axis('off')
        
        # Image préparée
        plt.subplot(1, 3, 2)
        plt.imshow(image_array, cmap='gray')
        plt.title('Image préparée\n(comme l\'IA la voit)')
        plt.axis('off')
        
        # Résultat
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.7, f'🤖 Prédiction', ha='center', fontsize=16, transform=plt.gca().transAxes)
        plt.text(0.5, 0.5, f'Chiffre: {prediction}', ha='center', fontsize=24, fontweight='bold', 
                 transform=plt.gca().transAxes)
        plt.text(0.5, 0.3, f'Confiance: {confidence:.1f}%', ha='center', fontsize=16, 
                 transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Prédiction terminée !")
        print(f"   🎯 Chiffre détecté : {prediction}")
        print(f"   📊 Confiance : {confidence:.1f}%")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de l'image : {e}")
        return None, 0

def creer_image_test(chiffre="7"):
    """Crée une image de test simple avec un chiffre"""
    print(f"🎨 Création d'une image de test avec le chiffre '{chiffre}'...")
    
    # Créer une image avec un chiffre
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.text(0.5, 0.5, str(chiffre), fontsize=60, ha='center', va='center', 
            transform=ax.transAxes, color='black', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Sauvegarder
    nom_fichier = f'chiffre_test_{chiffre}.png'
    plt.savefig(nom_fichier, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', dpi=100)
    plt.show()
    
    print(f"✅ Image test créée : '{nom_fichier}'")
    return nom_fichier

def menu_interactif():
    """Menu interactif pour utiliser l'IA"""
    print("\n" + "="*50)
    print("🤖 IA DE RECONNAISSANCE DE CHIFFRES")
    print("="*50)
    
    while True:
        print("\n📋 Que veux-tu faire ?")
        print("1. 🖼️  Tester une de mes images")
        print("2. 🎨 Créer une image de test")
        print("3. 📊 Voir les statistiques du modèle")
        print("4. 🚪 Quitter")
        
        choix = input("\n👉 Ton choix (1-4) : ").strip()
        
        if choix == "1":
            chemin = input("📁 Chemin vers ton image : ").strip()
            if chemin:
                predire_image_perso(chemin)
            else:
                print("❌ Chemin vide !")
        
        elif choix == "2":
            chiffre = input("🔢 Quel chiffre créer ? (0-9, défaut=7) : ").strip()
            if not chiffre:
                chiffre = "7"
            if chiffre in "0123456789":
                nom_fichier = creer_image_test(chiffre)
                tester = input(f"🧪 Tester l'image créée ? (o/n) : ").strip().lower()
                if tester in ['o', 'oui', 'y', 'yes']:
                    predire_image_perso(nom_fichier)
            else:
                print("❌ Chiffre invalide ! Utilise 0-9")
        
        elif choix == "3":
            print("\n📊 Informations sur le modèle :")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   🧠 Paramètres totaux : {total_params:,}")
            print(f"   🖥️  Dispositif : {device}")
            print(f"   📁 Fichier : modele_chiffres_pytorch.pth")
        
        elif choix == "4":
            print("👋 Au revoir !")
            break
        
        else:
            print("❌ Choix invalide ! Utilise 1, 2, 3 ou 4")

# Lancer le menu interactif
if __name__ == "__main__":
    menu_interactif()