# Interface pour tester l'IA de reconnaissance de chiffres - Version incrémentale
# Compatible avec la nouvelle version qui sauvegarde l'historique

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Vérifier si GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Utilisation de : {device}")

# Configuration des fichiers (même que dans le script principal)
MODELE_FICHIER = 'modele_chiffres_pytorch.pth'
HISTORIQUE_FICHIER = 'historique_entrainement.json'
OPTIMISEUR_FICHIER = 'optimiseur_pytorch.pth'

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
    model.load_state_dict(torch.load(MODELE_FICHIER, map_location=device))
    model.eval()
    print("✅ Modèle chargé avec succès !")
except FileNotFoundError:
    print("❌ Fichier de modèle non trouvé !")
    print("   Lance d'abord le script d'entraînement principal.")
    exit()

def charger_historique():
    """Charge l'historique d'entraînement si disponible"""
    if os.path.exists(HISTORIQUE_FICHIER):
        try:
            with open(HISTORIQUE_FICHIER, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

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
            
            # Obtenir aussi le top 3 des prédictions
            probs = torch.exp(output)
            top3_probs, top3_indices = torch.topk(probs, 3)
            top3_predictions = [(top3_indices[0][i].item(), top3_probs[0][i].item() * 100) 
                              for i in range(3)]
        
        # Affichage des résultats
        plt.figure(figsize=(15, 4))
        
        # Image originale
        plt.subplot(1, 4, 1)
        plt.imshow(image_originale, cmap='gray' if image_originale.mode == 'L' else None)
        plt.title('Image originale')
        plt.axis('off')
        
        # Image préparée
        plt.subplot(1, 4, 2)
        plt.imshow(image_array, cmap='gray')
        plt.title('Image préparée\n(comme l\'IA la voit)')
        plt.axis('off')
        
        # Résultat principal
        plt.subplot(1, 4, 3)
        plt.text(0.5, 0.8, f'🤖 Prédiction principale', ha='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, 0.6, f'Chiffre: {prediction}', ha='center', fontsize=24, fontweight='bold', 
                 transform=plt.gca().transAxes)
        plt.text(0.5, 0.4, f'Confiance: {confidence:.1f}%', ha='center', fontsize=16, 
                 transform=plt.gca().transAxes)
        
        # Indicateur de confiance par couleur
        if confidence > 90:
            color = 'green'
            emoji = '🟢'
        elif confidence > 70:
            color = 'orange'
            emoji = '🟡'
        else:
            color = 'red'
            emoji = '🔴'
        
        plt.text(0.5, 0.2, f'{emoji} {color.upper()}', ha='center', fontsize=12, 
                 transform=plt.gca().transAxes, color=color)
        plt.axis('off')
        
        # Top 3 des prédictions
        plt.subplot(1, 4, 4)
        plt.text(0.5, 0.9, 'Top 3 des prédictions', ha='center', fontsize=12, fontweight='bold',
                 transform=plt.gca().transAxes)
        
        for i, (digit, prob) in enumerate(top3_predictions):
            y_pos = 0.7 - i * 0.2
            plt.text(0.5, y_pos, f'{i+1}. Chiffre {digit} : {prob:.1f}%', 
                     ha='center', fontsize=11, transform=plt.gca().transAxes)
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Prédiction terminée !")
        print(f"   🎯 Chiffre détecté : {prediction}")
        print(f"   📊 Confiance : {confidence:.1f}%")
        print(f"   🏆 Top 3 : {', '.join([f'{d}({p:.1f}%)' for d, p in top3_predictions])}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de l'image : {e}")
        return None, 0

def predire_plusieurs_images(dossier_path):
    """Prédit tous les chiffres dans un dossier d'images"""
    if not os.path.exists(dossier_path):
        print(f"❌ Dossier non trouvé : {dossier_path}")
        return
    
    # Extensions d'images supportées
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    # Trouver toutes les images
    images = []
    for fichier in os.listdir(dossier_path):
        if any(fichier.lower().endswith(ext) for ext in extensions):
            images.append(os.path.join(dossier_path, fichier))
    
    if not images:
        print(f"❌ Aucune image trouvée dans {dossier_path}")
        return
    
    print(f"🔍 Analyse de {len(images)} image(s)...")
    
    resultats = []
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)}: {os.path.basename(image_path)} ---")
        prediction, confidence = predire_image_perso(image_path)
        if prediction is not None:
            resultats.append({
                'fichier': os.path.basename(image_path),
                'prediction': prediction,
                'confidence': confidence
            })
    
    # Résumé
    print(f"\n📋 RÉSUMÉ DE L'ANALYSE")
    print("="*40)
    for r in resultats:
        print(f"📄 {r['fichier']} → Chiffre {r['prediction']} ({r['confidence']:.1f}%)")
    
    return resultats

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

def afficher_historique_complet():
    """Affiche l'historique complet d'entraînement avec graphiques"""
    historique = charger_historique()
    
    if not historique:
        print("❌ Aucun historique d'entraînement trouvé.")
        return
    
    print("\n📊 HISTORIQUE COMPLET D'ENTRAÎNEMENT")
    print("="*50)
    print(f"🎯 Précision actuelle : {historique.get('precision_finale', 0):.2f}%")
    print(f"📈 Nombre d'époques : {historique.get('epoch_actuelle', 0)}")
    print(f"📅 Dernière sauvegarde : {historique.get('derniere_sauvegarde', 'Inconnue')}")
    
    # Calculs d'amélioration
    test_acc = historique.get('test_accuracies', [])
    train_acc = historique.get('train_accuracies', [])
    
    if len(test_acc) > 1:
        print(f"🚀 Amélioration totale : +{test_acc[-1] - test_acc[0]:.2f}%")
        print(f"📉 Meilleure époque : Époque {test_acc.index(max(test_acc)) + 1} ({max(test_acc):.2f}%)")
    
    # Graphique si on a des données
    if test_acc and train_acc:
        plt.figure(figsize=(12, 4))
        
        epochs = range(1, len(test_acc) + 1)
        
        # Évolution de la précision
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_acc, 'bo-', label='Entraînement', alpha=0.7)
        plt.plot(epochs, test_acc, 'ro-', label='Test', alpha=0.7)
        plt.title('Évolution de la précision')
        plt.xlabel('Époque')
        plt.ylabel('Précision (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Évolution des pertes
        train_losses = historique.get('train_losses', [])
        if train_losses:
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_losses, 'go-', alpha=0.7)
            plt.title('Évolution de la perte')
            plt.xlabel('Époque')
            plt.ylabel('Perte')
            plt.grid(True, alpha=0.3)
        
        # Comparaison finale
        plt.subplot(1, 3, 3)
        if len(test_acc) > 5:
            # Tendance récente (5 dernières époques)
            recent_epochs = epochs[-5:]
            recent_test = test_acc[-5:]
            recent_train = train_acc[-5:]
            plt.plot(recent_epochs, recent_train, 'bo-', label='Entraînement', alpha=0.7)
            plt.plot(recent_epochs, recent_test, 'ro-', label='Test', alpha=0.7)
            plt.title('Tendance récente (5 dernières)')
        else:
            plt.plot(epochs, train_acc, 'bo-', label='Entraînement', alpha=0.7)
            plt.plot(epochs, test_acc, 'ro-', label='Test', alpha=0.7)
            plt.title('Évolution complète')
        
        plt.xlabel('Époque')
        plt.ylabel('Précision (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def menu_interactif():
    """Menu interactif pour utiliser l'IA"""
    print("\n" + "="*60)
    print("🤖 IA DE RECONNAISSANCE DE CHIFFRES - VERSION INCRÉMENTALE")
    print("="*60)
    
    # Afficher info du modèle au démarrage
    historique = charger_historique()
    if historique:
        print(f"📊 Modèle actuel : {historique.get('epoch_actuelle', 0)} époques, "
              f"précision {historique.get('precision_finale', 0):.2f}%")
    
    while True:
        print("\n📋 Que veux-tu faire ?")
        print("1. 🖼️  Tester une de mes images")
        print("2. 📁 Tester toutes les images d'un dossier")
        print("3. 🎨 Créer une image de test")
        print("4. 📊 Voir l'historique complet d'entraînement")
        print("5. 🔧 Informations sur le modèle")
        print("6. 🚪 Quitter")
        
        choix = input("\n👉 Ton choix (1-6) : ").strip()
        
        if choix == "1":
            chemin = input("📁 Chemin vers ton image : ").strip()
            if chemin:
                predire_image_perso(chemin)
            else:
                print("❌ Chemin vide !")
        
        elif choix == "2":
            dossier = input("📁 Chemin vers le dossier d'images : ").strip()
            if dossier:
                predire_plusieurs_images(dossier)
            else:
                print("❌ Chemin vide !")
        
        elif choix == "3":
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
        
        elif choix == "4":
            afficher_historique_complet()
        
        elif choix == "5":
            print("\n🔧 INFORMATIONS SUR LE MODÈLE")
            print("="*40)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   🧠 Paramètres totaux : {total_params:,}")
            print(f"   🖥️  Dispositif : {device}")
            print(f"   📁 Fichiers du modèle :")
            
            for fichier in [MODELE_FICHIER, HISTORIQUE_FICHIER, OPTIMISEUR_FICHIER]:
                if os.path.exists(fichier):
                    taille = os.path.getsize(fichier) / 1024  # Ko
                    print(f"      ✅ {fichier} ({taille:.1f} Ko)")
                else:
                    print(f"      ❌ {fichier} (manquant)")
            
            if historique:
                print(f"   📈 Statistiques d'entraînement :")
                print(f"      - Époques : {historique.get('epoch_actuelle', 0)}")
                print(f"      - Précision : {historique.get('precision_finale', 0):.2f}%")
                print(f"      - Dernière MAJ : {historique.get('derniere_sauvegarde', 'Inconnue')}")
        
        elif choix == "6":
            print("👋 Au revoir !")
            break
        
        else:
            print("❌ Choix invalide ! Utilise 1, 2, 3, 4, 5 ou 6")

# Lancer le menu interactif
if __name__ == "__main__":
    menu_interactif()