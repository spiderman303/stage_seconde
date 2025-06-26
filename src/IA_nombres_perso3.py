# IA de reconnaissance de chiffres manuscrits avec PyTorch
# Version avec sauvegarde incrémentale - Compatible Python 3.13.2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Vérifier si GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Utilisation de : {device}")

print("🤖 IA de reconnaissance de chiffres manuscrits - Version incrémentale")
print("=" * 70)

# Configuration des fichiers de sauvegarde
MODELE_FICHIER = 'modele_chiffres_pytorch.pth'
HISTORIQUE_FICHIER = 'historique_entrainement.json'
OPTIMISEUR_FICHIER = 'optimiseur_pytorch.pth'

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
print("📚 Chargement des données MNIST...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"✅ Données chargées !")
print(f"   - Images d'entraînement : {len(train_dataset)}")
print(f"   - Images de test : {len(test_dataset)}")

# 2. DÉFINITION DU RÉSEAU DE NEURONES (identique)
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

# 3. FONCTIONS DE SAUVEGARDE ET CHARGEMENT
def sauvegarder_modele_complet(model, optimizer, epoch, train_losses, train_accuracies, test_accuracies):
    """Sauvegarde le modèle, l'optimiseur et l'historique d'entraînement"""
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), MODELE_FICHIER)
    
    # Sauvegarde de l'optimiseur
    torch.save(optimizer.state_dict(), OPTIMISEUR_FICHIER)
    
    # Sauvegarde de l'historique
    historique = {
        'epoch_actuelle': epoch,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'derniere_sauvegarde': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'precision_finale': test_accuracies[-1] if test_accuracies else 0
    }
    
    with open(HISTORIQUE_FICHIER, 'w') as f:
        json.dump(historique, f, indent=2)
    
    print(f"💾 Modèle sauvegardé (Époque {epoch}, Précision: {test_accuracies[-1]:.2f}%)")

def charger_modele_existant(model, optimizer):
    """Charge un modèle existant s'il existe"""
    
    # Initialiser les variables par défaut
    epoch_debut = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Vérifier si les fichiers existent
    if os.path.exists(MODELE_FICHIER) and os.path.exists(HISTORIQUE_FICHIER):
        try:
            # Charger le modèle
            model.load_state_dict(torch.load(MODELE_FICHIER, map_location=device))
            print("✅ Modèle précédent chargé avec succès !")
            
            # Charger l'optimiseur si disponible
            if os.path.exists(OPTIMISEUR_FICHIER):
                optimizer.load_state_dict(torch.load(OPTIMISEUR_FICHIER, map_location=device))
                print("✅ État de l'optimiseur chargé !")
            
            # Charger l'historique
            with open(HISTORIQUE_FICHIER, 'r') as f:
                historique = json.load(f)
            
            epoch_debut = historique.get('epoch_actuelle', 0)
            train_losses = historique.get('train_losses', [])
            train_accuracies = historique.get('train_accuracies', [])
            test_accuracies = historique.get('test_accuracies', [])
            
            print(f"📈 Historique chargé :")
            print(f"   - Époque précédente : {epoch_debut}")
            print(f"   - Dernière précision : {historique.get('precision_finale', 0):.2f}%")
            print(f"   - Dernière sauvegarde : {historique.get('derniere_sauvegarde', 'Inconnue')}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement : {e}")
            print("🔄 Démarrage d'un nouvel entraînement...")
            epoch_debut = 0
            train_losses = []
            train_accuracies = []
            test_accuracies = []
    else:
        print("🆕 Aucun modèle précédent trouvé. Démarrage d'un nouvel entraînement.")
    
    return epoch_debut, train_losses, train_accuracies, test_accuracies

# 4. CRÉATION ET CHARGEMENT DU MODÈLE
print("\n🧠 Création/Chargement du modèle...")

model = ReseauChiffres().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Charger le modèle existant si disponible
epoch_debut, train_losses, train_accuracies, test_accuracies = charger_modele_existant(model, optimizer)

total_params = sum(p.numel() for p in model.parameters())
print(f"   - Nombre total de paramètres : {total_params:,}")

# 5. FONCTIONS D'ENTRAÎNEMENT ET TEST (identiques)
def entrainer_une_epoque(epoch_num):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 200 == 0:
            print(f'   Époque {epoch_num} - Batch {batch_idx:3d}/{len(train_loader)} - '
                  f'Perte: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def tester():
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_dataset)
    return accuracy

# 6. ENTRAÎNEMENT PRINCIPAL AVEC SAUVEGARDE INCRÉMENTALE
print(f"\n🏋️ Entraînement de l'IA (à partir de l'époque {epoch_debut + 1})...")

# Demander le nombre d'époques à ajouter
print(f"📝 Combien d'époques voulez-vous ajouter ? (Recommandé: 3-5)")
try:
    nouvelles_epoques = int(input("Nombre d'époques : ") or "3")
except:
    nouvelles_epoques = 3
    print(f"⚠️ Valeur par défaut utilisée : {nouvelles_epoques} époques")

epochs_fin = epoch_debut + nouvelles_epoques

for epoch in range(epoch_debut + 1, epochs_fin + 1):
    print(f"\n--- Époque {epoch} ---")
    
    # Entraînement
    train_loss, train_acc = entrainer_une_epoque(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Test
    test_acc = tester()
    test_accuracies.append(test_acc)
    
    print(f"✅ Époque {epoch} terminée:")
    print(f"   - Précision entraînement: {train_acc:.2f}%")
    print(f"   - Précision test: {test_acc:.2f}%")
    
    # Sauvegarde après chaque époque
    sauvegarder_modele_complet(model, optimizer, epoch, train_losses, train_accuracies, test_accuracies)

print(f"\n🎉 Entraînement terminé !")
print(f"🎯 Précision finale : {test_accuracies[-1]:.2f}%")

# 7. GRAPHIQUES D'APPRENTISSAGE COMPLETS
print("\n📈 Création des graphiques d'apprentissage...")

if len(train_accuracies) > 0:
    plt.figure(figsize=(15, 5))
    
    epochs_range = range(1, len(train_accuracies) + 1)
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Entraînement')
    plt.plot(epochs_range, test_accuracies, 'ro-', label='Test')
    plt.title('Évolution de la précision (Historique complet)')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_losses, 'go-')
    plt.title('Évolution de la perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.grid(True)
    
    # Zoom sur les dernières époques si il y en a beaucoup
    plt.subplot(1, 3, 3)
    if len(train_accuracies) > 10:
        derniers_10 = epochs_range[-10:]
        plt.plot(derniers_10, train_accuracies[-10:], 'bo-', label='Entraînement')
        plt.plot(derniers_10, test_accuracies[-10:], 'ro-', label='Test')
        plt.title('Évolution récente (10 dernières époques)')
    else:
        plt.plot(epochs_range, train_accuracies, 'bo-', label='Entraînement')
        plt.plot(epochs_range, test_accuracies, 'ro-', label='Test')
        plt.title('Évolution complète')
    
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 8. FONCTIONS DE TEST (identiques mais adaptées)
def tester_chiffre(index):
    """Teste l'IA sur un chiffre spécifique du dataset de test"""
    if index >= len(test_dataset):
        print(f"❌ Index trop grand ! Maximum: {len(test_dataset)-1}")
        return
    
    model.eval()
    data, true_label = test_dataset[index]
    
    with torch.no_grad():
        data_batch = data.unsqueeze(0).to(device)
        output = model(data_batch)
        prediction = output.argmax(dim=1).item()
        confidence = torch.exp(output.max()).item() * 100
    
    plt.figure(figsize=(6, 4))
    img = data.squeeze().numpy()
    plt.imshow(img, cmap='gray')
    
    color = 'green' if prediction == true_label else 'red'
    plt.title(f'Chiffre réel: {true_label}\n'
              f'Prédiction IA: {prediction}\n'
              f'Confiance: {confidence:.1f}%', 
              fontsize=12, color=color)
    plt.axis('off')
    plt.show()
    
    return prediction, confidence

def predire_image_perso(chemin_image):
    """Prédit le chiffre dans une image personnalisée"""
    try:
        if not os.path.exists(chemin_image):
            print(f"❌ Fichier non trouvé : {chemin_image}")
            return None, 0
        
        image_originale = Image.open(chemin_image)
        
        if image_originale.mode != 'L':
            image_gris = image_originale.convert('L')
        else:
            image_gris = image_originale
        
        image_redimensionnee = image_gris.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = np.array(image_redimensionnee)
        
        if np.mean(image_array) > 127:
            image_array = 255 - image_array
        
        image_array = image_array.astype(np.float32) / 255.0
        image_array = (image_array - 0.1307) / 0.3081
        
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
            confidence = torch.exp(output.max()).item() * 100
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_originale, cmap='gray' if image_originale.mode == 'L' else None)
        plt.title('Image originale')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image_array, cmap='gray')
        plt.title('Image préparée\n(comme l\'IA la voit)')
        plt.axis('off')
        
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

def afficher_historique():
    """Affiche l'historique complet d'entraînement"""
    if os.path.exists(HISTORIQUE_FICHIER):
        with open(HISTORIQUE_FICHIER, 'r') as f:
            historique = json.load(f)
        
        print("\n📊 HISTORIQUE D'ENTRAÎNEMENT")
        print("=" * 40)
        print(f"🎯 Précision actuelle : {historique.get('precision_finale', 0):.2f}%")
        print(f"📈 Nombre d'époques : {historique.get('epoch_actuelle', 0)}")
        print(f"📅 Dernière sauvegarde : {historique.get('derniere_sauvegarde', 'Inconnue')}")
        
        if len(historique.get('test_accuracies', [])) > 1:
            precisions = historique['test_accuracies']
            print(f"🚀 Amélioration : +{precisions[-1] - precisions[0]:.2f}% depuis le début")
    else:
        print("❌ Aucun historique trouvé.")

def reinitialiser_modele():
    """Remet le modèle à zéro (supprime les sauvegardes)"""
    response = input("⚠️ Êtes-vous sûr de vouloir supprimer le modèle existant ? (oui/non): ")
    if response.lower() in ['oui', 'yes', 'o', 'y']:
        fichiers_a_supprimer = [MODELE_FICHIER, HISTORIQUE_FICHIER, OPTIMISEUR_FICHIER]
        for fichier in fichiers_a_supprimer:
            if os.path.exists(fichier):
                os.remove(fichier)
                print(f"🗑️ {fichier} supprimé")
        print("✅ Modèle réinitialisé ! Relancez le script pour un nouvel entraînement.")
    else:
        print("❌ Réinitialisation annulée.")

# 9. INFORMATIONS FINALES
print("\n" + "="*70)
print("🎉 ENTRAÎNEMENT TERMINÉ !")
print("="*70)

# Afficher l'historique
afficher_historique()

print("\n📋 Commandes disponibles :")
print("   - tester_chiffre(123) : teste l'exemple n°123")
print("   - predire_image_perso('image.png') : teste ton image")
print("   - afficher_historique() : voir l'historique complet")
print("   - reinitialiser_modele() : remet à zéro (attention !)")

print("\n💡 Avantages de cette version :")
print("   ✅ Sauvegarde automatique après chaque époque")
print("   ✅ Reprise de l'entraînement où vous vous êtes arrêté")
print("   ✅ Historique complet conservé")
print("   ✅ Amélioration progressive des performances")
print("   ✅ Possibilité d'ajouter quelques époques à la fois")

print(f"\n🎯 Pour continuer l'entraînement, relancez simplement ce script !")