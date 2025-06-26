# Entraînement amélioré avec gestion des sauvegardes multiples
# Compatible Python 3.13.2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Utilisation de : {device}")

# Créer un dossier pour les modèles
if not os.path.exists('modeles'):
    os.makedirs('modeles')

print("🤖 Entraînement amélioré de l'IA")
print("=" * 60)

# 1. CONFIGURATION DE L'ENTRAÎNEMENT
print("⚙️ Configuration de l'entraînement...")

# Paramètres modifiables
EPOCHS = 20  # Plus d'époques = meilleur apprentissage
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.8

print(f"   📊 Époques : {EPOCHS}")
print(f"   📦 Taille des lots : {BATCH_SIZE}")
print(f"   🎯 Taux d'apprentissage : {LEARNING_RATE}")
print(f"   🛡️ Dropout : {DROPOUT_RATE}")

# 2. CHARGEMENT DES DONNÉES AVEC AUGMENTATION
print("\n📚 Chargement des données avec augmentation...")

# Transformations pour l'entraînement (avec augmentation)
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # Rotation aléatoire
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translation
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Transformations pour le test (sans augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Datasets
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST('data', train=False, transform=transform_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"✅ Données chargées avec augmentation !")

# 3. RÉSEAU AMÉLIORÉ
print("\n🧠 Construction du réseau amélioré...")

class ReseauChiffresAmeliore(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ReseauChiffresAmeliore, self).__init__()
        
        # Plus de couches pour plus de puissance
        self.fc1 = nn.Linear(28*28, 256)  # Plus de neurones
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        
        # Normalisation par lots
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        
        # Couche 1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        # Couche 2
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # Couche 3
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # Couche 4
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        # Couche de sortie
        x = F.log_softmax(self.fc5(x), dim=1)
        return x

# Créer le modèle
model = ReseauChiffresAmeliore(dropout_rate=DROPOUT_RATE).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Réseau amélioré créé ! Paramètres : {total_params:,}")

# 4. CONFIGURATION AVANCÉE
print("\n⚙️ Configuration avancée...")

# Optimiseur avec scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
criterion = nn.NLLLoss()

# Variables pour tracking
train_losses = []
train_accuracies = []
test_accuracies = []
best_accuracy = 0
best_model_path = ""

# 5. FONCTIONS D'ENTRAÎNEMENT AMÉLIORÉES
def entrainer_une_epoque(epoch):
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
        
        # Affichage moins fréquent
        if batch_idx % 300 == 0:
            print(f'   Époque {epoch}, Batch {batch_idx:3d}/{len(train_loader)} - '
                  f'Perte: {loss.item():.4f}, Précision: {100. * correct / total:.1f}%')
    
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

# 6. ENTRAÎNEMENT PRINCIPAL
print(f"\n🏋️ Entraînement sur {EPOCHS} époques...")
print("   (Plus d'époques = meilleure performance)")

# Timestamp pour les noms de fichiers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Époque {epoch}/{EPOCHS} ---")
    
    # Entraînement
    train_loss, train_acc = entrainer_une_epoque(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Test
    test_acc = tester()
    test_accuracies.append(test_acc)
    
    # Mise à jour du learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"✅ Époque {epoch} terminée:")
    print(f"   - Précision entraînement: {train_acc:.2f}%")
    print(f"   - Précision test: {test_acc:.2f}%")
    print(f"   - Learning rate: {current_lr:.6f}")
    
    # Sauvegarder le meilleur modèle
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model_path = f'modeles/meilleur_modele_{timestamp}_{test_acc:.1f}pct.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"   🏆 Nouveau record ! Sauvegardé : {best_model_path}")
    
    # Sauvegarder aussi à chaque époque
    epoch_path = f'modeles/modele_epoque_{epoch}_{timestamp}_{test_acc:.1f}pct.pth'
    torch.save(model.state_dict(), epoch_path)

# 7. SAUVEGARDE FINALE
final_path = f'modeles/modele_final_{timestamp}_{test_accuracies[-1]:.1f}pct.pth'
torch.save(model.state_dict(), final_path)

# Aussi sauvegarder comme fichier principal (pour compatibilité)
torch.save(model.state_dict(), 'modele_chiffres_pytorch.pth')

print(f"\n🎉 Entraînement terminé !")
print(f"🎯 Précision finale : {test_accuracies[-1]:.2f}%")
print(f"🏆 Meilleure précision : {best_accuracy:.2f}%")

# 8. GRAPHIQUES DÉTAILLÉS
print("\n📈 Création des graphiques...")
plt.figure(figsize=(18, 6))

# Précision
plt.subplot(1, 3, 1)
epochs_range = range(1, EPOCHS + 1)
plt.plot(epochs_range, train_accuracies, 'bo-', label='Entraînement', linewidth=2)
plt.plot(epochs_range, test_accuracies, 'ro-', label='Test', linewidth=2)
plt.title('Évolution de la précision', fontsize=14)
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Perte
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_losses, 'go-', linewidth=2)
plt.title('Évolution de la perte', fontsize=14)
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.grid(True, alpha=0.3)

# Comparaison finale
plt.subplot(1, 3, 3)
plt.bar(['Entraînement', 'Test'], [train_accuracies[-1], test_accuracies[-1]], 
        color=['blue', 'red'], alpha=0.7)
plt.title('Précision finale', fontsize=14)
plt.ylabel('Précision (%)')
plt.ylim(0, 100)

# Ajouter les valeurs sur les barres
for i, v in enumerate([train_accuracies[-1], test_accuracies[-1]]):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 9. RÉSUMÉ DES FICHIERS SAUVEGARDÉS
print("\n📁 Fichiers sauvegardés :")
print(f"   🏆 Meilleur modèle : {best_model_path}")
print(f"   📊 Modèle final : {final_path}")
print(f"   🔄 Modèle principal : modele_chiffres_pytorch.pth")
print(f"   📂 Dossier : modeles/")

# 10. CONSEILS POUR AMÉLIORER
print("\n💡 Conseils pour améliorer :")
if test_accuracies[-1] < 95:
    print("   - Augmente le nombre d'époques (EPOCHS)")
    print("   - Réduis le learning rate (LEARNING_RATE)")
    print("   - Augmente la taille du réseau")
elif test_accuracies[-1] < 98:
    print("   - Ajoute plus d'augmentation de données")
    print("   - Ajuste le dropout")
    print("   - Essaie différents optimiseurs")
else:
    print("   🎉 Excellent ! Ton IA est très performante !")

print(f"\n📊 Statistiques finales :")
print(f"   - Amélioration : {test_accuracies[-1] - test_accuracies[0]:.1f}%")
print(f"   - Écart train/test : {abs(train_accuracies[-1] - test_accuracies[-1]):.1f}%")
if abs(train_accuracies[-1] - test_accuracies[-1]) > 5:
    print("   ⚠️ Possible surapprentissage - augmente le dropout")