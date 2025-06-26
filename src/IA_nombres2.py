# IA de reconnaissance de chiffres manuscrits avec PyTorch
# Compatible Python 3.13.2 - Projet pour classe de seconde

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Vérifier si GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Utilisation de : {device}")

print("🤖 Création d'une IA pour reconnaître les chiffres manuscrits")
print("=" * 60)

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
print("📚 Chargement des données MNIST...")

# Transformations pour normaliser les données
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit en tensor PyTorch
    transforms.Normalize((0.1307,), (0.3081,))  # Normalisation MNIST standard
])

# Téléchargement des données
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

# Création des loaders pour traiter les données par petits groupes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"✅ Données chargées !")
print(f"   - Images d'entraînement : {len(train_dataset)}")
print(f"   - Images de test : {len(test_dataset)}")

# 2. VISUALISATION DE QUELQUES EXEMPLES
print("\n🖼️ Affichage de quelques exemples...")
plt.figure(figsize=(12, 6))

# Prendre quelques exemples du dataset
examples = []
labels = []
for i in range(10):
    img, label = train_dataset[i]
    examples.append(img)
    labels.append(label)

for i in range(10):
    plt.subplot(2, 5, i+1)
    # Convertir le tensor en numpy et retirer la dimension couleur
    img_np = examples[i].squeeze().numpy()
    plt.imshow(img_np, cmap='gray')
    plt.title(f'Chiffre: {labels[i]}')
    plt.axis('off')

plt.suptitle('Exemples de chiffres manuscrits', fontsize=16)
plt.tight_layout()
plt.show()

# 3. DÉFINITION DU RÉSEAU DE NEURONES
print("\n🧠 Construction du réseau de neurones...")

class ReseauChiffres(nn.Module):
    def __init__(self):
        super(ReseauChiffres, self).__init__()
        # Couches fully connected (entièrement connectées)
        self.fc1 = nn.Linear(28*28, 128)  # 784 -> 128
        self.fc2 = nn.Linear(128, 64)     # 128 -> 64
        self.fc3 = nn.Linear(64, 10)      # 64 -> 10 (0-9)
        self.dropout = nn.Dropout(0.2)    # Pour éviter le surapprentissage
    
    def forward(self, x):
        # Aplatir l'image 28x28 en vecteur de 784
        x = x.view(-1, 28*28)
        
        # Première couche + fonction d'activation ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Deuxième couche + ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Couche de sortie (pas d'activation, on utilise log_softmax)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Création du modèle
model = ReseauChiffres().to(device)
print("✅ Réseau de neurones créé !")

# Compter les paramètres
total_params = sum(p.numel() for p in model.parameters())
print(f"   - Nombre total de paramètres : {total_params:,}")

# 4. CONFIGURATION DE L'ENTRAÎNEMENT
print("\n⚙️ Configuration de l'entraînement...")

# Optimiseur et fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss

# Listes pour sauvegarder les métriques
train_losses = []
train_accuracies = []
test_accuracies = []

# 5. FONCTION D'ENTRAÎNEMENT
def entrainer_une_epoque():
    model.train()  # Mode entraînement
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Remettre les gradients à zéro
        optimizer.zero_grad()
        
        # Prédiction
        output = model(data)
        
        # Calcul de la perte
        loss = criterion(output, target)
        
        # Rétropropagation
        loss.backward()
        optimizer.step()
        
        # Statistiques
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Affichage du progrès
        if batch_idx % 200 == 0:
            print(f'   Batch {batch_idx:3d}/{len(train_loader)} - '
                  f'Perte: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    return avg_loss, accuracy

# 6. FONCTION DE TEST
def tester():
    model.eval()  # Mode évaluation
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Pas de calcul de gradients
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_dataset)
    test_accuracies.append(accuracy)
    
    return accuracy

# 7. ENTRAÎNEMENT PRINCIPAL
print("\n🏋️ Entraînement de l'IA en cours...")
print("   (Cela peut prendre quelques minutes)")

epochs = 5
for epoch in range(1, epochs + 1):
    print(f"\n--- Époque {epoch}/{epochs} ---")
    
    # Entraînement
    train_loss, train_acc = entrainer_une_epoque()
    
    # Test
    test_acc = tester()
    
    print(f"✅ Époque {epoch} terminée:")
    print(f"   - Précision entraînement: {train_acc:.2f}%")
    print(f"   - Précision test: {test_acc:.2f}%")

print(f"\n🎉 Entraînement terminé !")
print(f"🎯 Précision finale : {test_accuracies[-1]:.2f}%")

# 8. GRAPHIQUES D'APPRENTISSAGE
print("\n📈 Création des graphiques d'apprentissage...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, epochs+1), train_accuracies, 'bo-', label='Entraînement')
plt.plot(range(1, epochs+1), test_accuracies, 'ro-', label='Test')
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, epochs+1), train_losses, 'go-')
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.grid(True)

plt.subplot(1, 3, 3)
epochs_range = range(1, epochs+1)
plt.plot(epochs_range, train_accuracies, label='Entraînement')
plt.plot(epochs_range, test_accuracies, label='Test')
plt.fill_between(epochs_range, train_accuracies, alpha=0.3)
plt.fill_between(epochs_range, test_accuracies, alpha=0.3)
plt.title('Comparaison précision')
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. TEST SUR DES EXEMPLES SPÉCIFIQUES
print("\n🔍 Test de l'IA sur quelques exemples...")

# Prendre 10 exemples du dataset de test
model.eval()
examples_data = []
examples_labels = []
predictions_data = []

with torch.no_grad():
    for i in range(10):
        data, label = test_dataset[i]
        examples_data.append(data)
        examples_labels.append(label)
        
        # Prédiction
        data_batch = data.unsqueeze(0).to(device)  # Ajouter dimension batch
        output = model(data_batch)
        prediction = output.argmax(dim=1).item()
        confidence = torch.exp(output.max()).item() * 100
        
        predictions_data.append((prediction, confidence))

# Affichage des résultats
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    
    # Image
    img = examples_data[i].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    
    # Informations
    true_digit = examples_labels[i]
    pred_digit, confidence = predictions_data[i]
    
    # Couleur selon si c'est correct
    color = 'green' if pred_digit == true_digit else 'red'
    
    plt.title(f'Vrai: {true_digit}\nIA: {pred_digit} ({confidence:.1f}%)', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Prédictions de l\'IA (vert = correct, rouge = erreur)', fontsize=14)
plt.tight_layout()
plt.show()

# 10. FONCTION POUR TESTER UN EXEMPLE SPÉCIFIQUE
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
    
    # Affichage
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

# 11. SAUVEGARDE DU MODÈLE
print("\n💾 Sauvegarde du modèle...")
torch.save(model.state_dict(), 'modele_chiffres_pytorch.pth')
print("✅ Modèle sauvegardé sous 'modele_chiffres_pytorch.pth'")

# 12. INFORMATIONS FINALES
print("\n" + "="*60)
print("🎉 PROJET TERMINÉ !")
print("="*60)
print(f"🎯 Précision finale : {test_accuracies[-1]:.2f}%")
print("🔧 Pour tester un exemple : tester_chiffre(42)")
print("📁 Modèle sauvegardé : modele_chiffres_pytorch.pth")
print("\n📋 Commandes utiles :")
print("   - tester_chiffre(123) : teste l'exemple n°123")
print("   - model.train() : passer en mode entraînement")   
print("   - model.eval() : passer en mode évaluation")