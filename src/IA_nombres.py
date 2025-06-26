# IA de reconnaissance de chiffres manuscrits
# Projet pour classe de seconde

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("🤖 Création d'une IA pour reconnaître les chiffres manuscrits")
print("=" * 60)

# 1. CHARGEMENT DES DONNÉES
print("📚 Chargement des données d'entraînement...")
# MNIST = base de données avec 70 000 images de chiffres dessinés à la main
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"✅ Données chargées !")
print(f"   - Images d'entraînement : {x_train.shape[0]}")
print(f"   - Images de test : {x_test.shape[0]}")
print(f"   - Taille de chaque image : {x_train.shape[1]}x{x_train.shape[2]} pixels")

# 2. VISUALISATION DE QUELQUES EXEMPLES
print("\n🖼️ Affichage de quelques exemples...")
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Chiffre: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Exemples de chiffres manuscrits', fontsize=16)
plt.tight_layout()
plt.show()

# 3. PRÉPARATION DES DONNÉES
print("\n🔧 Préparation des données...")
# Normalisation : convertir les valeurs de 0-255 vers 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Aplatir les images 28x28 en vecteurs de 784 valeurs
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

print(f"✅ Données préparées !")
print(f"   - Forme finale des données : {x_train.shape}")

# 4. CRÉATION DU MODÈLE D'IA
print("\n🧠 Construction du réseau de neurones...")
model = keras.Sequential([
    # Couche d'entrée : 784 neurones (un par pixel)
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    # Couche cachée : 128 neurones
    keras.layers.Dense(64, activation='relu'),
    
    # Couche de sortie : 10 neurones (un par chiffre 0-9)
    keras.layers.Dense(10, activation='softmax')
])

# Configuration de l'apprentissage
model.compile(
    optimizer='adam',      # Algorithme d'optimisation
    loss='sparse_categorical_crossentropy',  # Fonction de perte
    metrics=['accuracy']   # Mesure de performance
)

print("✅ Réseau de neurones créé !")
print(f"   - Nombre total de paramètres : {model.count_params():,}")

# 5. ENTRAÎNEMENT DE L'IA
print("\n🏋️ Entraînement de l'IA en cours...")
print("   (Cela peut prendre quelques minutes)")

# Entraînement sur 5 époques (5 passages sur toutes les données)
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,  # 10% des données pour validation
    verbose=1
)

print("✅ Entraînement terminé !")

# 6. ÉVALUATION DES PERFORMANCES
print("\n📊 Évaluation des performances...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Précision sur les données de test : {test_accuracy:.2%}")

# 7. GRAPHIQUE DE L'APPRENTISSAGE
print("\n📈 Création du graphique d'apprentissage...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision entraînement')
plt.plot(history.history['val_accuracy'], label='Précision validation')
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()

# 8. TEST SUR QUELQUES EXEMPLES
print("\n🔍 Test de l'IA sur quelques exemples...")
predictions = model.predict(x_test[:10])

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    
    # Reformer l'image 28x28 pour l'affichage
    image = x_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    
    # Prédiction de l'IA
    predicted_digit = np.argmax(predictions[i])
    confidence = np.max(predictions[i]) * 100
    
    # Vraie réponse
    true_digit = y_test[i]
    
    # Couleur : vert si correct, rouge si incorrect
    color = 'green' if predicted_digit == true_digit else 'red'
    
    plt.title(f'Vrai: {true_digit}\nIA: {predicted_digit} ({confidence:.1f}%)', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Prédictions de l\'IA (vert = correct, rouge = erreur)', fontsize=14)
plt.tight_layout()
plt.show()

# 9. FONCTION POUR TESTER UN CHIFFRE SPÉCIFIQUE
def tester_chiffre(index):
    """Teste l'IA sur un chiffre spécifique"""
    image = x_test[index].reshape(28, 28)
    prediction = model.predict(x_test[index:index+1])
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    true_digit = y_test[index]
    
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f'Chiffre réel: {true_digit}\n'
              f'Prédiction IA: {predicted_digit}\n'
              f'Confiance: {confidence:.1f}%', fontsize=12)
    plt.axis('off')
    plt.show()
    
    return predicted_digit, confidence

# 10. SAUVEGARDE DU MODÈLE
print("\n💾 Sauvegarde du modèle...")
model.save('modele_chiffres.h5')
print("✅ Modèle sauvegardé sous 'modele_chiffres.h5'")

print("\n🎉 PROJET TERMINÉ !")
print("=" * 60)
print(f"🎯 Précision finale : {test_accuracy:.2%}")
print("🔧 Pour tester un exemple : tester_chiffre(42)")
print("📁 Modèle sauvegardé : modele_chiffres.h5")