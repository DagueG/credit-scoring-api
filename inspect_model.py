import pickle
import os

model_path = r'z:\openclassroom\credit-scoring-api\model\model.pkl'

# Charger avec pickle
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Fichier chargé avec pickle")
except Exception as e:
    print(f"Erreur pickle: {e}")
    exit(1)

# Inspection du modèle
print("\n" + "="*60)
print("INSPECTION DU MODÈLE")
print("="*60)

print(f"\n1. Type du modèle:")
print(f"   {type(model)}")
print(f"   {type(model).__name__}")

# Vérifier si c'est un pipeline
print(f"\n2. Est-ce un Pipeline sklearn?")
if hasattr(model, 'steps'):
    print(f"   ✓ Oui, c'est un Pipeline")
    print(f"\n   Étapes:")
    for i, (name, step) in enumerate(model.steps):
        print(f"     {i+1}. {name}: {type(step).__name__}")
    
    if hasattr(model, 'named_steps'):
        print(f"\n   Named steps disponibles: {list(model.named_steps.keys())}")
else:
    print(f"   ✗ Non, ce n'est pas un Pipeline")
    print(f"   Type: {type(model).__name__}")

# Features attendues
print(f"\n3. Features attendues en entrée:")
if hasattr(model, 'feature_names_in_'):
    print(f"   ✓ Disponible: {list(model.feature_names_in_)}")
    print(f"   Nombre de features: {len(model.feature_names_in_)}")
else:
    print(f"   ✗ Pas d'attribut 'feature_names_in_'")
    if hasattr(model, 'n_features_in_'):
        print(f"   Mais n_features_in_ = {model.n_features_in_}")

# Méthodes de prédiction
print(f"\n4. Méthodes de prédiction disponibles:")
methods = ['predict', 'predict_proba', 'predict_log_proba', 'decision_function']
for method in methods:
    if hasattr(model, method):
        print(f"   ✓ {method}")
    else:
        print(f"   ✗ {method}")

# Afficher les attributs importants
print(f"\n5. Attributs du modèle:")
attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
for attr in attrs[:15]:  # Les 15 premiers
    try:
        val = getattr(model, attr)
        if not isinstance(val, (list, dict)) or len(str(val)) < 100:
            print(f"   {attr}: {val}")
    except:
        pass

print("\n" + "="*60)
