"""
Script de téléchargement des fichiers de modèle et données pour HF Spaces.
Les fichiers binaires ne peuvent pas être pushés directement sur HF Spaces,
donc on les télécharge depuis GitHub releases au besoin.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Configuration
GITHUB_REPO = "DagueG/credit-scoring-api"
GITHUB_BRANCH = "main"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"

FILES_TO_DOWNLOAD = {
    "model/model.pkl": f"{BASE_URL}/model/model.pkl",
    "data/clients_reference.parquet": f"{BASE_URL}/data/clients_reference.parquet",
    "data/drift_baseline.parquet": f"{BASE_URL}/data/drift_baseline.parquet",
}

def download_file(url: str, dest: Path) -> bool:
    """Télécharge un fichier depuis une URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Téléchargement: {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✅ {dest.name} ({size_mb:.2f} MB) téléchargé")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Télécharge les fichiers manquants."""
    print("\n🔄 Vérification des fichiers de modèle et données...\n")
    
    missing_files = []
    for local_path, url in FILES_TO_DOWNLOAD.items():
        path = Path(local_path)
        if not path.exists():
            missing_files.append((path, url))
            print(f"⚠️  Fichier manquant: {local_path}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ Trouvé: {local_path} ({size_mb:.2f} MB)")
    
    if not missing_files:
        print("\n✨ Tous les fichiers sont présents!\n")
        return 0
    
    print(f"\n📥 Téléchargement de {len(missing_files)} fichier(s)...\n")
    
    for path, url in missing_files:
        if not download_file(url, path):
            print(f"❌ Impossible de télécharger {path}")
            print(f"   URL: {url}")
            return 1
    
    print("\n✨ Tous les fichiers ont été téléchargés avec succès!\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
