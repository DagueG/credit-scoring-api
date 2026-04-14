#!/usr/bin/env python
"""
Entrypoint pour le container Docker - vérifie/télécharge les fichiers avant de lancer l'API.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Main entrypoint."""
    app_dir = Path("/app")
    
    # Télécharger les fichiers manquants
    print("\n🔄 Vérification des fichiers...")
    result = subprocess.run([sys.executable, "scripts/download_models.py"], cwd=app_dir)
    
    if result.returncode != 0:
        print("\n❌ Erreur lors du téléchargement des fichiers")
        sys.exit(1)
    
    print("\n🚀 Démarrage de l'API...")
    
    # Lancer uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "7860"
    ], cwd=app_dir)

if __name__ == "__main__":
    main()
