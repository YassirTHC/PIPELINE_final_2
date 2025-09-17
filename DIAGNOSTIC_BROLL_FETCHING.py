#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 DIAGNOSTIC COMPLET SYSTÈME B-ROLL
Identifie pourquoi les B-rolls ne sont pas téléchargés et propose des solutions
"""

import os
import sys
from pathlib import Path
import requests
import time

def check_api_keys():
    """Vérifie les clés API configurées"""
    print("🔑 VÉRIFICATION DES CLÉS API")
    print("=" * 50)
    
    api_keys = {
        'PEXELS_API_KEY': os.getenv('PEXELS_API_KEY'),
        'PIXABAY_API_KEY': os.getenv('PIXABAY_API_KEY'),
        'UNSPLASH_ACCESS_KEY': os.getenv('UNSPLASH_ACCESS_KEY'),
        'GIPHY_API_KEY': os.getenv('GIPHY_API_KEY')
    }
    
    configured = 0
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"✅ {key_name}: {key_value[:8]}****** (configurée)")
            configured += 1
        else:
            print(f"❌ {key_name}: NON CONFIGURÉE")
    
    print(f"\n📊 Résultat: {configured}/4 clés API configurées")
    
    if configured == 0:
        print("\n🚨 PROBLÈME MAJEUR: Aucune clé API configurée!")
        print("🔧 SOLUTION: Configurez au moins une clé API pour activer le téléchargement")
        print("\n📋 INSTRUCTIONS:")
        print("1. Créez un compte gratuit sur Pexels.com")
        print("2. Obtenez votre clé API Pexels")
        print("3. Définissez la variable d'environnement: set PEXELS_API_KEY=votre_cle")
        print("4. Ou ajoutez dans .env: PEXELS_API_KEY=votre_cle")
        return False
    
    return True

def test_api_connection():
    """Test la connexion aux APIs configurées"""
    print("\n🌐 TEST CONNEXION APIs")
    print("=" * 50)
    
    # Test Pexels
    pexels_key = os.getenv('PEXELS_API_KEY')
    if pexels_key:
        print("🔍 Test Pexels API...")
        try:
            headers = {"Authorization": pexels_key}
            response = requests.get(
                "https://api.pexels.com/videos/search?query=nature&per_page=1",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Pexels API: FONCTIONNELLE")
                return True
            else:
                print(f"❌ Pexels API: Erreur {response.status_code}")
                print(f"   Réponse: {response.text[:100]}")
        except Exception as e:
            print(f"❌ Pexels API: Erreur connexion - {e}")
    
    # Test Pixabay
    pixabay_key = os.getenv('PIXABAY_API_KEY')
    if pixabay_key:
        print("🔍 Test Pixabay API...")
        try:
            response = requests.get(
                f"https://pixabay.com/api/videos/?key={pixabay_key}&q=nature&per_page=3",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Pixabay API: FONCTIONNELLE")
                return True
            else:
                print(f"❌ Pixabay API: Erreur {response.status_code}")
        except Exception as e:
            print(f"❌ Pixabay API: Erreur connexion - {e}")
    
    print("❌ Aucune API fonctionnelle trouvée")
    return False

def check_cache_directories():
    """Vérifie les dossiers de cache B-roll"""
    print("\n📁 VÉRIFICATION DOSSIERS CACHE")
    print("=" * 50)
    
    cache_dirs = [
        "AI-B-roll/broll_library",
        "AI-B-roll/broll_library/fetched",
        "cache/broll",
        "cache/broll/pexels",
        "cache/broll/pixabay"
    ]
    
    total_files = 0
    for cache_dir in cache_dirs:
        path = Path(cache_dir)
        if path.exists():
            files = list(path.rglob("*.mp4")) + list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
            print(f"✅ {cache_dir}: {len(files)} fichiers")
            total_files += len(files)
        else:
            print(f"❌ {cache_dir}: N'EXISTE PAS")
    
    print(f"\n📊 Total: {total_files} fichiers B-roll en cache")
    
    if total_files == 0:
        print("\n🚨 PROBLÈME: Aucun B-roll en cache!")
        print("🔧 SOLUTION: Le système doit télécharger automatiquement lors du premier usage")
        return False
    
    return True

def check_fetching_enabled():
    """Vérifie si le fetching est activé"""
    print("\n⚙️ VÉRIFICATION CONFIGURATION FETCHING")
    print("=" * 50)
    
    # Charger la configuration depuis video_processor
    try:
        sys.path.insert(0, '.')
        from video_processor import Config
        
        print(f"✅ BROLL_FETCH_ENABLE: {Config.BROLL_FETCH_ENABLE}")
        print(f"✅ BROLL_FETCH_PROVIDER: {Config.BROLL_FETCH_PROVIDER}")
        print(f"✅ BROLL_FETCH_MAX_PER_KEYWORD: {Config.BROLL_FETCH_MAX_PER_KEYWORD}")
        print(f"✅ BROLL_FETCH_ALLOW_VIDEOS: {Config.BROLL_FETCH_ALLOW_VIDEOS}")
        print(f"✅ BROLL_FETCH_ALLOW_IMAGES: {Config.BROLL_FETCH_ALLOW_IMAGES}")
        print(f"✅ PEXELS_API_KEY: {'✅ Configurée' if Config.PEXELS_API_KEY else '❌ Manquante'}")
        print(f"✅ PIXABAY_API_KEY: {'✅ Configurée' if Config.PIXABAY_API_KEY else '❌ Manquante'}")
        
        if not Config.BROLL_FETCH_ENABLE:
            print("\n🚨 PROBLÈME: BROLL_FETCH_ENABLE = False")
            print("🔧 SOLUTION: Activez le fetching avec BROLL_FETCH_ENABLE=True")
            return False
            
        if not (Config.PEXELS_API_KEY or Config.PIXABAY_API_KEY):
            print("\n🚨 PROBLÈME: Aucune clé API configurée")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur chargement configuration: {e}")
        return False

def test_manual_fetch():
    """Test un téléchargement B-roll manuel"""
    print("\n🧪 TEST TÉLÉCHARGEMENT MANUEL")
    print("=" * 50)
    
    pexels_key = os.getenv('PEXELS_API_KEY')
    if not pexels_key:
        print("❌ Pas de clé Pexels pour le test")
        return False
    
    try:
        print("🔍 Test téléchargement 'therapy' depuis Pexels...")
        
        # Appel API Pexels
        headers = {"Authorization": pexels_key}
        response = requests.get(
            "https://api.pexels.com/videos/search?query=therapy&per_page=1",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Erreur API: {response.status_code}")
            return False
        
        data = response.json()
        videos = data.get('videos', [])
        
        if not videos:
            print("❌ Aucune vidéo trouvée")
            return False
        
        video = videos[0]
        video_files = video.get('video_files', [])
        
        if not video_files:
            print("❌ Aucun fichier vidéo disponible")
            return False
        
        # Choisir la meilleure qualité
        best_file = max(video_files, key=lambda x: x.get('width', 0) * x.get('height', 0))
        download_url = best_file['link']
        
        print(f"📥 Téléchargement: {download_url[:50]}...")
        
        # Créer le dossier de test
        test_dir = Path("test_broll_download")
        test_dir.mkdir(exist_ok=True)
        
        # Télécharger
        download_response = requests.get(download_url, stream=True, timeout=30)
        download_response.raise_for_status()
        
        test_file = test_dir / "test_therapy.mp4"
        with open(test_file, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if test_file.exists() and test_file.stat().st_size > 1000:
            print(f"✅ Téléchargement réussi: {test_file.stat().st_size} bytes")
            print(f"✅ Fichier: {test_file}")
            return True
        else:
            print("❌ Téléchargement échoué")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test téléchargement: {e}")
        return False

def generate_solution_steps():
    """Génère les étapes de solution"""
    print("\n🔧 ÉTAPES DE RÉSOLUTION")
    print("=" * 60)
    
    print("1. 🔑 CONFIGURER UNE CLÉ API:")
    print("   - Allez sur https://www.pexels.com/api/")
    print("   - Créez un compte gratuit")
    print("   - Obtenez votre clé API")
    print("   - Ajoutez: set PEXELS_API_KEY=votre_cle_ici")
    print()
    
    print("2. ⚙️ VÉRIFIER LA CONFIGURATION:")
    print("   - BROLL_FETCH_ENABLE=True")
    print("   - BROLL_FETCH_PROVIDER=pexels")
    print("   - BROLL_FETCH_ALLOW_VIDEOS=True")
    print()
    
    print("3. 🧪 TESTER LE SYSTÈME:")
    print("   - Lancez une vidéo de test")
    print("   - Vérifiez les logs pour 'Fetch B-roll'")
    print("   - Contrôlez le dossier AI-B-roll/broll_library/fetched/")
    print()
    
    print("4. 🗑️ NETTOYAGE AUTOMATIQUE:")
    print("   - Les B-rolls sont supprimés après utilisation (BROLL_DELETE_AFTER_USE=True)")
    print("   - Cache intelligent garde les récents (30 jours)")
    print("   - Nettoyage périodique via interface GUI")

def main():
    """Diagnostic principal"""
    print("🔍 DIAGNOSTIC SYSTÈME B-ROLL FETCHING")
    print("=" * 60)
    print(f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        ("Clés API", check_api_keys),
        ("Connexion APIs", test_api_connection),
        ("Dossiers Cache", check_cache_directories),
        ("Configuration Fetching", check_fetching_enabled),
        ("Test Téléchargement", test_manual_fetch)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DIAGNOSTIC")
    print("=" * 60)
    
    total_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "✅ OK" if ok else "❌ PROBLÈME"
        print(f"{status:<12} {name}")
    
    print(f"\n🎯 Score: {total_ok}/{len(results)} checks réussis")
    
    if total_ok < len(results):
        generate_solution_steps()
    else:
        print("\n🎉 SYSTÈME B-ROLL FONCTIONNEL !")
        print("✅ Tous les checks sont passés")
        print("🚀 Les B-rolls devraient se télécharger automatiquement")

if __name__ == "__main__":
    main() 