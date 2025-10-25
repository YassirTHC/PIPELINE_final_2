#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ðŸ” DIAGNOSTIC COMPLET SYSTÃˆME B-ROLL
Identifie pourquoi les B-rolls ne sont pas tÃ©lÃ©chargÃ©s et propose des solutions
"""

import os
import sys
from pathlib import Path
import requests
import time

def check_api_keys():
    """VÃ©rifie les clÃ©s API configurÃ©es"""
    print("ðŸ”‘ VÃ‰RIFICATION DES CLÃ‰S API")
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
            print(f"âœ… {key_name}: {key_value[:8]}****** (configurÃ©e)")
            configured += 1
        else:
            print(f"âŒ {key_name}: NON CONFIGURÃ‰E")
    
    print(f"\nðŸ“Š RÃ©sultat: {configured}/4 clÃ©s API configurÃ©es")
    
    if configured == 0:
        print("\nðŸš¨ PROBLÃˆME MAJEUR: Aucune clÃ© API configurÃ©e!")
        print("ðŸ”§ SOLUTION: Configurez au moins une clÃ© API pour activer le tÃ©lÃ©chargement")
        print("\nðŸ“‹ INSTRUCTIONS:")
        print("1. CrÃ©ez un compte gratuit sur Pexels.com")
        print("2. Obtenez votre clÃ© API Pexels")
        print("3. DÃ©finissez la variable d'environnement: set PEXELS_API_KEY=votre_cle")
        print("4. Ou ajoutez dans .env: PEXELS_API_KEY=votre_cle")
        return False
    
    return True

def test_api_connection():
    """Test la connexion aux APIs configurÃ©es"""
    print("\nðŸŒ TEST CONNEXION APIs")
    print("=" * 50)
    
    # Test Pexels
    pexels_key = os.getenv('PEXELS_API_KEY')
    if pexels_key:
        print("ðŸ” Test Pexels API...")
        try:
            headers = {"Authorization": pexels_key}
            response = requests.get(
                "https://api.pexels.com/videos/search?query=nature&per_page=1",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                print("âœ… Pexels API: FONCTIONNELLE")
                return True
            else:
                print(f"âŒ Pexels API: Erreur {response.status_code}")
                print(f"   RÃ©ponse: {response.text[:100]}")
        except Exception as e:
            print(f"âŒ Pexels API: Erreur connexion - {e}")
    
    # Test Pixabay
    pixabay_key = os.getenv('PIXABAY_API_KEY')
    if pixabay_key:
        print("ðŸ” Test Pixabay API...")
        try:
            response = requests.get(
                f"https://pixabay.com/api/videos/?key={pixabay_key}&q=nature&per_page=3",
                timeout=10
            )
            if response.status_code == 200:
                print("âœ… Pixabay API: FONCTIONNELLE")
                return True
            else:
                print(f"âŒ Pixabay API: Erreur {response.status_code}")
        except Exception as e:
            print(f"âŒ Pixabay API: Erreur connexion - {e}")
    
    print("âŒ Aucune API fonctionnelle trouvÃ©e")
    return False

def check_cache_directories():
    """VÃ©rifie les dossiers de cache B-roll"""
    print("\nðŸ“ VÃ‰RIFICATION DOSSIERS CACHE")
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
            print(f"âœ… {cache_dir}: {len(files)} fichiers")
            total_files += len(files)
        else:
            print(f"âŒ {cache_dir}: N'EXISTE PAS")
    
    print(f"\nðŸ“Š Total: {total_files} fichiers B-roll en cache")
    
    if total_files == 0:
        print("\nðŸš¨ PROBLÃˆME: Aucun B-roll en cache!")
        print("ðŸ”§ SOLUTION: Le systÃ¨me doit tÃ©lÃ©charger automatiquement lors du premier usage")
        return False
    
    return True

def check_fetching_enabled():
    """VÃ©rifie si le fetching est activÃ©"""
    print("\nâš™ï¸ VÃ‰RIFICATION CONFIGURATION FETCHING")
    print("=" * 50)
    
    # Charger la configuration depuis video_processor
    try:
        sys.path.insert(0, '.')
        from video_processor import Config
        
        print(f"âœ… BROLL_FETCH_ENABLE: {Config.BROLL_FETCH_ENABLE}")
        print(f"âœ… BROLL_FETCH_PROVIDER: {Config.BROLL_FETCH_PROVIDER}")
        print(f"âœ… BROLL_FETCH_MAX_PER_KEYWORD: {Config.BROLL_FETCH_MAX_PER_KEYWORD}")
        print(f"âœ… BROLL_FETCH_ALLOW_VIDEOS: {Config.BROLL_FETCH_ALLOW_VIDEOS}")
        print(f"âœ… BROLL_FETCH_ALLOW_IMAGES: {Config.BROLL_FETCH_ALLOW_IMAGES}")
        print(f"âœ… PEXELS_API_KEY: {'âœ… ConfigurÃ©e' if Config.PEXELS_API_KEY else 'âŒ Manquante'}")
        print(f"âœ… PIXABAY_API_KEY: {'âœ… ConfigurÃ©e' if Config.PIXABAY_API_KEY else 'âŒ Manquante'}")
        
        if not Config.BROLL_FETCH_ENABLE:
            print("\nðŸš¨ PROBLÃˆME: BROLL_FETCH_ENABLE = False")
            print("ðŸ”§ SOLUTION: Activez le fetching avec BROLL_FETCH_ENABLE=True")
            return False
            
        if not (Config.PEXELS_API_KEY or Config.PIXABAY_API_KEY):
            print("\nðŸš¨ PROBLÃˆME: Aucune clÃ© API configurÃ©e")
            return False
            
        return True
        
    except Exception as e:
        print(f"âŒ Erreur chargement configuration: {e}")
        return False

def test_manual_fetch():
    """Test un tÃ©lÃ©chargement B-roll manuel"""
    print("\nðŸ§ª TEST TÃ‰LÃ‰CHARGEMENT MANUEL")
    print("=" * 50)
    
    pexels_key = os.getenv('PEXELS_API_KEY')
    if not pexels_key:
        print("âŒ Pas de clÃ© Pexels pour le test")
        return False
    
    try:
        print("ðŸ” Test tÃ©lÃ©chargement 'therapy' depuis Pexels...")
        
        # Appel API Pexels
        headers = {"Authorization": pexels_key}
        response = requests.get(
            "https://api.pexels.com/videos/search?query=therapy&per_page=1",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"âŒ Erreur API: {response.status_code}")
            return False
        
        data = response.json()
        videos = data.get('videos', [])
        
        if not videos:
            print("âŒ Aucune vidÃ©o trouvÃ©e")
            return False
        
        video = videos[0]
        video_files = video.get('video_files', [])
        
        if not video_files:
            print("âŒ Aucun fichier vidÃ©o disponible")
            return False
        
        # Choisir la meilleure qualitÃ©
        best_file = max(video_files, key=lambda x: x.get('width', 0) * x.get('height', 0))
        download_url = best_file['link']
        
        print(f"ðŸ“¥ TÃ©lÃ©chargement: {download_url[:50]}...")
        
        # CrÃ©er le dossier de test
        test_dir = Path("test_broll_download")
        test_dir.mkdir(exist_ok=True)
        
        # TÃ©lÃ©charger
        download_response = requests.get(download_url, stream=True, timeout=30)
        download_response.raise_for_status()
        
        test_file = test_dir / "test_therapy.mp4"
        with open(test_file, 'wb') as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if test_file.exists() and test_file.stat().st_size > 1000:
            print(f"âœ… TÃ©lÃ©chargement rÃ©ussi: {test_file.stat().st_size} bytes")
            print(f"âœ… Fichier: {test_file}")
            return True
        else:
            print("âŒ TÃ©lÃ©chargement Ã©chouÃ©")
            return False
            
    except Exception as e:
        print(f"âŒ Erreur test tÃ©lÃ©chargement: {e}")
        return False

def generate_solution_steps():
    """GÃ©nÃ¨re les Ã©tapes de solution"""
    print("\nðŸ”§ Ã‰TAPES DE RÃ‰SOLUTION")
    print("=" * 60)
    
    print("1. ðŸ”‘ CONFIGURER UNE CLÃ‰ API:")
    print("   - Allez sur https://www.pexels.com/api/")
    print("   - CrÃ©ez un compte gratuit")
    print("   - Obtenez votre clÃ© API")
    print("   - Ajoutez: set PEXELS_API_KEY=votre_cle_ici")
    print()
    
    print("2. âš™ï¸ VÃ‰RIFIER LA CONFIGURATION:")
    print("   - BROLL_FETCH_ENABLE=True")
    print("   - BROLL_FETCH_PROVIDER=pexels")
    print("   - BROLL_FETCH_ALLOW_VIDEOS=True")
    print()
    
    print("3. ðŸ§ª TESTER LE SYSTÃˆME:")
    print("   - Lancez une vidÃ©o de test")
    print("   - VÃ©rifiez les logs pour 'Fetch B-roll'")
    print("   - ContrÃ´lez le dossier AI-B-roll/broll_library/fetched/")
    print()
    
    print("4. ðŸ—‘ï¸ NETTOYAGE AUTOMATIQUE:")
    print("   - Les B-rolls sont supprimÃ©s aprÃ¨s utilisation (BROLL_DELETE_AFTER_USE=True)")
    print("   - Cache intelligent garde les rÃ©cents (30 jours)")
    print("   - Nettoyage pÃ©riodique via interface GUI")

def main():
    """Diagnostic principal"""
    print("ðŸ” DIAGNOSTIC SYSTÃˆME B-ROLL FETCHING")
    print("=" * 60)
    print(f"ðŸ•’ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        ("ClÃ©s API", check_api_keys),
        ("Connexion APIs", test_api_connection),
        ("Dossiers Cache", check_cache_directories),
        ("Configuration Fetching", check_fetching_enabled),
        ("Test TÃ©lÃ©chargement", test_manual_fetch)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"âŒ Erreur {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("ðŸ“Š RÃ‰SUMÃ‰ DIAGNOSTIC")
    print("=" * 60)
    
    total_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "âœ… OK" if ok else "âŒ PROBLÃˆME"
        print(f"{status:<12} {name}")
    
    print(f"\nðŸŽ¯ Score: {total_ok}/{len(results)} checks rÃ©ussis")
    
    if total_ok < len(results):
        generate_solution_steps()
    else:
        print("\nðŸŽ‰ SYSTÃˆME B-ROLL FONCTIONNEL !")
        print("âœ… Tous les checks sont passÃ©s")
        print("ðŸš€ Les B-rolls devraient se tÃ©lÃ©charger automatiquement")

if __name__ == "__main__":
    main() 
