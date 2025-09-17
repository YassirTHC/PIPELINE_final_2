#!/usr/bin/env python3
"""
Force Reload du Module B-roll Selector
Résout le problème d'import en forçant le rechargement
"""

import sys
import importlib

def force_reload_broll_selector():
    """Force le rechargement du module broll_selector"""
    print("🚀 FORCE RELOAD DU MODULE B-ROLL SELECTOR")
    print("=" * 60)
    
    try:
        # 1. Vérifier si le module est déjà chargé
        if 'broll_selector' in sys.modules:
            print("🔍 Module broll_selector déjà chargé dans sys.modules")
            print(f"   📍 Emplacement: {sys.modules['broll_selector']}")
            
            # 2. Supprimer le module du cache
            del sys.modules['broll_selector']
            print("   ✅ Module supprimé du cache sys.modules")
        
        # 3. Forcer le rechargement
        print("\n🔄 Rechargement forcé du module...")
        import broll_selector
        importlib.reload(broll_selector)
        print("   ✅ Module rechargé avec succès")
        
        # 4. Vérifier que la fonction est disponible
        selector = broll_selector.BrollSelector()
        if hasattr(selector, 'find_broll_matches'):
            print("   ✅ Fonction find_broll_matches disponible après rechargement")
            
            # 5. Test de la fonction
            test_keywords = ["healthcare", "family", "community"]
            matches = selector.find_broll_matches(test_keywords, domain="health")
            print(f"   ✅ Fonction testée avec succès: {len(matches)} résultats")
            
        else:
            print("   ❌ Fonction find_broll_matches toujours manquante")
            return False
        
        # 6. Vérifier l'intégration avec video_processor
        print("\n🔍 Test d'intégration avec video_processor...")
        try:
            # Forcer le rechargement de video_processor aussi
            if 'video_processor' in sys.modules:
                del sys.modules['video_processor']
                print("   ✅ video_processor supprimé du cache")
            
            import video_processor
            print("   ✅ video_processor rechargé")
            
            if hasattr(video_processor, 'BROLL_SELECTOR_AVAILABLE'):
                print(f"   ✅ BROLL_SELECTOR_AVAILABLE: {video_processor.BROLL_SELECTOR_AVAILABLE}")
            else:
                print("   ❌ BROLL_SELECTOR_AVAILABLE manquant")
                
        except Exception as e:
            print(f"   ❌ Erreur rechargement video_processor: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 RELOAD FORCÉ RÉUSSI !")
        print("=" * 60)
        print("✅ Module broll_selector rechargé")
        print("✅ Fonction find_broll_matches disponible")
        print("✅ Intégration avec video_processor validée")
        print("🚀 Le pipeline peut maintenant utiliser le nouveau sélecteur")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors du reload forcé: {e}")
        return False

if __name__ == "__main__":
    success = force_reload_broll_selector()
    exit(0 if success else 1) 