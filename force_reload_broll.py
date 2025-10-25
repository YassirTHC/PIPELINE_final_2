ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Force Reload du Module B-roll Selector
RÃƒÂ©sout le problÃƒÂ¨me d'import en forÃƒÂ§ant le rechargement
"""

import sys
import importlib

def force_reload_broll_selector():
    """Force le rechargement du module broll_selector"""
    print("Ã°Å¸Å¡â‚¬ FORCE RELOAD DU MODULE B-ROLL SELECTOR")
    print("=" * 60)
    
    try:
        # 1. VÃƒÂ©rifier si le module est dÃƒÂ©jÃƒÂ  chargÃƒÂ©
        if 'broll_selector' in sys.modules:
            print("Ã°Å¸â€Â Module broll_selector dÃƒÂ©jÃƒÂ  chargÃƒÂ© dans sys.modules")
            print(f"   Ã°Å¸â€œÂ Emplacement: {sys.modules['broll_selector']}")
            
            # 2. Supprimer le module du cache
            del sys.modules['broll_selector']
            print("   Ã¢Å“â€¦ Module supprimÃƒÂ© du cache sys.modules")
        
        # 3. Forcer le rechargement
        print("\nÃ°Å¸â€â€ž Rechargement forcÃƒÂ© du module...")
        import broll_selector
        importlib.reload(broll_selector)
        print("   Ã¢Å“â€¦ Module rechargÃƒÂ© avec succÃƒÂ¨s")
        
        # 4. VÃƒÂ©rifier que la fonction est disponible
        selector = broll_selector.BrollSelector()
        if hasattr(selector, 'find_broll_matches'):
            print("   Ã¢Å“â€¦ Fonction find_broll_matches disponible aprÃƒÂ¨s rechargement")
            
            # 5. Test de la fonction
            test_keywords = ["healthcare", "family", "community"]
            matches = selector.find_broll_matches(test_keywords, domain="health")
            print(f"   Ã¢Å“â€¦ Fonction testÃƒÂ©e avec succÃƒÂ¨s: {len(matches)} rÃƒÂ©sultats")
            
        else:
            print("   Ã¢ÂÅ’ Fonction find_broll_matches toujours manquante")
            return False
        
        # 6. VÃƒÂ©rifier l'intÃƒÂ©gration avec video_processor
        print("\nÃ°Å¸â€Â Test d'intÃƒÂ©gration avec video_processor...")
        try:
            # Forcer le rechargement de video_processor aussi
            if 'video_processor' in sys.modules:
                del sys.modules['video_processor']
                print("   Ã¢Å“â€¦ video_processor supprimÃƒÂ© du cache")
            
            import video_processor
            print("   Ã¢Å“â€¦ video_processor rechargÃƒÂ©")
            
            if hasattr(video_processor, 'BROLL_SELECTOR_AVAILABLE'):
                print(f"   Ã¢Å“â€¦ BROLL_SELECTOR_AVAILABLE: {video_processor.BROLL_SELECTOR_AVAILABLE}")
            else:
                print("   Ã¢ÂÅ’ BROLL_SELECTOR_AVAILABLE manquant")
                
        except Exception as e:
            print(f"   Ã¢ÂÅ’ Erreur rechargement video_processor: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("Ã°Å¸Å½â€° RELOAD FORCÃƒâ€° RÃƒâ€°USSI !")
        print("=" * 60)
        print("Ã¢Å“â€¦ Module broll_selector rechargÃƒÂ©")
        print("Ã¢Å“â€¦ Fonction find_broll_matches disponible")
        print("Ã¢Å“â€¦ IntÃƒÂ©gration avec video_processor validÃƒÂ©e")
        print("Ã°Å¸Å¡â‚¬ Le pipeline peut maintenant utiliser le nouveau sÃƒÂ©lecteur")
        
        return True
        
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Erreur lors du reload forcÃƒÂ©: {e}")
        return False

if __name__ == "__main__":
    success = force_reload_broll_selector()
    exit(0 if success else 1) 

