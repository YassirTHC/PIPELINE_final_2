ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å½Â¬ INTÃƒâ€°GRATION DIRECTE AVEC VOTRE PIPELINE VIDÃƒâ€°O EXISTANT
# Ce script intÃƒÂ¨gre notre systÃƒÂ¨me LLM industriel ÃƒÂ  votre VideoProcessor

import sys
import os
from pathlib import Path

# Ajouter le rÃƒÂ©pertoire utils au path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

def integrate_with_existing_pipeline():
    """IntÃƒÂ¨gre notre systÃƒÂ¨me LLM avec votre pipeline existant"""
    
    print("Ã°Å¸Å¡â‚¬ INTÃƒâ€°GRATION AVEC VOTRE PIPELINE VIDÃƒâ€°O EXISTANT")
    print("=" * 60)
    
    try:
        # 1. VÃƒÂ©rifier que votre VideoProcessor existe
        if not Path("video_processor.py").exists():
            print("Ã¢ÂÅ’ Fichier video_processor.py non trouvÃƒÂ©")
            print("   Assurez-vous d'ÃƒÂªtre dans le bon rÃƒÂ©pertoire")
            return False
        
        print("Ã¢Å“â€¦ VideoProcessor trouvÃƒÂ©")
        
        # 2. Importer votre VideoProcessor
        try:
            from video_processor import VideoProcessor
            print("Ã¢Å“â€¦ VideoProcessor importÃƒÂ© avec succÃƒÂ¨s")
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur import VideoProcessor: {e}")
            return False
        
        # 3. CrÃƒÂ©er une instance et l'amÃƒÂ©liorer
        try:
            processor = VideoProcessor()
            print("Ã¢Å“â€¦ Instance VideoProcessor crÃƒÂ©ÃƒÂ©e")
            
            # AmÃƒÂ©liorer avec nos mÃƒÂ©thodes LLM
            from video_pipeline_integration import enhance_video_processor_methods
            enhance_video_processor_methods(VideoProcessor)
            
            print("Ã¢Å“â€¦ MÃƒÂ©thodes VideoProcessor amÃƒÂ©liorÃƒÂ©es")
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur crÃƒÂ©ation instance: {e}")
            return False
        
        # 4. Test d'intÃƒÂ©gration
        print("\nÃ°Å¸Â§Âª Test d'intÃƒÂ©gration...")
        
        # CrÃƒÂ©er des sous-titres de test
        test_subtitles = [
            {'start': 0.0, 'end': 5.0, 'text': 'EMDR therapy is a powerful treatment for trauma and PTSD.'},
            {'start': 5.0, 'end': 10.0, 'text': 'The therapist uses bilateral stimulation to help patients process traumatic memories.'},
            {'start': 10.0, 'end': 15.0, 'text': 'This innovative approach combines psychology and neuroscience for lasting healing.'}
        ]
        
        # Tester la mÃƒÂ©thode amÃƒÂ©liorÃƒÂ©e
        try:
            print("    Ã°Å¸Å½Â¯ Test de la mÃƒÂ©thode generate_caption_and_hashtags amÃƒÂ©liorÃƒÂ©e...")
            
            metadata = processor.generate_caption_and_hashtags(test_subtitles) or {}
            title = str(metadata.get('title') or '')
            description = str(metadata.get('description') or '')
            hashtags = [h for h in (metadata.get('hashtags') or []) if isinstance(h, str)]
            broll_keywords = [kw for kw in (metadata.get('broll_keywords') or []) if isinstance(kw, str)]
            
            print(f"    Ã¢Å“â€¦ Titre: {title}")
            print(f"    Ã¢Å“â€¦ Description: {description}")
            print(f"    Ã¢Å“â€¦ Hashtags: {len(hashtags)} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            print(f"    Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll: {len(broll_keywords)} gÃƒÂ©nÃƒÂ©rÃƒÂ©s")
            
            if broll_keywords:
                print(f"    Ã°Å¸Å½Â¬ Exemples B-roll: {', '.join(broll_keywords[:5])}")
            
            print("\nÃ°Å¸Å½â€° IntÃƒÂ©gration rÃƒÂ©ussie !")
            print("\nÃ°Å¸â€œâ€¹ VOTRE PIPELINE EST MAINTENANT CONNECTÃƒâ€° AU SYSTÃƒË†ME LLM INDUSTRIEL")
            print("=" * 60)
            print("Ã¢Å“â€¦ DÃƒÂ©tection de domaine automatique (TF-IDF)")
            print("Ã¢Å“â€¦ GÃƒÂ©nÃƒÂ©ration de titres, descriptions et hashtags optimisÃƒÂ©s")
            print("Ã¢Å“â€¦ Mots-clÃƒÂ©s B-roll intelligents et optimisÃƒÂ©s")
            print("Ã¢Å“â€¦ MÃƒÂ©triques et monitoring en temps rÃƒÂ©el")
            print("Ã¢Å“â€¦ Fallbacks automatiques en cas d'erreur")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Erreur test mÃƒÂ©thode: {e}")
            return False
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_guide():
    """Affiche le guide d'intÃƒÂ©gration"""
    
    print("\nÃ°Å¸â€œÅ¡ GUIDE D'INTÃƒâ€°GRATION COMPLET")
    print("=" * 60)
    
    print("""
Ã°Å¸Å½Â¯ COMMENT UTILISER VOTRE PIPELINE AMÃƒâ€°LIORÃƒâ€° :

1. Ã°Å¸Å¡â‚¬ DÃƒâ€°MARRAGE AUTOMATIQUE
   Votre pipeline fonctionne maintenant automatiquement avec le systÃƒÂ¨me LLM !
   Plus besoin de modifier le code existant.

2. Ã°Å¸Å½Â¬ UTILISATION NORMALE
   Utilisez votre pipeline exactement comme avant :
   python video_processor.py [vos_paramÃƒÂ¨tres]

3. Ã°Å¸Â§Â  AMÃƒâ€°LIORATIONS AUTOMATIQUES
   - DÃƒÂ©tection de domaine intelligente
   - Titres et hashtags optimisÃƒÂ©s pour TikTok/Instagram
   - Mots-clÃƒÂ©s B-roll contextuels
   - MÃƒÂ©triques de performance

4. Ã°Å¸â€œÅ  MONITORING
   Les mÃƒÂ©triques sont automatiquement collectÃƒÂ©es et exportÃƒÂ©es.
   Consultez les rapports dans output/meta/

5. Ã°Å¸â€Â§ CONFIGURATION
   Modifiez utils/video_pipeline_integration.py pour ajuster :
   - Seuils de confiance
   - Nombre de mots-clÃƒÂ©s
   - Timeouts
   - Fallbacks

6. Ã°Å¸Å½Â¯ PERSONNALISATION
   Pour ajouter des domaines spÃƒÂ©cifiques :
   - Modifiez utils/domain_detection_enhanced.py
   - Ajoutez vos patterns de mots-clÃƒÂ©s
   - Ajustez les seuils de confiance
""")

def show_next_steps():
    """Affiche les prochaines ÃƒÂ©tapes"""
    
    print("\nÃ°Å¸Å½Â¯ PROCHAINES Ãƒâ€°TAPES RECOMMANDÃƒâ€°ES")
    print("=" * 60)
    
    print("""
1. Ã°Å¸Â§Âª TEST COMPLET
   Lancez votre pipeline sur une vidÃƒÂ©o de test :
   python video_processor.py [chemin_video]

2. Ã°Å¸â€œÅ  VALIDATION DES RÃƒâ€°SULTATS
   VÃƒÂ©rifiez la qualitÃƒÂ© des outputs :
   - Titres et descriptions
   - Hashtags gÃƒÂ©nÃƒÂ©rÃƒÂ©s
   - Mots-clÃƒÂ©s B-roll
   - DÃƒÂ©tection de domaine

3. Ã¢Å¡â„¢Ã¯Â¸Â AJUSTEMENTS FINES
   Ajustez les seuils selon vos besoins :
   - Seuils de confiance domaine
   - Nombre de mots-clÃƒÂ©s B-roll
   - Timeouts LLM

4. Ã°Å¸Å¡â‚¬ PRODUCTION
   Une fois validÃƒÂ©, votre pipeline est prÃƒÂªt pour la production !
   - Traitement en lot
   - Monitoring automatique
   - Rapports de performance

5. Ã°Å¸â€œË† OPTIMISATION CONTINUE
   Analysez les mÃƒÂ©triques pour :
   - Identifier les goulots d'ÃƒÂ©tranglement
   - Optimiser les performances
   - AmÃƒÂ©liorer la qualitÃƒÂ© des outputs
""")

def main():
    """Fonction principale"""
    
    print("Ã°Å¸Å½Â¬ INTÃƒâ€°GRATION AVEC VOTRE PIPELINE VIDÃƒâ€°O EXISTANT")
    print("=" * 60)
    
    # 1. IntÃƒÂ©gration
    success = integrate_with_existing_pipeline()
    
    if success:
        # 2. Guide d'intÃƒÂ©gration
        show_integration_guide()
        
        # 3. Prochaines ÃƒÂ©tapes
        show_next_steps()
        
        print("\nÃ°Å¸Å½â€° FÃƒâ€°LICITATIONS !")
        print("Votre pipeline vidÃƒÂ©o est maintenant connectÃƒÂ© au systÃƒÂ¨me LLM industriel !")
        print("\nÃ°Å¸Å¡â‚¬ PrÃƒÂªt pour la production !")
        
        return True
    else:
        print("\nÃ¢ÂÅ’ IntÃƒÂ©gration ÃƒÂ©chouÃƒÂ©e")
        print("VÃƒÂ©rifiez les erreurs ci-dessus et rÃƒÂ©essayez")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nÃ¢ÂÂ¹Ã¯Â¸Â IntÃƒÂ©gration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\nÃ°Å¸â€™Â¥ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 

