# 🎬 INTÉGRATION DIRECTE AVEC VOTRE PIPELINE VIDÉO EXISTANT
# Ce script intègre notre système LLM industriel à votre VideoProcessor

import sys
import os
from pathlib import Path

# Ajouter le répertoire utils au path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

def integrate_with_existing_pipeline():
    """Intègre notre système LLM avec votre pipeline existant"""
    
    print("🚀 INTÉGRATION AVEC VOTRE PIPELINE VIDÉO EXISTANT")
    print("=" * 60)
    
    try:
        # 1. Vérifier que votre VideoProcessor existe
        if not Path("video_processor.py").exists():
            print("❌ Fichier video_processor.py non trouvé")
            print("   Assurez-vous d'être dans le bon répertoire")
            return False
        
        print("✅ VideoProcessor trouvé")
        
        # 2. Importer votre VideoProcessor
        try:
            from video_processor import VideoProcessor
            print("✅ VideoProcessor importé avec succès")
        except Exception as e:
            print(f"❌ Erreur import VideoProcessor: {e}")
            return False
        
        # 3. Créer une instance et l'améliorer
        try:
            processor = VideoProcessor()
            print("✅ Instance VideoProcessor créée")
            
            # Améliorer avec nos méthodes LLM
            from video_pipeline_integration import enhance_video_processor_methods
            enhance_video_processor_methods(VideoProcessor)
            
            print("✅ Méthodes VideoProcessor améliorées")
            
        except Exception as e:
            print(f"❌ Erreur création instance: {e}")
            return False
        
        # 4. Test d'intégration
        print("\n🧪 Test d'intégration...")
        
        # Créer des sous-titres de test
        test_subtitles = [
            {'start': 0.0, 'end': 5.0, 'text': 'EMDR therapy is a powerful treatment for trauma and PTSD.'},
            {'start': 5.0, 'end': 10.0, 'text': 'The therapist uses bilateral stimulation to help patients process traumatic memories.'},
            {'start': 10.0, 'end': 15.0, 'text': 'This innovative approach combines psychology and neuroscience for lasting healing.'}
        ]
        
        # Tester la méthode améliorée
        try:
            print("    🎯 Test de la méthode generate_caption_and_hashtags améliorée...")
            
            title, description, hashtags, broll_keywords = processor.generate_caption_and_hashtags(test_subtitles)
            
            print(f"    ✅ Titre: {title}")
            print(f"    ✅ Description: {description}")
            print(f"    ✅ Hashtags: {len(hashtags)} générés")
            print(f"    ✅ Mots-clés B-roll: {len(broll_keywords)} générés")
            
            if broll_keywords:
                print(f"    🎬 Exemples B-roll: {', '.join(broll_keywords[:5])}")
            
            print("\n🎉 Intégration réussie !")
            print("\n📋 VOTRE PIPELINE EST MAINTENANT CONNECTÉ AU SYSTÈME LLM INDUSTRIEL")
            print("=" * 60)
            print("✅ Détection de domaine automatique (TF-IDF)")
            print("✅ Génération de titres, descriptions et hashtags optimisés")
            print("✅ Mots-clés B-roll intelligents et optimisés")
            print("✅ Métriques et monitoring en temps réel")
            print("✅ Fallbacks automatiques en cas d'erreur")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"    ❌ Erreur test méthode: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_guide():
    """Affiche le guide d'intégration"""
    
    print("\n📚 GUIDE D'INTÉGRATION COMPLET")
    print("=" * 60)
    
    print("""
🎯 COMMENT UTILISER VOTRE PIPELINE AMÉLIORÉ :

1. 🚀 DÉMARRAGE AUTOMATIQUE
   Votre pipeline fonctionne maintenant automatiquement avec le système LLM !
   Plus besoin de modifier le code existant.

2. 🎬 UTILISATION NORMALE
   Utilisez votre pipeline exactement comme avant :
   python video_processor.py [vos_paramètres]

3. 🧠 AMÉLIORATIONS AUTOMATIQUES
   - Détection de domaine intelligente
   - Titres et hashtags optimisés pour TikTok/Instagram
   - Mots-clés B-roll contextuels
   - Métriques de performance

4. 📊 MONITORING
   Les métriques sont automatiquement collectées et exportées.
   Consultez les rapports dans output/meta/

5. 🔧 CONFIGURATION
   Modifiez utils/video_pipeline_integration.py pour ajuster :
   - Seuils de confiance
   - Nombre de mots-clés
   - Timeouts
   - Fallbacks

6. 🎯 PERSONNALISATION
   Pour ajouter des domaines spécifiques :
   - Modifiez utils/domain_detection_enhanced.py
   - Ajoutez vos patterns de mots-clés
   - Ajustez les seuils de confiance
""")

def show_next_steps():
    """Affiche les prochaines étapes"""
    
    print("\n🎯 PROCHAINES ÉTAPES RECOMMANDÉES")
    print("=" * 60)
    
    print("""
1. 🧪 TEST COMPLET
   Lancez votre pipeline sur une vidéo de test :
   python video_processor.py [chemin_video]

2. 📊 VALIDATION DES RÉSULTATS
   Vérifiez la qualité des outputs :
   - Titres et descriptions
   - Hashtags générés
   - Mots-clés B-roll
   - Détection de domaine

3. ⚙️ AJUSTEMENTS FINES
   Ajustez les seuils selon vos besoins :
   - Seuils de confiance domaine
   - Nombre de mots-clés B-roll
   - Timeouts LLM

4. 🚀 PRODUCTION
   Une fois validé, votre pipeline est prêt pour la production !
   - Traitement en lot
   - Monitoring automatique
   - Rapports de performance

5. 📈 OPTIMISATION CONTINUE
   Analysez les métriques pour :
   - Identifier les goulots d'étranglement
   - Optimiser les performances
   - Améliorer la qualité des outputs
""")

def main():
    """Fonction principale"""
    
    print("🎬 INTÉGRATION AVEC VOTRE PIPELINE VIDÉO EXISTANT")
    print("=" * 60)
    
    # 1. Intégration
    success = integrate_with_existing_pipeline()
    
    if success:
        # 2. Guide d'intégration
        show_integration_guide()
        
        # 3. Prochaines étapes
        show_next_steps()
        
        print("\n🎉 FÉLICITATIONS !")
        print("Votre pipeline vidéo est maintenant connecté au système LLM industriel !")
        print("\n🚀 Prêt pour la production !")
        
        return True
    else:
        print("\n❌ Intégration échouée")
        print("Vérifiez les erreurs ci-dessus et réessayez")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 