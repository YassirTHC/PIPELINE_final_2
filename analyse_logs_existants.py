ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸â€Â ANALYSE DES LOGS EXISTANTS - COMPRÃƒâ€°HENSION DU FONCTIONNEMENT
Analyse des logs existants pour comprendre comment le pipeline fonctionne
"""

import json
import os
from pathlib import Path

def analyser_logs_existants():
    """Analyser les logs existants pour comprendre le fonctionnement"""
    print("Ã°Å¸â€Â ANALYSE DES LOGS EXISTANTS - COMPRÃƒâ€°HENSION DU FONCTIONNEMENT")
    print("=" * 70)
    
    try:
        # Analyser les mÃƒÂ©tadonnÃƒÂ©es existantes
        print("\nÃ°Å¸â€œÅ  ANALYSE DES MÃƒâ€°TADONNÃƒâ€°ES EXISTANTES:")
        print("=" * 50)
        
        # MÃƒÂ©tadonnÃƒÂ©es intelligentes
        meta_path = Path("output/meta/reframed_intelligent_broll_metadata.json")
        if meta_path.exists():
            print(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es intelligentes trouvÃƒÂ©es: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            intelligent_analysis = meta_data.get('intelligent_analysis', {})
            print(f"   Ã°Å¸Å½Â¯ Contexte dÃƒÂ©tectÃƒÂ©: {intelligent_analysis.get('main_theme', 'N/A')}")
            print(f"   Ã°Å¸Â§Â¬ Sujets: {', '.join(intelligent_analysis.get('key_topics', [])[:5])}")
            print(f"   Ã°Å¸ËœÅ  Sentiment: {intelligent_analysis.get('sentiment', 'N/A')}")
            print(f"   Ã°Å¸â€œÅ  ComplexitÃƒÂ©: {intelligent_analysis.get('complexity', 'N/A')}")
            
            keywords = intelligent_analysis.get('keywords', [])
            print(f"   Ã°Å¸â€â€˜ Mots-clÃƒÂ©s: {len(keywords)} termes")
            if keywords:
                print(f"   Ã°Å¸â€œÂ DÃƒÂ©tail: {', '.join(keywords[:10])}{'...' if len(keywords) > 10 else ''}")
                
                # Analyse de la qualitÃƒÂ©
                generic_words = ['this', 'that', 'the', 'and', 'or', 'but', 'for', 'with', 'your', 'my', 'his', 'her', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall']
                generic_count = sum(1 for kw in keywords if kw.lower() in generic_words)
                print(f"   Ã¢Å¡Â Ã¯Â¸Â Mots gÃƒÂ©nÃƒÂ©riques: {generic_count}/{len(keywords)} ({generic_count/len(keywords)*100:.1f}%)")
                
                # VÃƒÂ©rifier la structure
                has_hierarchical = any('synonyms' in str(kw) for kw in keywords)
                print(f"   Ã°Å¸Ââ€”Ã¯Â¸Â Structure hiÃƒÂ©rarchique: {'Ã¢Å“â€¦ DÃƒÂ©tectÃƒÂ©e' if has_hierarchical else 'Ã¢ÂÅ’ NON dÃƒÂ©tectÃƒÂ©e'}")
                
        else:
            print(f"Ã¢ÂÅ’ MÃƒÂ©tadonnÃƒÂ©es intelligentes non trouvÃƒÂ©es: {meta_path}")
        
        # Rapport de sÃƒÂ©lection B-roll
        selection_path = Path("output/meta/reframed_broll_selection_report.json")
        if selection_path.exists():
            print(f"\nÃ¢Å“â€¦ Rapport de sÃƒÂ©lection B-roll trouvÃƒÂ©: {selection_path}")
            with open(selection_path, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)
            
            print(f"   Ã°Å¸â€œÅ  SÃƒÂ©lection: {selection_data.get('num_selected', 0)}/{selection_data.get('num_candidates', 0)} B-rolls")
            print(f"   Ã°Å¸Å½Â¯ Top score: {selection_data.get('top_score', 0.0):.3f}")
            print(f"   Ã°Å¸â€œÂ Seuil appliquÃƒÂ©: {selection_data.get('min_score', 0.0):.3f}")
            print(f"   Ã°Å¸â€ Ëœ Fallback utilisÃƒÂ©: {selection_data.get('fallback_used', False)}")
            print(f"   Ã°Å¸ÂÂ·Ã¯Â¸Â Tier fallback: {selection_data.get('fallback_tier', 'N/A')}")
            
        else:
            print(f"Ã¢ÂÅ’ Rapport de sÃƒÂ©lection B-roll non trouvÃƒÂ©: {selection_path}")
        
        # Analyser les logs du pipeline
        print(f"\nÃ°Å¸â€œâ€¹ ANALYSE DES LOGS DU PIPELINE:")
        print("=" * 50)
        
        log_path = Path("output/pipeline.log.jsonl")
        if log_path.exists():
            print(f"Ã¢Å“â€¦ Logs du pipeline trouvÃƒÂ©s: {log_path}")
            
            # Lire les derniÃƒÂ¨res lignes du log
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"   Ã°Å¸â€œÅ  Nombre total de lignes: {len(lines)}")
            
            # Analyser les derniÃƒÂ¨res entrÃƒÂ©es
            if lines:
                print(f"\nÃ°Å¸â€Â DERNIÃƒË†RES ENTREES DU LOG:")
                for i, line in enumerate(lines[-5:], 1):
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp = log_entry.get('timestamp', 'N/A')
                        level = log_entry.get('level', 'INFO')
                        message = log_entry.get('message', 'N/A')
                        print(f"   {i}. [{timestamp}] {level}: {message[:100]}{'...' if len(message) > 100 else ''}")
                    except json.JSONDecodeError:
                        print(f"   {i}. [ERREUR JSON] {line.strip()[:100]}...")
        else:
            print(f"Ã¢ÂÅ’ Logs du pipeline non trouvÃƒÂ©s: {log_path}")
        
        # Analyser la structure des dossiers
        print(f"\nÃ°Å¸â€œÂ ANALYSE DE LA STRUCTURE DES DOSSIERS:")
        print("=" * 50)
        
        # Dossier clips
        clips_dir = Path("clips")
        if clips_dir.exists():
            clips_files = list(clips_dir.glob("*.mp4"))
            print(f"Ã¢Å“â€¦ Dossier clips: {len(clips_files)} vidÃƒÂ©os")
            for clip in clips_files[:3]:
                print(f"   Ã°Å¸â€œÂ¹ {clip.name} ({clip.stat().st_size / (1024*1024):.1f} MB)")
            if len(clips_files) > 3:
                print(f"   ... et {len(clips_files) - 3} autres")
        else:
            print("Ã¢ÂÅ’ Dossier clips non trouvÃƒÂ©")
        
        # Dossier output
        output_dir = Path("output")
        if output_dir.exists():
            output_files = list(output_dir.rglob("*.mp4"))
            print(f"Ã¢Å“â€¦ Dossier output: {len(output_files)} vidÃƒÂ©os traitÃƒÂ©es")
            for output in output_files[:3]:
                print(f"   Ã°Å¸Å½Â¬ {output.relative_to(output_dir)} ({output.stat().st_size / (1024*1024):.1f} MB)")
            if len(output_files) > 3:
                print(f"   ... et {len(output_files) - 3} autres")
        else:
            print("Ã¢ÂÅ’ Dossier output non trouvÃƒÂ©")
        
        # Dossier AI-B-roll
        ai_broll_dir = Path("AI-B-roll")
        if ai_broll_dir.exists():
            broll_library = ai_broll_dir / "broll_library"
            if broll_library.exists():
                clip_dirs = list(broll_library.glob("clip_*"))
                print(f"Ã¢Å“â€¦ Dossier AI-B-roll: {len(clip_dirs)} dossiers de clips")
                for clip_dir in clip_dirs[:3]:
                    print(f"   Ã°Å¸â€œÂ {clip_dir.name}")
                if len(clip_dirs) > 3:
                    print(f"   ... et {len(clip_dirs) - 3} autres")
            else:
                print("   Ã¢ÂÅ’ Sous-dossier broll_library non trouvÃƒÂ©")
        else:
            print("Ã¢ÂÅ’ Dossier AI-B-roll non trouvÃƒÂ©")
        
        # RÃƒÂ©sumÃƒÂ© de l'analyse
        print(f"\nÃ°Å¸â€œÅ  RÃƒâ€°SUMÃƒâ€° DE L'ANALYSE:")
        print("=" * 50)
        
        if meta_path.exists() and selection_path.exists():
            print("Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es et rapports disponibles")
            print("Ã¢Å“â€¦ Analyse du fonctionnement possible")
            
            # DÃƒÂ©terminer le statut
            if selection_data.get('fallback_used', False):
                print("Ã¢Å¡Â Ã¯Â¸Â  Fallback activÃƒÂ© - ProblÃƒÂ¨me de sÃƒÂ©lection B-roll")
            else:
                print("Ã¢Å“â€¦ SÃƒÂ©lection B-roll normale")
                
            if generic_count > len(keywords) * 0.3:  # Plus de 30% de mots gÃƒÂ©nÃƒÂ©riques
                print("Ã¢ÂÅ’ QualitÃƒÂ© des mots-clÃƒÂ©s mÃƒÂ©diocre (trop de mots gÃƒÂ©nÃƒÂ©riques)")
            else:
                print("Ã¢Å“â€¦ QualitÃƒÂ© des mots-clÃƒÂ©s acceptable")
                
        else:
            print("Ã¢ÂÅ’ DonnÃƒÂ©es insuffisantes pour l'analyse")
            
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyser_logs_existants() 

