#!/usr/bin/env python3
"""
🔍 ANALYSE DES LOGS EXISTANTS - COMPRÉHENSION DU FONCTIONNEMENT
Analyse des logs existants pour comprendre comment le pipeline fonctionne
"""

import json
import os
from pathlib import Path

def analyser_logs_existants():
    """Analyser les logs existants pour comprendre le fonctionnement"""
    print("🔍 ANALYSE DES LOGS EXISTANTS - COMPRÉHENSION DU FONCTIONNEMENT")
    print("=" * 70)
    
    try:
        # Analyser les métadonnées existantes
        print("\n📊 ANALYSE DES MÉTADONNÉES EXISTANTES:")
        print("=" * 50)
        
        # Métadonnées intelligentes
        meta_path = Path("output/meta/reframed_intelligent_broll_metadata.json")
        if meta_path.exists():
            print(f"✅ Métadonnées intelligentes trouvées: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            intelligent_analysis = meta_data.get('intelligent_analysis', {})
            print(f"   🎯 Contexte détecté: {intelligent_analysis.get('main_theme', 'N/A')}")
            print(f"   🧬 Sujets: {', '.join(intelligent_analysis.get('key_topics', [])[:5])}")
            print(f"   😊 Sentiment: {intelligent_analysis.get('sentiment', 'N/A')}")
            print(f"   📊 Complexité: {intelligent_analysis.get('complexity', 'N/A')}")
            
            keywords = intelligent_analysis.get('keywords', [])
            print(f"   🔑 Mots-clés: {len(keywords)} termes")
            if keywords:
                print(f"   📝 Détail: {', '.join(keywords[:10])}{'...' if len(keywords) > 10 else ''}")
                
                # Analyse de la qualité
                generic_words = ['this', 'that', 'the', 'and', 'or', 'but', 'for', 'with', 'your', 'my', 'his', 'her', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall']
                generic_count = sum(1 for kw in keywords if kw.lower() in generic_words)
                print(f"   ⚠️ Mots génériques: {generic_count}/{len(keywords)} ({generic_count/len(keywords)*100:.1f}%)")
                
                # Vérifier la structure
                has_hierarchical = any('synonyms' in str(kw) for kw in keywords)
                print(f"   🏗️ Structure hiérarchique: {'✅ Détectée' if has_hierarchical else '❌ NON détectée'}")
                
        else:
            print(f"❌ Métadonnées intelligentes non trouvées: {meta_path}")
        
        # Rapport de sélection B-roll
        selection_path = Path("output/meta/reframed_broll_selection_report.json")
        if selection_path.exists():
            print(f"\n✅ Rapport de sélection B-roll trouvé: {selection_path}")
            with open(selection_path, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)
            
            print(f"   📊 Sélection: {selection_data.get('num_selected', 0)}/{selection_data.get('num_candidates', 0)} B-rolls")
            print(f"   🎯 Top score: {selection_data.get('top_score', 0.0):.3f}")
            print(f"   📏 Seuil appliqué: {selection_data.get('min_score', 0.0):.3f}")
            print(f"   🆘 Fallback utilisé: {selection_data.get('fallback_used', False)}")
            print(f"   🏷️ Tier fallback: {selection_data.get('fallback_tier', 'N/A')}")
            
        else:
            print(f"❌ Rapport de sélection B-roll non trouvé: {selection_path}")
        
        # Analyser les logs du pipeline
        print(f"\n📋 ANALYSE DES LOGS DU PIPELINE:")
        print("=" * 50)
        
        log_path = Path("output/pipeline.log.jsonl")
        if log_path.exists():
            print(f"✅ Logs du pipeline trouvés: {log_path}")
            
            # Lire les dernières lignes du log
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"   📊 Nombre total de lignes: {len(lines)}")
            
            # Analyser les dernières entrées
            if lines:
                print(f"\n🔍 DERNIÈRES ENTREES DU LOG:")
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
            print(f"❌ Logs du pipeline non trouvés: {log_path}")
        
        # Analyser la structure des dossiers
        print(f"\n📁 ANALYSE DE LA STRUCTURE DES DOSSIERS:")
        print("=" * 50)
        
        # Dossier clips
        clips_dir = Path("clips")
        if clips_dir.exists():
            clips_files = list(clips_dir.glob("*.mp4"))
            print(f"✅ Dossier clips: {len(clips_files)} vidéos")
            for clip in clips_files[:3]:
                print(f"   📹 {clip.name} ({clip.stat().st_size / (1024*1024):.1f} MB)")
            if len(clips_files) > 3:
                print(f"   ... et {len(clips_files) - 3} autres")
        else:
            print("❌ Dossier clips non trouvé")
        
        # Dossier output
        output_dir = Path("output")
        if output_dir.exists():
            output_files = list(output_dir.rglob("*.mp4"))
            print(f"✅ Dossier output: {len(output_files)} vidéos traitées")
            for output in output_files[:3]:
                print(f"   🎬 {output.relative_to(output_dir)} ({output.stat().st_size / (1024*1024):.1f} MB)")
            if len(output_files) > 3:
                print(f"   ... et {len(output_files) - 3} autres")
        else:
            print("❌ Dossier output non trouvé")
        
        # Dossier AI-B-roll
        ai_broll_dir = Path("AI-B-roll")
        if ai_broll_dir.exists():
            broll_library = ai_broll_dir / "broll_library"
            if broll_library.exists():
                clip_dirs = list(broll_library.glob("clip_*"))
                print(f"✅ Dossier AI-B-roll: {len(clip_dirs)} dossiers de clips")
                for clip_dir in clip_dirs[:3]:
                    print(f"   📁 {clip_dir.name}")
                if len(clip_dirs) > 3:
                    print(f"   ... et {len(clip_dirs) - 3} autres")
            else:
                print("   ❌ Sous-dossier broll_library non trouvé")
        else:
            print("❌ Dossier AI-B-roll non trouvé")
        
        # Résumé de l'analyse
        print(f"\n📊 RÉSUMÉ DE L'ANALYSE:")
        print("=" * 50)
        
        if meta_path.exists() and selection_path.exists():
            print("✅ Métadonnées et rapports disponibles")
            print("✅ Analyse du fonctionnement possible")
            
            # Déterminer le statut
            if selection_data.get('fallback_used', False):
                print("⚠️  Fallback activé - Problème de sélection B-roll")
            else:
                print("✅ Sélection B-roll normale")
                
            if generic_count > len(keywords) * 0.3:  # Plus de 30% de mots génériques
                print("❌ Qualité des mots-clés médiocre (trop de mots génériques)")
            else:
                print("✅ Qualité des mots-clés acceptable")
                
        else:
            print("❌ Données insuffisantes pour l'analyse")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyser_logs_existants() 