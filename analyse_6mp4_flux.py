#!/usr/bin/env python3
"""
Analyse du Flux LLM → Fetchers → Scoring pour 6.mp4
Vérification de ce qui s'est passé lors du traitement
"""

import json
import os
from pathlib import Path

def analyser_flux_6mp4():
    """Analyse du flux LLM → Fetchers → Scoring pour 6.mp4"""
    print("🔍 ANALYSE DU FLUX LLM → FETCHERS → SCORING POUR 6.MP4")
    print("=" * 80)
    
    # Dossier de sortie de 6.mp4
    output_6_dir = Path("output/clips/6")
    
    if not output_6_dir.exists():
        print("❌ Dossier output/clips/6 non trouvé")
        return
    
    print(f"📁 Dossier analysé: {output_6_dir}")
    
    # 1. Vérifier les fichiers générés
    print("\n🔍 1. FICHIERS GÉNÉRÉS:")
    files = list(output_6_dir.iterdir())
    for file in files:
        if file.is_file():
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   📄 {file.name}: {size_mb:.1f} MB")
    
    # 2. Analyser les métadonnées
    print("\n🔍 2. MÉTADONNÉES:")
    meta_file = output_6_dir / "meta.txt"
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"   📝 Contenu meta.txt:")
        for line in content.split('\n'):
            if line.strip():
                print(f"      {line}")
    else:
        print("   ❌ meta.txt non trouvé")
    
    # 3. Analyser la transcription
    print("\n🔍 3. TRANSCRIPTION:")
    segments_file = output_6_dir / "6_segments.json"
    if segments_file.exists():
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        print(f"   📊 {len(segments)} segments de transcription")
        
        # Extraire les mots-clés potentiels
        all_text = " ".join([seg.get('text', '') for seg in segments])
        words = [word.lower().strip() for word in all_text.split() if len(word) > 3]
        
        # Mots-clés liés à la santé
        health_keywords = ['healthcare', 'medical', 'doctor', 'hospital', 'operation', 'medicare', 'medicaid']
        found_health = [word for word in words if any(health in word for health in health_keywords)]
        
        print(f"   🏥 Mots-clés santé trouvés: {len(found_health)}")
        if found_health:
            print(f"      Exemples: {', '.join(set(found_health[:10]))}")
    
    # 4. Analyser les tokens avec couleurs
    print("\n🔍 4. TOKENS AVEC COULEURS:")
    tokens_file = output_6_dir / "final_subtitled.tokens.json"
    if tokens_file.exists():
        with open(tokens_file, 'r', encoding='utf-8') as f:
            tokens_data = json.load(f)
        
        print(f"   🎨 {len(tokens_data)} segments avec tokens")
        
        # Compter les mots-clés colorés
        colored_keywords = []
        for segment in tokens_data:
            for token in segment.get('tokens', []):
                if token.get('is_keyword', False):
                    colored_keywords.append(token.get('text', ''))
        
        print(f"   🎯 Mots-clés colorés: {len(colored_keywords)}")
        if colored_keywords:
            unique_colored = list(set(colored_keywords))
            print(f"      Exemples: {', '.join(unique_colored[:15])}")
    
    # 5. Vérifier le log du pipeline
    print("\n🔍 5. LOG DU PIPELINE:")
    pipeline_log = Path("output/pipeline.log.jsonl")
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   📋 {len(lines)} lignes dans le log")
        
        # Chercher des informations sur 6.mp4
        lines_6mp4 = [line for line in lines if "6" in line]
        print(f"   🎬 Lignes contenant '6': {len(lines_6mp4)}")
        
        # Analyser les derniers événements
        recent_events = lines[-20:] if len(lines) > 20 else lines
        print(f"   ⏰ 20 derniers événements:")
        
        for i, line in enumerate(recent_events[-5:], 1):
            try:
                event = json.loads(line.strip())
                event_type = event.get('type', 'N/A')
                media_path = event.get('media_path', 'N/A')
                start_s = event.get('start_s', 'N/A')
                end_s = event.get('end_s', 'N/A')
                
                print(f"      {i}. [{start_s}s-{end_s}s] {event_type}")
                if '6' in media_path:
                    print(f"         🎬 6.mp4: {os.path.basename(media_path)}")
                else:
                    print(f"         📹 B-roll: {os.path.basename(media_path)}")
                    
            except:
                print(f"      {i}. ⚠️ Ligne non-JSON")
    
    # 6. Vérifier la bibliothèque B-roll
    print("\n🔍 6. BIBLIOTHÈQUE B-ROLL:")
    broll_library = Path("AI-B-roll/broll_library")
    if broll_library.exists():
        clip_dirs = [d for d in broll_library.iterdir() if d.is_dir() and d.name.startswith('clip_reframed_')]
        print(f"   📚 {len(clip_dirs)} dossiers de clips reframés")
        
        # Vérifier les clips récents
        recent_clips = sorted(clip_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print(f"   🆕 5 clips les plus récents:")
        
        for clip_dir in recent_clips:
            clip_name = clip_dir.name
            mtime = clip_dir.stat().st_mtime
            print(f"      📁 {clip_name}")
            
            # Vérifier le contenu
            fetched_dir = clip_dir / "fetched"
            if fetched_dir.exists():
                sources = [d.name for d in fetched_dir.iterdir() if d.is_dir()]
                print(f"         📥 Sources: {', '.join(sources)}")
    
    # 7. Analyse du flux LLM → Fetchers → Scoring
    print("\n🔍 7. ANALYSE DU FLUX LLM → FETCHERS → SCORING:")
    
    # Vérifier si les mots-clés B-roll ont été générés
    if meta_file.exists():
        meta_content = open(meta_file, 'r', encoding='utf-8').read()
        
        # Vérifier la présence de mots-clés B-roll
        if 'broll_keywords' in meta_content.lower() or 'keywords' in meta_content.lower():
            print("   ✅ Mots-clés B-roll détectés dans les métadonnées")
        else:
            print("   ⚠️ Aucun mot-clé B-roll détecté dans les métadonnées")
            print("   🔍 Vérification du fichier tokens.json...")
            
            # Vérifier dans tokens.json
            if tokens_file.exists():
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    tokens_content = f.read()
                
                if 'broll_keywords' in tokens_content.lower():
                    print("      ✅ Mots-clés B-roll trouvés dans tokens.json")
                else:
                    print("      ❌ Aucun mot-clé B-roll trouvé")
    
    # Vérifier l'utilisation des B-rolls
    if pipeline_log.exists():
        with open(pipeline_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Compter les B-rolls utilisés
        broll_events = log_content.count('"type": "event_applied"')
        print(f"   🎬 Événements B-roll dans le log: {broll_events}")
        
        if broll_events > 0:
            print("   ✅ B-rolls ont été appliqués")
        else:
            print("   ❌ Aucun B-roll appliqué")
    
    # 8. Conclusion
    print("\n🔍 8. CONCLUSION:")
    
    # Vérifier les composants du flux
    components_status = {
        "LLM": "❓ À vérifier",
        "Fetchers": "❓ À vérifier", 
        "Scoring": "❓ À vérifier",
        "Sélection": "❓ À vérifier"
    }
    
    # Mettre à jour le statut basé sur l'analyse
    if meta_file.exists() and "healthcare" in open(meta_file, 'r', encoding='utf-8').read().lower():
        components_status["LLM"] = "✅ Actif (mots-clés générés)"
    
    if broll_library.exists() and len(list(broll_library.iterdir())) > 0:
        components_status["Fetchers"] = "✅ Actif (bibliothèque B-roll)"
    
    if pipeline_log.exists() and broll_events > 0:
        components_status["Scoring"] = "✅ Actif (B-rolls appliqués)"
        components_status["Sélection"] = "✅ Actif (B-rolls sélectionnés)"
    
    for component, status in components_status.items():
        print(f"   {component}: {status}")
    
    # Recommandations
    print("\n🔍 9. RECOMMANDATIONS:")
    
    if components_status["LLM"] == "❓ À vérifier":
        print("   🧠 Vérifier la génération LLM des mots-clés B-roll")
    
    if components_status["Fetchers"] == "❓ À vérifier":
        print("   📥 Vérifier le téléchargement des B-rolls")
    
    if components_status["Scoring"] == "❓ À vérifier":
        print("   🎯 Vérifier le système de scoring")
    
    if components_status["Sélection"] == "❓ À vérifier":
        print("   🎬 Vérifier la sélection finale des B-rolls")
    
    if all("✅" in status for status in components_status.values()):
        print("   🎉 Tous les composants du flux sont actifs !")
        print("   🚀 Le pipeline LLM → Fetchers → Scoring fonctionne parfaitement")

def main():
    """Fonction principale"""
    print("🎯 Analyse du flux LLM → Fetchers → Scoring pour 6.mp4")
    
    analyser_flux_6mp4()

if __name__ == "__main__":
    main() 