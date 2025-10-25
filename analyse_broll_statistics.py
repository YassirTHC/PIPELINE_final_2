ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Analyse des Statistiques B-roll du Pipeline
Comprendre le comportement et la densitÃƒÂ© des B-rolls
"""

import json
from pathlib import Path
from collections import defaultdict

def analyser_statistiques_broll():
    """Analyse les statistiques B-roll du pipeline"""
    print("\nÃ°Å¸â€œÅ  ANALYSE DES STATISTIQUES B-ROLL DU PIPELINE")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Comprendre le comportement et la densitÃƒÂ© des B-rolls")
    
    # VÃƒÂ©rifier les logs
    log_file = Path("output/pipeline.log.jsonl")
    if not log_file.exists():
        print("   Ã¢ÂÅ’ Fichier de logs non trouvÃƒÂ©")
        return False
    
    print(f"   Ã°Å¸â€œÂ Fichier de logs: {log_file.name}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Analyser les B-rolls appliquÃƒÂ©s
        broll_events = []
        video_segments = defaultdict(list)
        
        for line in lines:
            try:
                event = json.loads(line.strip())
                if event.get('type') == 'event_applied':
                    broll_events.append(event)
                    
                    # Grouper par vidÃƒÂ©o (extraire du chemin)
                    media_path = event.get('media_path', '')
                    if 'clip_reframed_' in media_path:
                        # Extraire l'ID de la vidÃƒÂ©o
                        parts = media_path.split('clip_reframed_')
                        if len(parts) > 1:
                            video_id = parts[1].split('\\')[0]
                            video_segments[video_id].append(event)
            except:
                continue
        
        print(f"\n   Ã°Å¸â€œÅ  STATISTIQUES GÃƒâ€°NÃƒâ€°RALES:")
        print(f"      Ã¢â‚¬Â¢ Total B-rolls appliquÃƒÂ©s: {len(broll_events)}")
        print(f"      Ã¢â‚¬Â¢ VidÃƒÂ©os traitÃƒÂ©es: {len(video_segments)}")
        
        # Analyser par vidÃƒÂ©o
        print(f"\n   Ã°Å¸Å½Â¬ ANALYSE PAR VIDÃƒâ€°O:")
        for video_id, events in video_segments.items():
            print(f"\n      Ã°Å¸â€œÂ¹ VidÃƒÂ©o {video_id}:")
            print(f"         Ã¢â‚¬Â¢ B-rolls appliquÃƒÂ©s: {len(events)}")
            
            # Analyser la distribution temporelle
            if events:
                start_times = [e.get('start_s', 0) for e in events]
                end_times = [e.get('end_s', 0) for e in events]
                durations = [e.get('duration_effective', 0) for e in events]
                
                total_duration = max(end_times) if end_times else 0
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                print(f"         Ã¢â‚¬Â¢ DurÃƒÂ©e totale: {total_duration:.1f}s")
                print(f"         Ã¢â‚¬Â¢ DurÃƒÂ©e moyenne B-roll: {avg_duration:.1f}s")
                print(f"         Ã¢â‚¬Â¢ DensitÃƒÂ© B-roll: {len(events) / (total_duration / 60):.1f} B-rolls/minute")
                
                # Analyser la distribution temporelle
                print(f"         Ã¢â‚¬Â¢ Distribution temporelle:")
                for i, event in enumerate(events[:5]):  # Afficher les 5 premiers
                    start = event.get('start_s', 0)
                    end = event.get('end_s', 0)
                    duration = event.get('duration_effective', 0)
                    print(f"            {i+1}. {start:.1f}s Ã¢â€ â€™ {end:.1f}s ({duration:.1f}s)")
                
                if len(events) > 5:
                    print(f"            ... et {len(events) - 5} autres B-rolls")
        
        # Analyser la qualitÃƒÂ© des B-rolls
        print(f"\n   Ã°Å¸Å½Â¯ ANALYSE DE LA QUALITÃƒâ€°:")
        
        # Compter par provider
        providers = defaultdict(int)
        themes = defaultdict(int)
        
        for event in broll_events:
            media_path = event.get('media_path', '')
            
            # Provider
            if 'pexels' in media_path:
                providers['pexels'] += 1
            elif 'pixabay' in media_path:
                providers['pixabay'] += 1
            elif 'archive' in media_path:
                providers['archive'] += 1
            
            # ThÃƒÂ¨me
            if 'fetched' in media_path:
                parts = media_path.split('fetched\\')
                if len(parts) > 1:
                    theme_part = parts[1].split('\\')[1] if len(parts[1].split('\\')) > 1 else 'unknown'
                    themes[theme_part] += 1
        
        print(f"      Ã¢â‚¬Â¢ RÃƒÂ©partition par provider:")
        for provider, count in providers.items():
            print(f"         - {provider}: {count} B-rolls")
        
        print(f"      Ã¢â‚¬Â¢ Top 10 des thÃƒÂ¨mes:")
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        for theme, count in sorted_themes[:10]:
            print(f"         - {theme}: {count} B-rolls")
        
        # Analyser les gaps entre B-rolls
        print(f"\n   Ã¢ÂÂ±Ã¯Â¸Â ANALYSE DES GAPS:")
        for video_id, events in video_segments.items():
            if len(events) > 1:
                print(f"\n      Ã°Å¸â€œÂ¹ VidÃƒÂ©o {video_id}:")
                sorted_events = sorted(events, key=lambda x: x.get('start_s', 0))
                
                gaps = []
                for i in range(1, len(sorted_events)):
                    prev_end = sorted_events[i-1].get('end_s', 0)
                    curr_start = sorted_events[i].get('start_s', 0)
                    gap = curr_start - prev_end
                    gaps.append(gap)
                
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    min_gap = min(gaps)
                    max_gap = max(gaps)
                    
                    print(f"         Ã¢â‚¬Â¢ Gap moyen: {avg_gap:.1f}s")
                    print(f"         Ã¢â‚¬Â¢ Gap minimum: {min_gap:.1f}s")
                    print(f"         Ã¢â‚¬Â¢ Gap maximum: {max_gap:.1f}s")
        
        # Recommandations
        print(f"\n   Ã°Å¸â€™Â¡ RECOMMANDATIONS POUR 1 MINUTE:")
        print(f"      Ã¢â‚¬Â¢ DensitÃƒÂ© optimale: 3-6 B-rolls/minute")
        print(f"      Ã¢â‚¬Â¢ DurÃƒÂ©e B-roll: 2-4 secondes")
        print(f"      Ã¢â‚¬Â¢ Gaps recommandÃƒÂ©s: 8-15 secondes")
        print(f"      Ã¢â‚¬Â¢ Facteurs clÃƒÂ©s: mots-clÃƒÂ©s, contexte, diversitÃƒÂ©")
        
        return True
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur lors de l'analyse: {e}")
        return False

def main():
    """Fonction principale"""
    print("Ã°Å¸Å¡â‚¬ ANALYSE DES STATISTIQUES B-ROLL")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Comprendre le comportement du pipeline")
    
    # ExÃƒÂ©cuter l'analyse
    result = analyser_statistiques_broll()
    
    if result:
        print("\n" + "=" * 70)
        print("Ã¢Å“â€¦ ANALYSE TERMINÃƒâ€°E AVEC SUCCÃƒË†S")
        print("=" * 70)
        print("Ã°Å¸â€™Â¡ Le pipeline optimise automatiquement la densitÃƒÂ© B-roll")
        print("Ã°Å¸Å½Â¯ La qualitÃƒÂ© prime sur la quantitÃƒÂ©")
        print("Ã¢ÂÂ±Ã¯Â¸Â Les gaps sont calculÃƒÂ©s intelligemment")
    else:
        print("\n" + "=" * 70)
        print("Ã¢ÂÅ’ ANALYSE Ãƒâ€°CHOUÃƒâ€°E")
        print("=" * 70)
    
    return result

if __name__ == "__main__":
    main() 

