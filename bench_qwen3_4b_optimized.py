#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ðŸ§ª BENCH QWEN3:4B OPTIMISÃ‰ - PROMPT DIRECTIF
Test de qwen3:4b avec un prompt optimisÃ© pour Ã©viter le mode "thinking"
"""

import time
import requests
import json
import psutil
from datetime import datetime

def bench_qwen3_4b_optimized():
    """Benchmark de qwen3:4b avec prompt optimisÃ©"""
    
    print("ðŸ§ª BENCH QWEN3:4B OPTIMISÃ‰ - PROMPT DIRECTIF")
    print("=" * 60)
    print(f"â° DÃ©but: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Configuration
    URL = "http://localhost:11434/api/generate"
    MODEL = "qwen3:4b"
    
    # Prompt ultra-directif (pas de "thinking")
    PROMPT = '''Generate 5 filmable keywords for "family playing in park".
Return ONLY valid JSON: {"keywords":["k1","k2","k3","k4","k5"]}
No explanations, no thinking, just JSON.'''
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0.1,  # Plus dÃ©terministe
        "max_tokens": 100,   # RÃ©duit pour accÃ©lÃ©rer
        "stream": False,
        "top_p": 0.9,       # ContrÃ´le de la crÃ©ativitÃ©
        "top_k": 40         # Limite les choix
    }
    
    print(f"ðŸŽ¯ ModÃ¨le: {MODEL}")
    print(f"ðŸ“ Prompt: {len(PROMPT)} caractÃ¨res")
    print(f"â±ï¸ Timeout: 60s (rÃ©duit)")
    print(f"ðŸ“Š Max tokens: 100")
    print(f"ðŸŒ¡ï¸ Temperature: 0.1 (dÃ©terministe)")
    print()
    
    # Monitoring systÃ¨me avant
    print("ðŸ“Š MONITORING SYSTÃˆME - AVANT")
    print("-" * 40)
    mem_before = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"ðŸ’¾ RAM disponible: {mem_before.available / 1e9:.2f} GB")
    print(f"ðŸ’¾ RAM utilisÃ©e: {mem_before.used / 1e9:.2f} GB")
    print(f"ðŸ”„ CPU: {cpu_percent}%")
    print()
    
    # Test LLM
    print("ðŸš€ TEST QWEN3:4B OPTIMISÃ‰ EN COURS...")
    print("-" * 40)
    
    try:
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=60)
        t1 = time.time()
        
        elapsed = t1 - t0
        status = r.status_code
        
        print(f"âœ… Statut: {status}")
        print(f"â±ï¸ Temps total: {elapsed:.2f}s")
        print(f"ðŸ“Š Latence: {elapsed*1000:.0f}ms")
        
        if r.status_code == 200:
            try:
                response_data = r.json()
                response_text = response_data.get('response', '')
                print(f"ðŸ“ RÃ©ponse: {len(response_text)} caractÃ¨res")
                print(f"ðŸ” DÃ©but rÃ©ponse: {response_text[:200]}...")
                
                # Test parsing JSON
                try:
                    parsed_json = json.loads(response_text)
                    print("âœ… JSON valide dÃ©tectÃ©")
                    
                    # Analyse de la structure
                    if 'keywords' in parsed_json:
                        keywords = parsed_json['keywords']
                        print(f"ðŸŽ¯ Mots-clÃ©s trouvÃ©s: {len(keywords)}")
                        print(f"ðŸ“ Mots-clÃ©s: {keywords}")
                        
                        if isinstance(keywords, list) and len(keywords) >= 5:
                            print("âœ… Nombre de mots-clÃ©s OK (â‰¥5)")
                        else:
                            print("âš ï¸ Nombre de mots-clÃ©s insuffisant")
                    else:
                        print("âš ï¸ Structure 'keywords' manquante")
                        
                except json.JSONDecodeError as e:
                    print(f"âŒ JSON invalide: {e}")
                    print("ðŸ” Tentative de rÃ©paration...")
                    
                    # Tentative de rÃ©paration simple
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            parsed_json = json.loads(json_str)
                            print("âœ… JSON rÃ©parÃ© avec succÃ¨s")
                        else:
                            print("âŒ Impossible de rÃ©parer le JSON")
                    except:
                        print("âŒ RÃ©paration JSON Ã©chouÃ©e")
                        
            except Exception as e:
                print(f"âŒ Erreur parsing rÃ©ponse: {e}")
                print(f"ðŸ“ RÃ©ponse brute: {r.text[:200]}...")
        else:
            print(f"âŒ Erreur HTTP: {r.text}")
            
    except requests.exceptions.Timeout:
        print("â±ï¸ TIMEOUT aprÃ¨s 60s")
        elapsed = 60
        status = "TIMEOUT"
    except Exception as e:
        print(f"âŒ Erreur: {str(e)}")
        elapsed = 0
        status = "ERROR"
    
    # Monitoring systÃ¨me aprÃ¨s
    print()
    print("ðŸ“Š MONITORING SYSTÃˆME - APRÃˆS")
    print("-" * 40)
    mem_after = psutil.virtual_memory()
    cpu_percent_after = psutil.cpu_percent(interval=1)
    
    mem_delta = mem_before.available - mem_after.available
    print(f"ðŸ’¾ RAM delta: {mem_delta / 1e6:.1f} MB")
    print(f"ðŸ’¾ RAM disponible: {mem_after.available / 1e9:.2f} GB")
    print(f"ðŸ”„ CPU: {cpu_percent_after}%")
    
    # Analyse des rÃ©sultats
    print()
    print("ðŸ” ANALYSE DES RÃ‰SULTATS")
    print("=" * 60)
    
    if elapsed < 10:
        print("âœ… QWEN3:4B OPTIMISÃ‰ - Temps excellent (<10s)")
        print("ðŸŽ¯ ProblÃ¨me rÃ©solu: Prompt optimisÃ© fonctionne")
    elif elapsed < 30:
        print("âš ï¸ QWEN3:4B OPTIMISÃ‰ - Temps acceptable (10-30s)")
        print("ðŸŽ¯ AmÃ©lioration significative, peut Ãªtre optimisÃ©")
    elif elapsed < 60:
        print("âŒ QWEN3:4B OPTIMISÃ‰ - Temps Ã©levÃ© (30-60s)")
        print("ðŸŽ¯ ProblÃ¨me persiste, vÃ©rifier configuration")
    else:
        print("ðŸš¨ QWEN3:4B OPTIMISÃ‰ - Timeout atteint")
        print("ðŸŽ¯ ProblÃ¨me critique, modÃ¨le inutilisable")
    
    print()
    print("ðŸ“‹ RECOMMANDATIONS")
    print("-" * 40)
    
    if elapsed < 30:
        print("1. âœ… QWEN3:4B fonctionne avec prompt optimisÃ©")
        print("2. ðŸŽ¯ ImplÃ©menter ce prompt dans le pipeline")
        print("3. ðŸ”§ Supprimer qwen3:8b inutile")
    else:
        print("1. âš ï¸ VÃ©rifier configuration Ollama")
        print("2. ðŸ” Tester avec d'autres paramÃ¨tres")
        print("3. ðŸš¨ ConsidÃ©rer un modÃ¨le plus lÃ©ger")
    
    return elapsed, status

if __name__ == "__main__":
    elapsed, status = bench_qwen3_4b_optimized()
    
    print()
    print("=" * 60)
    print(f"ðŸ BENCH OPTIMISÃ‰ TERMINÃ‰ - Temps: {elapsed:.2f}s, Statut: {status}")
    
    input("\nAppuyez sur EntrÃ©e pour continuer...") 
