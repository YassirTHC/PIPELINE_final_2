ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ DIAGNOSTIC QWEN3:8B Ãƒâ€°TAPE 1 - CAPTURE RÃƒâ€°PONSE BRUTE

import requests
import json
import time
from prompts_hybrides_etapes import get_prompt_etape_1

def test_qwen3_8b_etape1():
    """Test Qwen3:8B avec l'ÃƒÂ©tape 1 et capture la rÃƒÂ©ponse brute"""
    
    # Transcript de test
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking things"
    
    # Prompt de l'ÃƒÂ©tape 1
    prompt = get_prompt_etape_1(test_transcript)
    
    print("Ã°Å¸Å¡â‚¬ DIAGNOSTIC QWEN3:8B Ãƒâ€°TAPE 1")
    print("=" * 50)
    print(f"Ã°Å¸â€œÂ Transcript: {test_transcript}")
    print(f"Ã°Å¸â€œÂ Prompt: {len(prompt)} caractÃƒÂ¨res")
    print(f"Ã¢ÂÂ±Ã¯Â¸Â Timeout: 300s")
    print()
    
    try:
        print("Ã°Å¸Â¤â€“ Appel ÃƒÂ  Qwen3:8B...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:8b",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Ã¢Å“â€¦ RÃƒÂ©ponse reÃƒÂ§ue en {duration:.1f}s")
        print(f"Ã°Å¸â€œÅ  Status HTTP: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Sauvegarde de la rÃƒÂ©ponse brute
            with open("qwen3_8b_reponse_brute.txt", "w", encoding="utf-8") as f:
                f.write("=== RÃƒâ€°PONSE BRUTE QWEN3:8B ===\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"DurÃƒÂ©e: {duration:.1f}s\n")
                f.write(f"Status: {response.status_code}\n")
                f.write("=" * 50 + "\n")
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
                f.write("\n" + "=" * 50 + "\n")
            
            print("Ã°Å¸â€™Â¾ RÃƒÂ©ponse brute sauvegardÃƒÂ©e dans 'qwen3_8b_reponse_brute.txt'")
            
            # Analyse de la rÃƒÂ©ponse
            if "response" in result:
                llm_response = result["response"]
                print(f"Ã°Å¸â€œÂ RÃƒÂ©ponse LLM: {len(llm_response)} caractÃƒÂ¨res")
                print("Ã°Å¸â€Â Contenu de la rÃƒÂ©ponse:")
                print("-" * 30)
                print(llm_response)
                print("-" * 30)
                
                # Test de parsing JSON
                try:
                    parsed = json.loads(llm_response)
                    print("Ã¢Å“â€¦ JSON valide dÃƒÂ©tectÃƒÂ© !")
                    print(f"Ã°Å¸â€œâ€¹ ClÃƒÂ©s trouvÃƒÂ©es: {list(parsed.keys())}")
                    
                    if "title" in parsed:
                        print(f"   Titres: {len(parsed['title'])}")
                    if "hashtags" in parsed:
                        print(f"   Hashtags: {len(parsed['hashtags'])}")
                        
                except json.JSONDecodeError as e:
                    print(f"Ã¢ÂÅ’ JSON invalide: {e}")
                    print("Ã°Å¸â€Â Analyse de l'erreur:")
                    
                    if not llm_response.strip():
                        print("   Ã¢â€ â€™ RÃƒÂ©ponse vide ou uniquement des espaces")
                    elif llm_response.startswith("Voici") or "Voici" in llm_response:
                        print("   Ã¢â€ â€™ LLM gÃƒÂ©nÃƒÂ¨re du texte explicatif au lieu de JSON")
                    elif "{" not in llm_response or "}" not in llm_response:
                        print("   Ã¢â€ â€™ RÃƒÂ©ponse ne contient pas d'accolades JSON")
                    else:
                        print("   Ã¢â€ â€™ Autre problÃƒÂ¨me de formatage JSON")
                        
            else:
                print("Ã¢ÂÅ’ Pas de clÃƒÂ© 'response' dans la rÃƒÂ©ponse")
                print(f"Ã°Å¸â€œâ€¹ ClÃƒÂ©s disponibles: {list(result.keys())}")
                
        else:
            print(f"Ã¢ÂÅ’ Erreur HTTP: {response.status_code}")
            print(f"Ã°Å¸â€œÂ Contenu: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Ã¢ÂÂ±Ã¯Â¸Â Timeout aprÃƒÂ¨s 300s")
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur: {str(e)}")

if __name__ == "__main__":
    test_qwen3_8b_etape1() 

