# 🚀 DIAGNOSTIC QWEN3:8B ÉTAPE 1 - CAPTURE RÉPONSE BRUTE

import requests
import json
import time
from prompts_hybrides_etapes import get_prompt_etape_1

def test_qwen3_8b_etape1():
    """Test Qwen3:8B avec l'étape 1 et capture la réponse brute"""
    
    # Transcript de test
    test_transcript = "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking things"
    
    # Prompt de l'étape 1
    prompt = get_prompt_etape_1(test_transcript)
    
    print("🚀 DIAGNOSTIC QWEN3:8B ÉTAPE 1")
    print("=" * 50)
    print(f"📝 Transcript: {test_transcript}")
    print(f"📝 Prompt: {len(prompt)} caractères")
    print(f"⏱️ Timeout: 300s")
    print()
    
    try:
        print("🤖 Appel à Qwen3:8B...")
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
        
        print(f"✅ Réponse reçue en {duration:.1f}s")
        print(f"📊 Status HTTP: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Sauvegarde de la réponse brute
            with open("qwen3_8b_reponse_brute.txt", "w", encoding="utf-8") as f:
                f.write("=== RÉPONSE BRUTE QWEN3:8B ===\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Durée: {duration:.1f}s\n")
                f.write(f"Status: {response.status_code}\n")
                f.write("=" * 50 + "\n")
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
                f.write("\n" + "=" * 50 + "\n")
            
            print("💾 Réponse brute sauvegardée dans 'qwen3_8b_reponse_brute.txt'")
            
            # Analyse de la réponse
            if "response" in result:
                llm_response = result["response"]
                print(f"📝 Réponse LLM: {len(llm_response)} caractères")
                print("🔍 Contenu de la réponse:")
                print("-" * 30)
                print(llm_response)
                print("-" * 30)
                
                # Test de parsing JSON
                try:
                    parsed = json.loads(llm_response)
                    print("✅ JSON valide détecté !")
                    print(f"📋 Clés trouvées: {list(parsed.keys())}")
                    
                    if "title" in parsed:
                        print(f"   Titres: {len(parsed['title'])}")
                    if "hashtags" in parsed:
                        print(f"   Hashtags: {len(parsed['hashtags'])}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON invalide: {e}")
                    print("🔍 Analyse de l'erreur:")
                    
                    if not llm_response.strip():
                        print("   → Réponse vide ou uniquement des espaces")
                    elif llm_response.startswith("Voici") or "Voici" in llm_response:
                        print("   → LLM génère du texte explicatif au lieu de JSON")
                    elif "{" not in llm_response or "}" not in llm_response:
                        print("   → Réponse ne contient pas d'accolades JSON")
                    else:
                        print("   → Autre problème de formatage JSON")
                        
            else:
                print("❌ Pas de clé 'response' dans la réponse")
                print(f"📋 Clés disponibles: {list(result.keys())}")
                
        else:
            print(f"❌ Erreur HTTP: {response.status_code}")
            print(f"📝 Contenu: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout après 300s")
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    test_qwen3_8b_etape1() 