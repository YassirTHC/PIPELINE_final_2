ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸â€Â CAPTURE JSON BRUT LLAMA2:13B
Capture et affiche la rÃƒÂ©ponse JSON complÃƒÂ¨te pour analyse de conformitÃƒÂ©
"""

import requests
import json
import time

def capture_json_llama2_13b():
    """Capture la rÃƒÂ©ponse JSON brute de llama2:13b"""
    print("Ã°Å¸â€Â CAPTURE JSON BRUT LLAMA2:13B")
    print("=" * 50)
    
    # Prompt simplifiÃƒÂ© (votre version actuelle)
    prompt = (
        "You are a JSON generator for social media content. Generate ONLY valid JSON.\n\n"
        "REQUIRED: Create a JSON object with these exact keys:\n"
        "- title: single catchy title (Ã¢â€°Â¤60 chars)\n"
        "- description: single description with call-to-action\n"
        "- hashtags: array of 10-14 hashtags (#keyword format)\n"
        "- broll_keywords: array of 20-25 keyword objects\n\n"
        "BROLL KEYWORDS STRUCTURE:\n"
        "Each keyword object must have:\n"
        "{\n"
        '  "category": "VISUAL ACTIONS|PEOPLE & ROLES|ENVIRONMENTS & PLACES|OBJECTS & PROPS|EMOTIONAL/CONTEXTUAL",\n'
        '  "base": "main keyword",\n'
        '  "synonyms": ["syn1", "syn2", "syn3"]\n'
        "}\n\n"
        "CATEGORIES:\n"
        "- VISUAL ACTIONS (8-12): running, exercising, meditating, writing\n"
        "- PEOPLE & ROLES (8-10): therapist, patient, family, professional\n"
        "- ENVIRONMENTS (8-10): hospital, clinic, gym, office, nature\n"
        "- OBJECTS (6-8): equipment, notebook, phone, weights\n"
        "- EMOTIONS (6-8): healing, stress relief, growth, recovery\n\n"
        "RULES:\n"
        "1. Output ONLY valid JSON\n"
        "2. No explanations or text outside JSON\n"
        "3. Use proper JSON syntax with double quotes\n"
        "4. Ensure all arrays have correct brackets\n"
        "5. Match transcript context (healthcare, therapy, EMDR)\n\n"
        "Transcript: EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking thing while stress and rationale coupling a low stress state the recall of trauma it's gonna allow people reshape relationship trauma it's a tolerate that discomfort and EMDR clinical colleagues tell me works best fairly well defined traumas\n\n"
        "JSON:"
    )
    
    print(f"Ã°Å¸â€œÂ Prompt: {len(prompt)} caractÃƒÂ¨res")
    print(f"Ã°Å¸Å½Â¯ ModÃƒÂ¨le: llama2:13b")
    print(f"Ã¢ÂÂ³ Test en cours...")
    
    try:
        # Appel direct ÃƒÂ  l'API Ollama
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama2:13b",
            "prompt": prompt,
            "temperature": 0.7,
            "stream": False
        }
        
        print(f"Ã°Å¸Å¡â‚¬ Envoi ÃƒÂ  Ollama...")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=600)  # 10 minutes
        response.raise_for_status()
        
        end_time = time.time()
        response_time = end_time - start_time
        
        data = response.json()
        raw_response = data.get("response", "")
        
        print(f"Ã¢Å“â€¦ RÃƒÂ©ponse reÃƒÂ§ue en {response_time:.1f}s")
        print(f"Ã°Å¸â€œÅ  Taille: {len(raw_response)} caractÃƒÂ¨res")
        
        # Sauvegarder la rÃƒÂ©ponse brute
        with open("llama2_13b_json_brut.txt", "w", encoding="utf-8") as f:
            f.write(raw_response)
        
        print(f"\nÃ°Å¸â€œÂ RÃƒÂ©ponse sauvegardÃƒÂ©e dans 'llama2_13b_json_brut.txt'")
        
        # Analyse de la rÃƒÂ©ponse
        print(f"\nÃ°Å¸â€Â ANALYSE DE LA RÃƒâ€°PONSE:")
        print("=" * 50)
        
        # 1. Recherche de JSON
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}")
        
        if json_start != -1 and json_end != -1:
            print(f"Ã¢Å“â€¦ JSON dÃƒÂ©tectÃƒÂ©: position {json_start} ÃƒÂ  {json_end}")
            json_content = raw_response[json_start:json_end+1]
            
            # Sauvegarder le JSON extrait
            with open("llama2_13b_json_extrait.txt", "w", encoding="utf-8") as f:
                f.write(json_content)
            
            print(f"Ã°Å¸â€œÂ JSON extrait sauvegardÃƒÂ© dans 'llama2_13b_json_extrait.txt'")
            
            # Test de validation JSON
            try:
                parsed_json = json.loads(json_content)
                print(f"Ã¢Å“â€¦ JSON valide !")
                print(f"Ã°Å¸â€œâ€¹ ClÃƒÂ©s trouvÃƒÂ©es: {list(parsed_json.keys())}")
                
                # Analyse dÃƒÂ©taillÃƒÂ©e des clÃƒÂ©s
                print(f"\nÃ°Å¸â€Â ANALYSE DÃƒâ€°TAILLÃƒâ€°E:")
                print("=" * 30)
                
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        print(f"   {key}: {len(value)} ÃƒÂ©lÃƒÂ©ments")
                        if key == "hashtags" and len(value) < 10:
                            print(f"      Ã¢Å¡Â Ã¯Â¸Â INSUFFISANT: {len(value)} hashtags (attendu: 10-14)")
                        elif key == "broll_keywords" and len(value) < 20:
                            print(f"      Ã¢Å¡Â Ã¯Â¸Â INSUFFISANT: {len(value)} keywords (attendu: 20-25)")
                    else:
                        print(f"   {key}: {type(value).__name__} = '{value}'")
                
                # Affichage complet du JSON
                print(f"\nÃ°Å¸â€œâ€ž JSON COMPLET GÃƒâ€°NÃƒâ€°RÃƒâ€°:")
                print("=" * 30)
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Ã¢ÂÅ’ JSON invalide: {e}")
                
        else:
            print(f"Ã¢ÂÅ’ Aucun JSON dÃƒÂ©tectÃƒÂ© dans la rÃƒÂ©ponse")
            print(f"Ã°Å¸â€Â Contenu de la rÃƒÂ©ponse:")
            print(f"   DÃƒÂ©but: {raw_response[:200]}...")
            print(f"   Fin: ...{raw_response[-200:]}")
        
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur: {e}")
        return False

if __name__ == "__main__":
    capture_json_llama2_13b() 

