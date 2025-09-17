#!/usr/bin/env python3
"""
🔍 CAPTURE JSON BRUT LLAMA2:13B
Capture et affiche la réponse JSON complète pour analyse de conformité
"""

import requests
import json
import time

def capture_json_llama2_13b():
    """Capture la réponse JSON brute de llama2:13b"""
    print("🔍 CAPTURE JSON BRUT LLAMA2:13B")
    print("=" * 50)
    
    # Prompt simplifié (votre version actuelle)
    prompt = (
        "You are a JSON generator for social media content. Generate ONLY valid JSON.\n\n"
        "REQUIRED: Create a JSON object with these exact keys:\n"
        "- title: single catchy title (≤60 chars)\n"
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
    
    print(f"📝 Prompt: {len(prompt)} caractères")
    print(f"🎯 Modèle: llama2:13b")
    print(f"⏳ Test en cours...")
    
    try:
        # Appel direct à l'API Ollama
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama2:13b",
            "prompt": prompt,
            "temperature": 0.7,
            "stream": False
        }
        
        print(f"🚀 Envoi à Ollama...")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=600)  # 10 minutes
        response.raise_for_status()
        
        end_time = time.time()
        response_time = end_time - start_time
        
        data = response.json()
        raw_response = data.get("response", "")
        
        print(f"✅ Réponse reçue en {response_time:.1f}s")
        print(f"📊 Taille: {len(raw_response)} caractères")
        
        # Sauvegarder la réponse brute
        with open("llama2_13b_json_brut.txt", "w", encoding="utf-8") as f:
            f.write(raw_response)
        
        print(f"\n📁 Réponse sauvegardée dans 'llama2_13b_json_brut.txt'")
        
        # Analyse de la réponse
        print(f"\n🔍 ANALYSE DE LA RÉPONSE:")
        print("=" * 50)
        
        # 1. Recherche de JSON
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}")
        
        if json_start != -1 and json_end != -1:
            print(f"✅ JSON détecté: position {json_start} à {json_end}")
            json_content = raw_response[json_start:json_end+1]
            
            # Sauvegarder le JSON extrait
            with open("llama2_13b_json_extrait.txt", "w", encoding="utf-8") as f:
                f.write(json_content)
            
            print(f"📁 JSON extrait sauvegardé dans 'llama2_13b_json_extrait.txt'")
            
            # Test de validation JSON
            try:
                parsed_json = json.loads(json_content)
                print(f"✅ JSON valide !")
                print(f"📋 Clés trouvées: {list(parsed_json.keys())}")
                
                # Analyse détaillée des clés
                print(f"\n🔍 ANALYSE DÉTAILLÉE:")
                print("=" * 30)
                
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        print(f"   {key}: {len(value)} éléments")
                        if key == "hashtags" and len(value) < 10:
                            print(f"      ⚠️ INSUFFISANT: {len(value)} hashtags (attendu: 10-14)")
                        elif key == "broll_keywords" and len(value) < 20:
                            print(f"      ⚠️ INSUFFISANT: {len(value)} keywords (attendu: 20-25)")
                    else:
                        print(f"   {key}: {type(value).__name__} = '{value}'")
                
                # Affichage complet du JSON
                print(f"\n📄 JSON COMPLET GÉNÉRÉ:")
                print("=" * 30)
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON invalide: {e}")
                
        else:
            print(f"❌ Aucun JSON détecté dans la réponse")
            print(f"🔍 Contenu de la réponse:")
            print(f"   Début: {raw_response[:200]}...")
            print(f"   Fin: ...{raw_response[-200:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    capture_json_llama2_13b() 