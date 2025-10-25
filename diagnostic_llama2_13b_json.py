ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Ã°Å¸â€Â DIAGNOSTIC COMPLET LLAMA2:13B - ANALYSE JSON
Capture et analyse la rÃƒÂ©ponse complÃƒÂ¨te pour identifier les problÃƒÂ¨mes de formatage
"""

import requests
import json
import time

def test_llama2_13b_raw_response():
    """Test direct avec l'API Ollama pour capturer la rÃƒÂ©ponse brute"""
    print("Ã°Å¸â€Â DIAGNOSTIC COMPLET LLAMA2:13B - RÃƒâ€°PONSE BRUTE")
    print("=" * 70)
    
    # Prompt de test (votre prompt complet)
    prompt = (
        "You are a social media strategist and B-roll content expert for TikTok and Instagram.\n"
        "From the provided transcript, generate all elements optimized for virality AND B-roll video selection.\n"
        "Write in English only. Ensure all keywords can be directly matched to stock video footage on Pexels, Pixabay, Archive.org, or other platforms.\n\n"
        "Your goal is to **maximize visual relevance, search engine performance, and engagement potential** on social media platforms.\n\n"
        "REQUIRED OUTPUT:\n"
        "1. title: short, catchy (<= 60 characters), TikTok/Instagram Reels style, generate 2-3 variants\n"
        "2. description: 1-2 punchy sentences with implicit call-to-action, generate 2-3 variants\n"
        "3. hashtags: 10-14 varied hashtags, exact format #keyword (no spaces), mix niche + trending, include seasonal/trend variations\n"
        "4. broll_keywords: 20-25 base keywords, each with 2-3 synonyms, distributed across:\n\n"
        "   VISUAL ACTIONS (8-12): specific, filmable actions (e.g., running, crying, laughing, writing, exercising, meditating, cooking, driving) with synonyms\n"
        "   PEOPLE & ROLES (8-10): specific person types (therapist, patient, family, athlete, student, professional, elderly, child) with diverse representations and synonyms\n"
        "   ENVIRONMENTS & PLACES (8-10): concrete locations (hospital room, therapy office, gym, nature trail, city street, home office) including indoor/outdoor, with synonyms\n"
        "   OBJECTS & PROPS (6-8): tangible items appearing in stock footage (notebook, medical equipment, exercise mat, car, phone, weights) with synonyms\n"
        "   EMOTIONAL/CONTEXTUAL (6-8): visually representable concepts (healing, stress relief, growth, trauma recovery, wellness) with synonyms and emotion-rich variations\n\n"
        "OPTIMIZATION RULES:\n"
        "- Maintain **domain-relevant terminology** based on transcript context (healthcare, finance, sports, education, business)\n"
        "- Include **2-3 synonym variations per keyword** for broader stock matching\n"
        "- Generate **fallback visuals** if transcript is vague: universal, dynamic, engaging, contextually relevant (people interacting, active nature, urban lifestyle, group activities)\n"
        "- Avoid static or overly generic clips (plain landscapes, empty streets)\n"
        "- Favor **emotionally resonant and trend-adjacent terms** for viral engagement\n"
        "- Include seasonal, cultural, and trend-aware keywords where appropriate\n"
        "- Generate **multiple options per category** when possible for selection/scoring\n"
        "- Use **hierarchical JSON format**: {\"base\": \"keyword\", \"synonyms\": [\"syn1\", \"syn2\", \"syn3\"]}\n"
        "- Check **coherence** of keywords with transcript; avoid off-context or abstract terms\n\n"
        "ENHANCED FEATURES FOR LLAMA2:13B:\n"
        "- Provide **2-3 variants of title, description, and keyword sets** for A/B testing\n"
        "- Include **emotionally rich, story-driven B-roll suggestions** for narrative impact\n"
        "- Suggest **trend-aware hashtags and visual keywords**\n"
        "- Score or rank B-roll suggestions by **visual engagement potential**\n"
        "- Ensure **strict hierarchical JSON output** leveraging llama2:13b's long-context reliability\n"
        "- Provide a **reportable structure**: number of keywords per category, synonyms included, fallback usage, quality indicators\n\n"
        "JSON STRUCTURE:\n"
        "- Output **compact, hierarchical JSON** only, with keys: title, description, hashtags, broll_keywords\n"
        "- Ensure **all categories are represented** and structured\n"
        "- Example format:\n"
        "{\n"
        "  \"title\": [\"Title variant 1\", \"Title variant 2\"],\n"
        "  \"description\": [\"Description variant 1\", \"Description variant 2\"],\n"
        "  \"hashtags\": [\"#keyword1\", \"#keyword2\", \"#keyword3\"],\n"
        "  \"broll_keywords\": [\n"
        "    {\"category\": \"VISUAL ACTIONS\", \"base\": \"running\", \"synonyms\": [\"jogging\", \"sprinting\", \"marathon\"]},\n"
        "    {\"category\": \"PEOPLE & ROLES\", \"base\": \"therapist\", \"synonyms\": [\"counselor\", \"mental health professional\"]},\n"
        "    {\"category\": \"ENVIRONMENTS & PLACES\", \"base\": \"gym\", \"synonyms\": [\"fitness center\", \"training facility\"]},\n"
        "    {\"category\": \"OBJECTS & PROPS\", \"base\": \"notebook\", \"synonyms\": [\"journal\", \"planner\", \"writing pad\"]},\n"
        "    {\"category\": \"EMOTIONAL/CONTEXTUAL\", \"base\": \"stress relief\", \"synonyms\": [\"relaxation\", \"calmness\", \"mental wellness\"]}\n"
        "  ]\n"
        "}\n\n"
        "Respond ONLY in this JSON format. **Do not modify any other part of the pipeline or code.** Focus exclusively on generating titles, descriptions, hashtags, and B-roll keywords according to these specifications.\n\n"
        "Transcript:\n"
        "EMDR movement sensation reprocessing lateralized movements people doing clinic got goofy looking thing while stress and rationale coupling a low stress state the recall of trauma it's gonna allow people reshape relationship trauma it's a tolerate that discomfort and EMDR clinical colleagues tell me works best fairly well defined traumas\n\n"
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
        with open("llama2_13b_response_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_response)
        
        print(f"\nÃ°Å¸â€œÂ RÃƒÂ©ponse sauvegardÃƒÂ©e dans 'llama2_13b_response_raw.txt'")
        
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
            with open("llama2_13b_json_extracted.txt", "w", encoding="utf-8") as f:
                f.write(json_content)
            
            print(f"Ã°Å¸â€œÂ JSON extrait sauvegardÃƒÂ© dans 'llama2_13b_json_extracted.txt'")
            
            # Test de validation JSON
            try:
                parsed_json = json.loads(json_content)
                print(f"Ã¢Å“â€¦ JSON valide !")
                print(f"Ã°Å¸â€œâ€¹ ClÃƒÂ©s trouvÃƒÂ©es: {list(parsed_json.keys())}")
                
                # Analyse des clÃƒÂ©s
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        print(f"   {key}: {len(value)} ÃƒÂ©lÃƒÂ©ments")
                    else:
                        print(f"   {key}: {type(value).__name__}")
                
            except json.JSONDecodeError as e:
                print(f"Ã¢ÂÅ’ JSON invalide: {e}")
                print(f"Ã°Å¸â€Â ProblÃƒÂ¨me ÃƒÂ  la ligne: {e.lineno}, colonne: {e.colno}")
                print(f"Ã°Å¸â€œÂ Message: {e.msg}")
                
                # Afficher le contexte du problÃƒÂ¨me
                lines = json_content.split('\n')
                if e.lineno <= len(lines):
                    problem_line = lines[e.lineno - 1]
                    print(f"Ã°Å¸Å¡Â¨ Ligne problÃƒÂ©matique: {problem_line}")
                
        else:
            print(f"Ã¢ÂÅ’ Aucun JSON dÃƒÂ©tectÃƒÂ© dans la rÃƒÂ©ponse")
            print(f"Ã°Å¸â€Â Contenu de la rÃƒÂ©ponse:")
            print(f"   DÃƒÂ©but: {raw_response[:200]}...")
            print(f"   Fin: ...{raw_response[-200:]}")
        
        # 2. Analyse du format
        print(f"\nÃ°Å¸â€œÅ  ANALYSE DU FORMAT:")
        print("=" * 30)
        
        has_curly_braces = "{" in raw_response and "}" in raw_response
        has_square_brackets = "[" in raw_response and "]" in raw_response
        has_quotes = '"' in raw_response
        
        print(f"   Accolades {{}}: {'Ã¢Å“â€¦' if has_curly_braces else 'Ã¢ÂÅ’'}")
        print(f"   Crochets []: {'Ã¢Å“â€¦' if has_square_brackets else 'Ã¢ÂÅ’'}")
        print(f"   Guillemets \": {'Ã¢Å“â€¦' if has_quotes else 'Ã¢ÂÅ’'}")
        
        # 3. Recherche de patterns
        print(f"\nÃ°Å¸â€Â PATTERNS DÃƒâ€°TECTÃƒâ€°S:")
        print("=" * 30)
        
        if "title" in raw_response.lower():
            print(f"   Ã¢Å“â€¦ 'title' trouvÃƒÂ©")
        else:
            print(f"   Ã¢ÂÅ’ 'title' manquant")
            
        if "description" in raw_response.lower():
            print(f"   Ã¢Å“â€¦ 'description' trouvÃƒÂ©")
        else:
            print(f"   Ã¢ÂÅ’ 'description' manquant")
            
        if "hashtags" in raw_response.lower():
            print(f"   Ã¢Å“â€¦ 'hashtags' trouvÃƒÂ©")
        else:
            print(f"   Ã¢ÂÅ’ 'hashtags' manquant")
            
        if "broll_keywords" in raw_response.lower():
            print(f"   Ã¢Å“â€¦ 'broll_keywords' trouvÃƒÂ©")
        else:
            print(f"   Ã¢ÂÅ’ 'broll_keywords' manquant")
        
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur: {e}")
        return False

if __name__ == "__main__":
    test_llama2_13b_raw_response() 

