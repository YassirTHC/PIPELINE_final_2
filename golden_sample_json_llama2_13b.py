#!/usr/bin/env python3
"""
🏆 GOLDEN SAMPLE JSON POUR LLAMA2:13B
Exemple JSON parfait avec exactement 20-25 keywords et 10-14 hashtags
"""

def generate_golden_sample():
    """Génère le golden sample JSON parfait"""
    
    golden_sample = {
        "title": "EMDR Movement Sensation: Transform Trauma Through Motion",
        "description": "Discover how lateralized movements and EMDR therapy can reshape your relationship with trauma and unlock lasting stress relief.",
        "hashtags": [
            "#EMDRtherapy", "#traumarecovery", "#stressrelief", "#mentalhealth",
            "#movementtherapy", "#healing", "#mindfulness", "#wellness",
            "#therapy", "#growth", "#recovery", "#mentalwellness"
        ],
        "broll_keywords": [
            # VISUAL ACTIONS (8-12 keywords)
            {
                "category": "VISUAL ACTIONS",
                "base": "lateralized movements",
                "synonyms": ["side-to-side", "front-to-back", "up-down", "diagonal"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "exercising",
                "synonyms": ["stretching", "yoga", "meditation", "breathing"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "writing",
                "synonyms": ["journaling", "note-taking", "planning", "reflecting"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "walking",
                "synonyms": ["strolling", "hiking", "jogging", "running"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "talking",
                "synonyms": ["conversing", "discussing", "sharing", "communicating"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "crying",
                "synonyms": ["sobbing", "weeping", "emotional release", "tears"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "laughing",
                "synonyms": ["smiling", "chuckling", "joy", "happiness"]
            },
            {
                "category": "VISUAL ACTIONS",
                "base": "cooking",
                "synonyms": ["preparing", "chopping", "stirring", "serving"]
            },
            
            # PEOPLE & ROLES (8-10 keywords)
            {
                "category": "PEOPLE & ROLES",
                "base": "therapist",
                "synonyms": ["counselor", "mental health professional", "coach", "guide"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "patient",
                "synonyms": ["client", "individual", "person", "seeker"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "family",
                "synonyms": ["parents", "children", "siblings", "loved ones"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "athlete",
                "synonyms": ["runner", "yogi", "fitness enthusiast", "active person"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "student",
                "synonyms": ["learner", "scholar", "apprentice", "knowledge seeker"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "professional",
                "synonyms": ["expert", "specialist", "practitioner", "consultant"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "elderly",
                "synonyms": ["senior", "mature adult", "wise person", "experienced"]
            },
            {
                "category": "PEOPLE & ROLES",
                "base": "child",
                "synonyms": ["young person", "kid", "youth", "minor"]
            },
            
            # ENVIRONMENTS & PLACES (8-10 keywords)
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "clinic",
                "synonyms": ["hospital", "medical office", "therapy room", "treatment center"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "gym",
                "synonyms": ["fitness center", "training facility", "workout space", "exercise room"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "nature trail",
                "synonyms": ["hiking path", "forest", "park", "outdoor space"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "city street",
                "synonyms": ["urban landscape", "downtown", "metropolitan area", "city center"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "home office",
                "synonyms": ["workspace", "desk", "computer area", "study room"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "hospital room",
                "synonyms": ["medical facility", "examination room", "treatment area", "healthcare space"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "therapy office",
                "synonyms": ["counseling room", "mental health space", "consultation area", "healing environment"]
            },
            {
                "category": "ENVIRONMENTS & PLACES",
                "base": "outdoor space",
                "synonyms": ["open air", "natural setting", "landscape", "environment"]
            },
            
            # OBJECTS & PROPS (6-8 keywords)
            {
                "category": "OBJECTS & PROPS",
                "base": "equipment",
                "synonyms": ["medical devices", "therapy tools", "instruments", "apparatus"]
            },
            {
                "category": "OBJECTS & PROPS",
                "base": "notebook",
                "synonyms": ["journal", "planner", "writing pad", "diary"]
            },
            {
                "category": "OBJECTS & PROPS",
                "base": "phone",
                "synonyms": ["mobile device", "smartphone", "telephone", "communication device"]
            },
            {
                "category": "OBJECTS & PROPS",
                "base": "exercise mat",
                "synonyms": ["yoga mat", "fitness mat", "pilates mat", "workout surface"]
            },
            {
                "category": "OBJECTS & PROPS",
                "base": "weights",
                "synonyms": ["dumbbells", "resistance equipment", "fitness tools", "strength equipment"]
            },
            {
                "category": "OBJECTS & PROPS",
                "base": "car",
                "synonyms": ["vehicle", "automobile", "transportation", "motor vehicle"]
            },
            
            # EMOTIONAL/CONTEXTUAL (6-8 keywords)
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "healing",
                "synonyms": ["recovery", "therapy", "treatment", "restoration"]
            },
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "stress relief",
                "synonyms": ["relaxation", "calmness", "peace", "tranquility"]
            },
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "growth",
                "synonyms": ["self-improvement", "personal development", "advancement", "progress"]
            },
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "trauma recovery",
                "synonyms": ["overcoming trauma", "resilience", "empowerment", "transformation"]
            },
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "wellness",
                "synonyms": ["health", "well-being", "self-care", "vitality"]
            },
            {
                "category": "EMOTIONAL/CONTEXTUAL",
                "base": "mindfulness",
                "synonyms": ["awareness", "presence", "consciousness", "attentiveness"]
            }
        ]
    }
    
    return golden_sample

def analyze_golden_sample():
    """Analyse le golden sample pour validation"""
    sample = generate_golden_sample()
    
    print("🏆 GOLDEN SAMPLE JSON POUR LLAMA2:13B")
    print("=" * 60)
    
    # Analyse des quantités
    hashtags_count = len(sample["hashtags"])
    keywords_count = len(sample["broll_keywords"])
    
    print(f"📊 ANALYSE QUANTITATIVE:")
    print(f"   Hashtags: {hashtags_count} (attendu: 10-14)")
    print(f"   Keywords B-roll: {keywords_count} (attendu: 20-25)")
    
    # Analyse des catégories
    categories = {}
    for kw in sample["broll_keywords"]:
        cat = kw["category"]
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print(f"\n📋 RÉPARTITION PAR CATÉGORIES:")
    for cat, count in categories.items():
        print(f"   {cat}: {count} keywords")
    
    # Validation
    hashtags_ok = 10 <= hashtags_count <= 14
    keywords_ok = 20 <= keywords_count <= 25
    categories_ok = all(count >= 4 for count in categories.values())
    
    print(f"\n✅ VALIDATION:")
    print(f"   Hashtags 10-14: {'✅' if hashtags_ok else '❌'}")
    print(f"   Keywords 20-25: {'✅' if keywords_ok else '❌'}")
    print(f"   Catégories ≥4: {'✅' if categories_ok else '❌'}")
    
    if hashtags_ok and keywords_ok and categories_ok:
        print(f"\n🏆 GOLDEN SAMPLE VALIDÉ - PRÊT POUR LLAMA2:13B !")
    
    return sample

def get_prompt_enhancement():
    """Génère le texte d'amélioration du prompt"""
    
    enhancement = """
⚠️ RÈGLES DE QUANTITÉ À RESPECTER STRICTEMENT :
1. "hashtags" doit contenir ENTRE 10 et 14 éléments EXACTEMENT (jamais moins, jamais plus).
2. "broll_keywords" doit contenir ENTRE 20 et 25 éléments EXACTEMENT.
3. Chaque "category" de "broll_keywords" doit avoir MINIMUM 4 mots-clés.
4. Si tu ne respectes pas ces règles, la réponse sera REJETÉE.

📋 EXEMPLE DE STRUCTURE PARFAITE (à reproduire exactement) :
{
  "title": "EMDR Movement Sensation: Transform Trauma Through Motion",
  "description": "Discover how lateralized movements and EMDR therapy can reshape your relationship with trauma and unlock lasting stress relief.",
  "hashtags": ["#EMDRtherapy", "#traumarecovery", "#stressrelief", "#mentalhealth", "#movementtherapy", "#healing", "#mindfulness", "#wellness", "#therapy", "#growth", "#recovery", "#mentalwellness"],
  "broll_keywords": [
    {"category": "VISUAL ACTIONS", "base": "lateralized movements", "synonyms": ["side-to-side", "front-to-back", "up-down"]},
    {"category": "PEOPLE & ROLES", "base": "therapist", "synonyms": ["counselor", "mental health professional", "coach"]}
    // ... TOTAL: 20-25 éléments répartis en 5 catégories
  ]
}

🚨 RESPECTE EXACTEMENT cette structure et ces quantités !
"""
    
    return enhancement

if __name__ == "__main__":
    # Générer et analyser le golden sample
    sample = analyze_golden_sample()
    
    # Afficher le JSON complet
    print(f"\n📄 GOLDEN SAMPLE JSON COMPLET:")
    print("=" * 60)
    import json
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    # Afficher l'amélioration du prompt
    print(f"\n🔧 AMÉLIORATION DU PROMPT:")
    print("=" * 60)
    print(get_prompt_enhancement())
    
    # Sauvegarder le golden sample
    with open("golden_sample_llama2_13b.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Golden sample sauvegardé dans 'golden_sample_llama2_13b.json'") 