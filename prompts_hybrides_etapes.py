# 🚀 PROMPTS UNIFIÉS QWEN3:8B (2 ÉTAPES) - CONTRAINTES SOUPLES

# ========================================
# ÉTAPE 1 : TITRE + HASHTAGS (Qwen3:8B)
# ========================================
PROMPT_ETAPE_1 = """⚠️ RÈGLES CRITIQUES — À RESPECTER ABSOLUMENT :
🚨 ENTRE 1 ET 3 titres.
🚨 ENTRE 10 ET 14 hashtags.
⚠️ Si tu ne respectes pas ces règles, la réponse sera REJETÉE.

You are a social media expert for TikTok and Instagram.
Generate ONLY title and hashtags from the transcript.

REQUIRED OUTPUT:
1. title: 1-3 short, catchy titles (≤60 chars), TikTok/Instagram style
2. hashtags: 10-14 varied hashtags (#keyword format), mix niche + trending

CRITICAL JSON OUTPUT REQUIREMENTS:
🚨 Output ONLY valid JSON, NO TEXT, NO EXPLANATIONS
🚨 JSON must start with {{ and end with }}
🚨 Pure JSON object only
🚨 ALL keys and values MUST be in double quotes
🚨 Example format: {{"title": ["Title 1"], "hashtags": ["#tag1", "#tag2"]}}

Transcript:
{text}

JSON:"""

# ========================================
# ÉTAPE 2 : DESCRIPTION + B-ROLL KEYWORDS (Qwen3:8B)
# ========================================
PROMPT_ETAPE_2 = """🚨 RÈGLES QUANTITATIVES CRITIQUES :
⚠️ TU DOIS GÉNÉRER ENTRE 1 ET 3 DESCRIPTIONS !
⚠️ TU DOIS GÉNÉRER ENTRE 24 ET 26 MOTS-CLÉS B-ROLL !
⚠️ CHAQUE CATÉGORIE DOIT CONTENIR ENTRE 4 ET 6 MOTS-CLÉS !
⚠️ SI TU NE RESPECTES PAS CES RÈGLES, LA RÉPONSE SERA REJETÉE !

You are a B-roll content expert for video production.
Generate ONLY description and B-roll keywords from the transcript.

REQUIRED OUTPUT:
1. description: 1-3 punchy sentences with call-to-action
2. broll_keywords: 24-26 keywords, 4-6 per category:
   - VISUAL ACTIONS: 4-6 mots-clés (1 base + 3-5 synonymes)
   - PEOPLE & ROLES: 4-6 mots-clés (1 base + 3-5 synonymes)
   - ENVIRONMENTS & PLACES: 4-6 mots-clés (1 base + 3-5 synonymes)
   - OBJECTS & PROPS: 4-6 mots-clés (1 base + 3-5 synonymes)
   - EMOTIONAL/CONTEXTUAL: 4-6 mots-clés (1 base + 3-5 synonymes)

RÈGLE ABSOLUE: 5 catégories × 4-6 mots-clés = 24-26 MOTS-CLÉS OBLIGATOIRES !

CRITICAL JSON OUTPUT REQUIREMENTS:
🚨 Output ONLY valid JSON, NO TEXT, NO EXPLANATIONS
🚨 JSON must start with {{ and end with }}
🚨 Pure JSON object only
🚨 ALL keys and values MUST be in double quotes
🚨 ENTRE 1-3 descriptions + 24-26 mots-clés répartis en 5 catégories de 4-6 mots-clés chacune

Transcript:
{text}

JSON:"""

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def get_prompt_etape_1(text):
    """Retourne le prompt de l'étape 1 avec le transcript"""
    return PROMPT_ETAPE_1.format(text=text)

def get_prompt_etape_2(text):
    """Retourne le prompt de l'étape 2 avec le transcript"""
    return PROMPT_ETAPE_2.format(text=text)

def get_prompt_info():
    """Retourne les informations sur les prompts"""
    return {
        "etape_1": {
            "taille": len(PROMPT_ETAPE_1),
            "modele_cible": "qwen3:8b",
            "objectif": "1-3 Titres + Hashtags (contraintes souples)"
        },
        "etape_2": {
            "taille": len(PROMPT_ETAPE_2),
            "modele_cible": "qwen3:8b", 
            "objectif": "1-3 Descriptions + 24-26 mots-clés B-roll (contraintes souples)"
        }
    }

if __name__ == "__main__":
    info = get_prompt_info()
    print("🚀 PROMPTS UNIFIÉS QWEN3:8B CRÉÉS (CONTRAINTES SOUPLES) :")
    print(f"📝 Étape 1: {info['etape_1']['taille']} caractères → {info['etape_1']['modele_cible']}")
    print(f"📝 Étape 2: {info['etape_2']['taille']} caractères → {info['etape_2']['modele_cible']}")
    print("🎯 Pipeline 100% Qwen3:8B + Normalisation JSON + Auto-correction")
    print("🎯 Stratégie: Prompt souple + Validation stricte côté code")
    print("🎯 Contraintes: 1-3 titres, 1-3 descriptions, 24-26 mots-clés B-roll") 