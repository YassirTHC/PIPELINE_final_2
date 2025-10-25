ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ PROMPTS UNIFIÃƒâ€°S QWEN3:8B (2 Ãƒâ€°TAPES) - CONTRAINTES SOUPLES

# ========================================
# Ãƒâ€°TAPE 1 : TITRE + HASHTAGS (Qwen3:8B)
# ========================================
PROMPT_ETAPE_1 = """Ã¢Å¡Â Ã¯Â¸Â RÃƒË†GLES CRITIQUES Ã¢â‚¬â€ Ãƒâ‚¬ RESPECTER ABSOLUMENT :
Ã°Å¸Å¡Â¨ ENTRE 1 ET 3 titres.
Ã°Å¸Å¡Â¨ ENTRE 10 ET 14 hashtags.
Ã¢Å¡Â Ã¯Â¸Â Si tu ne respectes pas ces rÃƒÂ¨gles, la rÃƒÂ©ponse sera REJETÃƒâ€°E.

You are a social media expert for TikTok and Instagram.
Generate ONLY title and hashtags from the transcript.

REQUIRED OUTPUT:
1. title: 1-3 short, catchy titles (Ã¢â€°Â¤60 chars), TikTok/Instagram style
2. hashtags: 10-14 varied hashtags (#keyword format), mix niche + trending

CRITICAL JSON OUTPUT REQUIREMENTS:
Ã°Å¸Å¡Â¨ Output ONLY valid JSON, NO TEXT, NO EXPLANATIONS
Ã°Å¸Å¡Â¨ JSON must start with {{ and end with }}
Ã°Å¸Å¡Â¨ Pure JSON object only
Ã°Å¸Å¡Â¨ ALL keys and values MUST be in double quotes
Ã°Å¸Å¡Â¨ Example format: {{"title": ["Title 1"], "hashtags": ["#tag1", "#tag2"]}}

Transcript:
{text}

JSON:"""

# ========================================
# Ãƒâ€°TAPE 2 : DESCRIPTION + B-ROLL KEYWORDS (Qwen3:8B)
# ========================================
PROMPT_ETAPE_2 = """Ã°Å¸Å¡Â¨ RÃƒË†GLES QUANTITATIVES CRITIQUES :
Ã¢Å¡Â Ã¯Â¸Â TU DOIS GÃƒâ€°NÃƒâ€°RER ENTRE 1 ET 3 DESCRIPTIONS !
Ã¢Å¡Â Ã¯Â¸Â TU DOIS GÃƒâ€°NÃƒâ€°RER ENTRE 24 ET 26 MOTS-CLÃƒâ€°S B-ROLL !
Ã¢Å¡Â Ã¯Â¸Â CHAQUE CATÃƒâ€°GORIE DOIT CONTENIR ENTRE 4 ET 6 MOTS-CLÃƒâ€°S !
Ã¢Å¡Â Ã¯Â¸Â SI TU NE RESPECTES PAS CES RÃƒË†GLES, LA RÃƒâ€°PONSE SERA REJETÃƒâ€°E !

You are a B-roll content expert for video production.
Generate ONLY description and B-roll keywords from the transcript.

REQUIRED OUTPUT:
1. description: 1-3 punchy sentences with call-to-action
2. broll_keywords: 24-26 keywords, 4-6 per category:
   - VISUAL ACTIONS: 4-6 mots-clÃƒÂ©s (1 base + 3-5 synonymes)
   - PEOPLE & ROLES: 4-6 mots-clÃƒÂ©s (1 base + 3-5 synonymes)
   - ENVIRONMENTS & PLACES: 4-6 mots-clÃƒÂ©s (1 base + 3-5 synonymes)
   - OBJECTS & PROPS: 4-6 mots-clÃƒÂ©s (1 base + 3-5 synonymes)
   - EMOTIONAL/CONTEXTUAL: 4-6 mots-clÃƒÂ©s (1 base + 3-5 synonymes)

RÃƒË†GLE ABSOLUE: 5 catÃƒÂ©gories Ãƒâ€” 4-6 mots-clÃƒÂ©s = 24-26 MOTS-CLÃƒâ€°S OBLIGATOIRES !

CRITICAL JSON OUTPUT REQUIREMENTS:
Ã°Å¸Å¡Â¨ Output ONLY valid JSON, NO TEXT, NO EXPLANATIONS
Ã°Å¸Å¡Â¨ JSON must start with {{ and end with }}
Ã°Å¸Å¡Â¨ Pure JSON object only
Ã°Å¸Å¡Â¨ ALL keys and values MUST be in double quotes
Ã°Å¸Å¡Â¨ ENTRE 1-3 descriptions + 24-26 mots-clÃƒÂ©s rÃƒÂ©partis en 5 catÃƒÂ©gories de 4-6 mots-clÃƒÂ©s chacune

Transcript:
{text}

JSON:"""

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def get_prompt_etape_1(text):
    """Retourne le prompt de l'ÃƒÂ©tape 1 avec le transcript"""
    return PROMPT_ETAPE_1.format(text=text)

def get_prompt_etape_2(text):
    """Retourne le prompt de l'ÃƒÂ©tape 2 avec le transcript"""
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
            "objectif": "1-3 Descriptions + 24-26 mots-clÃƒÂ©s B-roll (contraintes souples)"
        }
    }

if __name__ == "__main__":
    info = get_prompt_info()
    print("Ã°Å¸Å¡â‚¬ PROMPTS UNIFIÃƒâ€°S QWEN3:8B CRÃƒâ€°Ãƒâ€°S (CONTRAINTES SOUPLES) :")
    print(f"Ã°Å¸â€œÂ Ãƒâ€°tape 1: {info['etape_1']['taille']} caractÃƒÂ¨res Ã¢â€ â€™ {info['etape_1']['modele_cible']}")
    print(f"Ã°Å¸â€œÂ Ãƒâ€°tape 2: {info['etape_2']['taille']} caractÃƒÂ¨res Ã¢â€ â€™ {info['etape_2']['modele_cible']}")
    print("Ã°Å¸Å½Â¯ Pipeline 100% Qwen3:8B + Normalisation JSON + Auto-correction")
    print("Ã°Å¸Å½Â¯ StratÃƒÂ©gie: Prompt souple + Validation stricte cÃƒÂ´tÃƒÂ© code")
    print("Ã°Å¸Å½Â¯ Contraintes: 1-3 titres, 1-3 descriptions, 24-26 mots-clÃƒÂ©s B-roll") 

