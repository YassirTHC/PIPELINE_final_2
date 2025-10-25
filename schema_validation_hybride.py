ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ SCHÃƒâ€°MA DE VALIDATION PYDANTIC POUR PIPELINE HYBRIDE

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import json

# ========================================
# SCHÃƒâ€°MA Ãƒâ€°TAPE 1 : TITRES + HASHTAGS
# ========================================
class Etape1Schema(BaseModel):
    title: List[str] = Field(..., min_items=3, max_items=5, description="3-5 titres")
    hashtags: List[str] = Field(..., min_items=10, max_items=14, description="10-14 hashtags")
    
    @validator('title')
    def validate_titles(cls, v):
        for title in v:
            if len(title) > 60:
                raise ValueError(f"Titre trop long: {title} ({len(title)} > 60)")
            if not title.strip():
                raise ValueError("Titre vide dÃƒÂ©tectÃƒÂ©")
        return v
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        for hashtag in v:
            if not hashtag.startswith('#'):
                raise ValueError(f"Hashtag doit commencer par #: {hashtag}")
            if ' ' in hashtag:
                raise ValueError(f"Hashtag ne doit pas contenir d'espaces: {hashtag}")
        return v

# ========================================
# SCHÃƒâ€°MA Ãƒâ€°TAPE 2 : DESCRIPTIONS + B-ROLL KEYWORDS
# ========================================
class BrollKeywordItem(BaseModel):
    category: str = Field(..., description="CatÃƒÂ©gorie du mot-clÃƒÂ©")
    base: str = Field(..., description="Mot-clÃƒÂ© de base")
    synonyms: List[str] = Field(..., min_items=4, max_items=4, description="Exactement 4 synonymes")
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = [
            "VISUAL ACTIONS", "PEOPLE & ROLES", "ENVIRONMENTS & PLACES", 
            "OBJECTS & PROPS", "EMOTIONAL/CONTEXTUAL"
        ]
        if v not in valid_categories:
            raise ValueError(f"CatÃƒÂ©gorie invalide: {v}. Doit ÃƒÂªtre une de: {valid_categories}")
        return v

class Etape2Schema(BaseModel):
    description: List[str] = Field(..., min_items=2, max_items=3, description="2-3 descriptions")
    broll_keywords: List[BrollKeywordItem] = Field(..., min_items=25, max_items=25, description="Exactement 25 mots-clÃƒÂ©s")
    
    @validator('broll_keywords')
    def validate_broll_keywords_distribution(cls, v):
        # VÃƒÂ©rifier qu'on a exactement 5 mots-clÃƒÂ©s par catÃƒÂ©gorie
        categories = {}
        for item in v:
            if item.category not in categories:
                categories[item.category] = 0
            categories[item.category] += 1
        
        expected_categories = ["VISUAL ACTIONS", "PEOPLE & ROLES", "ENVIRONMENTS & PLACES", "OBJECTS & PROPS", "EMOTIONAL/CONTEXTUAL"]
        
        for category in expected_categories:
            if category not in categories:
                raise ValueError(f"CatÃƒÂ©gorie manquante: {category}")
            if categories[category] != 5:
                raise ValueError(f"CatÃƒÂ©gorie {category}: {categories[category]} mots-clÃƒÂ©s au lieu de 5")
        
        return v

# ========================================
# SCHÃƒâ€°MA FINAL COMBINÃƒâ€°
# ========================================
class FinalSchema(BaseModel):
    title: List[str] = Field(..., min_items=3, max_items=5)
    description: List[str] = Field(..., min_items=2, max_items=3)
    hashtags: List[str] = Field(..., min_items=10, max_items=14)
    broll_keywords: List[BrollKeywordItem] = Field(..., min_items=25, max_items=25)

# ========================================
# FONCTIONS DE VALIDATION
# ========================================
def validate_etape_1(json_str: str) -> Dict[str, Any]:
    """Valide et parse l'ÃƒÂ©tape 1"""
    try:
        data = json.loads(json_str)
        validated = Etape1Schema(**data)
        return {"success": True, "data": validated.dict(), "errors": None}
    except Exception as e:
        return {"success": False, "data": None, "errors": str(e)}

def validate_etape_2(json_str: str) -> Dict[str, Any]:
    """Valide et parse l'ÃƒÂ©tape 2"""
    try:
        data = json.loads(json_str)
        validated = Etape2Schema(**data)
        return {"success": True, "data": validated.dict(), "errors": None}
    except Exception as e:
        return {"success": False, "data": None, "errors": str(e)}

def combine_etapes(etape1_data: Dict, etape2_data: Dict) -> Dict[str, Any]:
    """Combine les rÃƒÂ©sultats des deux ÃƒÂ©tapes"""
    try:
        combined = {
            "title": etape1_data["title"],
            "hashtags": etape1_data["hashtags"],
            "description": etape2_data["description"],
            "broll_keywords": etape2_data["broll_keywords"]
        }
        validated = FinalSchema(**combined)
        return {"success": True, "data": validated.dict(), "errors": None}
    except Exception as e:
        return {"success": False, "data": None, "errors": str(e)}

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def get_schema_info():
    """Retourne les informations sur les schÃƒÂ©mas"""
    return {
        "etape_1": {
            "champs": ["title", "hashtags"],
            "contraintes": "3-5 titres, 10-14 hashtags",
            "validation": "Longueur titres Ã¢â€°Â¤60, format hashtags #keyword"
        },
        "etape_2": {
            "champs": ["description", "broll_keywords"],
            "contraintes": "2-3 descriptions, 25 mots-clÃƒÂ©s (5 par catÃƒÂ©gorie)",
            "validation": "5 catÃƒÂ©gories, 4 synonymes par mot-clÃƒÂ©"
        },
        "final": {
            "champs": ["title", "description", "hashtags", "broll_keywords"],
            "contraintes": "Toutes les contraintes des ÃƒÂ©tapes 1 et 2",
            "validation": "SchÃƒÂ©ma complet et cohÃƒÂ©rent"
        }
    }

if __name__ == "__main__":
    info = get_schema_info()
    print("Ã°Å¸Å¡â‚¬ SCHÃƒâ€°MAS DE VALIDATION CRÃƒâ€°Ãƒâ€°S :")
    for etape, details in info.items():
        print(f"Ã°Å¸â€œâ€¹ {etape.upper()}: {details['champs']}")
        print(f"   Contraintes: {details['contraintes']}")
        print(f"   Validation: {details['validation']}")
        print() 

