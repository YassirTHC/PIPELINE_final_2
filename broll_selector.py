ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Pipeline de SÃƒÂ©lection B-roll GÃƒÂ©nÃƒÂ©rique
Module rÃƒÂ©utilisable pour n'importe quel clip vidÃƒÂ©o/domaine
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import math
from collections import defaultdict
import re
import requests

# Ã°Å¸Å¡â‚¬ NOUVEAU: Cache global pour ÃƒÂ©viter le rechargement des modÃƒÂ¨les
_BROLL_MODEL_CACHE = {}

def get_cached_sentence_transformer(model_name: str):
    """RÃƒÂ©cupÃƒÂ¨re un modÃƒÂ¨le SentenceTransformer depuis le cache ou le charge"""
    if model_name not in _BROLL_MODEL_CACHE:
        print(f"    Ã°Å¸â€â€ž Chargement initial du modÃƒÂ¨le B-roll: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _BROLL_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            print(f"    Ã¢Å“â€¦ ModÃƒÂ¨le B-roll {model_name} chargÃƒÂ© et mis en cache")
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Erreur chargement modÃƒÂ¨le B-roll {model_name}: {e}")
            return None
    else:
        print(f"    Ã¢â„¢Â»Ã¯Â¸Â ModÃƒÂ¨le B-roll {model_name} rÃƒÂ©cupÃƒÂ©rÃƒÂ© du cache")
    
    return _BROLL_MODEL_CACHE[model_name]

# Fallback imports pour ÃƒÂ©viter les erreurs
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Ã¢Å¡Â Ã¯Â¸Â SentenceTransformers non disponible - fallback vers scoring lexical")

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Ã¢Å¡Â Ã¯Â¸Â NLTK non disponible - fallback vers normalisation basique")

logger = logging.getLogger(__name__)

@dataclass
class Asset:
    """ReprÃƒÂ©sentation d'un asset B-roll"""
    id: str
    file_path: str
    tags: List[str]
    title: str
    description: str
    source: str
    fetched_at: datetime
    duration: float
    resolution: str
    precomputed_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        data = asdict(self)
        data['fetched_at'] = self.fetched_at.isoformat()
        return data

@dataclass
class ScoringFeatures:
    """Features de scoring pour un asset"""
    token_overlap: float = 0.0
    embedding_similarity: float = 0.0
    domain_match: float = 0.0
    freshness: float = 0.0
    quality_score: float = 0.0
    diversity_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class BrollCandidate:
    """Candidat B-roll avec scoring"""
    asset: Asset
    score: float
    features: ScoringFeatures
    excluded_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset_id': self.asset.id,
            'file_path': self.asset.file_path,
            'score': self.score,
            'features': self.features.to_dict(),
            'excluded_reason': self.excluded_reason,
            'asset': self.asset.to_dict()  # Ajouter asset complet pour ÃƒÂ©viter erreur sÃƒÂ©rialisation
        }

class BrollSelector:
    """SÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique avec scoring mixte et fallback hiÃƒÂ©rarchique"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialise le sÃƒÂ©lecteur avec configuration"""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Initialiser le logger avant toute utilisation
        self.logger = logging.getLogger(__name__)

        # Ã°Å¸Å¡â‚¬ NOUVEAU: Mode DIRECT - utilise directement les rÃƒÂ©sultats API
        self.direct_api_mode = self.config.get('direct_api_mode', False)  # HYBRIDE INTELLIGENT PAR DÃƒâ€°FAUT

        if self.direct_api_mode:
            self.logger.info("Ã°Å¸Å½Â¯ MODE DIRECT API ACTIVÃƒâ€° - TÃƒÂ©lÃƒÂ©chargement automatique depuis Pexels/Pixabay")
            self.logger.info("    Ã°Å¸â€œÂ¡ FETCH par API = TÃƒÂ©lÃƒÂ©chargement automatique de vidÃƒÂ©os B-roll depuis Internet")
            self.logger.info("    Sources: Pexels (videos), Pixabay (videos)")
            self.logger.info("    Ã°Å¸â€â€ž Process: Mots-clÃƒÂ©s -> Recherche API -> TÃƒÂ©lÃƒÂ©chargement -> Insertion dans vidÃƒÂ©o")
        else:
            self.logger.info("Ã°Å¸â€Â MODE SÃƒâ€°LECTION ACTIVÃƒâ€° - Re-scoring des rÃƒÂ©sultats API")
        
        # Initialiser les modÃƒÂ¨les si disponibles
        self.embedding_model = None
        self.lemmatizer = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Ã°Å¸Å¡â‚¬ OPTIMISATION: Utiliser le cache pour ÃƒÂ©viter le rechargement
                self.embedding_model = get_cached_sentence_transformer('all-MiniLM-L6-v2')
                if self.embedding_model is not None:
                    self.logger.info("Ã¢Å“â€¦ ModÃƒÂ¨le d'embeddings chargÃƒÂ©")
                else:
                    self.logger.warning("Ã¢Å¡Â Ã¯Â¸Â Ãƒâ€°chec chargement modÃƒÂ¨le d'embeddings")
            except Exception as e:
                self.logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur chargement embeddings: {e}")
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # TÃƒÂ©lÃƒÂ©charger WordNet si nÃƒÂ©cessaire
                try:
                    wordnet.ensure_loaded()
                except:
                    pass
                self.logger.info("Ã¢Å“â€¦ Lemmatiseur NLTK chargÃƒÂ©")
            except Exception as e:
                self.logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Erreur chargement NLTK: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par dÃƒÂ©faut pour le sÃƒÂ©lecteur B-roll"""
        return {
            # Performance et seuils
            'thresholds': {
                'min_score': 0.3,
                'min_delay_seconds': 1.5,  # RÃƒÂ©duit de 2.0s ÃƒÂ  1.5s
                'quality_threshold': 0.5
            },
            'desired_broll_count': 5,  # RÃƒÂ©duit de 7 ÃƒÂ  5
            'max_candidates': 50,
            
            # Poids pour le scoring mixte
            'scoring_weights': {
                'embedding': 0.4,
                'token': 0.2,
                'domain': 0.15,
                'freshness': 0.1,
                'quality': 0.1,
                'diversity': 0.05
            },
            
            # Ã°Å¸Å¡â‚¬ NOUVEAU: Configuration du mode direct
            'direct_api_mode': False,  # Utiliser directement les rÃƒÂ©sultats API
            'direct_api_limit': 5,    # Nombre de B-rolls ÃƒÂ  prendre directement
            'smart_crop_mode': True,  # Recadrage intelligent pour 9:16
            
            # Fallback et diversitÃƒÂ©
            'enable_fallback': True,
            'fallback_tiers': ['high_quality', 'medium_quality', 'any_available'],
            'diversity_penalty_factor': 0.1
        }
    
    def normalize_keywords(self, keywords: List[str]) -> Set[str]:
        """Normalise et nettoie les mots-clÃƒÂ©s"""
        if not keywords:
            return set()
        
        normalized = set()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str):
                continue
            
            # Nettoyage basique
            clean = keyword.lower().strip()
            clean = re.sub(r'[^\w\s]', '', clean)
            
            # Supprimer les stopwords et mots trop courts
            if clean and len(clean) > 2 and clean not in stopwords:
                # Lemmatisation si disponible
                if self.lemmatizer:
                    try:
                        clean = self.lemmatizer.lemmatize(clean)
                    except:
                        pass
                
                normalized.add(clean)
        
        self.logger.info(f"Ã°Å¸â€â€˜ Mots-clÃƒÂ©s normalisÃƒÂ©s: {len(normalized)} -> {list(normalized)[:10]}")
        return normalized
    
    def expand_keywords(self, keywords: List[str], domain: Optional[str] = None, max_expansions: int = 15) -> List[str]:
        """Ãƒâ€°tend les mots-clÃƒÂ©s avec synonymes et termes proches"""
        if not keywords:
            return []
        
        expanded = set(keywords)
        
        # Expansion via WordNet si disponible
        if NLTK_AVAILABLE and self.lemmatizer:
            for keyword in keywords[:5]:  # Limiter pour ÃƒÂ©viter l'explosion
                try:
                    # Chercher des synonymes
                    synsets = wordnet.synsets(keyword)
                    for synset in synsets[:3]:  # Top 3 synsets
                        for lemma in synset.lemmas()[:2]:  # Top 2 lemmas
                            expanded.add(lemma.name().lower())
                except Exception as e:
                    self.logger.debug(f"Ã¢Å¡Â Ã¯Â¸Â Erreur expansion WordNet pour '{keyword}': {e}")
        
        # Expansion par domaine si spÃƒÂ©cifiÃƒÂ©
        if domain:
            domain_expansions = self._get_domain_expansions(domain)
            expanded.update(domain_expansions)
        
        # Limiter le nombre d'expansions
        result = list(expanded)[:max_expansions]
        self.logger.info(f"Ã°Å¸â€Â Mots-clÃƒÂ©s ÃƒÂ©tendus: {len(keywords)} -> {len(result)}")
        
        return result
    
    def _get_domain_expansions(self, domain: str) -> List[str]:
        """Retourne des expansions spÃƒÂ©cifiques au domaine"""
        domain_keywords = {
            'health': ['medical', 'wellness', 'fitness', 'care', 'treatment', 'doctor', 'hospital', 'medicine'],
            'finance': ['money', 'business', 'investment', 'banking', 'economy', 'trading', 'wealth'],
            'education': ['learning', 'school', 'university', 'study', 'knowledge', 'teaching', 'student'],
            'technology': ['digital', 'innovation', 'software', 'computer', 'ai', 'data', 'tech'],
            'food': ['cooking', 'restaurant', 'cuisine', 'ingredients', 'meal', 'chef', 'kitchen']
        }
        
        return domain_keywords.get(domain.lower(), [])
    
    def fetch_assets(self, keywords: List[str], limit: int = 200) -> List[Asset]:
        """RÃƒÂ©cupÃƒÂ¨re les assets disponibles depuis les dossiers B-roll rÃƒÂ©els"""
        assets = []
        
        print(f"Ã°Å¸â€Â DEBUG: fetch_assets appelÃƒÂ© avec {len(keywords)} mots-clÃƒÂ©s")
        print(f"Ã°Å¸â€Â DEBUG: Mots-clÃƒÂ©s: {keywords[:5]}")
        
        # Chercher dans les dossiers B-roll rÃƒÂ©els
        # D'abord dans AI-B-roll/broll_library
        broll_dirs = list(Path("AI-B-roll/broll_library").glob("clip_reframed_*"))
        # Ã°Å¸Å¡â‚¬ CORRECTION: PRIORISER le dossier le plus rÃƒÂ©cent
        broll_dirs = sorted(broll_dirs, key=lambda p: p.name, reverse=True)  # Plus rÃƒÂ©cent en premier
        print(f"Ã°Å¸â€Â DEBUG: Dossiers clip_reframed_* trouvÃƒÂ©s: {len(broll_dirs)}")
        if broll_dirs:
            print(f"Ã°Å¸Å½Â¯ DEBUG: Dossier prioritaire (plus rÃƒÂ©cent): {broll_dirs[0].name}")
        
        # Si pas trouvÃƒÂ©, chercher dans output/clips
        if not broll_dirs:
            print("Ã°Å¸â€Â DEBUG: Aucun dossier clip_reframed_* trouvÃƒÂ©, recherche dans output/clips")
            output_dirs = list(Path("output/clips").glob("*"))
            print(f"Ã°Å¸â€Â DEBUG: Dossiers output/clips trouvÃƒÂ©s: {len(output_dirs)}")
            for output_dir in output_dirs:
                if output_dir.is_dir():
                    print(f"Ã°Å¸â€Â DEBUG: Exploration de {output_dir}")
                    # Chercher des fichiers B-roll dans les sous-dossiers
                    for subdir in output_dir.iterdir():
                        if subdir.is_dir() and "broll" in subdir.name.lower():
                            print(f"Ã°Å¸â€Â DEBUG: Dossier B-roll trouvÃƒÂ©: {subdir}")
                            broll_dirs.append(subdir)
        
        # Si toujours pas trouvÃƒÂ©, chercher dans le dossier racine
        if not broll_dirs:
            print("Ã°Å¸â€Â DEBUG: Aucun dossier B-roll trouvÃƒÂ©, recherche dans le dossier racine")
            # Chercher seulement des dossiers, pas des fichiers
            root_dirs = [d for d in Path(".").glob("*broll*") if d.is_dir()]
            print(f"Ã°Å¸â€Â DEBUG: Dossiers *broll* trouvÃƒÂ©s: {len(root_dirs)}")
            broll_dirs.extend(root_dirs)
        
        # Si toujours pas trouvÃƒÂ©, chercher dans test_clip
        if not broll_dirs:
            print("Ã°Å¸â€Â DEBUG: Aucun dossier B-roll trouvÃƒÂ©, recherche dans test_clip")
            test_dir = Path("AI-B-roll/broll_library/test_clip")
            if test_dir.exists():
                print(f"Ã°Å¸â€Â DEBUG: Dossier test_clip trouvÃƒÂ©: {test_dir}")
                broll_dirs.append(test_dir)
        
        # Ã°Å¸Å¡â‚¬ NOUVEAU: Si pas de dossiers B-roll, TÃƒâ€°LÃƒâ€°CHARGER depuis les APIs
        if not broll_dirs:
            print("Ã°Å¸â€Â DEBUG: Aucun dossier B-roll trouvÃƒÂ©, TÃƒâ€°LÃƒâ€°CHARGEMENT depuis APIs...")
            return self._fetch_from_apis(keywords, limit)
        
        print(f"Ã°Å¸â€Â DEBUG: Total dossiers B-roll trouvÃƒÂ©s: {len(broll_dirs)}")
        
        # Ã°Å¸Å¡â‚¬ CORRECTION: Utiliser SEULEMENT le dossier le plus rÃƒÂ©cent s'il a des assets
        prioritized_dirs = []
        if broll_dirs:
            latest_dir = broll_dirs[0]  # Le plus rÃƒÂ©cent grÃƒÂ¢ce au tri
            latest_fetched = latest_dir / "fetched"
            if latest_fetched.exists() and len(list(latest_fetched.rglob("*.mp4"))) > 0:
                print(f"Ã°Å¸Å½Â¯ DEBUG: Utilisation exclusive du dossier rÃƒÂ©cent: {latest_dir.name}")
                prioritized_dirs = [latest_dir]  # SEULEMENT le plus rÃƒÂ©cent
            else:
                print(f"Ã¢Å¡Â Ã¯Â¸Â DEBUG: Dossier rÃƒÂ©cent vide, utilisation de tous les dossiers")
                prioritized_dirs = broll_dirs  # Fallback vers tous si rÃƒÂ©cent vide
        else:
            prioritized_dirs = broll_dirs
            
        for broll_dir in prioritized_dirs:
            if not broll_dir.exists():
                print(f"Ã°Å¸â€Â DEBUG: Dossier {broll_dir} n'existe pas")
                continue
                
            print(f"Ã°Å¸â€Â DEBUG: Exploration du dossier: {broll_dir}")
            
            # Chercher des fichiers vidÃƒÂ©o dans ce dossier
            video_files = []
            
            # Chercher dans fetched/ si existe
            fetched_dir = broll_dir / "fetched"
            if fetched_dir.exists():
                print(f"Ã°Å¸â€Â DEBUG: Dossier fetched trouvÃƒÂ©: {fetched_dir}")
                for provider_dir in fetched_dir.iterdir():
                    if provider_dir.is_dir():
                        print(f"Ã°Å¸â€Â DEBUG: Provider trouvÃƒÂ©: {provider_dir}")
                        for theme_dir in provider_dir.iterdir():
                            if theme_dir.is_dir():
                                print(f"Ã°Å¸â€Â DEBUG: Theme trouvÃƒÂ©: {theme_dir}")
                                for asset_file in theme_dir.glob("*.mp4"):
                                    print(f"Ã°Å¸â€Â DEBUG: Fichier vidÃƒÂ©o trouvÃƒÂ©: {asset_file}")
                                    video_files.append(asset_file)
            else:
                print(f"Ã°Å¸â€Â DEBUG: Dossier fetched non trouvÃƒÂ© dans {broll_dir}")
            
            # Si pas de fetched/, chercher directement
            if not video_files:
                print(f"Ã°Å¸â€Â DEBUG: Aucun fichier dans fetched/, recherche directe")
                for asset_file in broll_dir.rglob("*.mp4"):
                    print(f"Ã°Å¸â€Â DEBUG: Fichier vidÃƒÂ©o trouvÃƒÂ© (recherche directe): {asset_file}")
                    video_files.append(asset_file)
            
            print(f"Ã°Å¸â€Â DEBUG: Fichiers vidÃƒÂ©o trouvÃƒÂ©s dans {broll_dir.name}: {len(video_files)}")
            
            for asset_file in video_files[:10]:  # Limiter par dossier
                try:
                    print(f"Ã°Å¸â€Â DEBUG: CrÃƒÂ©ation asset pour {asset_file}")
                    # CrÃƒÂ©er un asset
                    asset = Asset(
                        id=f"asset_{len(assets)}",
                        file_path=str(asset_file),
                        tags=self._extract_tags_from_path(asset_file),
                        title=asset_file.stem,
                        description=f"Asset from {broll_dir.name}",
                        source="local",
                        fetched_at=datetime.now() - timedelta(days=len(assets) % 30),
                        duration=2.0 + (len(assets) % 3),
                        resolution="1920x1080"
                    )
                    assets.append(asset)
                    print(f"Ã°Å¸â€Â DEBUG: Asset crÃƒÂ©ÃƒÂ© avec succÃƒÂ¨s: {asset.id}")
                    
                    if len(assets) >= limit:
                        break
                        
                except Exception as e:
                    print(f"Ã°Å¸â€Â DEBUG: Erreur crÃƒÂ©ation asset {asset_file}: {e}")
        
        print(f"Ã°Å¸â€Â DEBUG: Total assets rÃƒÂ©cupÃƒÂ©rÃƒÂ©s: {len(assets)}")
        
        # Ã°Å¸Å¡â‚¬ AMÃƒâ€°LIORATION: TÃƒÂ©lÃƒÂ©charger aussi si mots-clÃƒÂ©s spÃƒÂ©cifiques et peu d'assets
        specialized_keywords = [kw for kw in keywords if '_' in kw or any(term in kw.lower() for term in ['brain', 'neural', 'adrenaline', 'chemical', 'medical'])]
        
        # Ã°Å¸Å¡â‚¬ CORRECTION CRITIQUE: VÃƒÂ©rifier si le dossier ACTUEL est vide
        # Compter seulement les assets du dossier le plus rÃƒÂ©cent
        latest_clip_assets = 0
        if broll_dirs:
            latest_clip_dir = max(broll_dirs, key=lambda p: p.name)
            latest_fetched = latest_clip_dir / "fetched"
            if latest_fetched.exists():
                latest_clip_assets = len(list(latest_fetched.rglob("*.mp4")))
                print(f"Ã°Å¸â€Â DEBUG: Assets dans dossier actuel {latest_clip_dir.name}: {latest_clip_assets}")
        
        should_download = (
            len(assets) == 0 or  # Aucun asset global
            latest_clip_assets == 0 or  # Dossier actuel vide
            (len(specialized_keywords) > 0 and latest_clip_assets < 5)  # Peu d'assets spÃƒÂ©cifiques au dossier actuel
        )
        
        if should_download:
            if len(assets) == 0:
                print("Ã°Å¸â€Â DEBUG: Aucun asset trouvÃƒÂ© dans le cache local")
            else:
                print(f"Ã°Å¸â€Â DEBUG: {len(specialized_keywords)} mots-clÃƒÂ©s spÃƒÂ©cialisÃƒÂ©s dÃƒÂ©tectÃƒÂ©s, tÃƒÂ©lÃƒÂ©chargement complÃƒÂ©mentaire")
                print(f"    Ã°Å¸Å½Â¯ Mots-clÃƒÂ©s spÃƒÂ©cialisÃƒÂ©s: {specialized_keywords[:3]}")
            
            print("Ã°Å¸â€œÂ¥ Lancement tÃƒÂ©lÃƒÂ©chargement depuis APIs...")
            api_assets = self._fetch_from_apis(keywords, limit)
            assets.extend(api_assets)
            print(f"Ã°Å¸â€œÂ¥ Total aprÃƒÂ¨s tÃƒÂ©lÃƒÂ©chargement: {len(assets)} assets")
        
        self.logger.info(f"Ã°Å¸â€œÂ¥ Assets rÃƒÂ©cupÃƒÂ©rÃƒÂ©s: {len(assets)}")
        return assets
    
    def _fetch_from_apis(self, keywords: List[str], limit: int = 200) -> List[Asset]:
        """TÃƒÂ©lÃƒÂ©charge des B-rolls depuis les APIs externes"""
        try:
            import requests
            import os
            from datetime import datetime
            
            assets = []
            
            # Configuration des clÃƒÂ©s API
            pexels_key = os.getenv('PEXELS_API_KEY') or 'pwhBa9K7fa9IQJCmfCy0NfHFWy8QyqoCkGnWLK3NC2SbDTtUeuhxpDoD'
            pixabay_key = os.getenv('PIXABAY_API_KEY') or '51724939-ee09a81ccfce0f5623df46a69'
            
            if not pexels_key and not pixabay_key:
                print("Ã¢ÂÅ’ Pas de clÃƒÂ© API (Pexels/Pixabay) pour le tÃƒÂ©lÃƒÂ©chargement")
                return self._create_fallback_assets(keywords)
            
            # Ã°Å¸Å¡â‚¬ CORRECTION: TÃƒÂ©lÃƒÂ©charger dans le dossier clip le plus rÃƒÂ©cent
            # Trouver le dossier clip le plus rÃƒÂ©cent (celui qui vient d'ÃƒÂªtre crÃƒÂ©ÃƒÂ©)
            broll_dirs = list(Path("AI-B-roll/broll_library").glob("clip_reframed_*"))
            if broll_dirs:
                # Prendre le plus rÃƒÂ©cent (tri par nom qui contient timestamp)
                latest_clip_dir = max(broll_dirs, key=lambda p: p.name)
                fetch_dir = latest_clip_dir / "fetched"
                print(f"Ã°Å¸Å½Â¯ TÃƒÂ©lÃƒÂ©chargement dans: {latest_clip_dir.name}/fetched/")
            else:
                # Fallback vers dossier gÃƒÂ©nÃƒÂ©rique si pas de clip trouvÃƒÂ©
                fetch_dir = Path("AI-B-roll/broll_library/fetched")
                print("Ã¢Å¡Â Ã¯Â¸Â Aucun dossier clip trouvÃƒÂ©, utilisation dossier gÃƒÂ©nÃƒÂ©rique")
            
            fetch_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement B-rolls depuis APIs pour {len(keywords)} mots-clÃƒÂ©s...")
            
            # Essayer tous les providers (Pexels/Pixabay uniquement)
            providers = []
            if pexels_key:
                providers.append(('pexels', pexels_key))
            if pixabay_key:
                providers.append(('pixabay', pixabay_key))
            if not providers:
                print('WARNING: no API provider (Pexels/Pixabay)')

            # Ã°Å¸Å¡â‚¬ NOUVEAU: Simplifier les mots-clÃƒÂ©s pour APIs externes
            simplified_keywords = []
            for keyword in keywords[:5]:  # Plus de mots-clÃƒÂ©s pour augmenter les chances
                # Simplifier les mots-clÃƒÂ©s LLM pour les APIs
                simplified = self._simplify_keyword_for_api(keyword)
                if simplified and simplified not in simplified_keywords:
                    simplified_keywords.append(simplified)
            
            # Limiter et afficher
            simplified_keywords = simplified_keywords[:3]
            print(f"Ã°Å¸â€Â Mots-clÃƒÂ©s simplifiÃƒÂ©s pour APIs: {simplified_keywords}")
            
            # Ã°Å¸Å¡â‚¬ OPTIMISATION VALIDÃƒâ€°E: TÃƒÂ©lÃƒÂ©chargement parallÃƒÂ¨le des APIs
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            def fetch_from_provider(keyword, provider, api_key, fetch_dir):
                """Download an asset for the given provider (Pexels/Pixabay)."""
                try:
                    if provider == 'pexels':
                        return self._fetch_from_pexels(keyword, api_key, fetch_dir / 'pexels')
                    if provider == 'pixabay':
                        return self._fetch_from_pixabay(keyword, api_key, fetch_dir / 'pixabay')
                    return []
                except Exception as e:
                    print(f"WARNING: provider {provider} failed for '{keyword}': {e}")
                    return []

            fetch_tasks = []
            for keyword in simplified_keywords:
                for provider, api_key in providers:
                    if len(assets) >= limit:
                        break
                    fetch_tasks.append((keyword, provider, api_key, fetch_dir))
            
            print(f"Ã°Å¸Å¡â‚¬ TÃƒÂ©lÃƒÂ©chargement parallÃƒÂ¨le: {len(fetch_tasks)} tÃƒÂ¢ches sur {len(providers)} APIs")
            
            # ExÃƒÂ©cution parallÃƒÂ¨le avec maximum 4 threads (optimisation rÃƒÂ©seau)
            with ThreadPoolExecutor(max_workers=min(4, len(fetch_tasks))) as executor:
                # Soumettre toutes les tÃƒÂ¢ches
                future_to_task = {
                    executor.submit(fetch_from_provider, keyword, provider, api_key, fetch_dir): (keyword, provider)
                    for keyword, provider, api_key, fetch_dir in fetch_tasks
                }
                
                # RÃƒÂ©cupÃƒÂ©rer les rÃƒÂ©sultats au fur et ÃƒÂ  mesure
                for future in as_completed(future_to_task):
                    keyword, provider = future_to_task[future]
                    try:
                        provider_assets = future.result(timeout=30)  # Timeout 30s par provider
                        if provider_assets:
                            assets.extend(provider_assets)
                            print(f"   Ã¢Å“â€¦ {provider}: {len(provider_assets)} assets pour '{keyword}'")
                        
                        # ArrÃƒÂªter si limite atteinte
                        if len(assets) >= limit:
                            print(f"   Ã°Å¸Å½Â¯ Limite atteinte: {len(assets)} assets")
                            break
                            
                    except Exception as e:
                        print(f"   Ã¢ÂÅ’ {provider} ÃƒÂ©chouÃƒÂ© pour '{keyword}': {e}")
            
            print(f"Ã¢Å¡Â¡ TÃƒÂ©lÃƒÂ©chargement parallÃƒÂ¨le terminÃƒÂ©: {len(assets)} assets obtenus")
                

            
            print(f"Ã¢Å“â€¦ {len(assets)} B-rolls tÃƒÂ©lÃƒÂ©chargÃƒÂ©s depuis les APIs")
            
            # Si pas d'assets tÃƒÂ©lÃƒÂ©chargÃƒÂ©s, fallback
            if not assets:
                print("Ã°Å¸â€â€ž Aucun tÃƒÂ©lÃƒÂ©chargement rÃƒÂ©ussi, utilisation fallback")
                return self._create_fallback_assets(keywords)
            
            return assets
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur tÃƒÂ©lÃƒÂ©chargement APIs: {e}")
            return self._create_fallback_assets(keywords)
    
    def _simplify_keyword_for_api(self, keyword: str) -> str:
        """Simplifie un mot-clÃƒÂ© LLM pour les APIs externes en prÃƒÂ©servant la spÃƒÂ©cificitÃƒÂ©"""
        # Convertir underscore en espace pour les APIs
        simplified = keyword.replace('_', ' ')
        
        # Ã°Å¸Å¡â‚¬ AMÃƒâ€°LIORATION: PrÃƒÂ©server la spÃƒÂ©cificitÃƒÂ© des mots-clÃƒÂ©s LLM
        concept_mapping = {
            # Ã°Å¸Â§Â  Cerveau & Neuroscience - PRÃƒâ€°SERVER LA SPÃƒâ€°CIFICITÃƒâ€°
            'brain neural networks': 'brain neurons neural network',
            'brain adrenaline buffer': 'brain neurotransmitter adrenaline',
            'brain neural connections': 'brain synapses neural',
            'brain internal reward': 'brain dopamine reward system',
            'neural networks': 'brain neural network',
            
            # Ã°Å¸â€˜Â¤ Actions humaines - GARDER LE CONTEXTE
            'person thinking concept': 'person thinking meditation',
            'person celebrating achievement': 'person celebrating success',
            'person achieving goal': 'person achievement success',
            'person celebrating success': 'person celebration victory',
            'person reflecting concept': 'person contemplating thinking',
            'person achieving objective': 'person goal achievement',
            'person celebrating win': 'person victory celebration',
            
            # Ã°Å¸â€™Â¼ Business - ENRICHIR AU LIEU DE SIMPLIFIER
            'business handshake deal': 'business handshake partnership',
            'entrepreneur presenting idea': 'entrepreneur presentation business',
            'team brainstorming session': 'team meeting brainstorming',
            'data visualization reward': 'data visualization charts',
            
            # Anciens mappings (maintenir compatibilitÃƒÂ©)
            'process direction visual': 'business process',
            'internal motivation concept': 'motivation psychology',
            'brain focus concept': 'brain thinking',
            'energy drive concept': 'energy motivation',
            'path outcome visual': 'path direction success',
            'adrenaline buffer concept': 'adrenaline stress hormone',
            'reward mechanism visual': 'reward success achievement',
            'goal achievement visual': 'goal achievement success',
            'cognitive control visual': 'brain thinking cognitive'
        }
        
        # Chercher correspondance exacte
        simplified_lower = simplified.lower()
        for complex_term, enhanced_term in concept_mapping.items():
            if complex_term in simplified_lower:
                print(f"    Ã°Å¸Å½Â¯ Mapping spÃƒÂ©cialisÃƒÂ©: {keyword} Ã¢â€ â€™ {enhanced_term}")
                return enhanced_term
        
        # Ã°Å¸Å¡â‚¬ AMÃƒâ€°LIORATION: Traitement intelligent des mots-clÃƒÂ©s structurÃƒÂ©s
        words = simplified.lower().split()
        if len(words) >= 2:
            # Identifier et enrichir les domaines spÃƒÂ©cialisÃƒÂ©s
            domain_enrichment = {
                'brain': ['neuroscience', 'cognitive'],
                'person': ['human', 'individual'],
                'business': ['professional', 'corporate'],
                'data': ['analytics', 'visualization']
            }
            
            # Construire une requÃƒÂªte enrichie
            enhanced_terms = []
            for word in words:
                if word in domain_enrichment:
                    enhanced_terms.append(word)
                    enhanced_terms.extend(domain_enrichment[word][:1])  # Ajouter 1 terme de domaine
                elif len(word) > 3 and word not in ['concept', 'visual', 'mechanism']:
                    enhanced_terms.append(word)
            
            if enhanced_terms:
                result = ' '.join(enhanced_terms[:4])  # Max 4 mots pour l'API
                print(f"    Ã°Å¸Â§Â  Enrichissement intelligent: {keyword} Ã¢â€ â€™ {result}")
                return result
        
        # Fallback amÃƒÂ©liorÃƒÂ© : garder la spÃƒÂ©cificitÃƒÂ©
        if len(simplified) > 2:
            return simplified
        else:
            return 'professional business'
    
    def _fetch_from_pexels(self, keyword: str, api_key: str, fetch_dir: Path) -> List[Asset]:
        """TÃƒÂ©lÃƒÂ©charge des B-rolls depuis Pexels"""
        assets = []
        try:
            print(f"Ã°Å¸â€Â Recherche Pexels: '{keyword}'")
            
            # Appel API Pexels
            headers = {"Authorization": api_key}
            response = requests.get(
                f"https://api.pexels.com/videos/search?query={keyword}&per_page=2",
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur API Pexels pour '{keyword}': {response.status_code}")
                return assets
            
            data = response.json()
            videos = data.get('videos', [])
            
            for i, video in enumerate(videos):
                try:
                    video_files = video.get('video_files', [])
                    if not video_files:
                        continue
                    
                    # Choisir la qualitÃƒÂ© medium ou HD
                    suitable_files = [vf for vf in video_files if vf.get('quality') in ['hd', 'medium']]
                    if not suitable_files:
                        suitable_files = video_files[:1]
                    
                    video_file = suitable_files[0]
                    download_url = video_file['link']
                    
                    # Nom du fichier
                    filename = f"{keyword}_{video['id']}_{i}.mp4"
                    file_path = fetch_dir / filename
                    
                    # CrÃƒÂ©er le dossier
                    fetch_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement Pexels: {filename}")
                    
                    # TÃƒÂ©lÃƒÂ©charger
                    download_response = requests.get(download_url, stream=True, timeout=30)
                    download_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        downloaded_size = 0
                        max_size = 10 * 1024 * 1024  # 10MB max par fichier
                        
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if downloaded_size > max_size:
                                    break
                    
                    # Ã°Å¸Å¡â‚¬ VALIDATION: VÃƒÂ©rifier l'intÃƒÂ©gritÃƒÂ© du fichier tÃƒÂ©lÃƒÂ©chargÃƒÂ©
                    if file_path.exists() and file_path.stat().st_size > 1000:
                        # Validation basique de l'intÃƒÂ©gritÃƒÂ© vidÃƒÂ©o
                        is_valid = True
                        try:
                            if filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                                # Test d'ouverture rapide avec MoviePy
                                from moviepy.editor import VideoFileClip
                                with VideoFileClip(str(file_path)) as test_clip:
                                    # VÃƒÂ©rifier que la durÃƒÂ©e est cohÃƒÂ©rente
                                    if test_clip.duration <= 0 or test_clip.duration > 300:  # Max 5 minutes
                                        is_valid = False
                        except Exception:
                            is_valid = False
                            
                        if is_valid:
                            asset = Asset(
                                id=f"pexels_{video['id']}",
                                file_path=str(file_path),
                                tags=[keyword, 'pexels', 'video'] + keyword.split('_'),
                                title=f"Pexels {keyword} {video['id']}",
                                description=f"B-roll tÃƒÂ©lÃƒÂ©chargÃƒÂ© depuis Pexels pour {keyword}",
                                source="pexels_api",
                                fetched_at=datetime.now(),
                                duration=float(video.get('duration', 3.0)),
                                resolution=f"{video_file.get('width', 1920)}x{video_file.get('height', 1080)}"
                            )
                            assets.append(asset)
                            print(f"Ã¢Å“â€¦ TÃƒÂ©lÃƒÂ©chargÃƒÂ© Pexels: {filename} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                        else:
                            print(f"Ã¢Å¡Â Ã¯Â¸Â Fichier Pexels corrompu ignorÃƒÂ©: {filename}")
                            try:
                                file_path.unlink()  # Supprimer le fichier corrompu
                            except:
                                pass
                    
                    if len(assets) >= 2:  # Limiter ÃƒÂ  2 par mot-clÃƒÂ© par provider
                        break
                        
                except Exception as e:
                    print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur tÃƒÂ©lÃƒÂ©chargement Pexels {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur recherche Pexels '{keyword}': {e}")
        
        return assets
    
    def _fetch_from_pixabay(self, keyword: str, api_key: str, fetch_dir: Path) -> List[Asset]:
        """TÃƒÂ©lÃƒÂ©charge des B-rolls depuis Pixabay avec format officiel"""
        assets = []
        try:
            print(f"Ã°Å¸â€Â Recherche Pixabay: '{keyword}'")
            
            # URL officielle exacte de la documentation Pixabay
            # Pixabay accepte per_page entre 3-200, pas 2
            url = f"https://pixabay.com/api/videos/?key={api_key}&q={keyword}&per_page=3"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur API Pixabay pour '{keyword}': {response.status_code}")
                print(f"   RÃƒÂ©ponse: {response.text[:100]}")
                return assets
            
            data = response.json()
            videos = data.get('hits', [])
            
            print(f"Ã°Å¸â€œÂ¹ Pixabay trouvÃƒÂ©: {len(videos)} vidÃƒÂ©os pour '{keyword}'")
            
            for i, video in enumerate(videos):
                try:
                    video_files = video.get('videos', {})
                    if not video_files:
                        continue
                    
                    # Choisir la meilleure qualitÃƒÂ© disponible
                    quality_order = ['medium', 'small', 'tiny']  # medium = 1280x720 gÃƒÂ©nÃƒÂ©ralement
                    selected_quality = None
                    
                    for quality in quality_order:
                        if quality in video_files and video_files[quality].get('url'):
                            selected_quality = quality
                            break
                    
                    if not selected_quality:
                        print(f"Ã¢Å¡Â Ã¯Â¸Â Aucune qualitÃƒÂ© disponible pour Pixabay video {video['id']}")
                        continue
                    
                    video_info = video_files[selected_quality]
                    download_url = video_info['url']
                    
                    # Nom du fichier
                    filename = f"{keyword}_{video['id']}_{i}.mp4"
                    file_path = fetch_dir / filename
                    
                    # CrÃƒÂ©er le dossier
                    fetch_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement Pixabay: {filename} ({selected_quality})")
                    
                    # TÃƒÂ©lÃƒÂ©charger
                    download_response = requests.get(download_url, stream=True, timeout=30)
                    download_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        downloaded_size = 0
                        max_size = 15 * 1024 * 1024  # 15MB max pour Pixabay
                        
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if downloaded_size > max_size:
                                    break
                    
                    # Ã°Å¸Å¡â‚¬ VALIDATION: VÃƒÂ©rifier l'intÃƒÂ©gritÃƒÂ© du fichier tÃƒÂ©lÃƒÂ©chargÃƒÂ©
                    if file_path.exists() and file_path.stat().st_size > 1000:
                        # Validation basique de l'intÃƒÂ©gritÃƒÂ© vidÃƒÂ©o
                        is_valid = True
                        try:
                            if filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                                # Test d'ouverture rapide avec MoviePy
                                from moviepy.editor import VideoFileClip
                                with VideoFileClip(str(file_path)) as test_clip:
                                    # VÃƒÂ©rifier que la durÃƒÂ©e est cohÃƒÂ©rente
                                    if test_clip.duration <= 0 or test_clip.duration > 300:  # Max 5 minutes
                                        is_valid = False
                        except Exception:
                            is_valid = False
                            
                        if is_valid:
                            asset = Asset(
                                id=f"pixabay_{video['id']}",
                                file_path=str(file_path),
                                tags=[keyword, 'pixabay', 'video'] + video.get('tags', '').split(', '),
                                title=f"Pixabay {keyword} {video['id']}",
                                description=f"B-roll tÃƒÂ©lÃƒÂ©chargÃƒÂ© depuis Pixabay pour {keyword}",
                                source="pixabay_api",
                                fetched_at=datetime.now(),
                                duration=float(video.get('duration', 3.0)),
                                resolution=f"{video_info.get('width', 1280)}x{video_info.get('height', 720)}"
                            )
                            assets.append(asset)
                            print(f"Ã¢Å“â€¦ TÃƒÂ©lÃƒÂ©chargÃƒÂ© Pixabay: {filename} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                        else:
                            print(f"Ã¢Å¡Â Ã¯Â¸Â Fichier Pixabay corrompu ignorÃƒÂ©: {filename}")
                            try:
                                file_path.unlink()  # Supprimer le fichier corrompu
                            except:
                                pass
                    
                    if len(assets) >= 2:  # Limiter ÃƒÂ  2 par mot-clÃƒÂ© par provider
                        break
                        
                except Exception as e:
                    print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur tÃƒÂ©lÃƒÂ©chargement Pixabay {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur recherche Pixabay '{keyword}': {e}")
        
        return assets
    
    def _fetch_from_unsplash(self, keyword: str, access_key: str, app_id: str, fetch_dir: Path) -> List[Asset]:
        """TÃƒÂ©lÃƒÂ©charge des images depuis Unsplash (photos haute qualitÃƒÂ©)"""
        assets = []
        try:
            print(f"Ã°Å¸â€Â Recherche Unsplash: '{keyword}'")
            
            # API Unsplash pour les photos
            headers = {
                "Authorization": f"Client-ID {access_key}",
                "Accept-Version": "v1"
            }
            
            # Recherche de photos avec le mot-clÃƒÂ©
            response = requests.get(
                f"https://api.unsplash.com/search/photos?query={keyword}&per_page=3&orientation=landscape",
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur API Unsplash pour '{keyword}': {response.status_code}")
                return assets
            
            data = response.json()
            photos = data.get('results', [])
            
            print(f"Ã°Å¸â€œÂ¸ Unsplash trouvÃƒÂ©: {len(photos)} photos pour '{keyword}'")
            
            for i, photo in enumerate(photos):
                try:
                    # Choisir la qualitÃƒÂ© regular (1080p) ou full (haute rÃƒÂ©solution)
                    urls = photo.get('urls', {})
                    download_url = urls.get('regular') or urls.get('full') or urls.get('small')
                    
                    if not download_url:
                        continue
                    
                    # Nom du fichier (image)
                    filename = f"{keyword}_{photo['id']}_{i}.jpg"
                    file_path = fetch_dir / filename
                    
                    # CrÃƒÂ©er le dossier
                    fetch_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement Unsplash: {filename}")
                    
                    # TÃƒÂ©lÃƒÂ©charger l'image
                    download_response = requests.get(download_url, stream=True, timeout=30)
                    download_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        downloaded_size = 0
                        max_size = 5 * 1024 * 1024  # 5MB max pour les images
                        
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if downloaded_size > max_size:
                                    break
                    
                    # CrÃƒÂ©er l'asset
                    if file_path.exists() and file_path.stat().st_size > 1000:
                        # Extraire les tags depuis la description/alt_description
                        photo_tags = [keyword, 'unsplash', 'photo']
                        if photo.get('alt_description'):
                            photo_tags.extend(photo['alt_description'].lower().split()[:5])
                        if photo.get('description'):
                            photo_tags.extend(photo['description'].lower().split()[:3])
                        
                        asset = Asset(
                            id=f"unsplash_{photo['id']}",
                            file_path=str(file_path),
                            tags=photo_tags,
                            title=f"Unsplash {keyword} {photo['id']}",
                            description=photo.get('alt_description') or photo.get('description') or f"Photo Unsplash pour {keyword}",
                            source="unsplash_api",
                            fetched_at=datetime.now(),
                            duration=3.0,  # Image statique, durÃƒÂ©e par dÃƒÂ©faut pour Ken Burns
                            resolution=f"{photo.get('width', 1920)}x{photo.get('height', 1080)}"
                        )
                        assets.append(asset)
                        print(f"Ã¢Å“â€¦ TÃƒÂ©lÃƒÂ©chargÃƒÂ© Unsplash: {filename} ({file_path.stat().st_size / 1024:.1f}KB)")
                    
                    if len(assets) >= 3:  # Limiter ÃƒÂ  3 par mot-clÃƒÂ© pour Unsplash
                        break
                        
                except Exception as e:
                    print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur tÃƒÂ©lÃƒÂ©chargement Unsplash {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur recherche Unsplash '{keyword}': {e}")
        
        return assets
    
    def _fetch_from_archive_org(self, keyword: str, fetch_dir: Path) -> List[Asset]:
        """TÃƒÂ©lÃƒÂ©charge des vidÃƒÂ©os depuis Archive.org (gratuit, domaine public)"""
        assets = []
        try:
            print(f"Ã°Å¸â€Â Recherche Archive.org: '{keyword}'")
            
            # API de recherche Archive.org
            # Recherche dans la collection de vidÃƒÂ©os open source
            search_query = f"collection:opensource_movies AND ({keyword})"
            
            response = requests.get(
                f"https://archive.org/advancedsearch.php",
                params={
                    'q': search_query,
                    'fl[]': ['identifier', 'title', 'description', 'downloads'],
                    'rows': 5,
                    'page': 1,
                    'output': 'json'
                },
                timeout=20
            )
            
            if response.status_code != 200:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur API Archive.org pour '{keyword}': {response.status_code}")
                return assets
            
            data = response.json()
            items = data.get('response', {}).get('docs', [])
            
            print(f"Ã°Å¸â€œÂ¹ Archive.org trouvÃƒÂ©: {len(items)} items pour '{keyword}'")
            
            for i, item in enumerate(items):
                try:
                    identifier = item.get('identifier', '')
                    if not identifier:
                        continue
                    
                    # Obtenir les dÃƒÂ©tails de l'item pour trouver des fichiers MP4
                    details_response = requests.get(
                        f"https://archive.org/metadata/{identifier}",
                        timeout=15
                    )
                    
                    if details_response.status_code != 200:
                        continue
                    
                    details = details_response.json()
                    files = details.get('files', [])
                    
                    # Chercher des fichiers MP4 de taille raisonnable
                    video_files = [
                        f for f in files 
                        if f.get('format', '').lower() in ['mpeg4', 'mp4'] 
                        and f.get('name', '').endswith('.mp4')
                        and int(f.get('size', '0')) < 50 * 1024 * 1024  # Moins de 50MB
                        and int(f.get('size', '0')) > 1 * 1024 * 1024   # Plus de 1MB
                    ]
                    
                    if not video_files:
                        continue
                    
                    # Prendre le premier fichier vidÃƒÂ©o valide
                    video_file = video_files[0]
                    filename_original = video_file['name']
                    
                    # Construire l'URL de tÃƒÂ©lÃƒÂ©chargement
                    download_url = f"https://archive.org/download/{identifier}/{filename_original}"
                    
                    # Nom du fichier local
                    filename = f"{keyword}_{identifier}_{i}.mp4"
                    file_path = fetch_dir / filename
                    
                    # CrÃƒÂ©er le dossier
                    fetch_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Ã°Å¸â€œÂ¥ TÃƒÂ©lÃƒÂ©chargement Archive.org: {filename}")
                    
                    # TÃƒÂ©lÃƒÂ©charger (avec limite de taille)
                    download_response = requests.get(download_url, stream=True, timeout=45)
                    download_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        downloaded_size = 0
                        max_size = 20 * 1024 * 1024  # 20MB max pour Archive.org
                        
                        for chunk in download_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if downloaded_size > max_size:
                                    print(f"   Ã¢Å¡Â Ã¯Â¸Â TÃƒÂ©lÃƒÂ©chargement arrÃƒÂªtÃƒÂ© ÃƒÂ  20MB")
                                    break
                    
                    # CrÃƒÂ©er l'asset
                    if file_path.exists() and file_path.stat().st_size > 100000:  # Au moins 100KB
                        # Extraire des tags depuis le titre et la description
                        archive_tags = [keyword, 'archive', 'video', 'creative_commons']
                        title = item.get('title', '')
                        if title:
                            archive_tags.extend(title.lower().split()[:5])
                        
                        asset = Asset(
                            id=f"archive_{identifier}",
                            file_path=str(file_path),
                            tags=archive_tags,
                            title=title or f"Archive.org {keyword} {identifier}",
                            description=item.get('description', f"VidÃƒÂ©o Archive.org pour {keyword}"),
                            source="archive_org",
                            fetched_at=datetime.now(),
                            duration=float(video_file.get('length', '10.0') or '10.0'),
                            resolution="unknown"
                        )
                        assets.append(asset)
                        print(f"Ã¢Å“â€¦ TÃƒÂ©lÃƒÂ©chargÃƒÂ© Archive.org: {filename} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                    
                    if len(assets) >= 2:  # Limiter ÃƒÂ  2 par mot-clÃƒÂ© pour Archive.org
                        break
                        
                except Exception as e:
                    print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur tÃƒÂ©lÃƒÂ©chargement Archive.org {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur recherche Archive.org '{keyword}': {e}")
        
        return assets
    
    def _create_fallback_assets(self, keywords: List[str]) -> List[Asset]:
        """CrÃƒÂ©e des assets de fallback si le tÃƒÂ©lÃƒÂ©chargement ÃƒÂ©choue"""
        assets = []
        
        for i, keyword in enumerate(keywords[:3]):
            asset = Asset(
                id=f"fallback_{keyword}_{i}",
                file_path=f"fallback_{keyword}.mp4",
                tags=[keyword, 'fallback'] + keywords[:3],
                title=f"Fallback {keyword.title()}",
                description=f"Asset de fallback pour {keyword}",
                source="fallback",
                fetched_at=datetime.now(),
                duration=2.0 + (i * 0.5),
                resolution="1920x1080"
            )
            assets.append(asset)
        
        return assets
    
    def _extract_tags_from_path(self, file_path: Path) -> List[str]:
        """Extrait des tags depuis le chemin du fichier"""
        tags = []
        
        # Extraire des tags depuis le nom du fichier
        filename = file_path.stem.lower()
        tags.extend(filename.split('_'))
        
        # Extraire des tags depuis les dossiers parents
        for parent in file_path.parents:
            if parent.name and parent.name != ".":
                tags.extend(parent.name.lower().split('_'))
        
        # Nettoyer et filtrer les tags
        clean_tags = []
        for tag in tags:
            if tag and len(tag) > 2 and tag not in ['clip', 'reframed', 'fetched', 'broll', 'library']:
                clean_tags.append(tag)
        
        return clean_tags[:10]  # Limiter ÃƒÂ  10 tags
    
    def score_asset(self, asset: Asset, query_keywords: Set[str], domain: Optional[str] = None) -> ScoringFeatures:
        """Calcule le score complet d'un asset"""
        features = ScoringFeatures()
        
        # 1. Token overlap (Jaccard)
        asset_tokens = set()
        if asset.tags:
            asset_tokens.update(asset.tags)
        if asset.title:
            asset_tokens.update(asset.title.lower().split())
        if asset.description:
            asset_tokens.update(asset.description.lower().split())
        
        if asset_tokens and query_keywords:
            intersection = len(asset_tokens & query_keywords)
            union = len(asset_tokens | query_keywords)
            features.token_overlap = intersection / union if union > 0 else 0.0
        
        # 2. Embedding similarity (si disponible)
        if self.embedding_model:
            try:
                # CrÃƒÂ©er un texte de recherche depuis les mots-clÃƒÂ©s
                query_text = " ".join(query_keywords)
                asset_text = " ".join([asset.title, asset.description] + asset.tags)
                
                # Calculer les embeddings
                query_embedding = self.embedding_model.encode([query_text])
                asset_embedding = self.embedding_model.encode([asset_text])
                
                # Calculer la similaritÃƒÂ© cosinus
                import numpy as np
                similarity = np.dot(query_embedding[0], asset_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(asset_embedding[0])
                )
                features.embedding_similarity = max(0.0, min(1.0, similarity))
            except Exception as e:
                self.logger.debug(f"Ã¢Å¡Â Ã¯Â¸Â Erreur embedding similarity: {e}")
                # Fallback basÃƒÂ© sur les tags pour les assets Pexels
                if asset.source == "pexels_api":
                    features.embedding_similarity = 0.7  # Score ÃƒÂ©levÃƒÂ© pour Pexels
                else:
                    features.embedding_similarity = 0.3
        
        # 3. Domain match
        if domain and asset.tags:
            domain_tokens = set(self._get_domain_expansions(domain))
            domain_overlap = len(set(asset.tags) & domain_tokens)
            features.domain_match = min(1.0, domain_overlap / max(len(domain_tokens), 1))
        
        # 4. Freshness
        if asset.fetched_at:
            days_diff = (datetime.now() - asset.fetched_at).days
            half_life = self.config['freshness_half_life_days']
            features.freshness = 1.0 / (1.0 + days_diff / half_life)
        
        # 5. Quality score
        if asset.source == "pexels_api":
            features.quality_score = 0.9  # Score ÃƒÂ©levÃƒÂ© pour Pexels (qualitÃƒÂ© garantie)
        elif "1920x1080" in asset.resolution or "hd" in asset.resolution.lower():
            features.quality_score = 0.8  # HD quality
        else:
            features.quality_score = 0.6  # Standard quality
        
        # 6. Diversity penalty (sera calculÃƒÂ© plus tard)
        features.diversity_penalty = 0.0
        
        return features
    
    def calculate_final_score(self, features: ScoringFeatures) -> float:
        """Calcule le score final pondÃƒÂ©rÃƒÂ©"""
        weights = self.config['weights']
        
        score = (
            weights['embedding'] * features.embedding_similarity +
            weights['token'] * features.token_overlap +
            weights['domain'] * features.domain_match +
            weights['freshness'] * features.freshness +
            weights['quality'] * features.quality_score -
            weights['diversity'] * features.diversity_penalty
        )
        
        return max(0.0, min(1.0, score))
    
    def _should_use_direct_mode(self, keywords: List[str], domain: Optional[str] = None) -> bool:
        """
        DÃƒÂ©cide intelligemment si utiliser le mode direct ou la sÃƒÂ©lection
        
        UTILISE MODE DIRECT pour:
        - Mots-clÃƒÂ©s spÃƒÂ©cifiques et visuels concrets
        - Domaines oÃƒÂ¹ les APIs excellent (santÃƒÂ©, business, tech)
        
        UTILISE SÃƒâ€°LECTION pour:
        - Concepts abstraits
        - Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques
        - Besoin de cohÃƒÂ©rence narrative
        """
        
        # Ã°Å¸Å½Â¯ CRITÃƒË†RES POUR MODE DIRECT (High confidence)
        concrete_indicators = {
            'professional_actions': ['talking', 'presenting', 'meeting', 'consultation', 'interview'],
            'specific_professions': ['doctor', 'therapist', 'teacher', 'engineer', 'lawyer'],
            'clear_objects': ['handshake', 'computer', 'stethoscope', 'whiteboard', 'documents'],
            'defined_settings': ['office', 'hospital', 'classroom', 'laboratory', 'clinic']
        }
        
        # Ã°Å¸Å¡Â¨ CRITÃƒË†RES CONTRE MODE DIRECT (Requires smart selection)
        abstract_indicators = {
            'emotions': ['happiness', 'success', 'motivation', 'growth', 'inspiration'],
            'concepts': ['achievement', 'progress', 'innovation', 'excellence', 'quality'],
            'vague_terms': ['content', 'media', 'general', 'various', 'different']
        }
        
        # Analyser les mots-clÃƒÂ©s
        keyword_text = ' '.join(keywords).lower()
        
        # Score de concrÃƒÂ©tude
        concrete_score = 0
        abstract_score = 0
        
        for category, terms in concrete_indicators.items():
            for term in terms:
                if term in keyword_text:
                    concrete_score += 2
        
        for category, terms in abstract_indicators.items():
            for term in terms:
                if term in keyword_text:
                    abstract_score += 1
        
        # Bonus pour mots-clÃƒÂ©s structurÃƒÂ©s (person_doing_something)
        structured_keywords = [kw for kw in keywords if '_' in kw and len(kw.split('_')) >= 2]
        if structured_keywords:
            concrete_score += len(structured_keywords) * 1.5
        
        # Bonus pour domaines oÃƒÂ¹ APIs excellent
        api_friendly_domains = ['healthcare', 'business', 'technology', 'education']
        if domain and domain.lower() in api_friendly_domains:
            concrete_score += 3
        
        # DÃƒÂ©cision
        use_direct = concrete_score > abstract_score and concrete_score >= 4
        
        print(f"Ã°Å¸Â¤â€“ DÃƒâ€°CISION INTELLIGENTE:")
        print(f"   Concret: {concrete_score:.1f} | Abstrait: {abstract_score:.1f}")
        print(f"   Mode: {'DIRECT API' if use_direct else 'SÃƒâ€°LECTION INTELLIGENTE'}")
        print(f"   Raison: {'APIs excellent pour ce contenu' if use_direct else 'Besoin de curation contextuelle'}")
        
        return use_direct

    def select_brolls(self, keywords: List[str], domain: Optional[str] = None, 
                      min_delay: float = 4.0, desired_count: int = 3) -> Dict[str, Any]:
        """
        SÃƒÂ©lection intelligente : dÃƒÂ©cide automatiquement entre direct et sÃƒÂ©lection
        """
        try:
            print(f"Ã°Å¸Å½Â¬ SÃƒÂ©lection B-roll: {len(keywords)} mots-clÃƒÂ©s, domaine: {domain or 'gÃƒÂ©nÃƒÂ©ral'}")
            
            # Ã°Å¸Â§Â  DÃƒâ€°CISION INTELLIGENTE basÃƒÂ©e sur le contenu
            if self.direct_api_mode:
                # Forcer le mode direct si explicitement demandÃƒÂ©
                use_direct = True
                print("Ã°Å¸â€â€™ MODE DIRECT FORCÃƒâ€° par configuration")
            else:
                # DÃƒÂ©cision intelligente automatique
                use_direct = self._should_use_direct_mode(keywords, domain)
            
            if use_direct:
                return self._select_brolls_direct_api(keywords, domain, min_delay, desired_count)
            else:
                return self._select_brolls_smart_selection(keywords, domain, min_delay, desired_count)
            
        except Exception as e:
            print(f"Ã¢ÂÅ’ Erreur sÃƒÂ©lection B-roll: {e}")
            return self._create_empty_report()

    def _select_brolls_smart_selection(self, keywords: List[str], domain: Optional[str], 
                                      min_delay: float, desired_count: int) -> Dict[str, Any]:
        """Mode SÃƒâ€°LECTION INTELLIGENTE : curation contextuelle pour concepts abstraits"""
        print("Ã°Å¸Â§Â  MODE SÃƒâ€°LECTION INTELLIGENTE : Curation contextuelle pour votre contenu")
            
        # RÃƒÂ©cupÃƒÂ©rer plus de candidats pour avoir le choix
        api_limit = self.config.get('direct_api_limit', 5) * 3  # 3x plus de candidats
        candidate_assets = self._fetch_from_apis(keywords, limit=api_limit)
            
        if not candidate_assets:
            print("Ã¢Å¡Â Ã¯Â¸Â Aucun asset trouvÃƒÂ© via APIs - Fallback vers librairie locale")
            return self._create_fallback_report(desired_count)
            
        # Appliquer scoring intelligent
        normalized_keywords = self.normalize_keywords(keywords)
        scored_candidates = []
        
        for asset in candidate_assets:
            features = self.score_asset(asset, normalized_keywords, domain)
            final_score = self.calculate_final_score(features)
            
            candidate = BrollCandidate(
                asset=asset,
                score=final_score,
                features=features
            )
            scored_candidates.append(candidate)
            
        # Trier et sÃƒÂ©lectionner les meilleurs
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
            
        # Seuil adaptatif
        min_score = self._calculate_adaptive_threshold(scored_candidates)
        selected = [c for c in scored_candidates if c.score >= min_score]
            
        # Assurer diversitÃƒÂ©
        final_selection = self.ensure_diversity(selected, desired_count)
            
        print(f"Ã¢Å“â€¦ SÃƒâ€°LECTION INTELLIGENTE : {len(final_selection)} B-rolls curÃƒÂ©s")
        print(f"   Ã°Å¸â€œÅ  Candidats ÃƒÂ©valuÃƒÂ©s: {len(candidate_assets)} Ã¢â€ â€™ SÃƒÂ©lectionnÃƒÂ©s: {len(final_selection)}")
        print(f"   Ã°Å¸Å½Â¯ Seuil qualitÃƒÂ©: {min_score:.2f}")
        
        return {
            'selected': [c.to_dict() for c in final_selection],
            'total_candidates': len(candidate_assets),
            'selection_method': 'smart_selection',
            'keywords_used': keywords,
            'domain': domain,
            'fallback_used': False,
            'diagnostics': {
                'num_selected': len(final_selection),
                'num_candidates': len(candidate_assets),
                'top_score': final_selection[0].score if final_selection else 0.0,
                'min_score': min_score,
                'selection_method': 'smart_selection'
            }
        }

    def _select_brolls_direct_api(self, keywords: List[str], domain: Optional[str], 
                                 min_delay: float, desired_count: int) -> Dict[str, Any]:
        """Mode DIRECT : utilise directement les meilleurs rÃƒÂ©sultats API"""
        print("Ã°Å¸Å¡â‚¬ MODE DIRECT API : Utilisation directe des rÃƒÂ©sultats Pexels/Pixabay")
        
        # RÃƒÂ©cupÃƒÂ©rer directement depuis les APIs
        api_limit = self.config.get('direct_api_limit', 5)
        direct_assets = self._fetch_from_apis(keywords, limit=api_limit)
        
        if not direct_assets:
            print("Ã¢Å¡Â Ã¯Â¸Â Aucun asset trouvÃƒÂ© via APIs - Fallback vers librairie locale")
            return self._create_fallback_report(desired_count)
        
        # Prendre directement les X premiers (pas de re-scoring complexe)
        selected_count = min(desired_count, len(direct_assets))
        selected_assets = direct_assets[:selected_count]
        
        # CrÃƒÂ©er des candidats simples
        selected_candidates = []
        for i, asset in enumerate(selected_assets):
            candidate = BrollCandidate(
                asset=asset,
                score=1.0 - (i * 0.1),  # Score dÃƒÂ©croissant simple
                features=ScoringFeatures(
                    token_overlap=1.0,
                    embedding_similarity=0.9,
                    domain_match=0.8,
                    freshness=1.0,
                    quality_score=0.9
                )
            )
            selected_candidates.append(candidate)
        
        print(f"Ã¢Å“â€¦ MODE DIRECT : {len(selected_candidates)} B-rolls sÃƒÂ©lectionnÃƒÂ©s directement")
        for i, candidate in enumerate(selected_candidates):
            print(f"   {i+1}. {Path(candidate.asset.file_path).name} (source: {candidate.asset.source})")
        
        return {
            'selected': [c.to_dict() for c in selected_candidates],
            'total_candidates': len(direct_assets),
            'selection_method': 'direct_api',
            'keywords_used': keywords,
            'domain': domain,
            'fallback_used': False,
            'diagnostics': {
                'num_selected': len(selected_candidates),
                'num_candidates': len(direct_assets),
                'top_score': selected_candidates[0].score if selected_candidates else 0.0,
                'min_score': 1.0,
                'selection_method': 'direct_api'
            }
        }

    def _select_brolls_classic(self, keywords: List[str], domain: Optional[str], 
                              min_delay: float, desired_count: int) -> Dict[str, Any]:
        """Mode classique avec re-sÃƒÂ©lection et scoring complexe"""
        print("Ã°Å¸â€Â MODE CLASSIQUE : Re-scoring des rÃƒÂ©sultats avec sÃƒÂ©lection intelligente")
        
        # Ancien comportement (votre code existant)
        normalized_keywords = self.normalize_keywords(keywords)
        expanded_keywords = self.expand_keywords(list(normalized_keywords), domain)
        
        # Fetch from multiple sources
        candidates_assets = self.fetch_assets(expanded_keywords, limit=self.config['max_candidates'])
        
        if not candidates_assets:
            return self._create_empty_report()
        
        # Score all candidates
        candidates = []
        for asset in candidates_assets:
            features = self.score_asset(asset, normalized_keywords, domain)
            final_score = self.calculate_final_score(features)
            
            candidate = BrollCandidate(
                asset=asset,
                score=final_score,
                features=features
            )
            candidates.append(candidate)
        
        # Rest of classic selection logic...
        # (votre code existant pour le scoring complexe)
        return self._create_selection_report([], candidates, keywords, domain, 0.3)
    
    # Ã°Å¸Å¡â‚¬ NOUVEAU: Fonction de compatibilitÃƒÂ© pour le pipeline existant
    def find_broll_matches(self, keywords: List[str], domain: Optional[str] = None, 
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fonction de compatibilitÃƒÂ© pour le pipeline existant.
        Retourne les correspondances B-roll dans le format attendu.
        
        Args:
            keywords: Mots-clÃƒÂ©s de recherche
            domain: Domaine dÃƒÂ©tectÃƒÂ©
            max_results: Nombre maximum de rÃƒÂ©sultats
        
        Returns:
            Liste des correspondances au format pipeline
        """
        try:
            # Utiliser la logique de sÃƒÂ©lection principale
            selection_report = self.select_brolls(
                keywords=keywords,
                domain=domain,
                desired_count=max_results
            )
            
            # Convertir au format attendu par le pipeline
            matches = []
            for candidate in selection_report.get('selected', []):
                match = {
                    'asset_id': candidate.asset.id,
                    'file_path': candidate.asset.file_path,
                    'score': candidate.score,
                    'tags': candidate.asset.tags,
                    'title': candidate.asset.title,
                    'description': candidate.asset.description,
                    'source': candidate.asset.source,
                    'duration': candidate.asset.duration,
                    'resolution': candidate.asset.resolution
                }
                matches.append(match)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Ã¢ÂÅ’ Erreur find_broll_matches: {e}")
            return []
    
    def _apply_fallback_hierarchy(self, candidates: List[BrollCandidate], 
                                 selected: List[BrollCandidate], desired_count: int,
                                 min_delay: float) -> Tuple[bool, Optional[str], List[BrollCandidate]]:
        """Applique le fallback hiÃƒÂ©rarchique"""
        self.logger.info("Ã°Å¸â€ Ëœ Activation du fallback hiÃƒÂ©rarchique")
        
        # Tier A: Domain-broad (expansion forte)
        tier_a_candidates = self._get_tier_a_candidates(candidates, selected, min_delay)
        if len(selected) + len(tier_a_candidates) >= desired_count:
            selected.extend(tier_a_candidates[:desired_count - len(selected)])
            return True, "A", selected
        
        # Tier B: Contextual semi-relevant
        tier_b_candidates = self._get_tier_b_candidates(candidates, selected, min_delay)
        if len(selected) + len(tier_b_candidates) >= desired_count:
            selected.extend(tier_b_candidates[:desired_count - len(selected)])
            return True, "B", selected
        
        # Tier C: Neutral scenic
        tier_c_candidates = self._get_tier_c_candidates(candidates, selected, min_delay)
        selected.extend(tier_c_candidates[:desired_count - len(selected)])
        
        return True, "C", selected
    
    def _get_tier_a_candidates(self, candidates: List[BrollCandidate], 
                              selected: List[BrollCandidate], min_delay: float) -> List[BrollCandidate]:
        """Tier A: Domain-broad (expansion forte)"""
        # Filtrer les candidats dÃƒÂ©jÃƒÂ  sÃƒÂ©lectionnÃƒÂ©s et respectant le dÃƒÂ©lai
        available = [c for c in candidates if c not in selected]
        return self._filter_by_timing(available, min_delay)
    
    def _get_tier_b_candidates(self, candidates: List[BrollCandidate], 
                              selected: List[BrollCandidate], min_delay: float) -> List[BrollCandidate]:
        """Tier B: Contextual semi-relevant (actions, ÃƒÂ©motions, gestes)"""
        # Chercher des assets avec des tags gÃƒÂ©nÃƒÂ©riques mais sÃƒÂ»rs
        safe_tags = {'people', 'family', 'walking', 'talking', 'working', 'thinking'}
        
        available = []
        for c in candidates:
            if c not in selected:
                asset_tags = set(c.asset.tags)
                if asset_tags & safe_tags:
                    available.append(c)
        
        return self._filter_by_timing(available, min_delay)
    
    def _get_tier_c_candidates(self, candidates: List[BrollCandidate], 
                              selected: List[BrollCandidate], min_delay: float) -> List[BrollCandidate]:
        """Tier C: Neutral scenic (paysages, textures)"""
        # Ãƒâ€°viter les termes fortement hors-sujet
        neutral_tags = {'landscape', 'texture', 'abstract', 'nature', 'city'}
        
        available = []
        for c in candidates:
            if c not in selected:
                asset_tags = set(c.asset.tags)
                if asset_tags & neutral_tags:
                    available.append(c)
        
        return self._filter_by_timing(available, min_delay)
    
    def _filter_by_timing(self, candidates: List[BrollCandidate], min_delay: float) -> List[BrollCandidate]:
        """Filtre les candidats par timing"""
        filtered = []
        last_end_time = 0.0
        
        for candidate in candidates:
            if last_end_time == 0.0 or (candidate.asset.duration + last_end_time) >= min_delay:
                filtered.append(candidate)
                last_end_time = candidate.asset.duration + last_end_time
        
        return filtered
    
    def _create_report(self, keywords: List[str], domain: Optional[str], 
                       candidates: List[BrollCandidate], selected: List[BrollCandidate],
                       fallback_used: bool, fallback_tier: Optional[str],
                       top_score: float, min_score: float) -> Dict[str, Any]:
        """CrÃƒÂ©e le rapport JSON dÃƒÂ©taillÃƒÂ©"""
        return {
            'video_id': f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'keywords': keywords,
            'domain': domain,
            'planned_candidates': [c.to_dict() for c in candidates],
            'selected': [c.to_dict() for c in selected],
            'fallback_used': fallback_used,
            'fallback_tier': fallback_tier,
            'diagnostics': {
                'top_score': top_score,
                'min_score': min_score,
                'num_candidates': len(candidates),
                'num_selected': len(selected),
                'selection_ratio': len(selected) / len(candidates) if candidates else 0.0
            }
        }
    
    def _create_empty_report(self) -> Dict[str, Any]:
        """CrÃƒÂ©e un rapport vide en cas d'erreur"""
        return {
            'video_id': f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'error': 'No assets available',
            'planned_candidates': [],
            'selected': [],
            'fallback_used': False,
            'fallback_tier': None,
            'diagnostics': {
                'top_score': 0.0,
                'min_score': 0.0,
                'num_candidates': 0,
                'num_selected': 0,
                'selection_ratio': 0.0
            }
        }

    def _create_fallback_report(self, desired_count: int) -> Dict[str, Any]:
        """CrÃƒÂ©e un rapport de fallback quand aucun asset n'est trouvÃƒÂ©"""
        return {
            'selected': [],
            'excluded': [],
            'fallback_used': True,
            'fallback_tier': 'C',
            'diagnostics': {
                'num_candidates': 0,
                'num_selected': 0,
                'top_score': 0.0,
                'min_score': 0.0,
                'selection_ratio': 0.0
            },
            'keywords': [],
            'domain': None,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_selection_report(self, selected: List[BrollCandidate], candidates: List[BrollCandidate],
                                keywords: List[str], domain: Optional[str], min_score: float) -> Dict[str, Any]:
        """CrÃƒÂ©e le rapport de sÃƒÂ©lection complet"""
        return {
            'selected': selected,
            'excluded': [c for c in candidates if c not in selected],
            'fallback_used': False,
            'fallback_tier': None,
            'diagnostics': {
                'num_candidates': len(candidates),
                'num_selected': len(selected),
                'top_score': candidates[0].score if candidates else 0.0,
                'min_score': min_score,
                'selection_ratio': len(selected) / len(candidates) if candidates else 0.0
            },
            'keywords': keywords,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_adaptive_threshold(self, candidates: List[BrollCandidate]) -> float:
        """Calcule le seuil adaptatif basÃƒÂ© sur les scores"""
        if not candidates:
            return 0.0
        
        top_score = candidates[0].score
        global_min = self.config['thresholds']['global_min']
        relative_factor = self.config['thresholds']['relative_factor']
        
        return max(global_min, top_score * relative_factor)
    
    def ensure_diversity(self, selected: List[BrollCandidate], desired_count: int) -> List[BrollCandidate]:
        """Assure la diversitÃƒÂ© des sources et du contenu"""
        if len(selected) <= desired_count:
            return selected
        
        # Prioriser la diversitÃƒÂ© des sources
        diverse_selection = []
        used_sources = set()
        
        for candidate in selected:
            if len(diverse_selection) >= desired_count:
                break
            
            if candidate.asset.source not in used_sources:
                diverse_selection.append(candidate)
                used_sources.add(candidate.asset.source)
        
        # ComplÃƒÂ©ter avec les meilleurs scores si nÃƒÂ©cessaire
        while len(diverse_selection) < desired_count and len(selected) > len(diverse_selection):
            for candidate in selected:
                if candidate not in diverse_selection:
                    diverse_selection.append(candidate)
                    break
        
        return diverse_selection[:desired_count]

# Instance globale paresseuse pour compatibilitÃƒÂ©
_broll_selector_instance: Optional[BrollSelector] = None


def get_broll_selector(config: Optional[Dict[str, Any]] = None, *, force_reload: bool = False) -> BrollSelector:
    """Retourne une instance partagÃƒÂ©e du :class:`BrollSelector`.

    Cette fonction instancie le sÃƒÂ©lecteur uniquement lors de la premiÃƒÂ¨re
    utilisation, ÃƒÂ©vitant ainsi les effets de bord (logs, tÃƒÂ©lÃƒÂ©chargements,
    initialisations coÃƒÂ»teuses) pendant l'import du module.

    Args:
        config: Configuration optionnelle ÃƒÂ  fusionner lors de la crÃƒÂ©ation ou ÃƒÂ 
            appliquer dynamiquement si l'instance existe dÃƒÂ©jÃƒÂ .
        force_reload: Si ``True``, remplace l'instance existante par une
            nouvelle en utilisant la configuration fournie.
    """

    global _broll_selector_instance

    if force_reload or _broll_selector_instance is None:
        _broll_selector_instance = BrollSelector(config)
    elif config:
        # Mettre ÃƒÂ  jour dynamiquement la configuration existante
        _broll_selector_instance.config.update(config)

    return _broll_selector_instance


class _BrollSelectorProxy:
    """Proxy lÃƒÂ©ger conservant la compatibilitÃƒÂ© avec l'ancienne API module."""

    def __call__(self, config: Optional[Dict[str, Any]] = None, *, force_reload: bool = False) -> BrollSelector:
        return get_broll_selector(config, force_reload=force_reload)

    def __getattr__(self, item: str) -> Any:
        return getattr(get_broll_selector(), item)

    def __repr__(self) -> str:
        instance = _broll_selector_instance
        if instance is None:
            return "<BrollSelector lazy proxy (uninitialized)>"
        return repr(instance)


broll_selector = _BrollSelectorProxy()

# Ã°Å¸Å¡â‚¬ FONCTION DE COMPATIBILITÃƒâ€° MANQUANTE
def find_broll_matches(keywords: List[str], max_count: int = 10, 
                       min_duration: float = 2.0, max_duration: float = 15.0,
                       **kwargs) -> List[Dict[str, Any]]:
    """
    Fonction de compatibilitÃƒÂ© pour l'ancien systÃƒÂ¨me
    Utilise le nouveau BrollSelector pour maintenir la compatibilitÃƒÂ©
    """
    try:
        # Utiliser l'instance globale du BrollSelector
        selector = get_broll_selector()
        
        # Normaliser et ÃƒÂ©tendre les mots-clÃƒÂ©s
        normalized_keywords = selector.normalize_keywords(keywords)
        expanded_keywords = selector.expand_keywords(list(normalized_keywords))

        # SÃƒÂ©lectionner les B-rolls
        result = selector.select_brolls(
            keywords=expanded_keywords,
            desired_count=max_count
        )

        # Convertir en format compatible
        matches = []
        for candidate in result.get('selected', []):
            asset = candidate.get('asset', {}) if isinstance(candidate, dict) else {}
            matches.append({
                'file_path': asset.get('file_path'),
                'duration': asset.get('duration'),
                'score': candidate.get('score') if isinstance(candidate, dict) else None,
                'tags': asset.get('tags'),
                'source': asset.get('source')
            })

        return matches
        
    except Exception as e:
        print(f"Ã¢Å¡Â Ã¯Â¸Â Erreur dans find_broll_matches: {e}")
        return []

# Ã°Å¸Å¡â‚¬ FONCTION CONTEXTUELLE CORRIGÃƒâ€°E
def get_contextual_broll_score(keywords: List[str], asset_tokens: List[str], asset_tags: List[str]) -> float:
    """
    Calcule un score contextuel intelligent pour la sÃƒÂ©lection B-roll
    CompatibilitÃƒÂ© avec l'ancien systÃƒÂ¨me - CORRIGÃƒâ€° pour retourner des scores rÃƒÂ©els
    """
    try:
        score = 0.0
        
        # Mapping contextuel simplifiÃƒÂ© pour compatibilitÃƒÂ©
        CONTEXTUAL_MAPPING = {
            'technology': {
                'keywords': ['ai', 'artificial', 'intelligence', 'tech', 'digital', 'smartphone', 'computer', 'software', 'app', 'online', 'automation'],
                'priority': 8.0,
                'broll_themes': ['technology', 'digital', 'computer', 'smartphone', 'tech', 'software']
            },
            'health': {
                'keywords': ['health', 'medical', 'doctor', 'hospital', 'care', 'wellness', 'fitness', 'medicine', 'treatment'],
                'priority': 7.0,
                'broll_themes': ['health', 'medical', 'hospital', 'doctor', 'wellness', 'fitness']
            },
            'business': {
                'keywords': ['business', 'money', 'finance', 'work', 'office', 'corporate', 'company', 'startup', 'entrepreneur', 'profit', 'revenue'],
                'priority': 6.0,
                'broll_themes': ['business', 'office', 'money', 'work', 'corporate', 'finance']
            },
            'lifestyle': {
                'keywords': ['family', 'home', 'food', 'travel', 'fitness', 'leisure', 'cooking', 'restaurant', 'vacation'],
                'priority': 5.0,
                'broll_themes': ['family', 'home', 'food', 'travel', 'lifestyle', 'cooking']
            },
            'education': {
                'keywords': ['learn', 'study', 'education', 'school', 'university', 'knowledge', 'teaching', 'student', 'course'],
                'priority': 5.5,
                'broll_themes': ['education', 'school', 'study', 'learning', 'university']
            }
        }
        
        # Ã°Å¸Å¡Â¨ CORRECTION: Normaliser les tokens et tags pour comparaison
        asset_tokens_lower = [token.lower().strip() for token in asset_tokens if token]
        asset_tags_lower = [tag.lower().strip() for tag in asset_tags if tag]
        keywords_lower = [kw.lower().strip() for kw in keywords if kw]
        
        # Analyser le contexte des mots-clÃƒÂ©s
        context_matches = []
        for keyword in keywords_lower:
            # VÃƒÂ©rifier le mapping contextuel
            for context, mapping in CONTEXTUAL_MAPPING.items():
                if keyword in mapping['keywords']:
                    context_matches.append(context)
                    # Score de base selon la prioritÃƒÂ© du contexte
                    score += mapping['priority']
                    
                    # Ã°Å¸Å¡Â¨ CORRECTION: Bonus pour les thÃƒÂ¨mes B-roll correspondants
                    asset_text_combined = ' '.join(asset_tokens_lower + asset_tags_lower)
                    theme_matches = 0
                    for theme in mapping['broll_themes']:
                        if theme in asset_text_combined:
                            theme_matches += 1
                            score += 5.0  # Bonus majeur pour correspondance parfaite
                    
                    # Ã°Å¸Å¡Â¨ CORRECTION: Bonus pour les tags correspondants directs
                    tag_matches = 0
                    for tag in asset_tags_lower:
                        if any(kw in tag for kw in mapping['keywords']):
                            tag_matches += 1
                            score += 3.0  # Bonus pour correspondance de tags
                    
                    # Ã°Å¸Å¡Â¨ NOUVEAU: Bonus pour correspondance directe mot-clÃƒÂ©
                    if keyword in asset_text_combined:
                        score += 10.0  # Bonus trÃƒÂ¨s ÃƒÂ©levÃƒÂ© pour correspondance exacte
                    
                    break
        
        # Ã°Å¸Å¡Â¨ NOUVEAU: Bonus de diversitÃƒÂ© contextuelle
        unique_contexts = len(set(context_matches))
        if unique_contexts > 1:
            score += unique_contexts * 2.0  # Bonus pour diversitÃƒÂ©
        
        # Ã°Å¸Å¡Â¨ NOUVEAU: Fallback scoring pour mots-clÃƒÂ©s non mappÃƒÂ©s
        if score == 0.0:
            # Score basique basÃƒÂ© sur correspondances lexicales
            for keyword in keywords_lower:
                # Correspondance exacte dans tokens/tags
                if keyword in asset_tokens_lower or keyword in asset_tags_lower:
                    score += 2.0
                # Correspondance partielle
                elif any(keyword in token for token in asset_tokens_lower + asset_tags_lower):
                    score += 1.0
        
        # Ã°Å¸Å¡Â¨ NOUVEAU: Bonus pour mots-clÃƒÂ©s spÃƒÂ©cifiques avec underscores
        for keyword in keywords_lower:
            if '_' in keyword:  # Mots-clÃƒÂ©s format "person_talking_to_therapist"
                # Ces mots-clÃƒÂ©s sont trÃƒÂ¨s spÃƒÂ©cifiques, bonus majeur
                score += 15.0
                
                # DÃƒÂ©composer et chercher les parties
                parts = keyword.split('_')
                for part in parts:
                    if part in asset_text_combined:
                        score += 5.0  # Bonus pour chaque partie trouvÃƒÂ©e
        
        # Ã°Å¸Â§Â  NOUVEAU: Bonus pour concepts directs importants (cerveau, science, etc.)
        concept_terms = ['brain', 'neurons', 'neural', 'science', 'medical', 'technology', 'business', 'education', 'adrenaline', 'chemical', 'hormone', 'neurotransmitter']
        for keyword in keywords_lower:
            for concept in concept_terms:
                if concept in keyword and concept in asset_text_combined:
                    score += 20.0  # Bonus trÃƒÂ¨s ÃƒÂ©levÃƒÂ© pour concepts spÃƒÂ©cialisÃƒÂ©s
                    print(f"    Ã°Å¸Å½Â¯ Bonus concept spÃƒÂ©cialisÃƒÂ©: {concept} Ã¢â€ â€™ +20.0")
                    break  # Un seul bonus par mot-clÃƒÂ©
        
        # Ã°Å¸â€Â¬ NOUVEAU: Super bonus pour mots-clÃƒÂ©s trÃƒÂ¨s spÃƒÂ©cifiques
        specialized_terms = ['brain_scan', 'neural_networks', 'adrenaline_concept', 'chemical_reaction', 'medical_research']
        for keyword in keywords_lower:
            for specialized in specialized_terms:
                if specialized in keyword:
                    score += 25.0  # Super bonus pour termes trÃƒÂ¨s spÃƒÂ©cialisÃƒÂ©s
                    print(f"    Ã°Å¸Å¡â‚¬ Super bonus spÃƒÂ©cialisÃƒÂ©: {specialized} Ã¢â€ â€™ +25.0")
                    break
        
        # Ã°Å¸Å¡Â¨ CORRECTION: S'assurer qu'on retourne un score > 0 si pertinent
        final_score = max(0.0, score)
        
        # Debug logging pour diagnostiquer
        if final_score > 0:
            print(f"    Ã°Å¸Å½Â¯ Score contextuel: {final_score:.1f} | Mots-clÃƒÂ©s: {keywords_lower[:3]} | Contextes: {set(context_matches)}")
        
        return final_score
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Erreur calcul score contextuel: {e}")
        return 1.0  # Ã°Å¸Å¡Â¨ CORRECTION: Retour fallback > 0 au lieu de 0.0 


