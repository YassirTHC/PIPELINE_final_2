"""
Récupération Parallèle des B-rolls depuis les Sources Gratuites
Système optimisé pour récupérer des assets de qualité depuis toutes les sources disponibles
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import hashlib
import time
from dataclasses import dataclass
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BrollAsset:
    """Asset B-roll avec métadonnées complètes"""
    id: str
    title: str
    description: str
    url: str
    file_path: Optional[Path]
    source: str
    license: str
    resolution: tuple
    duration: float
    file_size: int
    tags: List[str]
    categories: List[str]
    download_time: float
    quality_score: float

@dataclass
class FetchResult:
    """Résultat de la récupération d'une source"""
    source: str
    assets: List[BrollAsset]
    success: bool
    error_message: Optional[str]
    fetch_time: float
    assets_count: int

class EnhancedFreeFetcher:
    """Récupérateur avancé pour sources gratuites"""
    
    def __init__(self, cache_dir: str = "cache/broll"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des sources gratuites
        self.free_sources = {
            "pexels": {
                "enabled": True,
                "api_key_env": "PEXELS_API_KEY",
                "base_url": "https://api.pexels.com",
                "endpoints": {
                    "videos": "/videos/search",
                    "photos": "/v1/search"
                },
                "quality_bonus": 1.0,
                "max_per_query": 25
            },
            "pixabay": {
                "enabled": True,
                "api_key_env": "PIXABAY_API_KEY",
                "base_url": "https://pixabay.com/api",
                "endpoints": {
                    "videos": "/videos/",
                    "photos": "/"
                },
                "quality_bonus": 0.9,
                "max_per_query": 25
            },
            "unsplash": {
                "enabled": True,
                "api_key_env": "UNSPLASH_ACCESS_KEY",
                "base_url": "https://api.unsplash.com",
                "endpoints": {
                    "photos": "/search/photos"
                },
                "quality_bonus": 1.0,
                "max_per_query": 25
            },
            "giphy": {
                "enabled": True,
                "api_key_env": "GIPHY_API_KEY",
                "base_url": "https://api.giphy.com/v1/gifs",
                "endpoints": {
                    "gifs": "/search"
                },
                "quality_bonus": 0.8,
                "max_per_query": 25
            },
            "archive_org": {
                "enabled": True,
                "api_key_env": None,  # Pas d'API key requise
                "base_url": "https://archive.org",
                "endpoints": {
                    "search": "/advancedsearch.php"
                },
                "quality_bonus": 0.7,
                "max_per_query": 25
            },
            "wikimedia": {
                "enabled": True,
                "api_key_env": None,
                "base_url": "https://commons.wikimedia.org",
                "endpoints": {
                    "search": "/w/api.php"
                },
                "quality_bonus": 0.8,
                "max_per_query": 25
            },
            "nasa": {
                "enabled": True,
                "api_key_env": "NASA_API_KEY",
                "base_url": "https://api.nasa.gov",
                "endpoints": {
                    "images": "/planetary/apod"
                },
                "quality_bonus": 0.9,
                "max_per_query": 25
            },
            "wellcome": {
                "enabled": True,
                "api_key_env": None,
                "base_url": "https://wellcomecollection.org",
                "endpoints": {
                    "search": "/works"
                },
                "quality_bonus": 0.8,
                "max_per_query": 25
            }
        }
        
        # Cache des résultats
        self.cache = {}
        self.cache_ttl = 3600  # 1 heure
        
        # Statistiques de récupération
        self.stats = {
            "total_fetches": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "total_assets": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def fetch_candidates_from_free_providers(self, keywords: List[str], domain: str, 
                                                 max_assets: int = 50) -> List[BrollAsset]:
        """Récupération parallèle depuis toutes les sources gratuites"""
        try:
            logger.info(f"Début de la récupération parallèle pour: {keywords} (domaine: {domain})")
            start_time = time.time()
            
            # Vérifier le cache
            cache_key = self._generate_cache_key(keywords, domain)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    logger.info(f"Cache hit pour: {cache_key}")
                    return cached_result["assets"]
            
            self.stats["cache_misses"] += 1
            
            # Préparer les tâches de récupération
            tasks = []
            for source_name, source_config in self.free_sources.items():
                if source_config["enabled"]:
                    task = self._fetch_from_source(source_name, source_config, keywords, domain, max_assets)
                    tasks.append(task)
            
            # Exécution parallèle avec timeout
            timeout = 30  # 30 secondes max par source
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout * len(tasks)
            )
            
            # Traitement des résultats
            all_assets = []
            for result in results:
                if isinstance(result, FetchResult) and result.success:
                    all_assets.extend(result.assets)
                    self.stats["successful_fetches"] += 1
                    self.stats["total_assets"] += result.assets_count
                else:
                    self.stats["failed_fetches"] += 1
                    if isinstance(result, Exception):
                        logger.error(f"Erreur lors de la récupération: {result}")
            
            # Déduplication et tri par qualité
            unique_assets = self._deduplicate_assets(all_assets)
            sorted_assets = sorted(unique_assets, key=lambda x: x.quality_score, reverse=True)
            
            # Limiter le nombre d'assets
            final_assets = sorted_assets[:max_assets]
            
            # Mettre en cache
            self.cache[cache_key] = {
                "assets": final_assets,
                "timestamp": time.time()
            }
            
            # Mettre à jour les statistiques
            self.stats["total_fetches"] += 1
            fetch_time = time.time() - start_time
            
            logger.info(f"Récupération terminée: {len(final_assets)} assets en {fetch_time:.2f}s")
            logger.info(f"Statistiques: {self.stats}")
            
            return final_assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération parallèle: {e}")
            return []
    
    async def _fetch_from_source(self, source_name: str, source_config: Dict, 
                                keywords: List[str], domain: str, max_assets: int) -> FetchResult:
        """Récupération depuis une source spécifique"""
        try:
            start_time = time.time()
            logger.info(f"Début récupération depuis {source_name}")
            
            # Construire la requête optimisée
            query = self._build_optimized_query(keywords, domain, source_name)
            
            # Récupérer les assets selon le type de source
            if source_name in ["pexels", "pixabay", "unsplash"]:
                assets = await self._fetch_from_photo_api(source_name, source_config, query, max_assets)
            elif source_name == "giphy":
                assets = await self._fetch_from_giphy(source_config, query, max_assets)
            elif source_name == "archive_org":
                assets = await self._fetch_from_archive_org(source_config, query, max_assets)
            elif source_name == "wikimedia":
                assets = await self._fetch_from_wikimedia(source_config, query, max_assets)
            elif source_name == "nasa":
                assets = await self._fetch_from_nasa(source_config, query, max_assets)
            elif source_name == "wellcome":
                assets = await self._fetch_from_wellcome(source_config, query, max_assets)
            else:
                assets = []
            
            fetch_time = time.time() - start_time
            
            return FetchResult(
                source=source_name,
                assets=assets,
                success=True,
                error_message=None,
                fetch_time=fetch_time,
                assets_count=len(assets)
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis {source_name}: {e}")
            return FetchResult(
                source=source_name,
                assets=[],
                success=False,
                error_message=str(e),
                fetch_time=0.0,
                assets_count=0
            )
    
    def _build_optimized_query(self, keywords: List[str], domain: str, source: str) -> str:
        """Construction de requêtes optimisées selon la source et le domaine"""
        try:
            # Expansion des mots-clés selon le domaine
            from enhanced_keyword_expansion import expand_keywords_with_synonyms
            expanded_keywords = expand_keywords_with_synonyms(keywords[0], domain)
            
            # Sélection des mots-clés les plus pertinents pour la source
            if source in ["pexels", "pixabay", "unsplash"]:
                # Sources photo : privilégier les concepts visuels
                visual_keywords = [kw for kw in expanded_keywords if len(kw.split()) <= 2]
                selected_keywords = visual_keywords[:3]
            elif source == "giphy":
                # Giphy : privilégier les concepts dynamiques
                dynamic_keywords = [kw for kw in expanded_keywords if kw in ["innovation", "progress", "development", "growth"]]
                selected_keywords = dynamic_keywords[:2] if dynamic_keywords else expanded_keywords[:2]
            else:
                # Sources générales : mots-clés principaux
                selected_keywords = expanded_keywords[:3]
            
            # Construction de la requête
            query = " ".join(selected_keywords)
            logger.info(f"Requête optimisée pour {source}: '{query}'")
            
            return query
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction de la requête: {e}")
            return " ".join(keywords[:3])
    
    async def _fetch_from_photo_api(self, source_name: str, source_config: Dict, 
                                   query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis les APIs photo (Pexels, Pixabay, Unsplash)"""
        try:
            # Simulation de récupération (remplacer par les vraies APIs)
            assets = []
            
            # Créer des assets simulés pour la démonstration
            for i in range(min(max_assets, 10)):
                asset = BrollAsset(
                    id=f"{source_name}_{i}",
                    title=f"Asset {i} from {source_name}",
                    description=f"Description for asset {i}",
                    url=f"https://{source_name}.com/asset_{i}",
                    file_path=None,
                    source=source_name,
                    license="free_to_use",
                    resolution=(1920, 1080),
                    duration=3.0,
                    file_size=1024 * 1024,
                    tags=[query, source_name, "free"],
                    categories=["general"],
                    download_time=time.time(),
                    quality_score=0.8
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis {source_name}: {e}")
            return []
    
    async def _fetch_from_giphy(self, source_config: Dict, query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis Giphy"""
        try:
            assets = []
            
            # Simulation de récupération Giphy
            for i in range(min(max_assets, 8)):
                asset = BrollAsset(
                    id=f"giphy_{i}",
                    title=f"GIF {i} for {query}",
                    description=f"Animated GIF for {query}",
                    url=f"https://giphy.com/gif_{i}",
                    file_path=None,
                    source="giphy",
                    license="free_to_use",
                    resolution=(480, 270),  # Format GIF standard
                    duration=2.0,
                    file_size=512 * 1024,
                    tags=[query, "gif", "animated", "viral"],
                    categories=["entertainment"],
                    download_time=time.time(),
                    quality_score=0.7
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis Giphy: {e}")
            return []
    
    async def _fetch_from_archive_org(self, source_config: Dict, query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis Archive.org"""
        try:
            assets = []
            
            # Simulation de récupération Archive.org
            for i in range(min(max_assets, 6)):
                asset = BrollAsset(
                    id=f"archive_{i}",
                    title=f"Historical content {i} for {query}",
                    description=f"Historical content from archive.org",
                    url=f"https://archive.org/details/content_{i}",
                    file_path=None,
                    source="archive_org",
                    license="public_domain",
                    resolution=(1280, 720),
                    duration=4.0,
                    file_size=2048 * 1024,
                    tags=[query, "historical", "archive", "public_domain"],
                    categories=["history"],
                    download_time=time.time(),
                    quality_score=0.6
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis Archive.org: {e}")
            return []
    
    async def _fetch_from_wikimedia(self, source_config: Dict, query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis Wikimedia Commons"""
        try:
            assets = []
            
            # Simulation de récupération Wikimedia
            for i in range(min(max_assets, 5)):
                asset = BrollAsset(
                    id=f"wikimedia_{i}",
                    title=f"Wikimedia content {i} for {query}",
                    description=f"Educational content from Wikimedia Commons",
                    url=f"https://commons.wikimedia.org/wiki/File:content_{i}",
                    file_path=None,
                    source="wikimedia",
                    license="creative_commons",
                    resolution=(1600, 900),
                    duration=2.5,
                    file_size=1536 * 1024,
                    tags=[query, "educational", "commons", "creative_commons"],
                    categories=["education"],
                    download_time=time.time(),
                    quality_score=0.7
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis Wikimedia: {e}")
            return []
    
    async def _fetch_from_nasa(self, source_config: Dict, query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis NASA Images"""
        try:
            assets = []
            
            # Simulation de récupération NASA
            for i in range(min(max_assets, 4)):
                asset = BrollAsset(
                    id=f"nasa_{i}",
                    title=f"NASA content {i} for {query}",
                    description=f"Scientific content from NASA",
                    url=f"https://images.nasa.gov/details/content_{i}",
                    file_path=None,
                    source="nasa",
                    license="public_domain",
                    resolution=(1920, 1080),
                    duration=3.5,
                    file_size=3072 * 1024,
                    tags=[query, "nasa", "scientific", "space", "public_domain"],
                    categories=["science"],
                    download_time=time.time(),
                    quality_score=0.8
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis NASA: {e}")
            return []
    
    async def _fetch_from_wellcome(self, source_config: Dict, query: str, max_assets: int) -> List[BrollAsset]:
        """Récupération depuis Wellcome Collection"""
        try:
            assets = []
            
            # Simulation de récupération Wellcome
            for i in range(min(max_assets, 3)):
                asset = BrollAsset(
                    id=f"wellcome_{i}",
                    title=f"Wellcome content {i} for {query}",
                    description=f"Medical and scientific content from Wellcome Collection",
                    url=f"https://wellcomecollection.org/works/content_{i}",
                    file_path=None,
                    source="wellcome",
                    license="creative_commons",
                    resolution=(1440, 810),
                    duration=2.8,
                    file_size=1792 * 1024,
                    tags=[query, "medical", "scientific", "wellcome", "creative_commons"],
                    categories=["medical"],
                    download_time=time.time(),
                    quality_score=0.7
                )
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération depuis Wellcome: {e}")
            return []
    
    def _deduplicate_assets(self, assets: List[BrollAsset]) -> List[BrollAsset]:
        """Déduplication des assets basée sur l'URL et le titre"""
        try:
            seen_urls = set()
            seen_titles = set()
            unique_assets = []
            
            for asset in assets:
                # Vérifier l'URL
                if asset.url in seen_urls:
                    continue
                
                # Vérifier le titre (normalisé)
                normalized_title = asset.title.lower().strip()
                if normalized_title in seen_titles:
                    continue
                
                # Ajouter aux ensembles vus
                seen_urls.add(asset.url)
                seen_titles.add(normalized_title)
                unique_assets.append(asset)
            
            logger.info(f"Déduplication: {len(assets)} → {len(unique_assets)} assets uniques")
            return unique_assets
            
        except Exception as e:
            logger.error(f"Erreur lors de la déduplication: {e}")
            return assets
    
    def _generate_cache_key(self, keywords: List[str], domain: str) -> str:
        """Génération de la clé de cache"""
        try:
            # Combiner les mots-clés et le domaine
            key_string = f"{domain}:{':'.join(sorted(keywords))}"
            
            # Générer un hash MD5
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la clé de cache: {e}")
            return f"{domain}_{hash(str(keywords))}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de récupération"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Vide le cache"""
        self.cache.clear()
        logger.info("Cache vidé")

# Instance globale pour utilisation dans le pipeline
enhanced_fetcher = EnhancedFreeFetcher()

async def fetch_candidates_from_free_providers(keywords: List[str], domain: str, 
                                             max_assets: int = 50) -> List[BrollAsset]:
    """Fonction utilitaire pour la récupération parallèle"""
    return await enhanced_fetcher.fetch_candidates_from_free_providers(keywords, domain, max_assets)

def get_fetcher_stats() -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer les statistiques"""
    return enhanced_fetcher.get_stats() 