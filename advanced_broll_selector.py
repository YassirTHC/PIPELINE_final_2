ï»¿# -*- coding: utf-8 -*-
"""
SÃƒÂ©lecteur B-roll AvancÃƒÂ© avec Gestion VidÃƒÂ©o RÃƒÂ©elle
Version de production avec fichiers vidÃƒÂ©o, mÃƒÂ©tadonnÃƒÂ©es et analyse de contenu visuel
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import sqlite3
import hashlib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
import imageio

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('broll_selector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """MÃƒÂ©tadonnÃƒÂ©es complÃƒÂ¨tes d'une vidÃƒÂ©o B-roll"""
    id: str
    file_path: Path
    title: str
    description: str
    duration: float
    resolution: Tuple[int, int]
    fps: float
    bitrate: int
    file_size: int
    created_date: datetime
    modified_date: datetime
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    content_rating: str = "G"
    language: str = "en"
    source: str = "unknown"
    license: str = "unknown"
    thumbnail_path: Optional[Path] = None
    preview_path: Optional[Path] = None

@dataclass
class VisualAnalysis:
    """Analyse visuelle d'une vidÃƒÂ©o B-roll"""
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    brightness_level: float = 0.0
    contrast_level: float = 0.0
    motion_intensity: float = 0.0
    scene_changes: List[float] = field(default_factory=list)
    face_detected: bool = False
    text_detected: bool = False
    object_detection: List[Dict[str, Any]] = field(default_factory=list)
    visual_complexity: float = 0.0
    aesthetic_score: float = 0.0

@dataclass
class BrollCandidate:
    """Candidat B-roll avec analyse complÃƒÂ¨te"""
    metadata: VideoMetadata
    visual_analysis: VisualAnalysis
    semantic_similarity: float = 0.0
    context_relevance: float = 0.0
    quality_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0
    selection_reason: str = ""

@dataclass
class BrollSelection:
    """SÃƒÂ©lection B-roll finale avec alternatives"""
    primary_broll: BrollCandidate
    alternative_brolls: List[BrollCandidate] = field(default_factory=list)
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    context_match_score: float = 0.0
    diversity_score: float = 0.0

class AdvancedBrollSelector:
    """SÃƒÂ©lecteur B-roll avancÃƒÂ© avec gestion vidÃƒÂ©o rÃƒÂ©elle"""
    
    def __init__(self, database_path: str = "broll_database.db"):
        self.database_path = database_path
        self.db_connection = None
        self.video_cache = {}
        self.visual_analyzer = None
        self.semantic_matcher = None
        
        # Initialiser la base de donnÃƒÂ©es
        self._initialize_database()
        
        # Initialiser l'analyseur visuel
        self._initialize_visual_analyzer()
        
        logger.info("SÃƒÂ©lecteur B-roll avancÃƒÂ© initialisÃƒÂ©")

    def _initialize_database(self):
        """Initialise la base de donnÃƒÂ©es SQLite pour les B-rolls"""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            cursor = self.db_connection.cursor()
            
            # Table des mÃƒÂ©tadonnÃƒÂ©es vidÃƒÂ©o
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_metadata (
                    id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    duration REAL,
                    resolution_width INTEGER,
                    resolution_height INTEGER,
                    fps REAL,
                    bitrate INTEGER,
                    file_size INTEGER,
                    created_date TEXT,
                    modified_date TEXT,
                    tags TEXT,
                    categories TEXT,
                    content_rating TEXT,
                    language TEXT,
                    source TEXT,
                    license TEXT,
                    thumbnail_path TEXT,
                    preview_path TEXT
                )
            ''')
            
            # Table des analyses visuelles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visual_analysis (
                    video_id TEXT PRIMARY KEY,
                    dominant_colors TEXT,
                    brightness_level REAL,
                    contrast_level REAL,
                    motion_intensity REAL,
                    scene_changes TEXT,
                    face_detected BOOLEAN,
                    text_detected BOOLEAN,
                    object_detection TEXT,
                    visual_complexity REAL,
                    aesthetic_score REAL,
                    analysis_date TEXT,
                    FOREIGN KEY (video_id) REFERENCES video_metadata (id)
                )
            ''')
            
            # Table des scores de contexte
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_scores (
                    video_id TEXT,
                    context_type TEXT,
                    relevance_score REAL,
                    last_updated TEXT,
                    PRIMARY KEY (video_id, context_type),
                    FOREIGN KEY (video_id) REFERENCES video_metadata (id)
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON video_metadata(tags)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories ON video_metadata(categories)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_duration ON video_metadata(duration)')
            
            self.db_connection.commit()
            logger.info("Base de donnÃƒÂ©es B-roll initialisÃƒÂ©e")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de donnÃƒÂ©es: {e}")
            self.db_connection = None

    def _initialize_visual_analyzer(self):
        """Initialise l'analyseur visuel"""
        try:
            # VÃƒÂ©rifier si OpenCV est disponible
            import cv2
            self.visual_analyzer = {
                'opencv_available': True,
                'face_cascade': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            }
            logger.info("Analyseur visuel OpenCV initialisÃƒÂ©")
        except ImportError:
            logger.warning("OpenCV non disponible, analyse visuelle limitÃƒÂ©e")
            self.visual_analyzer = {'opencv_available': False}

    async def add_broll_to_database(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """Ajoute un nouveau B-roll ÃƒÂ  la base de donnÃƒÂ©es"""
        try:
            if not self.db_connection:
                logger.error("Base de donnÃƒÂ©es non initialisÃƒÂ©e")
                return False
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error(f"Fichier vidÃƒÂ©o non trouvÃƒÂ©: {file_path}")
                return False
            
            # GÃƒÂ©nÃƒÂ©rer un ID unique
            video_id = self._generate_video_id(file_path)
            
            # Extraire les mÃƒÂ©tadonnÃƒÂ©es vidÃƒÂ©o
            video_metadata = await self._extract_video_metadata(file_path_obj, metadata)
            
            # Analyser le contenu visuel
            visual_analysis = await self._analyze_video_visual(file_path_obj)
            
            # Sauvegarder en base
            cursor = self.db_connection.cursor()
            
            # InsÃƒÂ©rer les mÃƒÂ©tadonnÃƒÂ©es
            cursor.execute('''
                INSERT OR REPLACE INTO video_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                str(file_path_obj),
                video_metadata.title,
                video_metadata.description,
                video_metadata.duration,
                video_metadata.resolution[0],
                video_metadata.resolution[1],
                video_metadata.fps,
                video_metadata.bitrate,
                video_metadata.file_size,
                video_metadata.created_date.isoformat(),
                video_metadata.modified_date.isoformat(),
                json.dumps(video_metadata.tags),
                json.dumps(video_metadata.categories),
                video_metadata.content_rating,
                video_metadata.language,
                video_metadata.source,
                video_metadata.license,
                str(video_metadata.thumbnail_path) if video_metadata.thumbnail_path else None,
                str(video_metadata.preview_path) if video_metadata.preview_path else None
            ))
            
            # InsÃƒÂ©rer l'analyse visuelle
            cursor.execute('''
                INSERT OR REPLACE INTO visual_analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                json.dumps(visual_analysis.dominant_colors),
                visual_analysis.brightness_level,
                visual_analysis.contrast_level,
                visual_analysis.motion_intensity,
                json.dumps(visual_analysis.scene_changes),
                visual_analysis.face_detected,
                visual_analysis.text_detected,
                json.dumps(visual_analysis.object_detection),
                visual_analysis.visual_complexity,
                visual_analysis.aesthetic_score,
                datetime.now().isoformat()
            ))
            
            self.db_connection.commit()
            logger.info(f"B-roll ajoutÃƒÂ© ÃƒÂ  la base: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur ajout B-roll: {e}")
            return False

    def _generate_video_id(self, file_path: str) -> str:
        """GÃƒÂ©nÃƒÂ¨re un ID unique pour la vidÃƒÂ©o"""
        # Utiliser le hash du chemin + la taille du fichier
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            file_size = file_path_obj.stat().st_size
            content = f"{file_path}_{file_size}"
        else:
            content = file_path
        
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _extract_video_metadata(self, file_path: Path, user_metadata: Dict[str, Any]) -> VideoMetadata:
        """Extrait les mÃƒÂ©tadonnÃƒÂ©es vidÃƒÂ©o du fichier"""
        try:
            # MÃƒÂ©tadonnÃƒÂ©es de base du fichier
            stat = file_path.stat()
            created_date = datetime.fromtimestamp(stat.st_ctime)
            modified_date = datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size
            
            # MÃƒÂ©tadonnÃƒÂ©es vidÃƒÂ©o avec OpenCV
            video_metadata = VideoMetadata(
                id="",  # Sera dÃƒÂ©fini plus tard
                file_path=file_path,
                title=user_metadata.get('title', file_path.stem),
                description=user_metadata.get('description', ''),
                duration=0.0,
                resolution=(0, 0),
                fps=0.0,
                bitrate=0,
                file_size=file_size,
                created_date=created_date,
                modified_date=modified_date,
                tags=user_metadata.get('tags', []),
                categories=user_metadata.get('categories', []),
                content_rating=user_metadata.get('content_rating', 'G'),
                language=user_metadata.get('language', 'en'),
                source=user_metadata.get('source', 'unknown'),
                license=user_metadata.get('license', 'unknown')
            )
            
            # Extraire les mÃƒÂ©tadonnÃƒÂ©es vidÃƒÂ©o avec OpenCV
            if self.visual_analyzer and self.visual_analyzer.get('opencv_available'):
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    # RÃƒÂ©solution
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_metadata.resolution = (width, height)
                    
                    # FPS
                    video_metadata.fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # DurÃƒÂ©e
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if video_metadata.fps > 0:
                        video_metadata.duration = frame_count / video_metadata.fps
                    
                    # Bitrate (estimation)
                    video_metadata.bitrate = int((file_size * 8) / video_metadata.duration) if video_metadata.duration > 0 else 0
                    
                    cap.release()
            
            # CrÃƒÂ©er le thumbnail et la preview
            await self._create_video_previews(file_path, video_metadata)
            
            return video_metadata
            
        except Exception as e:
            logger.error(f"Erreur extraction mÃƒÂ©tadonnÃƒÂ©es: {e}")
            # Retourner des mÃƒÂ©tadonnÃƒÂ©es de base
            return VideoMetadata(
                id="",
                file_path=file_path,
                title=file_path.stem,
                description="Erreur lors de l'extraction des mÃƒÂ©tadonnÃƒÂ©es",
                duration=0.0,
                resolution=(0, 0),
                fps=0.0,
                bitrate=0,
                file_size=file_path.stat().st_size,
                created_date=datetime.now(),
                modified_date=datetime.now()
            )

    async def _create_video_previews(self, file_path: Path, metadata: VideoMetadata):
        """CrÃƒÂ©e les aperÃƒÂ§us vidÃƒÂ©o (thumbnail et preview)"""
        try:
            if not self.visual_analyzer or not self.visual_analyzer.get('opencv_available'):
                return
            
            # CrÃƒÂ©er le dossier des aperÃƒÂ§us
            preview_dir = Path("broll_previews")
            preview_dir.mkdir(exist_ok=True)
            
            # Thumbnail (premiÃƒÂ¨re frame)
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    thumbnail_path = preview_dir / f"{metadata.id}_thumb.jpg"
                    cv2.imwrite(str(thumbnail_path), frame)
                    metadata.thumbnail_path = thumbnail_path
                
                # Preview (frame au milieu)
                if metadata.duration > 0:
                    middle_frame = int(metadata.duration * metadata.fps / 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                    ret, frame = cap.read()
                    if ret:
                        preview_path = preview_dir / f"{metadata.id}_preview.jpg"
                        cv2.imwrite(str(preview_path), frame)
                        metadata.preview_path = preview_path
                
                cap.release()
                
        except Exception as e:
            logger.error(f"Erreur crÃƒÂ©ation aperÃƒÂ§us: {e}")

    async def _analyze_video_visual(self, file_path: Path) -> VisualAnalysis:
        """Analyse le contenu visuel de la vidÃƒÂ©o"""
        try:
            if not self.visual_analyzer or not self.visual_analyzer.get('opencv_available'):
                return VisualAnalysis()
            
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return VisualAnalysis()
            
            frames_analyzed = 0
            total_brightness = 0.0
            total_contrast = 0.0
            motion_frames = 0
            scene_changes = []
            faces_detected = 0
            text_detected = 0
            dominant_colors = []
            
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyser une frame sur 10 pour la performance
                if frame_count % 10 == 0:
                    # Couleurs dominantes
                    if len(dominant_colors) < 5:
                        colors = self._extract_dominant_colors(frame)
                        dominant_colors.extend(colors)
                    
                    # LuminositÃƒÂ© et contraste
                    brightness, contrast = self._analyze_frame_properties(frame)
                    total_brightness += brightness
                    total_contrast += contrast
                    
                    # DÃƒÂ©tection de visages
                    if self._detect_faces(frame):
                        faces_detected += 1
                    
                    # DÃƒÂ©tection de texte (approximation basÃƒÂ©e sur les contours)
                    if self._detect_text_like_content(frame):
                        text_detected += 1
                    
                    # DÃƒÂ©tection de mouvement
                    if prev_frame is not None:
                        motion = self._calculate_motion(prev_frame, frame)
                        if motion > 0.1:  # Seuil de mouvement
                            motion_frames += 1
                    
                    prev_frame = frame.copy()
                    frames_analyzed += 1
                
                frame_count += 1
            
            cap.release()
            
            # Calculer les scores moyens
            avg_brightness = total_brightness / frames_analyzed if frames_analyzed > 0 else 0.0
            avg_contrast = total_contrast / frames_analyzed if frames_analyzed > 0 else 0.0
            motion_intensity = motion_frames / frames_analyzed if frames_analyzed > 0 else 0.0
            
            # DÃƒÂ©tecter les changements de scÃƒÂ¨ne (approximation)
            scene_changes = self._detect_scene_changes(file_path)
            
            # Score de complexitÃƒÂ© visuelle
            visual_complexity = self._calculate_visual_complexity(
                frames_analyzed, motion_intensity, len(dominant_colors)
            )
            
            # Score esthÃƒÂ©tique
            aesthetic_score = self._calculate_aesthetic_score(
                avg_brightness, avg_contrast, motion_intensity, visual_complexity
            )
            
            return VisualAnalysis(
                dominant_colors=dominant_colors[:5],
                brightness_level=avg_brightness,
                contrast_level=avg_contrast,
                motion_intensity=motion_intensity,
                scene_changes=scene_changes,
                face_detected=faces_detected > 0,
                text_detected=text_detected > 0,
                object_detection=[],
                visual_complexity=visual_complexity,
                aesthetic_score=aesthetic_score
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse visuelle: {e}")
            return VisualAnalysis()

    def _extract_dominant_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extrait les couleurs dominantes d'une frame"""
        try:
            # Redimensionner pour la performance
            small_frame = cv2.resize(frame, (50, 50))
            
            # Convertir en RGB et aplatir
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pixels = rgb_frame.reshape(-1, 3)
            
            # Utiliser K-means pour trouver les couleurs dominantes
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixels)
            
            # Obtenir les couleurs dominantes
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except ImportError:
            # Fallback basique sans scikit-learn
            return [(128, 128, 128)]  # Gris par dÃƒÂ©faut
        except Exception as e:
            logger.error(f"Erreur extraction couleurs: {e}")
            return [(128, 128, 128)]

    def _analyze_frame_properties(self, frame: np.ndarray) -> Tuple[float, float]:
        """Analyse les propriÃƒÂ©tÃƒÂ©s d'une frame (luminositÃƒÂ©, contraste)"""
        try:
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # LuminositÃƒÂ© (moyenne des pixels)
            brightness = np.mean(gray) / 255.0
            
            # Contraste (ÃƒÂ©cart-type des pixels)
            contrast = np.std(gray) / 255.0
            
            return brightness, contrast
            
        except Exception as e:
            logger.error(f"Erreur analyse propriÃƒÂ©tÃƒÂ©s frame: {e}")
            return 0.5, 0.5

    def _detect_faces(self, frame: np.ndarray) -> bool:
        """DÃƒÂ©tecte les visages dans une frame"""
        try:
            if not self.visual_analyzer or not self.visual_analyzer.get('face_cascade'):
                return False
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.visual_analyzer['face_cascade'].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            return len(faces) > 0
            
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©tection visages: {e}")
            return False

    def _detect_text_like_content(self, frame: np.ndarray) -> bool:
        """DÃƒÂ©tecte le contenu ressemblant ÃƒÂ  du texte (approximation)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # DÃƒÂ©tecter les contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Chercher des contours rectangulaires qui pourraient ÃƒÂªtre du texte
            text_like_contours = 0
            for contour in contours:
                if len(contour) >= 4:
                    # Approximation du rectangle englobant
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    
                    # Ratio largeur/hauteur typique du texte
                    if 0.1 < width/height < 10 and cv2.contourArea(contour) > 100:
                        text_like_contours += 1
            
            return text_like_contours > 2
            
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©tection texte: {e}")
            return False

    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calcule le niveau de mouvement entre deux frames"""
        try:
            # DiffÃƒÂ©rence entre frames
            diff = cv2.absdiff(prev_frame, curr_frame)
            
            # Convertir en niveaux de gris
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Moyenne de la diffÃƒÂ©rence (plus c'est ÃƒÂ©levÃƒÂ©, plus il y a de mouvement)
            motion_score = np.mean(gray_diff) / 255.0
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Erreur calcul mouvement: {e}")
            return 0.0

    def _detect_scene_changes(self, file_path: Path) -> List[float]:
        """DÃƒÂ©tecte les changements de scÃƒÂ¨ne (approximation)"""
        try:
            # Pour l'instant, retourner des changements simulÃƒÂ©s
            # En production, utiliser des algorithmes plus sophistiquÃƒÂ©s
            return [0.0, 0.5, 1.0]  # Changements au dÃƒÂ©but, milieu et fin
            
        except Exception as e:
            logger.error(f"Erreur dÃƒÂ©tection changements de scÃƒÂ¨ne: {e}")
            return []

    def _calculate_visual_complexity(self, frames_analyzed: int, motion_intensity: float, color_variety: int) -> float:
        """Calcule la complexitÃƒÂ© visuelle de la vidÃƒÂ©o"""
        try:
            # Score basÃƒÂ© sur plusieurs facteurs
            motion_score = motion_intensity * 0.4
            color_score = min(color_variety / 5.0, 1.0) * 0.3
            frame_score = min(frames_analyzed / 100.0, 1.0) * 0.3
            
            complexity = motion_score + color_score + frame_score
            return min(1.0, complexity)
            
        except Exception as e:
            logger.error(f"Erreur calcul complexitÃƒÂ© visuelle: {e}")
            return 0.5

    def _calculate_aesthetic_score(self, brightness: float, contrast: float, motion: float, complexity: float) -> float:
        """Calcule un score esthÃƒÂ©tique basÃƒÂ© sur plusieurs critÃƒÂ¨res"""
        try:
            # Score basÃƒÂ© sur des critÃƒÂ¨res esthÃƒÂ©tiques
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal autour de 0.5
            contrast_score = contrast  # Plus de contraste = mieux
            motion_score = 1.0 - abs(motion - 0.3) * 2  # Mouvement modÃƒÂ©rÃƒÂ© optimal
            complexity_score = complexity  # ComplexitÃƒÂ© modÃƒÂ©rÃƒÂ©e = mieux
            
            # Moyenne pondÃƒÂ©rÃƒÂ©e
            aesthetic_score = (
                brightness_score * 0.3 +
                contrast_score * 0.3 +
                motion_score * 0.2 +
                complexity_score * 0.2
            )
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.error(f"Erreur calcul score esthÃƒÂ©tique: {e}")
            return 0.5

    async def select_contextual_brolls(self, 
                                     context_analysis: Dict[str, Any],
                                     segment_analysis: Dict[str, Any],
                                     max_results: int = 5) -> BrollSelection:
        """SÃƒÂ©lectionne les B-rolls contextuels appropriÃƒÂ©s"""
        try:
            logger.info(f"SÃƒÂ©lection B-roll contextuelle pour: {segment_analysis.get('semantic_context', 'unknown')}")
            
            # RÃƒÂ©cupÃƒÂ©rer les candidats de la base de donnÃƒÂ©es
            candidates = await self._get_broll_candidates_from_db(segment_analysis)
            
            if not candidates:
                logger.warning("Aucun candidat B-roll trouvÃƒÂ© en base")
                return None
            
            # Analyser et scorer chaque candidat
            scored_candidates = []
            for candidate in candidates:
                scored_candidate = await self._score_broll_candidate(
                    candidate, context_analysis, segment_analysis
                )
                scored_candidates.append(scored_candidate)
            
            # Trier par score final
            scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            # SÃƒÂ©lectionner le meilleur et les alternatives
            if not scored_candidates:
                return None
            
            primary_broll = scored_candidates[0]
            alternative_brolls = scored_candidates[1:max_results]
            
            # CrÃƒÂ©er la sÃƒÂ©lection finale
            selection = BrollSelection(
                primary_broll=primary_broll,
                alternative_brolls=alternative_brolls,
                selection_metadata={
                    'context_type': segment_analysis.get('semantic_context', 'unknown'),
                    'selection_timestamp': datetime.now().isoformat(),
                    'total_candidates': len(scored_candidates)
                },
                context_match_score=primary_broll.context_relevance,
                diversity_score=self._calculate_diversity_score(primary_broll, alternative_brolls)
            )
            
            logger.info(f"B-roll sÃƒÂ©lectionnÃƒÂ©: {primary_broll.metadata.title} (score: {primary_broll.final_score:.2f})")
            return selection
            
        except Exception as e:
            logger.error(f"Erreur sÃƒÂ©lection B-roll: {e}")
            return None

    # MÃƒâ€°THODES CRITIQUES MANQUANTES - IMPLÃƒâ€°MENTATION IMMÃƒâ€°DIATE
    def select_broll_for_segment(self, segment_analysis: Dict[str, Any], max_results: int = 5) -> List[BrollCandidate]:
        """SÃƒÂ©lection B-roll pour segment - Interface standard (SYNCHRONE)"""
        try:
            logger.info(f"SÃƒÂ©lection synchrone B-roll pour segment: {segment_analysis.get('semantic_context', 'unknown')}")
            
            # CrÃƒÂ©er un contexte d'analyse minimal
            context_analysis = {
                'semantic_context': segment_analysis.get('semantic_context', 'general'),
                'main_topics': segment_analysis.get('main_topics', ['general']),
                'sentiment': segment_analysis.get('sentiment_score', 0.0)
            }
            
            # Utiliser la mÃƒÂ©thode asynchrone existante dans un contexte synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.select_contextual_brolls(context_analysis, segment_analysis, max_results))
                if result:
                    return [result.primary_broll] + result.alternative_brolls
                else:
                    return []
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de la sÃƒÂ©lection synchrone B-roll: {e}")
            return []

    def calculate_diversity_score(self, broll_list: List[BrollCandidate]) -> float:
        """Calcul du score de diversitÃƒÂ© - Interface standard (SYNCHRONE)"""
        try:
            logger.info(f"Calcul synchrone du score de diversitÃƒÂ© pour {len(broll_list)} B-rolls")
            
            if len(broll_list) <= 1:
                return 1.0  # DiversitÃƒÂ© maximale pour 0 ou 1 ÃƒÂ©lÃƒÂ©ment
            
            # Calculer la diversitÃƒÂ© basÃƒÂ©e sur les mÃƒÂ©tadonnÃƒÂ©es
            diversity_score = self._calculate_visual_diversity_sync(broll_list)
            
            logger.info(f"Score de diversitÃƒÂ© calculÃƒÂ©: {diversity_score:.3f}")
            return diversity_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de diversitÃƒÂ©: {e}")
            return 0.5

    def get_broll_candidates(self, keywords: List[str], max_results: int = 10) -> List[BrollCandidate]:
        """Candidats B-roll - Interface standard (SYNCHRONE)"""
        try:
            logger.info(f"RÃƒÂ©cupÃƒÂ©ration synchrone de {max_results} candidats B-roll pour mots-clÃƒÂ©s: {keywords}")
            
            # CrÃƒÂ©er une analyse de segment basÃƒÂ©e sur les mots-clÃƒÂ©s
            segment_analysis = {
                'semantic_context': 'keyword_based',
                'key_phrases': keywords,
                'main_topics': keywords[:3] if len(keywords) >= 3 else keywords
            }
            
            # Utiliser la mÃƒÂ©thode asynchrone existante dans un contexte synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.select_contextual_brolls(
                    {'keywords': keywords}, segment_analysis, max_results
                ))
                if result:
                    return [result.primary_broll] + result.alternative_brolls
                else:
                    return []
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de la rÃƒÂ©cupÃƒÂ©ration des candidats B-roll: {e}")
            return []

    # MÃƒâ€°THODES DE SUPPORT POUR LES INTERFACES STANDARD
    def _calculate_visual_diversity_sync(self, broll_list: List[BrollCandidate]) -> float:
        """Calcul de la diversitÃƒÂ© visuelle (synchrone)"""
        try:
            if len(broll_list) <= 1:
                return 1.0
            
            # Calculer la diversitÃƒÂ© basÃƒÂ©e sur les couleurs dominantes
            dominant_colors = []
            for broll in broll_list:
                if hasattr(broll, 'visual_analysis') and broll.visual_analysis:
                    colors = broll.visual_analysis.dominant_colors
                    if colors:
                        dominant_colors.extend(colors)
            
            # Calculer la diversitÃƒÂ© des couleurs
            if dominant_colors:
                unique_colors = len(set(tuple(color) for color in dominant_colors))
                total_colors = len(dominant_colors)
                color_diversity = unique_colors / total_colors if total_colors > 0 else 0.0
            else:
                color_diversity = 0.5
            
            # DiversitÃƒÂ© basÃƒÂ©e sur la durÃƒÂ©e
            durations = []
            for broll in broll_list:
                if hasattr(broll, 'metadata') and broll.metadata:
                    durations.append(broll.metadata.duration)
            
            if durations:
                duration_variance = np.var(durations) if len(durations) > 1 else 0.0
                duration_diversity = min(1.0, duration_variance / 10.0)  # Normaliser
            else:
                duration_diversity = 0.5
            
            # Score de diversitÃƒÂ© final
            final_diversity = (color_diversity * 0.6) + (duration_diversity * 0.4)
            return max(0.0, min(1.0, final_diversity))
            
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de la diversitÃƒÂ© visuelle: {e}")
            return 0.5

    async def _get_broll_candidates_from_db(self, segment_analysis: Dict[str, Any]) -> List[BrollCandidate]:
        """RÃƒÂ©cupÃƒÂ¨re les candidats B-roll depuis la base de donnÃƒÂ©es"""
        try:
            if not self.db_connection:
                logger.error("Base de donnÃƒÂ©es non initialisÃƒÂ©e")
                return []
            
            cursor = self.db_connection.cursor()
            
            # Construire la requÃƒÂªte basÃƒÂ©e sur le contexte
            context_type = segment_analysis.get('semantic_context', 'general')
            
            # RequÃƒÂªte de base
            query = '''
                SELECT vm.*, va.* FROM video_metadata vm
                LEFT JOIN visual_analysis va ON vm.id = va.video_id
                WHERE 1=1
            '''
            params = []
            
            # Filtrer par catÃƒÂ©gories si disponibles
            if context_type != 'general':
                query += ' AND (vm.categories LIKE ? OR vm.tags LIKE ?)'
                context_pattern = f'%{context_type}%'
                params.extend([context_pattern, context_pattern])
            
            # Filtrer par durÃƒÂ©e appropriÃƒÂ©e
            target_duration = segment_analysis.get('duration', 10.0)
            query += ' AND vm.duration BETWEEN ? AND ?'
            min_duration = max(2.0, target_duration * 0.5)
            max_duration = target_duration * 2.0
            params.extend([min_duration, max_duration])
            
            # Limiter les rÃƒÂ©sultats
            query += ' LIMIT 20'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            candidates = []
            for row in rows:
                try:
                    # Reconstruire les objets depuis la base
                    metadata = self._reconstruct_metadata_from_db(row)
                    visual_analysis = self._reconstruct_visual_analysis_from_db(row)
                    
                    candidate = BrollCandidate(
                        metadata=metadata,
                        visual_analysis=visual_analysis
                    )
                    candidates.append(candidate)
                    
                except Exception as e:
                    logger.warning(f"Erreur reconstruction candidat: {e}")
                    continue
            
            return candidates
            
        except Exception as e:
            logger.error(f"Erreur rÃƒÂ©cupÃƒÂ©ration candidats: {e}")
            return []

    def _reconstruct_metadata_from_db(self, row: tuple) -> VideoMetadata:
        """Reconstruit un objet VideoMetadata depuis la base de donnÃƒÂ©es"""
        try:
            # Les colonnes sont dans l'ordre de la requÃƒÂªte JOIN
            metadata = VideoMetadata(
                id=row[0],
                file_path=Path(row[1]),
                title=row[2],
                description=row[3] or '',
                duration=row[4] or 0.0,
                resolution=(row[5] or 0, row[6] or 0),
                fps=row[7] or 0.0,
                bitrate=row[8] or 0,
                file_size=row[9] or 0,
                created_date=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
                modified_date=datetime.fromisoformat(row[11]) if row[11] else datetime.now(),
                tags=json.loads(row[12]) if row[12] else [],
                categories=json.loads(row[13]) if row[13] else [],
                content_rating=row[14] or 'G',
                language=row[15] or 'en',
                source=row[16] or 'unknown',
                license=row[17] or 'unknown',
                thumbnail_path=Path(row[18]) if row[18] else None,
                preview_path=Path(row[19]) if row[19] else None
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur reconstruction mÃƒÂ©tadonnÃƒÂ©es: {e}")
            raise

    def _reconstruct_visual_analysis_from_db(self, row: tuple) -> VisualAnalysis:
        """Reconstruit un objet VisualAnalysis depuis la base de donnÃƒÂ©es"""
        try:
            # Les colonnes commencent aprÃƒÂ¨s les mÃƒÂ©tadonnÃƒÂ©es (20+)
            visual_analysis = VisualAnalysis(
                dominant_colors=json.loads(row[21]) if row[21] else [],
                brightness_level=row[22] or 0.0,
                contrast_level=row[23] or 0.0,
                motion_intensity=row[24] or 0.0,
                scene_changes=json.loads(row[25]) if row[25] else [],
                face_detected=bool(row[26]),
                text_detected=bool(row[27]),
                object_detection=json.loads(row[28]) if row[28] else [],
                visual_complexity=row[29] or 0.0,
                aesthetic_score=row[30] or 0.0
            )
            
            return visual_analysis
            
        except Exception as e:
            logger.error(f"Erreur reconstruction analyse visuelle: {e}")
            return VisualAnalysis()

    async def _score_broll_candidate(self, 
                                   candidate: BrollCandidate,
                                   context_analysis: Dict[str, Any],
                                   segment_analysis: Dict[str, Any]) -> BrollCandidate:
        """Score un candidat B-roll selon plusieurs critÃƒÂ¨res avec scoring adaptatif"""
        try:
            # DÃƒÂ©tecter le domaine pour le scoring adaptatif
            domain = self._detect_domain_from_context(context_analysis, segment_analysis)
            context_complexity = self._detect_context_complexity(segment_analysis)
            
            # Utiliser le scoring adaptatif si disponible
            try:
                from enhanced_scoring import calculate_adaptive_scoring
                
                # PrÃƒÂ©parer les donnÃƒÂ©es du candidat pour le scoring adaptatif
                candidate_data = {
                    "id": candidate.metadata.id,
                    "title": candidate.metadata.title,
                    "description": candidate.metadata.description,
                    "url": str(candidate.metadata.file_path),
                    "source": candidate.metadata.source,
                    "license": candidate.metadata.license,
                    "resolution": candidate.metadata.resolution,
                    "duration": candidate.metadata.duration,
                    "file_size": candidate.metadata.file_size,
                    "tags": candidate.metadata.tags,
                    "categories": candidate.metadata.categories,
                    "keywords": candidate.metadata.tags  # Utiliser les tags comme mots-clÃƒÂ©s
                }
                
                # Contexte pour le scoring
                scoring_context = {
                    "keywords": segment_analysis.get('main_keywords', []),
                    "domain": domain,
                    "complexity": context_complexity,
                    "segment_duration": segment_analysis.get('end_time', 0) - segment_analysis.get('start_time', 0),
                    "used_sources": context_analysis.get('used_sources', []),
                    "used_categories": context_analysis.get('used_categories', []),
                    "recent_use": context_analysis.get('recent_use', {})
                }
                
                # Calculer le score adaptatif
                adaptive_result = calculate_adaptive_scoring(candidate_data, scoring_context, domain)
                
                # Mettre ÃƒÂ  jour le candidat avec les scores adaptatifs
                candidate.semantic_similarity = adaptive_result["semantic_score"]
                candidate.context_relevance = adaptive_result["context_relevance"]
                candidate.quality_score = adaptive_result["visual_score"]
                candidate.diversity_score = adaptive_result["diversity_score"]
                candidate.final_score = adaptive_result["final_score"]
                candidate.selection_reason = adaptive_result["selection_reason"]
                
                logger.info(f"Scoring adaptatif appliquÃƒÂ© pour le domaine '{domain}': {adaptive_result['final_score']:.3f}")
                return candidate
                
            except ImportError:
                logger.warning("Module de scoring adaptatif non disponible, utilisation du scoring standard")
                return await self._score_broll_candidate_standard(candidate, context_analysis, segment_analysis)
            except Exception as e:
                logger.warning(f"Erreur lors du scoring adaptatif: {e}, utilisation du scoring standard")
                return await self._score_broll_candidate_standard(candidate, context_analysis, segment_analysis)
            
        except Exception as e:
            logger.error(f"Erreur scoring candidat: {e}")
            candidate.final_score = 0.0
            return candidate
    
    async def _score_broll_candidate_standard(self, 
                                            candidate: BrollCandidate,
                                            context_analysis: Dict[str, Any],
                                            segment_analysis: Dict[str, Any]) -> BrollCandidate:
        """Scoring standard (mÃƒÂ©thode originale)"""
        try:
            # Score de similaritÃƒÂ© sÃƒÂ©mantique
            semantic_similarity = await self._calculate_semantic_similarity(
                candidate, segment_analysis
            )
            
            # Score de pertinence contextuelle
            context_relevance = await self._calculate_context_relevance(
                candidate, context_analysis, segment_analysis
            )
            
            # Score de qualitÃƒÂ© technique
            quality_score = self._calculate_quality_score(candidate)
            
            # Score de diversitÃƒÂ©
            diversity_score = self._calculate_diversity_score(candidate, [])
            
            # Score final pondÃƒÂ©rÃƒÂ©
            final_score = (
                semantic_similarity * 0.3 +
                context_relevance * 0.3 +
                quality_score * 0.2 +
                diversity_score * 0.2
            )
            
            # Mettre ÃƒÂ  jour le candidat
            candidate.semantic_similarity = semantic_similarity
            candidate.context_relevance = context_relevance
            candidate.quality_score = quality_score
            candidate.diversity_score = diversity_score
            candidate.final_score = final_score
            
            # GÃƒÂ©nÃƒÂ©rer la raison de sÃƒÂ©lection
            candidate.selection_reason = self._generate_selection_reason(
                semantic_similarity, context_relevance, quality_score, diversity_score
            )
            
            return candidate
            
        except Exception as e:
            logger.error(f"Erreur scoring candidat standard: {e}")
            candidate.final_score = 0.0
            return candidate
    
    def _detect_domain_from_context(self, context_analysis: Dict[str, Any], 
                                   segment_analysis: Dict[str, Any]) -> str:
        """DÃƒÂ©tecte le domaine ÃƒÂ  partir du contexte"""
        try:
            # PrioritÃƒÂ© 1: Domaine dÃƒÂ©tectÃƒÂ© par l'analyseur contextuel
            if 'global_analysis' in context_analysis:
                main_theme = context_analysis['global_analysis'].get('main_theme', '')
                if main_theme in ['neuroscience', 'science', 'technology', 'business', 'lifestyle', 'education']:
                    return main_theme
            
            # PrioritÃƒÂ© 2: Contexte sÃƒÂ©mantique du segment
            semantic_context = segment_analysis.get('semantic_context', '')
            if semantic_context in ['neuroscience', 'science', 'technology', 'business', 'lifestyle', 'education']:
                return semantic_context
            
            # PrioritÃƒÂ© 3: Analyse des mots-clÃƒÂ©s
            keywords = segment_analysis.get('main_keywords', [])
            if keywords:
                try:
                    from enhanced_keyword_expansion import analyze_domain_from_keywords
                    detected_domain = analyze_domain_from_keywords(keywords)
                    if detected_domain != "general":
                        return detected_domain
                except ImportError:
                    pass
            
            # Fallback: analyse basique des mots-clÃƒÂ©s
            text_lower = ' '.join(keywords).lower()
            if any(word in text_lower for word in ['brain', 'neural', 'cognitive', 'mental']):
                return 'neuroscience'
            elif any(word in text_lower for word in ['ai', 'artificial', 'intelligence', 'technology']):
                return 'technology'
            elif any(word in text_lower for word in ['research', 'discovery', 'experiment']):
                return 'science'
            elif any(word in text_lower for word in ['business', 'company', 'enterprise']):
                return 'business'
            
            return 'general'
            
        except Exception as e:
            logger.warning(f"Erreur lors de la dÃƒÂ©tection du domaine: {e}")
            return 'general'
    
    def _detect_context_complexity(self, segment_analysis: Dict[str, Any]) -> str:
        """DÃƒÂ©tecte la complexitÃƒÂ© du contexte"""
        try:
            # BasÃƒÂ© sur la complexitÃƒÂ© du segment
            complexity_score = segment_analysis.get('complexity_score', 0.5)
            
            if complexity_score > 0.7:
                return 'high'
            elif complexity_score < 0.3:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.warning(f"Erreur lors de la dÃƒÂ©tection de la complexitÃƒÂ©: {e}")
            return 'medium'

    async def _calculate_semantic_similarity(self, candidate: BrollCandidate, segment_analysis: Dict[str, Any]) -> float:
        """Calcule la similaritÃƒÂ© sÃƒÂ©mantique entre le B-roll et le segment"""
        try:
            # Score basÃƒÂ© sur la correspondance des tags et catÃƒÂ©gories
            segment_context = segment_analysis.get('semantic_context', 'general')
            segment_keywords = segment_analysis.get('main_keywords', [])
            
            # Correspondance des catÃƒÂ©gories
            category_match = 0.0
            if segment_context in candidate.metadata.categories:
                category_match = 1.0
            elif segment_context in candidate.metadata.tags:
                category_match = 0.8
            
            # Correspondance des mots-clÃƒÂ©s
            keyword_match = 0.0
            for keyword in segment_keywords:
                if keyword.lower() in [tag.lower() for tag in candidate.metadata.tags]:
                    keyword_match += 0.2
            
            keyword_match = min(1.0, keyword_match)
            
            # Score final
            semantic_score = (category_match * 0.6 + keyword_match * 0.4)
            return semantic_score
            
        except Exception as e:
            logger.error(f"Erreur calcul similaritÃƒÂ© sÃƒÂ©mantique: {e}")
            return 0.5

    async def _calculate_context_relevance(self, 
                                         candidate: BrollCandidate,
                                         context_analysis: Dict[str, Any],
                                         segment_analysis: Dict[str, Any]) -> float:
        """Calcule la pertinence contextuelle du B-roll"""
        try:
            # Score basÃƒÂ© sur la cohÃƒÂ©rence avec le contexte global
            global_context = context_analysis.get('global_analysis', {})
            main_theme = global_context.get('main_theme', 'general')
            
            # VÃƒÂ©rifier la cohÃƒÂ©rence avec le thÃƒÂ¨me principal
            theme_coherence = 0.0
            if main_theme in candidate.metadata.categories:
                theme_coherence = 1.0
            elif main_theme in candidate.metadata.tags:
                theme_coherence = 0.8
            
            # Score basÃƒÂ© sur la complexitÃƒÂ© du contenu
            complexity_match = 0.0
            segment_complexity = segment_analysis.get('complexity_level', 'medium')
            if segment_complexity == 'high' and candidate.visual_analysis.visual_complexity > 0.7:
                complexity_match = 1.0
            elif segment_complexity == 'medium' and 0.3 <= candidate.visual_analysis.visual_complexity <= 0.7:
                complexity_match = 1.0
            elif segment_complexity == 'low' and candidate.visual_analysis.visual_complexity < 0.3:
                complexity_match = 1.0
            
            # Score final
            context_score = (theme_coherence * 0.7 + complexity_match * 0.3)
            return context_score
            
        except Exception as e:
            logger.error(f"Erreur calcul pertinence contextuelle: {e}")
            return 0.5

    def _calculate_quality_score(self, candidate: BrollCandidate) -> float:
        """Calcule le score de qualitÃƒÂ© technique du B-roll"""
        try:
            # Score basÃƒÂ© sur plusieurs facteurs techniques
            resolution_score = 0.0
            if candidate.metadata.resolution[0] >= 1920 and candidate.metadata.resolution[1] >= 1080:
                resolution_score = 1.0
            elif candidate.metadata.resolution[0] >= 1280 and candidate.metadata.resolution[1] >= 720:
                resolution_score = 0.8
            elif candidate.metadata.resolution[0] >= 854 and candidate.metadata.resolution[1] >= 480:
                resolution_score = 0.6
            else:
                resolution_score = 0.4
            
            # Score de stabilitÃƒÂ© (FPS)
            fps_score = min(1.0, candidate.metadata.fps / 30.0) if candidate.metadata.fps > 0 else 0.0
            
            # Score esthÃƒÂ©tique
            aesthetic_score = candidate.visual_analysis.aesthetic_score
            
            # Score final
            quality_score = (resolution_score * 0.4 + fps_score * 0.3 + aesthetic_score * 0.3)
            return quality_score
            
        except Exception as e:
            logger.error(f"Erreur calcul score qualitÃƒÂ©: {e}")
            return 0.5

    def _calculate_diversity_score(self, primary: BrollCandidate, alternatives: List[BrollCandidate]) -> float:
        """Calcule le score de diversitÃƒÂ© entre les B-rolls sÃƒÂ©lectionnÃƒÂ©s"""
        try:
            if not alternatives:
                return 1.0  # Pas d'alternatives = diversitÃƒÂ© maximale
            
            # Calculer la diversitÃƒÂ© basÃƒÂ©e sur les catÃƒÂ©gories et tags
            all_candidates = [primary] + alternatives
            categories = set()
            tags = set()
            
            for candidate in all_candidates:
                categories.update(candidate.metadata.categories)
                tags.update(candidate.metadata.tags)
            
            # Score de diversitÃƒÂ© basÃƒÂ© sur la variÃƒÂ©tÃƒÂ©
            category_diversity = min(1.0, len(categories) / 5.0)  # Normaliser
            tag_diversity = min(1.0, len(tags) / 20.0)  # Normaliser
            
            diversity_score = (category_diversity * 0.6 + tag_diversity * 0.4)
            return diversity_score
            
        except Exception as e:
            logger.error(f"Erreur calcul score diversitÃƒÂ©: {e}")
            return 0.5

    def _generate_selection_reason(self, 
                                 semantic_similarity: float,
                                 context_relevance: float,
                                 quality_score: float,
                                 diversity_score: float) -> str:
        """GÃƒÂ©nÃƒÂ¨re la raison de sÃƒÂ©lection d'un B-roll"""
        try:
            reasons = []
            
            if semantic_similarity > 0.8:
                reasons.append("Excellente correspondance sÃƒÂ©mantique")
            elif semantic_similarity > 0.6:
                reasons.append("Bonne correspondance sÃƒÂ©mantique")
            
            if context_relevance > 0.8:
                reasons.append("Pertinence contextuelle ÃƒÂ©levÃƒÂ©e")
            elif context_relevance > 0.6:
                reasons.append("Pertinence contextuelle satisfaisante")
            
            if quality_score > 0.8:
                reasons.append("QualitÃƒÂ© technique ÃƒÂ©levÃƒÂ©e")
            elif quality_score > 0.6:
                reasons.append("QualitÃƒÂ© technique satisfaisante")
            
            if diversity_score > 0.7:
                reasons.append("Excellente diversitÃƒÂ© de contenu")
            
            if not reasons:
                reasons.append("SÃƒÂ©lection basÃƒÂ©e sur les critÃƒÂ¨res disponibles")
            
            return " | ".join(reasons)
            
        except Exception as e:
            logger.error(f"Erreur gÃƒÂ©nÃƒÂ©ration raison: {e}")
            return "SÃƒÂ©lection automatique"

    def get_database_stats(self) -> Dict[str, Any]:
        """Obtient les statistiques de la base de donnÃƒÂ©es B-roll"""
        try:
            if not self.db_connection:
                return {"error": "Base de donnÃƒÂ©es non initialisÃƒÂ©e"}
            
            cursor = self.db_connection.cursor()
            
            # Nombre total de B-rolls
            cursor.execute("SELECT COUNT(*) FROM video_metadata")
            total_brolls = cursor.fetchone()[0]
            
            # RÃƒÂ©partition par catÃƒÂ©gorie
            cursor.execute("SELECT categories FROM video_metadata WHERE categories IS NOT NULL")
            categories_data = cursor.fetchall()
            
            category_counts = {}
            for row in categories_data:
                try:
                    categories = json.loads(row[0])
                    for category in categories:
                        category_counts[category] = category_counts.get(category, 0) + 1
                except:
                    continue
            
            # Statistiques de durÃƒÂ©e
            cursor.execute("SELECT AVG(duration), MIN(duration), MAX(duration) FROM video_metadata")
            duration_stats = cursor.fetchone()
            
            # Statistiques de rÃƒÂ©solution
            cursor.execute("SELECT AVG(resolution_width), AVG(resolution_height) FROM video_metadata")
            resolution_stats = cursor.fetchone()
            
            return {
                "total_brolls": total_brolls,
                "category_distribution": category_counts,
                "duration_stats": {
                    "average": duration_stats[0] or 0.0,
                    "minimum": duration_stats[1] or 0.0,
                    "maximum": duration_stats[2] or 0.0
                },
                "resolution_stats": {
                    "average_width": resolution_stats[0] or 0,
                    "average_height": resolution_stats[1] or 0
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur statistiques base: {e}")
            return {"error": str(e)}

    def close_database(self):
        """Ferme la connexion ÃƒÂ  la base de donnÃƒÂ©es"""
        if self.db_connection:
            self.db_connection.close()
            logger.info("Connexion base de donnÃƒÂ©es fermÃƒÂ©e")

# Instance globale
advanced_broll_selector = AdvancedBrollSelector() 

