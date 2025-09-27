import os
from pathlib import Path

# Configuration principale pour compatibilité avec le pipeline
class Config:
    """Configuration principale du pipeline"""

    # Modèles Whisper
    WHISPER_MODEL = "base"
    
    # Dossiers principaux
    CLIPS_FOLDER = Path("clips")
    OUTPUT_FOLDER = Path("output")
    TEMP_FOLDER = Path("temp")
    
    # Configuration B-roll
    BROLL_SELECTOR_CONFIG_PATH = Path("config/broll_selector_config.yaml")
    
    # Dimensions cibles
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    
    # Options
    USE_HARDLINKS = True
    ENABLE_BROLL = True
    ENABLE_PIPELINE_CORE_FETCHER = False
    ENABLE_LEGACY_PIPELINE_FALLBACK = False




class AdvancedConfig:
    """Configuration avancée du pipeline"""
    
    # Dossiers
    BASE_DIR = Path(__file__).parent
    CLIPS_FOLDER = BASE_DIR / "clips"
    OUTPUT_FOLDER = BASE_DIR / "output"
    TEMP_FOLDER = BASE_DIR / "temp"
    SCRIPTS_FOLDER = BASE_DIR / "scripts"
    
    # Résolutions par plateforme
    PLATFORMS = {
        "tiktok": {"width": 1080, "height": 1920, "fps": 30},
        "instagram": {"width": 1080, "height": 1920, "fps": 30},
        "youtube_shorts": {"width": 1080, "height": 1920, "fps": 60},
    }
    
    # Paramètres Whisper
    WHISPER_MODELS = {
        "tiny": "Très rapide, précision moyenne",
        "base": "Bon compromis vitesse/précision", 
        "small": "Précision élevée, plus lent",
        "medium": "Très précis, assez lent",
        "large": "Maximum de précision, très lent"
    }
    
    # Styles de sous-titres prédéfinis
    SUBTITLE_STYLES = {
        "classic": {
            "fontsize": 60,
            "color": "white",
            "stroke_color": "black",
            "stroke_width": 3,
            "font": "Arial-Bold"
        },
        "trendy": {
            "fontsize": 70,
            "color": "yellow",
            "stroke_color": "red", 
            "stroke_width": 2,
            "font": "Impact"
        },
        "minimal": {
            "fontsize": 45,
            "color": "white",
            "stroke_color": "none",
            "stroke_width": 0,
            "font": "Helvetica"
        }
    }
    
    # Paramètres de qualité d'export
    EXPORT_PRESETS = {
        "high_quality": {
            "codec": "libx264",
            "bitrate": "8000k",
            "audio_codec": "aac",
            "audio_bitrate": "192k"
        },
        "medium_quality": {
            "codec": "libx264", 
            "bitrate": "4000k",
            "audio_codec": "aac",
            "audio_bitrate": "128k"
        },
        "fast_export": {
            "codec": "libx264",
            "bitrate": "2000k", 
            "audio_codec": "aac",
            "audio_bitrate": "96k"
        }
    }
    
    # Webhooks n8n (fix de votre erreur)
    N8N_WEBHOOKS = {
        "clip_created": "http://localhost:5678/webhook-test/clip-created",
        "processing_status": "http://localhost:5678/webhook-test/processing-status", 
        "pipeline_complete": "http://localhost:5678/webhook-test/pipeline-complete"
    }
    
    # Variables d'environnement
    @classmethod
    def from_env(cls):
        """Charge la config depuis les variables d'environnement"""
        return {
            "whisper_model": os.getenv("WHISPER_MODEL", "base"),
            "output_quality": os.getenv("EXPORT_QUALITY", "medium_quality"),
            "subtitle_style": os.getenv("SUBTITLE_STYLE", "classic"),
            "target_platform": os.getenv("TARGET_PLATFORM", "tiktok")
        }


# Configuration B-roll pour compatibilité avec le pipeline
class BrollConfig:
    """Configuration B-roll pour compatibilité avec le pipeline intelligent"""
    
    def __init__(self, input_video: str, output_video: str, broll_library: str, **kwargs):
        self.input_video = input_video
        self.output_video = output_video
        self.broll_library = broll_library
        
        # Paramètres hérités pour compatibilité
        self.srt_path = kwargs.get('srt_path')
        self.subtitle_font = kwargs.get('subtitle_font')
        self.subtitle_font_size = kwargs.get('subtitle_font_size', 72)
        self.subtitle_color = kwargs.get('subtitle_color', 'white')
        self.subtitle_stroke_color = kwargs.get('subtitle_stroke_color', 'black')
        self.subtitle_stroke_width = kwargs.get('subtitle_stroke_width', 2)
        self.render_subtitles = kwargs.get('render_subtitles', True)
        self.subtitle_safe_margin_px = kwargs.get('subtitle_safe_margin_px', 160)
        self.enable_emoji_subtitles = kwargs.get('enable_emoji_subtitles', False)
        self.emoji_inject_rate = kwargs.get('emoji_inject_rate', 0.2)
        self.emoji_overlay_only = kwargs.get('emoji_overlay_only', False)
        
        # === ÉQUILIBRE INTELLIGENT : VITESSE + QUALITÉ ===
        
        # B-roll selection (optimisée mais de qualité)
        self.max_broll_ratio = 0.40  # 🚀 AUGMENTÉ: 20% → 40% pour couvrir toute la vidéo
        self.min_broll_clip_s = 2.0  # Durée correcte
        self.max_broll_clip_s = 4.0  # Durée standard
        self.min_gap_between_broll_s = 2.5  # 📱 ATTENTION CURVE: 3.0s → 2.5s pour maintenir l'engagement constant
        
        # Sélection intelligente avec LLM
        self.enable_llm_reranking = True  # ACTIVÉ pour qualité maximale B-roll
        self.max_broll_insertions = 10  # 🚀 AUGMENTÉ: 6 → 10 pour plus de B-rolls
        self.fast_broll_search = False  # Recherche complète pour pertinence
        self.skip_similarity_check = False  # Garder vérifications qualité
        
        # Traitement vidéo intelligent (vitesse sans perte qualité)
        self.target_width = 1080  # Retour qualité HD pour meilleur rendu
        self.target_height = 1920  # Format 9:16 standard
        self.ffmpeg_preset = "fast"  # Compromis vitesse/qualité (vs ultrafast)
        self.crf = 23  # Meilleure qualité (vs 28)
        
        # Audio/analyse optimisée (garder l'essentiel)
        self.skip_audio_analysis = False  # RÉACTIVÉ pour placement intelligent
        self.simple_scene_detection = False  # Détection complète pour qualité
        self.fast_mode = False  # Mode complet pour qualité
        
        # === OPTIMISATIONS VIRALITÉ ===
        
        # Qualité B-roll pour engagement
        self.force_broll_diversity = True  # Diversité pour intérêt
        self.smart_cropping = True  # Cadrage intelligent
        self.min_duration_threshold_s = 1.5  # 🚀 RÉDUIT: 2.5s → 1.5s pour plus de B-rolls
        self.diversity_penalty = 0.3  # 🧠 RÉDUIT: 0.7 → 0.3 car LLM génère des mots-clés pertinents
        
        # 🧠 NOUVEAU: Analyse émotionnelle pour synchronisation
        self.enable_emotional_mapping = True  # Synchroniser B-rolls avec émotion du discours
        self.emotion_intensity_threshold = 0.6  # Seuil d'intensité émotionnelle
        self.emotion_broll_mapping = {
            'excitement': ['energetic', 'dynamic', 'fast-paced'],
            'calm': ['peaceful', 'serene', 'slow-motion'],
            'inspiration': ['motivational', 'aspirational', 'achievement'],
            'humor': ['funny', 'playful', 'light-hearted'],
            'serious': ['professional', 'focused', 'intense']
        }
        
        # 🧠 CORRECTION TikTok: Micro-moments désactivés pour durées optimales
        self.enable_micro_moments = False  # ❌ Désactivé: B-rolls courts nuisent à l'engagement TikTok
        self.micro_moment_duration = 1.5  # Si réactivé: durée minimale 1.5s pour TikTok
        self.micro_moment_frequency = 0.1  # Si réactivé: seulement 10% de micro-moments
        self.micro_moment_intensity = 0.8  # Intensité visuelle élevée pour micro-moments
        
        # Style viral
        self.emoji_style = "colorful"  # Emojis colorés activés
        self.dynamic_transitions = True  # Transitions fluides
        
        # === NOUVEAU : OPTIMISATIONS EXTRÊMES ===
        
        # Cache et mémoire
        self.enable_broll_cache = True  # Cache des résultats
        self.preload_popular_brolls = True  # Précharger les plus utilisés
        self.parallel_processing = True  # Traitement parallèle
        
        # Recherche de qualité avec LLM
        self.max_search_results = 25  # Plus de résultats pour meilleur choix
        self.quick_match_threshold = 0.6  # Seuil élevé pour qualité
        self.skip_complex_scoring = False  # Scoring complet pour pertinence
        
        # Export optimisé
        self.use_hardware_encoding = True  # GPU si disponible
        self.optimize_for_streaming = True  # Optimisé upload
        self.skip_quality_checks = True  # Ignorer vérifications finales 
        
        # Paramètres avancés pour compatibilité complète
        self.no_broll_before_s = kwargs.get('no_broll_before_s', 0.8)  # 🧠 HOOK PATTERN: 1.5s → 0.8s pour capturer l'attention immédiatement
        self.min_keywords_for_broll = kwargs.get('min_keywords_for_broll', 2)
        self.pad_with_blur = kwargs.get('pad_with_blur', True)
        self.threads = kwargs.get('threads', 0)
        self.use_whisper = kwargs.get('use_whisper', True)
        self.whisper_model = kwargs.get('whisper_model', 'base')
        self.use_transcript = kwargs.get('use_transcript', True)
        self.enable_fetcher = kwargs.get('enable_fetcher', False)
        self.fetch_provider = kwargs.get('fetch_provider')
        self.fetch_max_per_keyword = kwargs.get('fetch_max_per_keyword', 6)
        self.fetch_allow_videos = kwargs.get('fetch_allow_videos', True)
        self.fetch_allow_images = kwargs.get('fetch_allow_images', False)
        self.pexels_api_key = kwargs.get('pexels_api_key')
        self.pixabay_api_key = kwargs.get('pixabay_api_key')
        self.unsplash_access_key = kwargs.get('unsplash_access_key')
        self.use_embeddings = kwargs.get('use_embeddings', False)
        self.embedding_model_name = kwargs.get('embedding_model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.enable_crossfade = kwargs.get('enable_crossfade', True)
        self.crossfade_duration_s = kwargs.get('crossfade_duration_s', 0.2)
        self.occlude_main_under_broll = kwargs.get('occlude_main_under_broll', True)


