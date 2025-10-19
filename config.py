import os
import warnings
from pathlib import Path

from video_pipeline.config import get_settings as _typed_get_settings

warnings.warn(
    "config.py is deprecated; import video_pipeline.config instead.",
    DeprecationWarning,
    stacklevel=2,
)


def get_settings():
    """Return the shared typed settings instance (with env overrides)."""
    settings = _typed_get_settings()
    try:
        _apply_env_overrides(settings)
    except Exception:
        # si un override casse, on garde la conf d'origine
        pass
    return settings

# Configuration principale pour compatibilitÃ© avec le pipeline
class Config:
    """Configuration principale du pipeline"""

    # ModÃ¨les Whisper
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
    """Configuration avancÃ©e du pipeline"""
    
    # Dossiers
    BASE_DIR = Path(__file__).parent
    CLIPS_FOLDER = BASE_DIR / "clips"
    OUTPUT_FOLDER = BASE_DIR / "output"
    TEMP_FOLDER = BASE_DIR / "temp"
    SCRIPTS_FOLDER = BASE_DIR / "scripts"
    
    # RÃ©solutions par plateforme
    PLATFORMS = {
        "tiktok": {"width": 1080, "height": 1920, "fps": 30},
        "instagram": {"width": 1080, "height": 1920, "fps": 30},
        "youtube_shorts": {"width": 1080, "height": 1920, "fps": 60},
    }
    
    # ParamÃ¨tres Whisper
    WHISPER_MODELS = {
        "tiny": "TrÃ¨s rapide, prÃ©cision moyenne",
        "base": "Bon compromis vitesse/prÃ©cision", 
        "small": "PrÃ©cision Ã©levÃ©e, plus lent",
        "medium": "TrÃ¨s prÃ©cis, assez lent",
        "large": "Maximum de prÃ©cision, trÃ¨s lent"
    }
    
    # Styles de sous-titres prÃ©dÃ©finis
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
    
    # ParamÃ¨tres de qualitÃ© d'export
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


# Configuration B-roll pour compatibilitÃ© avec le pipeline
class BrollConfig:
    """Configuration B-roll pour compatibilitÃ© avec le pipeline intelligent"""
    
    def __init__(self, input_video: str, output_video: str, broll_library: str, **kwargs):
        self.input_video = input_video
        self.output_video = output_video
        self.broll_library = broll_library
        
        # ParamÃ¨tres hÃ©ritÃ©s pour compatibilitÃ©
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
        
        # === Ã‰QUILIBRE INTELLIGENT : VITESSE + QUALITÃ‰ ===
        
        # B-roll selection (optimisÃ©e mais de qualitÃ©)
        self.max_broll_ratio = 0.40  # ðŸš€ AUGMENTÃ‰: 20% â†’ 40% pour couvrir toute la vidÃ©o
        self.min_broll_clip_s = 2.0  # DurÃ©e correcte
        self.max_broll_clip_s = 4.0  # DurÃ©e standard
        self.min_gap_between_broll_s = 0.5  # Intervalle rÃ©duit pour enchaÃ®ner plus rapidement les B-rolls
        
        # SÃ©lection intelligente avec LLM
        self.enable_llm_reranking = True  # ACTIVÃ‰ pour qualitÃ© maximale B-roll
        self.max_broll_insertions = 10  # ðŸš€ AUGMENTÃ‰: 6 â†’ 10 pour plus de B-rolls
        self.fast_broll_search = False  # Recherche complÃ¨te pour pertinence
        self.skip_similarity_check = False  # Garder vÃ©rifications qualitÃ©
        
        # Traitement vidÃ©o intelligent (vitesse sans perte qualitÃ©)
        self.target_width = 1080  # Retour qualitÃ© HD pour meilleur rendu
        self.target_height = 1920  # Format 9:16 standard
        self.ffmpeg_preset = "fast"  # Compromis vitesse/qualitÃ© (vs ultrafast)
        self.crf = 23  # Meilleure qualitÃ© (vs 28)
        
        # Audio/analyse optimisÃ©e (garder l'essentiel)
        self.skip_audio_analysis = False  # RÃ‰ACTIVÃ‰ pour placement intelligent
        self.simple_scene_detection = False  # DÃ©tection complÃ¨te pour qualitÃ©
        self.fast_mode = False  # Mode complet pour qualitÃ©
        
        # === OPTIMISATIONS VIRALITÃ‰ ===
        
        # QualitÃ© B-roll pour engagement
        self.force_broll_diversity = True  # DiversitÃ© pour intÃ©rÃªt
        self.smart_cropping = True  # Cadrage intelligent
        self.min_duration_threshold_s = 1.5  # ðŸš€ RÃ‰DUIT: 2.5s â†’ 1.5s pour plus de B-rolls
        self.diversity_penalty = 0.3  # ðŸ§  RÃ‰DUIT: 0.7 â†’ 0.3 car LLM gÃ©nÃ¨re des mots-clÃ©s pertinents
        
        # ðŸ§  NOUVEAU: Analyse Ã©motionnelle pour synchronisation
        self.enable_emotional_mapping = True  # Synchroniser B-rolls avec Ã©motion du discours
        self.emotion_intensity_threshold = 0.6  # Seuil d'intensitÃ© Ã©motionnelle
        self.emotion_broll_mapping = {
            'excitement': ['energetic', 'dynamic', 'fast-paced'],
            'calm': ['peaceful', 'serene', 'slow-motion'],
            'inspiration': ['motivational', 'aspirational', 'achievement'],
            'humor': ['funny', 'playful', 'light-hearted'],
            'serious': ['professional', 'focused', 'intense']
        }
        
        # ðŸ§  CORRECTION TikTok: Micro-moments dÃ©sactivÃ©s pour durÃ©es optimales
        self.enable_micro_moments = False  # âŒ DÃ©sactivÃ©: B-rolls courts nuisent Ã  l'engagement TikTok
        self.micro_moment_duration = 1.5  # Si rÃ©activÃ©: durÃ©e minimale 1.5s pour TikTok
        self.micro_moment_frequency = 0.1  # Si rÃ©activÃ©: seulement 10% de micro-moments
        self.micro_moment_intensity = 0.8  # IntensitÃ© visuelle Ã©levÃ©e pour micro-moments
        
        # Style viral
        self.emoji_style = "colorful"  # Emojis colorÃ©s activÃ©s
        self.dynamic_transitions = True  # Transitions fluides
        
        # === NOUVEAU : OPTIMISATIONS EXTRÃŠMES ===
        
        # Cache et mÃ©moire
        self.enable_broll_cache = True  # Cache des rÃ©sultats
        self.preload_popular_brolls = True  # PrÃ©charger les plus utilisÃ©s
        self.parallel_processing = True  # Traitement parallÃ¨le
        
        # Recherche de qualitÃ© avec LLM
        self.max_search_results = 25  # Plus de rÃ©sultats pour meilleur choix
        self.quick_match_threshold = 0.6  # Seuil Ã©levÃ© pour qualitÃ©
        self.skip_complex_scoring = False  # Scoring complet pour pertinence
        
        # Export optimisÃ©
        self.use_hardware_encoding = True  # GPU si disponible
        self.optimize_for_streaming = True  # OptimisÃ© upload
        self.skip_quality_checks = True  # Ignorer vÃ©rifications finales 
        
        # ParamÃ¨tres avancÃ©s pour compatibilitÃ© complÃ¨te
        self.no_broll_before_s = kwargs.get('no_broll_before_s', 0.8)  # ðŸ§  HOOK PATTERN: 1.5s â†’ 0.8s pour capturer l'attention immÃ©diatement
        self.min_keywords_for_broll = kwargs.get('min_keywords_for_broll', 2)
        self.pad_with_blur = kwargs.get('pad_with_blur', True)
        self.threads = kwargs.get('threads', 0)
        self.use_whisper = kwargs.get('use_whisper', True)
        self.whisper_model = kwargs.get('whisper_model', 'base')
        self.use_transcript = kwargs.get('use_transcript', True)
        self.enable_fetcher = kwargs.get('enable_fetcher', False)


try:  # Align legacy Config paths with typed settings when available.
    _SETTINGS_SNAPSHOT = _typed_get_settings()
except Exception:  # pragma: no cover - defensive
    _SETTINGS_SNAPSHOT = None
else:
    try:
        Config.CLIPS_FOLDER = Path(_SETTINGS_SNAPSHOT.clips_dir)
        Config.OUTPUT_FOLDER = Path(_SETTINGS_SNAPSHOT.output_dir)
        Config.TEMP_FOLDER = Path(_SETTINGS_SNAPSHOT.temp_dir)
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        AdvancedConfig.CLIPS_FOLDER = Path(_SETTINGS_SNAPSHOT.clips_dir)
        AdvancedConfig.OUTPUT_FOLDER = Path(_SETTINGS_SNAPSHOT.output_dir)
        AdvancedConfig.TEMP_FOLDER = Path(_SETTINGS_SNAPSHOT.temp_dir)
    except Exception:  # pragma: no cover - defensive
        pass
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



def _apply_env_overrides(settings):
    """
    Applique des overrides simples depuis des variables d'environnement
    (compatible objets pydantic/dataclass et dicts).
    """
    import os
    from collections.abc import Mapping

    def _is_mapping(x): return isinstance(x, Mapping)
    def _hasattr(ns, name): 
        try: return hasattr(ns, name)
        except Exception: return False

    def _get(ns, name, default=None):
        if _is_mapping(ns):
            return ns.get(name, default)
        return getattr(ns, name, default)

    def _set(ns, name, value):
        if _is_mapping(ns):
            ns[name] = value
        else:
            try:
                setattr(ns, name, value)
            except Exception:
                # Pydantic v2: model fields peuvent nÃ©cessiter object.model_copy(update=...)
                if hasattr(ns, "model_copy"):
                    new = ns.model_copy(update={name: value})
                    # essayer de remettre sur le parent si connu
                    return new
        return ns

    def _ensure_dict_field(ns, name):
        cur = _get(ns, name)
        if cur is None:
            cur = {}
            _set(ns, name, cur)
        return cur

    # --- FETCH ---
    fetch = settings.fetch

    val = os.getenv("FETCH_TIMEOUT_S")
    if val:
        try: _set(fetch, "timeout_s", float(val))
        except Exception: pass

    val = os.getenv("FETCH_PROVIDER_LIMITS__PEXELS")
    if val:
        try:
            v = int(val)
            provider_limits = _ensure_dict_field(fetch, "provider_limits")
            provider_limits["pexels"] = v
        except Exception:
            pass

    val = os.getenv("FETCH_ALLOW_IMAGES")
    if val is not None:
        try: _set(fetch, "allow_images", bool(int(val)))
        except Exception: pass

    val = os.getenv("FETCH_ALLOW_VIDEOS")
    if val is not None:
        try: _set(fetch, "allow_videos", bool(int(val)))
        except Exception: pass

    # --- BROLL diversity / selection / backfill ---
    bd = settings.broll_diversity
    val = os.getenv("BROLL_DIVERSITY_ENABLE_MMR")
    if val is not None:
        try: _set(bd, "enable_mmr", bool(int(val)))
        except Exception: pass
    val = os.getenv("BROLL_DIVERSITY_REPEAT_PENALTY")
    if val:
        try: _set(bd, "repeat_penalty", float(val))
        except Exception: pass
    val = os.getenv("BROLL_DIVERSITY_REPEAT_WINDOW")
    if val:
        try: _set(bd, "repeat_window", int(val))
        except Exception: pass

    bs = settings.broll_selection
    val = os.getenv("BROLL_SELECTION_ENABLE_ADAPTIVE_TOPK")
    if val is not None:
        try: _set(bs, "enable_adaptive_topk", bool(int(val)))
        except Exception: pass

    bf = settings.broll_backfill
    val = os.getenv("BROLL_BACKFILL_ENABLE")
    if val is not None:
        try: _set(bf, "enable", bool(int(val)))
        except Exception: pass
    val = os.getenv("BROLL_BACKFILL_LOCAL_MAX_GAP_MULTIPLIER")
    if val:
        try: _set(bf, "local_max_gap_multiplier", float(val))
        except Exception: pass

    st = settings.scheduler_tuning
    val = os.getenv("SCHEDULER_TUNING_ENABLE_LOCAL_RELAX")
    if val is not None:
        try: _set(st, "enable_local_relax", bool(int(val)))
        except Exception: pass
    val = os.getenv("SCHEDULER_TUNING_MICRO_INSERT_MIN_S")
    if val:
        try: _set(st, "micro_insert_min_s", float(val))
        except Exception: pass
    val = os.getenv("SCHEDULER_TUNING_MICRO_INSERT_MAX_S")
    if val:
        try: _set(st, "micro_insert_max_s", float(val))
        except Exception: pass

    llm = settings.llm
    for key in ("LLM_DISABLE_DYNAMIC_SEGMENT", "LLM__DISABLE_DYNAMIC_SEGMENT"):
        val = os.getenv(key)
        if val is not None:
            try: _set(llm, "disable_dynamic_segment", bool(int(val)))
            except Exception: pass



def _apply_env_overrides(settings):
    """
    Applique des overrides simples depuis des variables d'environnement
    (compatible objets pydantic/dataclass et dicts).
    Exemples attendus :
      FETCH_PROVIDER_LIMITS__PEXELS=6
      FETCH_TIMEOUT_S=12.0
      BROLL_DIVERSITY_REPEAT_PENALTY=0.45
      BROLL_DIVERSITY_REPEAT_WINDOW=4
      BROLL_SELECTION_ENABLE_ADAPTIVE_TOPK=1
      BROLL_BACKFILL_ENABLE=1
      SCHEDULER_TUNING_ENABLE_LOCAL_RELAX=1
      SCHEDULER_TUNING_MICRO_INSERT_MIN_S=0.8
      SCHEDULER_TUNING_MICRO_INSERT_MAX_S=1.2
      LLM_DISABLE_DYNAMIC_SEGMENT=0  (ou LLM__DISABLE_DYNAMIC_SEGMENT=0)
    """
    import os
    from collections.abc import Mapping

    def _is_mapping(x): return isinstance(x, Mapping)

    def _get(ns, name, default=None):
        if _is_mapping(ns):
            return ns.get(name, default)
        return getattr(ns, name, default)

    def _set(ns, name, value):
        if _is_mapping(ns):
            ns[name] = value
        else:
            try:
                setattr(ns, name, value)
            except Exception:
                # Pydantic v2: si gelé, on tente une copie avec update
                if hasattr(ns, "model_copy"):
                    new = ns.model_copy(update={name: value})
                    # on **retourne** le nouvel objet au caller pour qu'il le replace si besoin
                    return new
        return ns

    def _ensure_dict_field(ns, name):
        cur = _get(ns, name)
        if cur is None:
            cur = {}
            _set(ns, name, cur)
        return cur

    # --- FETCH ---
    fetch = settings.fetch

    val = os.getenv("FETCH_TIMEOUT_S")
    if val:
        try: _set(fetch, "timeout_s", float(val))
        except Exception: pass

    val = os.getenv("FETCH_PROVIDER_LIMITS__PEXELS")
    if val:
        try:
            v = int(val)
            provider_limits = _ensure_dict_field(fetch, "provider_limits")
            provider_limits["pexels"] = v
        except Exception:
            pass

    val = os.getenv("FETCH_ALLOW_IMAGES")
    if val is not None:
        try: _set(fetch, "allow_images", bool(int(val)))
        except Exception: pass

    val = os.getenv("FETCH_ALLOW_VIDEOS")
    if val is not None:
        try: _set(fetch, "allow_videos", bool(int(val)))
        except Exception: pass

    # --- BROLL diversity / selection / backfill ---
    bd = settings.broll_diversity
    val = os.getenv("BROLL_DIVERSITY_ENABLE_MMR")
    if val is not None:
        try: _set(bd, "enable_mmr", bool(int(val)))
        except Exception: pass
    val = os.getenv("BROLL_DIVERSITY_REPEAT_PENALTY")
    if val:
        try: _set(bd, "repeat_penalty", float(val))
        except Exception: pass
    val = os.getenv("BROLL_DIVERSITY_REPEAT_WINDOW")
    if val:
        try: _set(bd, "repeat_window", int(val))
        except Exception: pass

    bs = settings.broll_selection
    val = os.getenv("BROLL_SELECTION_ENABLE_ADAPTIVE_TOPK")
    if val is not None:
        try: _set(bs, "enable_adaptive_topk", bool(int(val)))
        except Exception: pass

    bf = settings.broll_backfill
    val = os.getenv("BROLL_BACKFILL_ENABLE")
    if val is not None:
        try: _set(bf, "enable", bool(int(val)))
        except Exception: pass
    val = os.getenv("BROLL_BACKFILL_LOCAL_MAX_GAP_MULTIPLIER")
    if val:
        try: _set(bf, "local_max_gap_multiplier", float(val))
        except Exception: pass

    st = settings.scheduler_tuning
    val = os.getenv("SCHEDULER_TUNING_ENABLE_LOCAL_RELAX")
    if val is not None:
        try: _set(st, "enable_local_relax", bool(int(val)))
        except Exception: pass
    val = os.getenv("SCHEDULER_TUNING_MICRO_INSERT_MIN_S")
    if val:
        try: _set(st, "micro_insert_min_s", float(val))
        except Exception: pass
    val = os.getenv("SCHEDULER_TUNING_MICRO_INSERT_MAX_S")
    if val:
        try: _set(st, "micro_insert_max_s", float(val))
        except Exception: pass

    llm = settings.llm
    for key in ("LLM_DISABLE_DYNAMIC_SEGMENT", "LLM__DISABLE_DYNAMIC_SEGMENT"):
        val = os.getenv(key)
        if val is not None:
            try: _set(llm, "disable_dynamic_segment", bool(int(val)))
            except Exception: pass

