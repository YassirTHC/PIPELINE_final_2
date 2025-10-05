# CONFIG SCHEMA

## Table des matières
- [Variables d'environnement](#variables-denvironnement)
- [Dataclass `Settings` proposée](#dataclass-settings-proposee)
- [Initialisation & logs](#initialisation--logs)

## Variables d'environnement
| Variable | Description | Type / Domaine | Valeur par défaut | Référence |
|---|---|---|---|---|
| `PIPELINE_LLM_MODEL` | Modèle Ollama principal utilisé pour tous les appels si spécifique absent. | `str` non vide | `"qwen2.5:7b"` | 【F:pipeline_core/llm_service.py†L155-L166】 |
| `PIPELINE_LLM_MODEL_JSON` | Modèle dédié aux réponses JSON (fallback vers `PIPELINE_LLM_MODEL`). | `str` | hérite de `PIPELINE_LLM_MODEL` | 【F:pipeline_core/llm_service.py†L2607-L2610】 |
| `PIPELINE_LLM_MODEL_TEXT` | Modèle dédié texte libre. | `str` | hérite de `PIPELINE_LLM_MODEL` | 【F:pipeline_core/llm_service.py†L2607-L2615】 |
| `PIPELINE_LLM_ENDPOINT` / `PIPELINE_LLM_BASE_URL` | URL base Ollama. | URL | `http://localhost:11434` | 【F:pipeline_core/llm_service.py†L155-L166】 |
| `PIPELINE_LLM_KEEP_ALIVE` | Keep-alive transmis à Ollama. | `str` (durée ex `30m`) | `"30m"` | 【F:pipeline_core/llm_service.py†L1464-L1474】 |
| `PIPELINE_LLM_TIMEOUT_S` | Timeout global (recyclé pour fetchers). | `float > 0` | `60.0` (LLM) / `8.0` (fetchers) | 【F:pipeline_core/llm_service.py†L172-L184】【F:pipeline_core/configuration.py†L356-L365】 |
| `PIPELINE_LLM_NUM_PREDICT` | Nombre maximum de tokens générés. | `int >= 1` | `256` | 【F:pipeline_core/llm_service.py†L182-L183】 |
| `PIPELINE_LLM_TEMP` | Température. | `float >= 0` | `0.3` | 【F:pipeline_core/llm_service.py†L182-L184】 |
| `PIPELINE_LLM_TOP_P` | Top-p sampling. | `float 0-1` | `0.9` | 【F:pipeline_core/llm_service.py†L183-L184】 |
| `PIPELINE_LLM_REPEAT_PENALTY` | Pénalité répétition. | `float >=0` | `1.1` | 【F:pipeline_core/llm_service.py†L1405-L1409】 |
| `PIPELINE_LLM_NUM_CTX` | Taille du contexte. | `int` | `4096` | 【F:pipeline_core/llm_service.py†L1411-L1412】 |
| `PIPELINE_LLM_MIN_CHARS` | Garde basse streaming. | `int >=0` | `8` | 【F:pipeline_core/llm_service.py†L1486-L1491】 |
| `PIPELINE_LLM_FALLBACK_TRUNC` | Taille du prompt tronqué pour fallback. | `int >0` | `3500` | 【F:pipeline_core/llm_service.py†L1492-L1500】 |
| `PIPELINE_LLM_FORCE_NON_STREAM` | Force le mode non-stream. | Booléen (`1/true/...`) | `False` | 【F:pipeline_core/llm_service.py†L1501-L1511】 |
| `PIPELINE_LLM_KEYWORDS_FIRST` | Priorise mots-clés vs transcript. | Booléen | `True` si unset | 【F:pipeline_core/llm_service.py†L112-L119】 |
| `PIPELINE_LLM_DISABLE_HASHTAGS` | Retire hashtags des résultats. | Booléen | `False` | 【F:pipeline_core/llm_service.py†L127-L133】 |
| `PIPELINE_LLM_TARGET_LANG` | Langue cible des prompts. | `str` (code) | `"en"` | 【F:pipeline_core/llm_service.py†L119-L125】 |
| `PIPELINE_LLM_JSON_PROMPT` | Override complet du prompt JSON. | `str` | `None` | 【F:pipeline_core/llm_service.py†L1267-L1276】 |
| `PIPELINE_LLM_JSON_MODE` | Force mode JSON. | Booléen | `False` | 【F:pipeline_core/llm_service.py†L1310-L1326】 |
| `PIPELINE_LLM_JSON_TRANSCRIPT_LIMIT` | Longueur max transcript en JSON. | `int` | `None` | 【F:pipeline_core/llm_service.py†L929-L940】 |
| `PIPELINE_TFIDF_FALLBACK_DISABLED` | Désactive fallback TF-IDF. | Booléen | `False` | 【F:pipeline_core/configuration.py†L88-L118】 |
| `PIPELINE_DISABLE_TFIDF_FALLBACK` | Alias legacy. | Booléen | `False` + warning | 【F:pipeline_core/configuration.py†L88-L118】 |
| `PIPELINE_MAX_SEGMENTS_IN_FLIGHT` | Parallélisme segments. | `int >=1` | `1` | 【F:pipeline_core/configuration.py†L184-L188】 |
| `PIPELINE_LLM_MAX_QUERIES_PER_SEGMENT` | Nombre max de requêtes par segment. | `int >=1` | `3` | 【F:pipeline_core/configuration.py†L188-L190】 |
| `PIPELINE_FAST_TESTS` | Mode tests rapides (bypass import `utils`). | Booléen | `False` | 【F:pipeline_core/llm_service.py†L35-L47】 |
| `BROLL_FETCH_MAX_PER_KEYWORD` | Limite globale par provider. | `int >=1` | Config (`6`) | 【F:pipeline_core/configuration.py†L373-L421】 |
| `FETCH_MAX` | Limite par défaut si `BROLL_FETCH_MAX_PER_KEYWORD` absent. | `int >=1` | `8` | 【F:pipeline_core/configuration.py†L366-L379】 |
| `BROLL_FETCH_ALLOW_IMAGES` / `BROLL_FETCH_ALLOW_VIDEOS` | Autorise types média. | Booléen | Config (`True/True`) | 【F:pipeline_core/configuration.py†L203-L233】【F:pipeline_core/configuration.py†L423-L431】 |
| `BROLL_FETCH_PROVIDER` / `AI_BROLL_FETCH_PROVIDER` | Liste providers actifs (`pixabay`, `pexels`, `all`). | CSV | `pixabay` | 【F:pipeline_core/configuration.py†L379-L404】 |
| `BROLL_<PROVIDER>_MAX_PER_KEYWORD` | Override spécifique par provider. | `int >=1` | `None` | 【F:pipeline_core/configuration.py†L395-L413】 |
| `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `UNSPLASH_ACCESS_KEY` | Clés API providers. | `str` | `None` | 【F:pipeline_core/fetchers.py†L714-L775】 |
| `PIPELINE_BROLL_MIN_START_SECONDS` *(nouveau)* | Décalage minimal avant premier B-roll. | `float >=0` | `2.0` (proposé) | — |
| `PIPELINE_BROLL_MIN_GAP_SECONDS` *(nouveau)* | Intervalle minimal entre B-rolls. | `float >=0` | `1.5` (aligné config) | — |
| `PIPELINE_BROLL_NO_REPEAT_SECONDS` *(nouveau)* | Fenêtre anti-repeat par asset/keyword. | `float >=0` | `6.0` (proposé) | — |
| `PIPELINE_LLM_MAX_ATTEMPTS` *(nouveau)* | Nombre max tentatives streaming. | `int >=1` | `3` | — |

## Dataclass `Settings` proposée
```python
from dataclasses import dataclass
from pathlib import Path
import os

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUE:
        return True
    if value in FALSE:
        return False
    return default

@dataclass(slots=True)
class Settings:
    clips_dir: Path = Path(os.getenv("PIPELINE_CLIPS_DIR", "clips"))
    output_dir: Path = Path(os.getenv("PIPELINE_OUTPUT_DIR", "output"))
    temp_dir: Path = Path(os.getenv("PIPELINE_TEMP_DIR", "temp"))

    llm_model: str = os.getenv("PIPELINE_LLM_MODEL", "qwen2.5:7b")
    llm_model_json: str = os.getenv("PIPELINE_LLM_MODEL_JSON", "").strip()
    llm_model_text: str = os.getenv("PIPELINE_LLM_MODEL_TEXT", "").strip()
    llm_endpoint: str = os.getenv("PIPELINE_LLM_ENDPOINT") or os.getenv("PIPELINE_LLM_BASE_URL") or "http://localhost:11434"
    llm_keep_alive: str = os.getenv("PIPELINE_LLM_KEEP_ALIVE", "30m")
    llm_timeout_stream: float = float(os.getenv("PIPELINE_LLM_TIMEOUT_S", "60"))
    llm_timeout_fallback: float = float(os.getenv("PIPELINE_LLM_FALLBACK_TIMEOUT_S", "45"))
    llm_min_chars: int = int(os.getenv("PIPELINE_LLM_MIN_CHARS", "8"))
    llm_max_attempts: int = int(os.getenv("PIPELINE_LLM_MAX_ATTEMPTS", "3"))

    broll_min_start: float = float(os.getenv("PIPELINE_BROLL_MIN_START_SECONDS", "2.0"))
    broll_min_gap: float = float(os.getenv("PIPELINE_BROLL_MIN_GAP_SECONDS", "1.5"))
    broll_no_repeat: float = float(os.getenv("PIPELINE_BROLL_NO_REPEAT_SECONDS", "6.0"))
    fetch_max_per_keyword: int = int(os.getenv("BROLL_FETCH_MAX_PER_KEYWORD", os.getenv("FETCH_MAX", "8")))
    allow_images: bool = _env_bool("BROLL_FETCH_ALLOW_IMAGES", True)
    allow_videos: bool = _env_bool("BROLL_FETCH_ALLOW_VIDEOS", True)

    @property
    def llm_model_effective_json(self) -> str:
        return self.llm_model_json or self.llm_model

    @property
    def llm_model_effective_text(self) -> str:
        return self.llm_model_text or self.llm_model

    def as_dict(self) -> dict[str, object]:
        return {
            "clips_dir": str(self.clips_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "llm_model": self.llm_model,
            "llm_endpoint": self.llm_endpoint,
            "llm_timeout_stream": self.llm_timeout_stream,
            "broll_min_start": self.broll_min_start,
            "broll_min_gap": self.broll_min_gap,
            "broll_no_repeat": self.broll_no_repeat,
        }
```
Points clés :
- Lecture unique des variables, conversions typées, fallback pour les nouveaux paramètres.
- `llm_timeout_fallback` introduit pour séparer la fenêtre non-stream.
- Conversion booléenne robuste (`_env_bool`).

## Initialisation & logs
1. Instancier `Settings` dans `run_pipeline.py` avant tout import du pipeline et l'exposer via `os.environ`/monkeypatch si nécessaire.【F:run_pipeline.py†L64-L200】
2. Logger au démarrage (`logger.info("settings", extra=settings.as_dict())`) pour tracer les valeurs utilisées (sans clés).【F:pipeline_core/logging.py†L34-L118】
3. Invalider les caches dépendants (`get_shared_llm_service`) si le modèle change entre runs en comparant avec `Settings` (recréation du singleton).【F:pipeline_core/llm_service.py†L3876-L3943】
