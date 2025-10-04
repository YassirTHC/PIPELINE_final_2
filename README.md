# Pipeline Vidéo Intelligent - Viral Vertical Content

Un pipeline automatisé pour créer du contenu vidéo viral optimisé pour les plateformes verticales (TikTok, Instagram Reels, YouTube Shorts).

##  Fonctionnalités Principales

- **Génération automatique de B-roll** avec IA contextuelle
- **Système de sous-titres intelligent** avec animations
- **Sélection optimisée** basée sur l'analyse sémantique
- **Support multi-plateformes** (Pexels, Pixabay, Giphy)
- **Interface graphique intuitive** pour faciliter l'utilisation
- **Pipeline modulaire** et extensible

##  Prérequis

- Python 3.11+
- Clés API pour les services de médias (Pexels, Pixabay, Giphy)
- FFmpeg installé sur le système

##  Installation

1. **Cloner le repository**

   ```bash
   git clone https://github.com/YassirTHC/PIPELINE_final_2.git
   cd PIPELINE_final_2
   ```

2. **Créer un environnement virtuel**

   ```bash
   python -m venv venv311
   # Windows
   venv311\Scripts\activate
   # Linux/Mac
   source venv311/bin/activate
   ```

3. **Installer les dépendances**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**

   ```bash
   # Copier le fichier d'exemple
   copy .env.example .env
   # Éditer .env avec vos clés API
   ```

##  Configuration des Clés API

Créez un fichier `.env` basé sur `.env.example` :

```env
# Clés API B-roll
PEXELS_API_KEY=your_pexels_api_key_here
PIXABAY_API_KEY=your_pixabay_api_key_here
GIPHY_API_KEY=your_giphy_api_key_here

# Configuration pipeline
BROLL_FETCH_ENABLE=True
BROLL_FETCH_PROVIDER=pexels
BROLL_FETCH_ALLOW_VIDEOS=True
BROLL_FETCH_ALLOW_IMAGES=False
BROLL_FETCH_MAX_PER_KEYWORD=8
```

### Variables d'environnement supplémentaires

- `PIPELINE_TFIDF_FALLBACK_DISABLED` : lorsque défini sur une valeur vraie (`1`, `true`, `yes`...), le pipeline n'utilise
  plus le secours TF-IDF pour la génération de contexte dynamique et les indices de segments. Cela permet de forcer un
  échec explicite lorsque les réponses LLM ne sont pas disponibles ou jugées insuffisantes.
  - L'ancien drapeau `PIPELINE_DISABLE_TFIDF_FALLBACK` reste pris en charge pour compatibilité mais émet désormais un
    avertissement de dépréciation ; migrez vers le nouveau nom dès que possible.

##  Utilisation

### Interface Graphique (Recommandé)

```bash
python video_converter_gui.py
```

### Interface Ligne de Commande

```bash
python video_processor.py --video input/votre_video.mp4 --output output/
```

### Script de Lancement Windows

```bash
run_pipeline_updated.bat
```

##  Structure du Projet

```
PIPELINE_final_2/
 pipeline_core/          # Modules principaux
    fetchers.py        # Récupération de médias
    llm_service.py     # Services IA
    transcript.py      # Traitement des transcriptions
    configuration.py   # Configuration
 utils/                  # Utilitaires
    llm_metadata_generator.py
    optimized_llm.py
    video_pipeline_integration.py
 config/                 # Fichiers de configuration
 emoji_assets/          # Assets emoji
 tests/                 # Tests unitaires
 video_processor.py     # Processeur principal
 video_converter_gui.py # Interface graphique
 main.py               # Point d'entrée alternatif
```

##  Fonctionnalités Avancées

### Système de B-roll Intelligent
- Analyse contextuelle des mots-clés
- Scoring multi-critères pour la sélection
- Support de plusieurs fournisseurs de médias
- Cache intelligent pour optimiser les performances

### Génération de Sous-titres
- Détection automatique de la langue
- Animations personnalisables
- Synchronisation précise avec l'audio
- Support des emojis et caractères spéciaux

### Pipeline Modulaire
- Architecture extensible
- Support de plugins personnalisés
- Configuration flexible via YAML
- Logging détaillé pour le debugging

##  Tests

```bash
# Lancer tous les tests
pytest

# Tests spécifiques
pytest tests/test_video_processor.py
pytest tests/test_llm_singleton.py
```

##  Monitoring et Logs

Le système génère des logs détaillés dans le dossier logs/ :
- pipeline_events.jsonl : Événements du pipeline
- selection_report.json : Rapport de sélection B-roll
- debug.log : Logs de debugging

##  Sécurité

- **Aucune clé API n'est stockée dans le code**
- Utilisation de variables d'environnement
- Fichier .env exclu du versioning
- Configuration sécurisée par défaut

##  Contribution

1. Fork le projet
2. Créer une branche feature (git checkout -b feature/AmazingFeature)
3. Commit vos changements (git commit -m 'Add some AmazingFeature')
4. Push vers la branche (git push origin feature/AmazingFeature)
5. Ouvrir une Pull Request

##  Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

##  Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la documentation dans docs/
- Vérifier les logs pour le debugging

##  Changelog

### v1.0.0
- Pipeline de base fonctionnel
- Interface graphique
- Support multi-fournisseurs B-roll
- Système de sous-titres intelligent
- Architecture modulaire

---

**Développé avec  pour créer du contenu viral de qualité**
## How to run

1. Probe LLM (Ollama):

```powershell
python tools/llm_probe.py --models "gemma3:4b,llama3.1:8b,qwen2.5:7b"
```

2. Bench pipeline (vidéo-only d’abord):

```powershell
powershell -ExecutionPolicy Bypass -File tools\pipeline_bench.ps1 -Video "clips/121.mp4" -Models "gemma3:4b,llama3.1:8b,qwen2.5:7b" -AllowImages 0
```

3. Analyse & recommandation:

```powershell
python tools/analyze_bench.py
# Ouvre tools/out/summary.md → le modèle recommandé + ENV finaux s’y trouvent
```

4. (Optionnel) Re-bench avec images autorisées (diagnostic “plein”):

```powershell
powershell -ExecutionPolicy Bypass -File tools\pipeline_bench.ps1 -Video "clips/121.mp4" -Models "gemma3:4b,llama3.1:8b,qwen2.5:7b" -AllowImages 1
python tools/analyze_bench.py
```
### ENV conseillés (vertical + combo modèles)

# Orientation & seuils
setx SELECTION_PREFER_LANDSCAPE "0"
setx SELECTION_MIN_SCORE "0.25"

# LLM routing
setx PIPELINE_LLM_ENDPOINT "http://localhost:11434"
setx PIPELINE_LLM_MODEL_JSON "gemma3:4b"
setx PIPELINE_LLM_MODEL_TEXT "qwen2.5:7b"

# B-roll providers (ex)
setx BROLL_FETCH_PROVIDER "pexels,pixabay"
setx BROLL_FETCH_ALLOW_VIDEOS "1"
setx BROLL_FETCH_ALLOW_IMAGES "0"
