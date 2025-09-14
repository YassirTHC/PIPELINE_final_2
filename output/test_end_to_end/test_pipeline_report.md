# 📊 RAPPORT DE TEST PIPELINE END-TO-END

## 📅 Informations Générales
- **Date/Heure**: 2025-08-28T08:35:29.318784
- **Vidéo de test**: clips/11.mp4
- **Statut global**: CRITIQUE

## 🧪 Résultats des Tests

- ❌ **imports**: FAIL
- ❌ **configuration**: FAIL
- ❌ **context_analyzer**: FAIL
- ❌ **keyword_expansion**: FAIL
- ✅ **broll_selector**: PASS
- ✅ **pipeline_integration**: PASS
- ✅ **verification_system**: PASS
- ✅ **enhanced_features**: PASS
- ❌ **video_processing**: FAIL

## ❌ Erreurs
- Imports: No module named 'mediapipe'
- Configuration: cannot import name 'Config' from 'config' (C:\Users\Administrator\Desktop\video_pipeline - Copy\config.py)
- Context Analyzer: 'AdvancedContextAnalyzer' object has no attribute 'analyze_text'
- Keyword Expansion: cannot import name 'EnhancedKeywordExpansion' from 'enhanced_keyword_expansion' (C:\Users\Administrator\Desktop\video_pipeline - Copy\enhanced_keyword_expansion.py)
- Video Processing: No module named 'mediapipe'

## ⚠️ Avertissements
- Méthodes manquantes: ['process_video', 'analyze_video_content', 'plan_broll_insertion', 'execute_broll_plan']
- Méthodes de vérification manquantes: ['detect_visual_duplicates', 'evaluate_broll_quality', 'verify_context_relevance']
- Enhanced Fetcher: cannot import name 'EnhancedBrollFetcher' from 'enhanced_fetchers' (C:\Users\Administrator\Desktop\video_pipeline - Copy\enhanced_fetchers.py)
- Enhanced Scoring: cannot import name 'EnhancedBrollScoring' from 'enhanced_scoring' (C:\Users\Administrator\Desktop\video_pipeline - Copy\enhanced_scoring.py)

## 📈 Statistiques
- **Tests réussis**: 4/9
- **Taux de succès**: 44.4%

## 🎯 Recommandations
- Pipeline nécessite des corrections majeures
- Révision complète requise
- Priorité aux erreurs critiques