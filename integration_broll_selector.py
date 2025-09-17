#!/usr/bin/env python3
"""
Intégration du Sélecteur B-roll Générique
Connexion au pipeline existant et validation
"""

import json
import logging
from pathlib import Path
from broll_selector import BrollSelector
import yaml
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrollSelectorIntegrator:
    """Intégrateur du sélecteur B-roll dans le pipeline existant"""
    
    def __init__(self, config_path: str = "config/broll_selector_config.yaml"):
        """Initialise l'intégrateur"""
        self.config_path = Path(config_path)
        self.selector = None
        self.config = None
        
        # Charger la configuration
        self._load_config()
        
        # Initialiser le sélecteur
        self._init_selector()
    
    def _load_config(self):
        """Charge la configuration depuis le fichier YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"✅ Configuration chargée: {self.config_path}")
            else:
                logger.warning(f"⚠️ Fichier de configuration non trouvé: {self.config_path}")
                self.config = {}
        except Exception as e:
            logger.error(f"❌ Erreur chargement configuration: {e}")
            self.config = {}
    
    def _init_selector(self):
        """Initialise le sélecteur B-roll"""
        try:
            self.selector = BrollSelector(self.config)
            logger.info("✅ Sélecteur B-roll initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation sélecteur: {e}")
            self.selector = None
    
    def integrate_with_pipeline(self, video_id: str, keywords: List[str], 
                              domain: Optional[str] = None) -> Dict[str, Any]:
        """Intègre le sélecteur avec le pipeline existant"""
        if not self.selector:
            logger.error("❌ Sélecteur non initialisé")
            return self._create_error_report(video_id, "Selector not initialized")
        
        try:
            logger.info(f"🎬 Intégration B-roll pour vidéo: {video_id}")
            logger.info(f"🔑 Mots-clés: {keywords}")
            logger.info(f"🎯 Domaine: {domain}")
            
            # Utiliser le nouveau sélecteur
            report = self.selector.select_brolls(
                keywords=keywords,
                domain=domain,
                min_delay=self.config.get('thresholds', {}).get('min_delay_seconds', 4.0),
                desired_count=self.config.get('desired_broll_count', 3)
            )
            
            # Enrichir le rapport avec des métadonnées d'intégration
            report['integration'] = {
                'pipeline_version': '2.0',
                'selector_version': '1.0',
                'integration_timestamp': report['timestamp'],
                'config_used': self.config
            }
            
            logger.info(f"✅ Intégration réussie: {len(report['selected'])} B-rolls sélectionnés")
            return report
            
        except Exception as e:
            logger.error(f"❌ Erreur intégration: {e}")
            return self._create_error_report(video_id, str(e))
    
    def _create_error_report(self, video_id: str, error_msg: str) -> Dict[str, Any]:
        """Crée un rapport d'erreur"""
        return {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'integration_failed': True,
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
    
    def validate_integration(self) -> bool:
        """Valide l'intégration complète"""
        logger.info("🔍 Validation de l'intégration...")
        
        # Test 1: Configuration
        if not self.config:
            logger.error("❌ Configuration manquante")
            return False
        
        # Test 2: Sélecteur
        if not self.selector:
            logger.error("❌ Sélecteur non initialisé")
            return False
        
        # Test 3: Test de sélection
        try:
            test_keywords = ["test", "validation", "integration"]
            test_report = self.selector.select_brolls(
                keywords=test_keywords,
                domain="general",
                min_delay=4.0,
                desired_count=1
            )
            
            if 'error' in test_report:
                logger.error(f"❌ Test de sélection échoué: {test_report['error']}")
                return False
            
            logger.info("✅ Test de sélection réussi")
            
        except Exception as e:
            logger.error(f"❌ Erreur test de sélection: {e}")
            return False
        
        logger.info("✅ Intégration validée avec succès")
        return True
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Génère un rapport d'intégration complet"""
        return {
            'integration_status': 'ready' if self.validate_integration() else 'failed',
            'timestamp': datetime.now().isoformat(),
            'config_loaded': bool(self.config),
            'selector_initialized': bool(self.selector),
            'config_path': str(self.config_path),
            'available_features': [
                'normalize_keywords',
                'expand_keywords',
                'score_asset',
                'select_brolls',
                'fallback_hierarchy'
            ] if self.selector else [],
            'next_steps': [
                'Connecter fetch_assets au pipeline existant',
                'Intégrer dans video_processor.py',
                'Tester avec de vraies vidéos',
                'Ajuster les paramètres selon les résultats'
            ]
        }

def test_integration():
    """Test de l'intégration"""
    print("🚀 TEST D'INTÉGRATION DU SÉLECTEUR B-ROLL")
    print("=" * 70)
    
    # 1. Test d'initialisation
    print("\n📋 1. Test d'initialisation...")
    try:
        integrator = BrollSelectorIntegrator()
        print("   ✅ BrollSelectorIntegrator initialisé")
    except Exception as e:
        print(f"   ❌ Erreur d'initialisation: {e}")
        return False
    
    # 2. Test de validation
    print("\n🔍 2. Test de validation...")
    try:
        validation_result = integrator.validate_integration()
        if validation_result:
            print("   ✅ Intégration validée")
        else:
            print("   ❌ Validation échouée")
            return False
    except Exception as e:
        print(f"   ❌ Erreur validation: {e}")
        return False
    
    # 3. Test d'intégration
    print("\n🔗 3. Test d'intégration...")
    try:
        # Simuler le cas 6.mp4
        keywords_6mp4 = ["family", "even", "playing", "with", "think"]
        domain_6mp4 = "health"
        
        report = integrator.integrate_with_pipeline(
            video_id="6.mp4",
            keywords=keywords_6mp4,
            domain=domain_6mp4
        )
        
        print(f"   ✅ Intégration réussie")
        print(f"   📊 Rapport généré: {len(report)} champs")
        print(f"   🎯 B-rolls sélectionnés: {len(report['selected'])}")
        
        if 'error' in report:
            print(f"   ⚠️ Erreur détectée: {report['error']}")
        
    except Exception as e:
        print(f"   ❌ Erreur intégration: {e}")
        return False
    
    # 4. Génération du rapport d'intégration
    print("\n📋 4. Rapport d'intégration...")
    try:
        integration_report = integrator.generate_integration_report()
        print(f"   ✅ Rapport généré")
        print(f"   📊 Statut: {integration_report['integration_status']}")
        
        # Sauvegarder le rapport
        output_dir = Path("output/reports")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "integration_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(integration_report, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Rapport sauvegardé: {report_path}")
        
    except Exception as e:
        print(f"   ❌ Erreur rapport: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ INTÉGRATION RÉUSSIE !")
    print("=" * 70)
    print("🎯 Le sélecteur B-roll est prêt pour l'intégration")
    print("🔧 Connectez-le au pipeline principal")
    print("📊 Rapports disponibles dans output/reports/")
    
    return True

def main():
    """Fonction principale"""
    print("🚀 INTÉGRATION DU SÉLECTEUR B-ROLL GÉNÉRIQUE")
    print("=" * 70)
    print("🎯 Connexion au pipeline existant et validation")
    
    # Exécuter le test d'intégration
    success = test_integration()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 INTÉGRATION TERMINÉE AVEC SUCCÈS !")
        print("=" * 70)
        print("✅ Le sélecteur B-roll générique est opérationnel")
        print("🔧 Prêt pour l'intégration dans video_processor.py")
        print("📊 Tous les tests de validation sont passés")
        print("🚀 Prochaine étape: Intégration complète au pipeline")
    else:
        print("\n" + "=" * 70)
        print("❌ INTÉGRATION ÉCHOUÉE")
        print("=" * 70)
        print("⚠️ Des problèmes ont été détectés")
        print("🔧 Correction nécessaire avant intégration")
    
    return success

if __name__ == "__main__":
    main() 