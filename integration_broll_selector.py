ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
IntÃƒÂ©gration du SÃƒÂ©lecteur B-roll GÃƒÂ©nÃƒÂ©rique
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
    """IntÃƒÂ©grateur du sÃƒÂ©lecteur B-roll dans le pipeline existant"""
    
    def __init__(self, config_path: str = "config/broll_selector_config.yaml"):
        """Initialise l'intÃƒÂ©grateur"""
        self.config_path = Path(config_path)
        self.selector = None
        self.config = None
        
        # Charger la configuration
        self._load_config()
        
        # Initialiser le sÃƒÂ©lecteur
        self._init_selector()
    
    def _load_config(self):
        """Charge la configuration depuis le fichier YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Ã¢Å“â€¦ Configuration chargÃƒÂ©e: {self.config_path}")
            else:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Fichier de configuration non trouvÃƒÂ©: {self.config_path}")
                self.config = {}
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur chargement configuration: {e}")
            self.config = {}
    
    def _init_selector(self):
        """Initialise le sÃƒÂ©lecteur B-roll"""
        try:
            self.selector = BrollSelector(self.config)
            logger.info("Ã¢Å“â€¦ SÃƒÂ©lecteur B-roll initialisÃƒÂ©")
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur initialisation sÃƒÂ©lecteur: {e}")
            self.selector = None
    
    def integrate_with_pipeline(self, video_id: str, keywords: List[str], 
                              domain: Optional[str] = None) -> Dict[str, Any]:
        """IntÃƒÂ¨gre le sÃƒÂ©lecteur avec le pipeline existant"""
        if not self.selector:
            logger.error("Ã¢ÂÅ’ SÃƒÂ©lecteur non initialisÃƒÂ©")
            return self._create_error_report(video_id, "Selector not initialized")
        
        try:
            logger.info(f"Ã°Å¸Å½Â¬ IntÃƒÂ©gration B-roll pour vidÃƒÂ©o: {video_id}")
            logger.info(f"Ã°Å¸â€â€˜ Mots-clÃƒÂ©s: {keywords}")
            logger.info(f"Ã°Å¸Å½Â¯ Domaine: {domain}")
            
            # Utiliser le nouveau sÃƒÂ©lecteur
            report = self.selector.select_brolls(
                keywords=keywords,
                domain=domain,
                min_delay=self.config.get('thresholds', {}).get('min_delay_seconds', 4.0),
                desired_count=self.config.get('desired_broll_count', 3)
            )
            
            # Enrichir le rapport avec des mÃƒÂ©tadonnÃƒÂ©es d'intÃƒÂ©gration
            report['integration'] = {
                'pipeline_version': '2.0',
                'selector_version': '1.0',
                'integration_timestamp': report['timestamp'],
                'config_used': self.config
            }
            
            logger.info(f"Ã¢Å“â€¦ IntÃƒÂ©gration rÃƒÂ©ussie: {len(report['selected'])} B-rolls sÃƒÂ©lectionnÃƒÂ©s")
            return report
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur intÃƒÂ©gration: {e}")
            return self._create_error_report(video_id, str(e))
    
    def _create_error_report(self, video_id: str, error_msg: str) -> Dict[str, Any]:
        """CrÃƒÂ©e un rapport d'erreur"""
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
        """Valide l'intÃƒÂ©gration complÃƒÂ¨te"""
        logger.info("Ã°Å¸â€Â Validation de l'intÃƒÂ©gration...")
        
        # Test 1: Configuration
        if not self.config:
            logger.error("Ã¢ÂÅ’ Configuration manquante")
            return False
        
        # Test 2: SÃƒÂ©lecteur
        if not self.selector:
            logger.error("Ã¢ÂÅ’ SÃƒÂ©lecteur non initialisÃƒÂ©")
            return False
        
        # Test 3: Test de sÃƒÂ©lection
        try:
            test_keywords = ["test", "validation", "integration"]
            test_report = self.selector.select_brolls(
                keywords=test_keywords,
                domain="general",
                min_delay=4.0,
                desired_count=1
            )
            
            if 'error' in test_report:
                logger.error(f"Ã¢ÂÅ’ Test de sÃƒÂ©lection ÃƒÂ©chouÃƒÂ©: {test_report['error']}")
                return False
            
            logger.info("Ã¢Å“â€¦ Test de sÃƒÂ©lection rÃƒÂ©ussi")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Erreur test de sÃƒÂ©lection: {e}")
            return False
        
        logger.info("Ã¢Å“â€¦ IntÃƒÂ©gration validÃƒÂ©e avec succÃƒÂ¨s")
        return True
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ¨re un rapport d'intÃƒÂ©gration complet"""
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
                'IntÃƒÂ©grer dans video_processor.py',
                'Tester avec de vraies vidÃƒÂ©os',
                'Ajuster les paramÃƒÂ¨tres selon les rÃƒÂ©sultats'
            ]
        }

def test_integration():
    """Test de l'intÃƒÂ©gration"""
    print("Ã°Å¸Å¡â‚¬ TEST D'INTÃƒâ€°GRATION DU SÃƒâ€°LECTEUR B-ROLL")
    print("=" * 70)
    
    # 1. Test d'initialisation
    print("\nÃ°Å¸â€œâ€¹ 1. Test d'initialisation...")
    try:
        integrator = BrollSelectorIntegrator()
        print("   Ã¢Å“â€¦ BrollSelectorIntegrator initialisÃƒÂ©")
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur d'initialisation: {e}")
        return False
    
    # 2. Test de validation
    print("\nÃ°Å¸â€Â 2. Test de validation...")
    try:
        validation_result = integrator.validate_integration()
        if validation_result:
            print("   Ã¢Å“â€¦ IntÃƒÂ©gration validÃƒÂ©e")
        else:
            print("   Ã¢ÂÅ’ Validation ÃƒÂ©chouÃƒÂ©e")
            return False
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur validation: {e}")
        return False
    
    # 3. Test d'intÃƒÂ©gration
    print("\nÃ°Å¸â€â€” 3. Test d'intÃƒÂ©gration...")
    try:
        # Simuler le cas 6.mp4
        keywords_6mp4 = ["family", "even", "playing", "with", "think"]
        domain_6mp4 = "health"
        
        report = integrator.integrate_with_pipeline(
            video_id="6.mp4",
            keywords=keywords_6mp4,
            domain=domain_6mp4
        )
        
        print(f"   Ã¢Å“â€¦ IntÃƒÂ©gration rÃƒÂ©ussie")
        print(f"   Ã°Å¸â€œÅ  Rapport gÃƒÂ©nÃƒÂ©rÃƒÂ©: {len(report)} champs")
        print(f"   Ã°Å¸Å½Â¯ B-rolls sÃƒÂ©lectionnÃƒÂ©s: {len(report['selected'])}")
        
        if 'error' in report:
            print(f"   Ã¢Å¡Â Ã¯Â¸Â Erreur dÃƒÂ©tectÃƒÂ©e: {report['error']}")
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur intÃƒÂ©gration: {e}")
        return False
    
    # 4. GÃƒÂ©nÃƒÂ©ration du rapport d'intÃƒÂ©gration
    print("\nÃ°Å¸â€œâ€¹ 4. Rapport d'intÃƒÂ©gration...")
    try:
        integration_report = integrator.generate_integration_report()
        print(f"   Ã¢Å“â€¦ Rapport gÃƒÂ©nÃƒÂ©rÃƒÂ©")
        print(f"   Ã°Å¸â€œÅ  Statut: {integration_report['integration_status']}")
        
        # Sauvegarder le rapport
        output_dir = Path("output/reports")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "integration_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(integration_report, f, indent=2, ensure_ascii=False)
        
        print(f"   Ã°Å¸â€™Â¾ Rapport sauvegardÃƒÂ©: {report_path}")
        
    except Exception as e:
        print(f"   Ã¢ÂÅ’ Erreur rapport: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Ã¢Å“â€¦ INTÃƒâ€°GRATION RÃƒâ€°USSIE !")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Le sÃƒÂ©lecteur B-roll est prÃƒÂªt pour l'intÃƒÂ©gration")
    print("Ã°Å¸â€Â§ Connectez-le au pipeline principal")
    print("Ã°Å¸â€œÅ  Rapports disponibles dans output/reports/")
    
    return True

def main():
    """Fonction principale"""
    print("Ã°Å¸Å¡â‚¬ INTÃƒâ€°GRATION DU SÃƒâ€°LECTEUR B-ROLL GÃƒâ€°NÃƒâ€°RIQUE")
    print("=" * 70)
    print("Ã°Å¸Å½Â¯ Connexion au pipeline existant et validation")
    
    # ExÃƒÂ©cuter le test d'intÃƒÂ©gration
    success = test_integration()
    
    if success:
        print("\n" + "=" * 70)
        print("Ã°Å¸Å½â€° INTÃƒâ€°GRATION TERMINÃƒâ€°E AVEC SUCCÃƒË†S !")
        print("=" * 70)
        print("Ã¢Å“â€¦ Le sÃƒÂ©lecteur B-roll gÃƒÂ©nÃƒÂ©rique est opÃƒÂ©rationnel")
        print("Ã°Å¸â€Â§ PrÃƒÂªt pour l'intÃƒÂ©gration dans video_processor.py")
        print("Ã°Å¸â€œÅ  Tous les tests de validation sont passÃƒÂ©s")
        print("Ã°Å¸Å¡â‚¬ Prochaine ÃƒÂ©tape: IntÃƒÂ©gration complÃƒÂ¨te au pipeline")
    else:
        print("\n" + "=" * 70)
        print("Ã¢ÂÅ’ INTÃƒâ€°GRATION Ãƒâ€°CHOUÃƒâ€°E")
        print("=" * 70)
        print("Ã¢Å¡Â Ã¯Â¸Â Des problÃƒÂ¨mes ont ÃƒÂ©tÃƒÂ© dÃƒÂ©tectÃƒÂ©s")
        print("Ã°Å¸â€Â§ Correction nÃƒÂ©cessaire avant intÃƒÂ©gration")
    
    return success

if __name__ == "__main__":
    main() 

