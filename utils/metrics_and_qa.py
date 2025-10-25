# -*- coding: utf-8 -*-
# ðŸ“Š MÃ‰TRIQUES ET QA AUTOMATIQUE - SYSTÃˆME DE MESURE INDUSTRIEL
# DÃ©finit et mesure les mÃ©triques clÃ©s pour la qualitÃ© du systÃ¨me LLM

import time
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import statistics

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """MÃ©triques de qualitÃ© pour un segment/transcript"""
    segment_id: str
    transcript_length: int
    llm_success: bool
    llm_response_time: float
    keywords_generated: int
    keywords_quality_score: float
    domain_detected: str
    domain_confidence: float
    fallback_used: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """MÃ©triques globales du systÃ¨me"""
    total_segments: int
    successful_segments: int
    fallback_rate: float
    avg_response_time: float
    p95_response_time: float
    avg_keywords_per_segment: float
    domain_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    error_distribution: Dict[str, int]

class MetricsCollector:
    """Collecteur de mÃ©triques en temps rÃ©el"""
    
    def __init__(self):
        self.metrics_history: List[QualityMetrics] = []
        self.current_session = {
            'start_time': time.time(),
            'total_calls': 0,
            'successful_calls': 0,
            'total_response_time': 0.0,
            'response_times': []
        }
        
        # Seuils d'alerte
        self.alert_thresholds = {
            'fallback_rate': 0.10,      # 10% max
            'p95_latency': 60.0,        # 60s max
            'avg_latency': 30.0,        # 30s max
            'quality_threshold': 0.7     # 70% min
        }
    
    def record_llm_call(self, segment_id: str, transcript: str, 
                        success: bool, response_time: float, 
                        keywords: List[str], domain: str, 
                        confidence: float, fallback: bool = False,
                        error_type: Optional[str] = None,
                        error_message: Optional[str] = None) -> QualityMetrics:
        """
        Enregistre les mÃ©triques d'un appel LLM
        """
        # Calculer la qualitÃ© des mots-clÃ©s
        keywords_quality = self._calculate_keywords_quality(keywords, transcript)
        
        # CrÃ©er les mÃ©triques
        metrics = QualityMetrics(
            segment_id=segment_id,
            transcript_length=len(transcript),
            llm_success=success,
            llm_response_time=response_time,
            keywords_generated=len(keywords) if keywords else 0,
            keywords_quality_score=keywords_quality,
            domain_detected=domain,
            domain_confidence=confidence,
            fallback_used=fallback,
            error_type=error_type,
            error_message=error_message
        )
        
        # Ajouter Ã  l'historique
        self.metrics_history.append(metrics)
        
        # Mettre Ã  jour les mÃ©triques de session
        self.current_session['total_calls'] += 1
        if success:
            self.current_session['successful_calls'] += 1
        
        self.current_session['total_response_time'] += response_time
        self.current_session['response_times'].append(response_time)
        
        # VÃ©rifier les alertes
        self._check_alerts()
        
        logger.info(f"ðŸ“Š MÃ©triques enregistrÃ©es pour {segment_id}: succÃ¨s={success}, temps={response_time:.1f}s, qualitÃ©={keywords_quality:.2f}")
        return metrics
    
    def _calculate_keywords_quality(self, keywords: List[str], transcript: str) -> float:
        """
        Calcule un score de qualitÃ© pour les mots-clÃ©s
        """
        if not keywords:
            return 0.0
        
        # CritÃ¨res de qualitÃ©
        scores = []
        
        # 1. Longueur des mots-clÃ©s (3-15 caractÃ¨res = optimal)
        for kw in keywords:
            if 3 <= len(kw) <= 15:
                scores.append(1.0)
            elif len(kw) < 3:
                scores.append(0.3)
            else:
                scores.append(0.7)
        
        # 2. PrÃ©sence dans le transcript (mots-clÃ©s pertinents)
        transcript_lower = transcript.lower()
        relevance_score = 0.0
        for kw in keywords:
            if kw.lower() in transcript_lower:
                relevance_score += 1.0
        relevance_score = relevance_score / len(keywords) if keywords else 0.0
        
        # 3. DiversitÃ© (Ã©viter les doublons)
        unique_keywords = set(kw.lower() for kw in keywords)
        diversity_score = len(unique_keywords) / len(keywords) if keywords else 0.0
        
        # 4. Score final pondÃ©rÃ©
        length_score = statistics.mean(scores) if scores else 0.0
        final_score = (0.3 * length_score + 0.4 * relevance_score + 0.3 * diversity_score)
        
        return min(1.0, max(0.0, final_score))
    
    def _check_alerts(self):
        """
        VÃ©rifie les seuils d'alerte et gÃ©nÃ¨re des alertes si nÃ©cessaire
        """
        if self.current_session['total_calls'] < 5:  # Attendre quelques appels
            return
        
        # Calculer les mÃ©triques actuelles
        current_metrics = self.get_current_metrics()
        
        # VÃ©rifier le taux de fallback
        if current_metrics.fallback_rate > self.alert_thresholds['fallback_rate']:
            logger.warning(f"ðŸš¨ ALERTE: Taux de fallback Ã©levÃ©: {current_metrics.fallback_rate:.1%} > {self.alert_thresholds['fallback_rate']:.1%}")
        
        # VÃ©rifier la latence P95
        if current_metrics.p95_response_time > self.alert_thresholds['p95_latency']:
            logger.warning(f"ðŸš¨ ALERTE: Latence P95 Ã©levÃ©e: {current_metrics.p95_response_time:.1f}s > {self.alert_thresholds['p95_latency']:.1f}s")
        
        # VÃ©rifier la latence moyenne
        if current_metrics.avg_response_time > self.alert_thresholds['avg_latency']:
            logger.warning(f"ðŸš¨ ALERTE: Latence moyenne Ã©levÃ©e: {current_metrics.avg_response_time:.1f}s > {self.alert_thresholds['avg_latency']:.1f}s")
    
    def get_current_metrics(self) -> SystemMetrics:
        """
        Calcule les mÃ©triques actuelles du systÃ¨me
        """
        if not self.metrics_history:
            return SystemMetrics(
                total_segments=0, successful_segments=0, fallback_rate=0.0,
                avg_response_time=0.0, p95_response_time=0.0,
                avg_keywords_per_segment=0.0, domain_distribution={},
                quality_distribution={}, error_distribution={}
            )
        
        # MÃ©triques de base
        total_segments = len(self.metrics_history)
        successful_segments = sum(1 for m in self.metrics_history if m.llm_success)
        fallback_rate = 1.0 - (successful_segments / total_segments)
        
        # MÃ©triques de temps
        response_times = [m.llm_response_time for m in self.metrics_history if m.llm_success]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # P95 (95Ã¨me percentile)
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p95_response_time = sorted_times[p95_index]
        else:
            p95_response_time = 0.0
        
        # MÃ©triques de mots-clÃ©s
        keywords_counts = [m.keywords_generated for m in self.metrics_history if m.llm_success]
        avg_keywords_per_segment = statistics.mean(keywords_counts) if keywords_counts else 0.0
        
        # Distribution des domaines
        domain_counts = Counter(m.domain_detected for m in self.metrics_history)
        domain_distribution = dict(domain_counts)
        
        # Distribution de la qualitÃ©
        quality_scores = [m.keywords_quality_score for m in self.metrics_history if m.llm_success]
        quality_distribution = {
            'high': sum(1 for s in quality_scores if s >= 0.8),
            'medium': sum(1 for s in quality_scores if 0.6 <= s < 0.8),
            'low': sum(1 for s in quality_scores if s < 0.6)
        }
        
        # Distribution des erreurs
        error_counts = Counter(m.error_type for m in self.metrics_history if m.error_type)
        error_distribution = dict(error_counts)
        
        return SystemMetrics(
            total_segments=total_segments,
            successful_segments=successful_segments,
            fallback_rate=fallback_rate,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            avg_keywords_per_segment=avg_keywords_per_segment,
            domain_distribution=domain_distribution,
            quality_distribution=quality_distribution,
            error_distribution=error_distribution
        )
    
    def export_metrics(self, output_path: str = None) -> Dict[str, Any]:
        """
        Exporte toutes les mÃ©triques au format JSON
        """
        if not output_path:
            timestamp = int(time.time())
            output_path = f"metrics_export_{timestamp}.json"
        
        # MÃ©triques actuelles
        current_metrics = self.get_current_metrics()
        
        # DonnÃ©es complÃ¨tes
        export_data = {
            'export_timestamp': time.time(),
            'session_duration': time.time() - self.current_session['start_time'],
            'current_metrics': asdict(current_metrics),
            'detailed_metrics': [asdict(m) for m in self.metrics_history],
            'session_summary': self.current_session
        }
        
        # Sauvegarder
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"ðŸ“Š MÃ©triques exportÃ©es vers: {output_path}")
        except Exception as e:
            logger.error(f"âŒ Erreur export mÃ©triques: {e}")
        
        return export_data
    
    def generate_report(self) -> str:
        """
        GÃ©nÃ¨re un rapport textuel des mÃ©triques
        """
        metrics = self.get_current_metrics()
        
        report = f"""
ðŸ“Š RAPPORT DE MÃ‰TRIQUES SYSTÃˆME LLM
{'='*50}

ðŸŽ¯ PERFORMANCE GÃ‰NÃ‰RALE:
   â€¢ Segments traitÃ©s: {metrics.total_segments}
   â€¢ SuccÃ¨s: {metrics.successful_segments} ({metrics.successful_segments/metrics.total_segments*100:.1f}%)
   â€¢ Taux de fallback: {metrics.fallback_rate*100:.1f}%

â±ï¸ LATENCE:
   â€¢ Temps moyen: {metrics.avg_response_time:.1f}s
   â€¢ P95: {metrics.p95_response_time:.1f}s

ðŸ” QUALITÃ‰:
   â€¢ Mots-clÃ©s moyens par segment: {metrics.avg_keywords_per_segment:.1f}
   â€¢ Distribution qualitÃ©:
     - Haute (â‰¥80%): {metrics.quality_distribution.get('high', 0)}
     - Moyenne (60-80%): {metrics.quality_distribution.get('medium', 0)}
     - Faible (<60%): {metrics.quality_distribution.get('low', 0)}

ðŸŽ¯ DISTRIBUTION DES DOMAINES:
"""
        
        for domain, count in metrics.domain_distribution.items():
            percentage = count / metrics.total_segments * 100
            report += f"   â€¢ {domain}: {count} ({percentage:.1f}%)\n"
        
        if metrics.error_distribution:
            report += f"\nâŒ ERREURS DÃ‰TECTÃ‰ES:\n"
            for error_type, count in metrics.error_distribution.items():
                report += f"   â€¢ {error_type}: {count}\n"
        
        # Ã‰valuations
        report += f"\nðŸ“ˆ Ã‰VALUATIONS:\n"
        
        if metrics.fallback_rate <= 0.05:
            report += "   âœ… Taux de fallback: EXCELLENT (<5%)\n"
        elif metrics.fallback_rate <= 0.10:
            report += "   âš ï¸ Taux de fallback: BON (5-10%)\n"
        else:
            report += "   âŒ Taux de fallback: CRITIQUE (>10%)\n"
        
        if metrics.p95_response_time <= 30:
            report += "   âœ… Latence P95: EXCELLENTE (<30s)\n"
        elif metrics.p95_response_time <= 60:
            report += "   âš ï¸ Latence P95: ACCEPTABLE (30-60s)\n"
        else:
            report += "   âŒ Latence P95: CRITIQUE (>60s)\n"
        
        return report

class QualityAssurance:
    """SystÃ¨me de QA automatique pour valider la qualitÃ©"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_keywords': 5,
            'max_keywords': 25,
            'min_quality_score': 0.6,
            'max_fallback_rate': 0.10,
            'max_avg_latency': 30.0
        }
    
    def assess_system_health(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """
        Ã‰value la santÃ© globale du systÃ¨me
        """
        health_score = 0.0
        issues = []
        warnings = []
        
        # 1. Taux de fallback
        if metrics.fallback_rate <= 0.05:
            health_score += 25
        elif metrics.fallback_rate <= 0.10:
            health_score += 15
            warnings.append(f"Taux de fallback Ã©levÃ©: {metrics.fallback_rate:.1%}")
        else:
            issues.append(f"Taux de fallback critique: {metrics.fallback_rate:.1%}")
        
        # 2. Latence moyenne
        if metrics.avg_response_time <= 15:
            health_score += 25
        elif metrics.avg_response_time <= 30:
            health_score += 15
            warnings.append(f"Latence moyenne Ã©levÃ©e: {metrics.avg_response_time:.1f}s")
        else:
            issues.append(f"Latence moyenne critique: {metrics.avg_response_time:.1f}s")
        
        # 3. QualitÃ© des mots-clÃ©s
        high_quality_ratio = metrics.quality_distribution.get('high', 0) / max(metrics.successful_segments, 1)
        if high_quality_ratio >= 0.7:
            health_score += 25
        elif high_quality_ratio >= 0.5:
            health_score += 15
            warnings.append(f"QualitÃ© des mots-clÃ©s modÃ©rÃ©e: {high_quality_ratio:.1%}")
        else:
            issues.append(f"QualitÃ© des mots-clÃ©s faible: {high_quality_ratio:.1%}")
        
        # 4. StabilitÃ©
        if metrics.total_segments >= 10:  # Assez de donnÃ©es
            health_score += 25
        else:
            health_score += (metrics.total_segments / 10) * 25
            warnings.append(f"DonnÃ©es insuffisantes: {metrics.total_segments} segments")
        
        # Ã‰valuation globale
        if health_score >= 90:
            status = "EXCELLENT"
        elif health_score >= 75:
            status = "BON"
        elif health_score >= 60:
            status = "ACCEPTABLE"
        else:
            status = "CRITIQUE"
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': self._generate_recommendations(issues, warnings)
        }
    
    def _generate_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """
        GÃ©nÃ¨re des recommandations basÃ©es sur les problÃ¨mes dÃ©tectÃ©s
        """
        recommendations = []
        
        if any("fallback" in issue.lower() for issue in issues):
            recommendations.append("ðŸ”§ VÃ©rifier la stabilitÃ© du modÃ¨le LLM et ajuster les prompts")
            recommendations.append("ðŸ”§ ImplÃ©menter des fallbacks plus robustes")
        
        if any("latence" in issue.lower() for issue in issues):
            recommendations.append("âš¡ Optimiser les paramÃ¨tres du modÃ¨le (temperature, max_tokens)")
            recommendations.append("âš¡ VÃ©rifier les ressources systÃ¨me (CPU, RAM, GPU)")
        
        if any("qualitÃ©" in issue.lower() for issue in issues):
            recommendations.append("ðŸŽ¯ AmÃ©liorer la validation des mots-clÃ©s gÃ©nÃ©rÃ©s")
            recommendations.append("ðŸŽ¯ Ajuster les seuils de qualitÃ©")
        
        if warnings:
            recommendations.append("ðŸ“Š Surveiller les mÃ©triques et ajuster les seuils si nÃ©cessaire")
        
        return recommendations

# === INSTANCES GLOBALES ===
metrics_collector = MetricsCollector()
qa_system = QualityAssurance()

# === FONCTIONS UTILITAIRES ===
def record_llm_metrics(segment_id: str, transcript: str, success: bool, 
                       response_time: float, keywords: List[str], domain: str, 
                       confidence: float, fallback: bool = False,
                       error_type: Optional[str] = None,
                       error_message: Optional[str] = None) -> QualityMetrics:
    """Enregistre les mÃ©triques d'un appel LLM"""
    return metrics_collector.record_llm_call(
        segment_id, transcript, success, response_time, 
        keywords, domain, confidence, fallback, error_type, error_message
    )

def get_system_metrics() -> SystemMetrics:
    """RÃ©cupÃ¨re les mÃ©triques actuelles du systÃ¨me"""
    return metrics_collector.get_current_metrics()

def assess_system_health() -> Dict[str, Any]:
    """Ã‰value la santÃ© du systÃ¨me"""
    metrics = get_system_metrics()
    return qa_system.assess_system_health(metrics)

def export_metrics(output_path: str = None) -> Dict[str, Any]:
    """Exporte les mÃ©triques"""
    return metrics_collector.export_metrics(output_path)

def generate_metrics_report() -> str:
    """GÃ©nÃ¨re un rapport des mÃ©triques"""
    return metrics_collector.generate_report()

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("ðŸ§ª Test du systÃ¨me de mÃ©triques et QA...")
    
    # Simuler quelques appels LLM
    test_cases = [
        ("seg_001", "EMDR therapy for trauma healing", True, 8.5, ["therapy", "trauma", "healing"], "medical_psychology", 0.85),
        ("seg_002", "Business strategy for startups", True, 12.3, ["business", "strategy", "startup"], "business_entrepreneurship", 0.78),
        ("seg_003", "AI technology future", False, 45.2, [], "generic", 0.0, True, "timeout", "Request timeout"),
        ("seg_004", "Mindfulness wellness practice", True, 6.8, ["mindfulness", "wellness", "practice"], "lifestyle_wellness", 0.92),
        ("seg_005", "Investment portfolio management", True, 9.1, ["investment", "portfolio", "management"], "finance_investment", 0.81)
    ]
    
    for segment_id, transcript, success, response_time, keywords, domain, confidence, *args in test_cases:
        fallback = args[0] if len(args) > 0 else False
        error_type = args[1] if len(args) > 1 else None
        error_message = args[2] if len(args) > 2 else None
        
        metrics = record_llm_metrics(
            segment_id, transcript, success, response_time,
            keywords, domain, confidence, fallback, error_type, error_message
        )
    
    # Afficher les mÃ©triques
    print("\nðŸ“Š MÃ©triques du systÃ¨me:")
    system_metrics = get_system_metrics()
    print(f"   Total segments: {system_metrics.total_segments}")
    print(f"   SuccÃ¨s: {system_metrics.successful_segments}")
    print(f"   Taux de fallback: {system_metrics.fallback_rate:.1%}")
    print(f"   Temps moyen: {system_metrics.avg_response_time:.1f}s")
    
    # Ã‰valuer la santÃ©
    print("\nðŸ¥ SantÃ© du systÃ¨me:")
    health = assess_system_health()
    print(f"   Score: {health['health_score']:.1f}/100")
    print(f"   Status: {health['status']}")
    
    if health['issues']:
        print("   âŒ ProblÃ¨mes:")
        for issue in health['issues']:
            print(f"      â€¢ {issue}")
    
    if health['warnings']:
        print("   âš ï¸ Avertissements:")
        for warning in health['warnings']:
            print(f"      â€¢ {warning}")
    
    if health['recommendations']:
        print("   ðŸ”§ Recommandations:")
        for rec in health['recommendations']:
            print(f"      â€¢ {rec}")
    
    # GÃ©nÃ©rer le rapport
    print("\nðŸ“‹ Rapport complet:")
    report = generate_metrics_report()
    print(report)
    
    print("\nï¿½ï¿½ Test terminÃ© !") 
