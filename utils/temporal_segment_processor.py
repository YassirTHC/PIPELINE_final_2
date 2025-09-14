#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕒 PROCESSEUR DE SEGMENTS TEMPORELS AVANCÉ
Gestion granulaire des segments vidéo avec validation et optimisation
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import timedelta

logger = logging.getLogger(__name__)

@dataclass
class TemporalSegment:
    """Segment temporel enrichi avec métadonnées"""
    start: float
    end: float
    text: str
    confidence: float = 1.0
    keywords: List[str] = None
    domain: str = ""
    context: str = ""
    broll_keywords: List[str] = None
    search_queries: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.broll_keywords is None:
            self.broll_keywords = []
        if self.search_queries is None:
            self.search_queries = []
    
    @property
    def duration(self) -> float:
        """Durée du segment en secondes"""
        return max(0.0, self.end - self.start)
    
    @property
    def center(self) -> float:
        """Point central du segment"""
        return (self.start + self.end) / 2.0
    
    def overlaps_with(self, other: 'TemporalSegment', tolerance: float = 0.1) -> bool:
        """Vérifie si ce segment chevauche avec un autre"""
        return not (self.end + tolerance <= other.start or other.end + tolerance <= self.start)
    
    def merge_with(self, other: 'TemporalSegment') -> 'TemporalSegment':
        """Fusionne avec un autre segment"""
        merged_text = f"{self.text} {other.text}".strip()
        merged_keywords = list(set(self.keywords + other.keywords))
        merged_broll = list(set(self.broll_keywords + other.broll_keywords))
        merged_queries = list(set(self.search_queries + other.search_queries))
        
        return TemporalSegment(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text=merged_text,
            confidence=min(self.confidence, other.confidence),
            keywords=merged_keywords,
            domain=self.domain or other.domain,
            context=self.context or other.context,
            broll_keywords=merged_broll,
            search_queries=merged_queries
        )

class TemporalSegmentProcessor:
    """Processeur de segments temporels avec validation et optimisation"""
    
    def __init__(self):
        self.min_segment_duration = 0.5  # Durée minimale d'un segment
        self.max_segment_duration = 15.0  # Durée maximale d'un segment
        self.merge_threshold = 0.2  # Seuil pour fusionner des segments proches
        self.confidence_threshold = 0.7  # Seuil de confiance minimum
        
    def process_segments(self, raw_segments: List[Dict[str, Any]]) -> List[TemporalSegment]:
        """
        Traite une liste de segments bruts en segments temporels enrichis
        
        Args:
            raw_segments: Liste de segments bruts (format Whisper/transcription)
            
        Returns:
            Liste de segments temporels validés et enrichis
        """
        logger.info(f"🕒 Traitement de {len(raw_segments)} segments temporels")
        
        # 1. Conversion en segments temporels
        temporal_segments = self._convert_to_temporal_segments(raw_segments)
        logger.info(f"✅ {len(temporal_segments)} segments convertis")
        
        # 2. Validation et nettoyage
        validated_segments = self._validate_segments(temporal_segments)
        logger.info(f"✅ {len(validated_segments)} segments validés")
        
        # 3. Fusion des segments trop courts ou proches
        merged_segments = self._merge_close_segments(validated_segments)
        logger.info(f"✅ {len(merged_segments)} segments après fusion")
        
        # 4. Division des segments trop longs
        final_segments = self._split_long_segments(merged_segments)
        logger.info(f"✅ {len(final_segments)} segments finaux")
        
        # 5. Enrichissement avec métadonnées
        enriched_segments = self._enrich_segments(final_segments)
        logger.info(f"✅ Segments enrichis avec métadonnées")
        
        return enriched_segments
    
    def _convert_to_temporal_segments(self, raw_segments: List[Dict[str, Any]]) -> List[TemporalSegment]:
        """Convertit des segments bruts en segments temporels"""
        temporal_segments = []
        
        for i, segment in enumerate(raw_segments):
            try:
                # Extraction des données de base
                start = float(segment.get('start', 0.0))
                end = float(segment.get('end', start + 1.0))
                text = str(segment.get('text', '')).strip()
                
                # Validation de base
                if not text or end <= start:
                    continue
                
                # Création du segment temporel
                temporal_segment = TemporalSegment(
                    start=start,
                    end=end,
                    text=text,
                    confidence=float(segment.get('confidence', 1.0))
                )
                
                temporal_segments.append(temporal_segment)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Erreur conversion segment {i}: {e}")
                continue
        
        return temporal_segments
    
    def _validate_segments(self, segments: List[TemporalSegment]) -> List[TemporalSegment]:
        """Valide et filtre les segments"""
        validated = []
        
        for segment in segments:
            # Validation de la durée
            if segment.duration < self.min_segment_duration:
                logger.debug(f"⚠️ Segment trop court: {segment.duration:.2f}s")
                continue
            
            if segment.duration > self.max_segment_duration:
                logger.debug(f"⚠️ Segment trop long: {segment.duration:.2f}s (sera divisé)")
            
            # Validation de la confiance
            if segment.confidence < self.confidence_threshold:
                logger.debug(f"⚠️ Confiance faible: {segment.confidence:.2f}")
                # On garde le segment mais on le marque
                segment.confidence = max(0.1, segment.confidence)
            
            # Validation du texte
            if len(segment.text.strip()) < 3:
                logger.debug(f"⚠️ Texte trop court: '{segment.text}'")
                continue
            
            validated.append(segment)
        
        return validated
    
    def _merge_close_segments(self, segments: List[TemporalSegment]) -> List[TemporalSegment]:
        """Fusionne les segments trop proches ou courts"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            # Calculer l'écart entre les segments
            gap = next_segment.start - current.end
            
            # Fusionner si l'écart est petit ou si le segment actuel est trop court
            should_merge = (
                gap <= self.merge_threshold or 
                current.duration < self.min_segment_duration or
                next_segment.duration < self.min_segment_duration
            )
            
            if should_merge:
                logger.debug(f"🔗 Fusion segments: {current.duration:.1f}s + {next_segment.duration:.1f}s")
                current = current.merge_with(next_segment)
            else:
                merged.append(current)
                current = next_segment
        
        # Ajouter le dernier segment
        merged.append(current)
        
        return merged
    
    def _split_long_segments(self, segments: List[TemporalSegment]) -> List[TemporalSegment]:
        """Divise les segments trop longs en segments plus petits"""
        split_segments = []
        
        for segment in segments:
            if segment.duration <= self.max_segment_duration:
                split_segments.append(segment)
                continue
            
            # Calculer le nombre de sous-segments nécessaires
            num_parts = int(segment.duration / self.max_segment_duration) + 1
            part_duration = segment.duration / num_parts
            
            logger.debug(f"✂️ Division segment {segment.duration:.1f}s en {num_parts} parties")
            
            # Diviser le texte (approximatif)
            words = segment.text.split()
            words_per_part = max(1, len(words) // num_parts)
            
            for i in range(num_parts):
                part_start = segment.start + (i * part_duration)
                part_end = segment.start + ((i + 1) * part_duration)
                
                # Texte de cette partie
                start_word = i * words_per_part
                end_word = min(len(words), (i + 1) * words_per_part)
                part_text = ' '.join(words[start_word:end_word])
                
                if part_text.strip():
                    part_segment = TemporalSegment(
                        start=part_start,
                        end=part_end,
                        text=part_text,
                        confidence=segment.confidence,
                        keywords=segment.keywords.copy(),
                        domain=segment.domain,
                        context=segment.context
                    )
                    split_segments.append(part_segment)
        
        return split_segments
    
    def _enrich_segments(self, segments: List[TemporalSegment]) -> List[TemporalSegment]:
        """Enrichit les segments avec des métadonnées supplémentaires"""
        for i, segment in enumerate(segments):
            # Ajouter index de position
            segment.position_ratio = i / len(segments) if segments else 0.0
            
            # Calculer la densité de mots
            words = segment.text.split()
            segment.word_density = len(words) / segment.duration if segment.duration > 0 else 0
            
            # Marquer les segments de transition (début/fin)
            segment.is_intro = i < len(segments) * 0.1  # 10% premier
            segment.is_outro = i >= len(segments) * 0.9  # 10% dernier
            segment.is_middle = not (segment.is_intro or segment.is_outro)
        
        return segments
    
    def validate_temporal_consistency(self, segments: List[TemporalSegment]) -> Tuple[bool, List[str]]:
        """
        Valide la cohérence temporelle des segments
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if not segments:
            return True, []
        
        # Vérifier l'ordre temporel
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            
            if current.end > next_segment.start:
                issues.append(f"Chevauchement détecté: segment {i} ({current.end:.2f}s) > segment {i+1} ({next_segment.start:.2f}s)")
            
            if next_segment.start - current.end > 5.0:  # Gap de plus de 5 secondes
                issues.append(f"Gap important détecté: {next_segment.start - current.end:.2f}s entre segments {i} et {i+1}")
        
        # Vérifier la durée totale
        total_duration = segments[-1].end - segments[0].start if segments else 0
        content_duration = sum(seg.duration for seg in segments)
        coverage_ratio = content_duration / total_duration if total_duration > 0 else 0
        
        if coverage_ratio < 0.7:  # Moins de 70% de couverture
            issues.append(f"Couverture faible: {coverage_ratio:.1%} du temps total")
        
        return len(issues) == 0, issues
    
    def optimize_for_broll_insertion(self, segments: List[TemporalSegment], 
                                   target_broll_count: int = 6) -> List[Dict[str, Any]]:
        """
        Optimise les segments pour l'insertion de B-roll
        
        Returns:
            Liste des points d'insertion optimaux
        """
        if not segments:
            return []
        
        insertion_points = []
        total_duration = segments[-1].end - segments[0].start
        
        # Calculer les positions idéales pour les B-rolls
        for i in range(target_broll_count):
            # Position équilibrée sur toute la durée
            target_time = segments[0].start + (i + 1) * (total_duration / (target_broll_count + 1))
            
            # Trouver le segment le plus proche
            best_segment = min(segments, key=lambda s: abs(s.center - target_time))
            
            # Calculer le score pour ce point d'insertion
            score = self._calculate_insertion_score(best_segment, target_time, segments)
            
            insertion_point = {
                'target_time': target_time,
                'segment': best_segment,
                'score': score,
                'keywords': best_segment.broll_keywords or best_segment.keywords,
                'search_queries': best_segment.search_queries,
                'domain': best_segment.domain,
                'context': best_segment.context
            }
            
            insertion_points.append(insertion_point)
        
        # Trier par score décroissant
        insertion_points.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"🎬 {len(insertion_points)} points d'insertion optimisés")
        return insertion_points
    
    def _calculate_insertion_score(self, segment: TemporalSegment, target_time: float, 
                                 all_segments: List[TemporalSegment]) -> float:
        """Calcule un score pour un point d'insertion"""
        score = 0.0
        
        # Score basé sur la richesse en mots-clés
        keyword_score = len(segment.keywords) * 0.1
        broll_score = len(segment.broll_keywords) * 0.15
        
        # Score basé sur la position (préférence centre)
        position_ratio = segment.position_ratio if hasattr(segment, 'position_ratio') else 0.5
        position_score = 1.0 - abs(position_ratio - 0.5) * 2  # Maximum au centre
        
        # Score basé sur la durée du segment
        duration_score = min(1.0, segment.duration / 10.0)  # Segments de ~10s idéaux
        
        # Score basé sur la confiance
        confidence_score = segment.confidence
        
        # Score basé sur la proximité avec le temps cible
        time_diff = abs(segment.center - target_time)
        proximity_score = max(0.0, 1.0 - time_diff / 10.0)  # Proximité dans les 10s
        
        # Score total pondéré
        score = (
            keyword_score * 0.2 +
            broll_score * 0.25 +
            position_score * 0.15 +
            duration_score * 0.15 +
            confidence_score * 0.1 +
            proximity_score * 0.15
        )
        
        return score

# === FONCTIONS UTILITAIRES ===

def create_temporal_processor() -> TemporalSegmentProcessor:
    """Factory pour créer un processeur de segments temporels"""
    return TemporalSegmentProcessor()

def process_whisper_segments(whisper_segments: List[Dict[str, Any]]) -> List[TemporalSegment]:
    """Fonction utilitaire pour traiter des segments Whisper"""
    processor = create_temporal_processor()
    return processor.process_segments(whisper_segments)

def validate_segment_timeline(segments: List[TemporalSegment]) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour valider une timeline de segments"""
    processor = create_temporal_processor()
    return processor.validate_temporal_consistency(segments)

# === TEST RAPIDE ===
if __name__ == "__main__":
    print("🧪 Test du processeur de segments temporels...")
    
    # Test avec des segments simulés
    test_segments = [
        {"start": 0.0, "end": 3.5, "text": "EMDR therapy utilizes bilateral stimulation", "confidence": 0.95},
        {"start": 3.5, "end": 7.2, "text": "to process traumatic memories effectively", "confidence": 0.90},
        {"start": 7.3, "end": 11.1, "text": "The therapist guides the patient through eye movements", "confidence": 0.88},
        {"start": 11.2, "end": 15.8, "text": "while recalling distressing events safely", "confidence": 0.92}
    ]
    
    processor = create_temporal_processor()
    
    # Test du traitement
    temporal_segments = processor.process_segments(test_segments)
    print(f"✅ {len(temporal_segments)} segments traités")
    
    # Test de validation
    is_valid, issues = processor.validate_temporal_consistency(temporal_segments)
    print(f"✅ Validation: {'OK' if is_valid else 'Problèmes détectés'}")
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    
    # Test d'optimisation B-roll
    insertion_points = processor.optimize_for_broll_insertion(temporal_segments, 3)
    print(f"✅ {len(insertion_points)} points d'insertion optimisés")
    
    for i, point in enumerate(insertion_points):
        print(f"   {i+1}. Temps: {point['target_time']:.1f}s, Score: {point['score']:.2f}")
    
    print("\n�� Test terminé !") 