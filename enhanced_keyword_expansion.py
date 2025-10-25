ï»¿# -*- coding: utf-8 -*-
"""
Expansion Intelligente des Mots-ClÃƒÂ©s par Domaine
SystÃƒÂ¨me d'expansion sÃƒÂ©mantique pour amÃƒÂ©liorer la rÃƒÂ©cupÃƒÂ©ration des B-rolls
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DomainKeywords:
    """Configuration des mots-clÃƒÂ©s par domaine"""
    synonyms: List[str]
    related_concepts: List[str]
    visual_metaphors: List[str]
    search_variants: List[str]

class IntelligentKeywordExpander:
    """Expansion intelligente des mots-clÃƒÂ©s par domaine"""
    
    def __init__(self):
        # Configuration complÃƒÂ¨te des domaines
        self.domain_keywords = {
            "neuroscience": DomainKeywords(
                synonyms=["brain science", "cognitive science", "mental research", "brain study", "neural research"],
                related_concepts=["laboratory", "research", "discovery", "breakthrough", "study", "analysis", "experiment"],
                visual_metaphors=["microscope", "equipment", "technology", "innovation", "complexity", "structure"],
                search_variants=["brain", "mind", "cognition", "intelligence", "thinking", "mental", "neural"]
            ),
            "technology": DomainKeywords(
                synonyms=["innovation", "digital", "future", "progress", "advancement", "evolution"],
                related_concepts=["development", "transformation", "modernization", "upgrade", "enhancement"],
                visual_metaphors=["circuit", "connection", "network", "system", "architecture", "framework"],
                search_variants=["innovation", "digital", "future", "progress", "development"]
            ),
            "science": DomainKeywords(
                synonyms=["research", "discovery", "investigation", "exploration", "analysis"],
                related_concepts=["laboratory", "experiment", "methodology", "hypothesis", "theory"],
                visual_metaphors=["equipment", "tools", "instruments", "measurements", "data"],
                search_variants=["research", "discovery", "experiment", "analysis", "laboratory"]
            ),
            "business": DomainKeywords(
                synonyms=["enterprise", "corporation", "company", "organization", "firm"],
                related_concepts=["growth", "success", "strategy", "planning", "development"],
                visual_metaphors=["building", "office", "meeting", "collaboration", "progress"],
                search_variants=["enterprise", "growth", "success", "strategy", "development"]
            ),
            "lifestyle": DomainKeywords(
                synonyms=["wellness", "health", "fitness", "wellbeing", "vitality"],
                related_concepts=["balance", "harmony", "mindfulness", "self-care", "improvement"],
                visual_metaphors=["nature", "exercise", "meditation", "relaxation", "energy"],
                search_variants=["wellness", "health", "fitness", "balance", "harmony"]
            ),
            "education": DomainKeywords(
                synonyms=["learning", "teaching", "knowledge", "education", "instruction"],
                related_concepts=["development", "growth", "improvement", "skills", "expertise"],
                visual_metaphors=["books", "classroom", "study", "knowledge", "wisdom"],
                search_variants=["learning", "knowledge", "development", "skills", "expertise"]
            )
        }
        
        # Mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques ÃƒÂ  filtrer
        self.generic_words = {
            "background", "nature", "people", "abstract", "business", "office", 
            "city", "street", "technology", "very", "much", "many", "good", "bad",
            "new", "old", "big", "small", "fast", "slow", "reflexes", "speed"
        }
        
        # Mots-clÃƒÂ©s prioritaires par domaine
        self.priority_words = {
            "neuroscience": ["brain", "neural", "cognitive", "mental", "research", "laboratory"],
            "technology": ["innovation", "digital", "future", "progress", "development"],
            "science": ["research", "discovery", "experiment", "analysis", "laboratory"],
            "business": ["growth", "success", "strategy", "development", "enterprise"],
            "lifestyle": ["wellness", "health", "fitness", "balance", "harmony"],
            "education": ["learning", "knowledge", "development", "skills", "expertise"]
        }
    
    def expand_keywords(self, primary_keyword: str, domain: str = "general") -> List[str]:
        """Expansion complÃƒÂ¨te des mots-clÃƒÂ©s pour un domaine donnÃƒÂ©"""
        try:
            # Normaliser le domaine
            domain = domain.lower().strip()
            
            # Si le domaine n'est pas reconnu, utiliser "general"
            if domain not in self.domain_keywords:
                domain = "general"
                logger.info(f"Domaine '{domain}' non reconnu, utilisation du mode gÃƒÂ©nÃƒÂ©ral")
            
            # Expansion basÃƒÂ©e sur le domaine
            if domain in self.domain_keywords:
                domain_data = self.domain_keywords[domain]
                
                # Combiner tous les types de mots-clÃƒÂ©s
                expanded = [primary_keyword]
                expanded.extend(domain_data.synonyms)
                expanded.extend(domain_data.related_concepts)
                expanded.extend(domain_data.visual_metaphors)
                expanded.extend(domain_data.search_variants)
                
                # Filtrer les doublons et mots gÃƒÂ©nÃƒÂ©riques
                filtered = self._filter_keywords(expanded, domain)
                
                # Limiter ÃƒÂ  8-10 variantes optimales
                final_keywords = filtered[:10]
                
                logger.info(f"Expansion pour '{primary_keyword}' (domaine: {domain}): {len(final_keywords)} mots-clÃƒÂ©s")
                return final_keywords
            
            else:
                # Mode gÃƒÂ©nÃƒÂ©ral : expansion basique
                general_keywords = self._general_expansion(primary_keyword)
                
                # GARANTIE ABSOLUE : minimum 4 mots-clÃƒÂ©s quoi qu'il arrive
                if len(general_keywords) < 4:
                    generic_pool = ["innovation", "technology", "development", "research", "strategy", "health", "growth", "solution", "platform", "system"]
                    i = 0
                    while len(general_keywords) < 4 and i < len(generic_pool):
                        g = generic_pool[i]
                        if g.lower() not in [kw.lower() for kw in general_keywords]:
                            general_keywords.append(g)
                        i += 1
                    logger.info(f"Garantie absolue activÃƒÂ©e dans expand_keywords: {len(general_keywords)} mots-clÃƒÂ©s (minimum 4 garanti)")
                
                return general_keywords
                
        except Exception as e:
            logger.error(f"Erreur lors de l'expansion des mots-clÃƒÂ©s: {e}")
            return [primary_keyword]
    
    def expand_keywords_multi_domain(self, primary_keyword: str, keywords: List[str] = None) -> List[str]:
        """
        Pipeline robuste multi-domaines avec garantie d'expansion absolue.
        Assure un minimum par domaine et un minimum absolu global.
        """
        try:
            if keywords is None:
                keywords = [primary_keyword]
            
            # 1) DÃƒÂ©tection primaire des domaines
            domain_confidences = self.analyze_multiple_domains_from_keywords(keywords)
            
            # 2) Fallback token-level si vide ou faible confiance max
            max_conf = max(domain_confidences.values(), default=0.0)
            if not domain_confidences or max_conf < 0.20:
                logger.info("Fallback token-level activÃƒÂ©")
                token_confidences = self._fallback_domain_confidences_from_tokens(keywords, self.analyze_domain_from_keywords)
                # merge: garder les valeurs les plus ÃƒÂ©levÃƒÂ©es (token_confidences forcÃƒÂ© ÃƒÂ  0.75 si prÃƒÂ©sent)
                for d, c in token_confidences.items():
                    domain_confidences[d] = max(domain_confidences.get(d, 0.0), c)
                logger.info(f"Fallback token-level rÃƒÂ©ussi: {token_confidences}")
            
            # 3) Fallback intelligent renforcÃƒÂ© : assigner des domaines par dÃƒÂ©faut
            if not domain_confidences:
                logger.info("Fallback intelligent renforcÃƒÂ© activÃƒÂ©")
                default_domains = self._assign_default_domains(keywords)
                if default_domains:
                    domain_confidences = default_domains
                    logger.info(f"Domaines par dÃƒÂ©faut activÃƒÂ©s: {default_domains}")
            
            # 4) Si toujours rien Ã¢â€ â€™ dernier filet : expansion simple
            if not domain_confidences:
                logger.info("Fallback ultime Ã¢â€ â€™ expansion simple")
                return self.expand_keywords(primary_keyword, "general")
            
            # 5) SÃƒÂ©lection finale des domaines (ÃƒÂ©quilibres inclus)
            selected_domains = self._select_domains(domain_confidences, k=2)
            logger.info(f"Domaines sÃƒÂ©lectionnÃƒÂ©s: {selected_domains}")
            
            # 6) Dernier filet : si pas de domaines non-general, prendre les meilleurs (non-general) depuis domain_confidences
            if not selected_domains:
                ordered = [d for d, s in sorted(domain_confidences.items(), key=lambda x: x[1], reverse=True) if d != "general"]
                selected_domains = ordered[:2]
                logger.info(f"Dernier filet: domaines sÃƒÂ©lectionnÃƒÂ©s {selected_domains}")
            
            # 7) Expansion par domaine avec minima garantis
            expanded_keywords = [primary_keyword]
            
            for idx, domain in enumerate(selected_domains):
                confidence = domain_confidences.get(domain, 0.30)
                expansion_count = max(self._expansion_factor(confidence, idx), 2)  # Minimum 2 mots-clÃƒÂ©s
                
                domain_data = self.domain_keywords.get(domain)
                if domain_data:
                    # Expansion basÃƒÂ©e sur le facteur calculÃƒÂ©
                    if expansion_count >= 4:  # Expansion forte
                        expanded_keywords.extend(domain_data.synonyms[:4])
                        expanded_keywords.extend(domain_data.related_concepts[:3])
                        expanded_keywords.extend(domain_data.visual_metaphors[:3])
                        expanded_keywords.extend(domain_data.search_variants[:3])
                        logger.info(f"Expansion forte domaine '{domain}' (rang {idx+1}, confiance {confidence:.2f}): +{4+3+3+3} mots-clÃƒÂ©s")
                    
                    elif expansion_count >= 3:  # Expansion modÃƒÂ©rÃƒÂ©e
                        expanded_keywords.extend(domain_data.synonyms[:3])
                        expanded_keywords.extend(domain_data.related_concepts[:2])
                        expanded_keywords.extend(domain_data.visual_metaphors[:2])
                        logger.info(f"Expansion modÃƒÂ©rÃƒÂ©e domaine '{domain}' (rang {idx+1}, confiance {confidence:.2f}): +{3+2+2} mots-clÃƒÂ©s")
                    
                    else:  # Expansion minimale (Ã¢â€°Â¥2)
                        expanded_keywords.extend(domain_data.synonyms[:2])
                        expanded_keywords.extend(domain_data.related_concepts[:1])
                        logger.info(f"Expansion minimale domaine '{domain}' (rang {idx+1}, confiance {confidence:.2f}): +{2+1} mots-clÃƒÂ©s")
                    
                    # Garantir au moins 2 mots-clÃƒÂ©s par domaine
                    current_domain_keywords = [kw for kw in expanded_keywords if kw != primary_keyword]
                    if len(current_domain_keywords) < 2:
                        additional_keywords = domain_data.synonyms[2:4] if len(domain_data.synonyms) > 2 else []
                        if additional_keywords:
                            expanded_keywords.extend(additional_keywords[:2])
                            logger.info(f"Garantie d'expansion minimale: +{len(additional_keywords[:2])} mots-clÃƒÂ©s supplÃƒÂ©mentaires")
            
            # 8) Ajouter des mots-clÃƒÂ©s croisÃƒÂ©s entre domaines
            if len(selected_domains) > 1:
                cross_domain_keywords = self._generate_cross_domain_keywords(
                    [(domain, domain_confidences.get(domain, 0)) for domain in selected_domains]
                )
                expanded_keywords.extend(cross_domain_keywords)
                logger.info(f"Expansion croisÃƒÂ©e multi-domaines: +{len(cross_domain_keywords)} mots-clÃƒÂ©s")
            
            # 9) Normalisation et dÃƒÂ©doublonnage
            seen = set()
            deduped = []
            for kw in expanded_keywords:
                k = kw.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    deduped.append(kw)
            
            # 10) Garantie absolue : minimum global garanti
            absolute_min_total = 4
            if len(deduped) < absolute_min_total:
                generic_pool = ["innovation", "technology", "development", "research", "strategy", "health", "growth", "solution", "platform", "system"]
                i = 0
                while len(deduped) < absolute_min_total and i < len(generic_pool):
                    g = generic_pool[i]
                    if g not in seen:
                        deduped.append(g)
                        seen.add(g)
                    i += 1
                logger.info(f"Garantie absolue activÃƒÂ©e: {len(deduped)} mots-clÃƒÂ©s (minimum {absolute_min_total} garanti)")
            
            logger.info(f"Expansion multi-domaines pour '{primary_keyword}': {len(deduped)} mots-clÃƒÂ©s totaux (dÃƒÂ©dupliquÃƒÂ©s)")
            return deduped
            
        except Exception as e:
            logger.error(f"Erreur lors de l'expansion multi-domaines: {e}")
            # Fallback vers l'expansion simple
            return self.expand_keywords(primary_keyword, "general")
    
    def _generate_cross_domain_keywords(self, top_domains: List[tuple]) -> List[str]:
        """GÃƒÂ©nÃƒÂ¨re des mots-clÃƒÂ©s croisÃƒÂ©s entre domaines"""
        try:
            cross_keywords = []
            
            if len(top_domains) >= 2:
                domain1, confidence1 = top_domains[0]
                domain2, confidence2 = top_domains[1]
                
                # Combinaisons croisÃƒÂ©es intelligentes
                if domain1 == "technology" and domain2 == "neuroscience":
                    cross_keywords.extend(["brain-computer interface", "neural technology", "cognitive computing", "AI neuroscience"])
                elif domain1 == "technology" and domain2 == "business":
                    cross_keywords.extend(["digital transformation", "tech innovation", "business technology", "digital business"])
                elif domain1 == "technology" and domain2 == "education":
                    cross_keywords.extend(["educational technology", "edtech", "digital learning", "tech education"])
                elif domain1 == "science" and domain2 == "technology":
                    cross_keywords.extend(["scientific technology", "tech research", "scientific innovation", "research technology"])
                elif domain1 == "business" and domain2 == "education":
                    cross_keywords.extend(["business education", "corporate learning", "business training", "educational business"])
                elif domain1 == "healthcare" and domain2 == "technology":
                    cross_keywords.extend(["healthtech", "medical technology", "digital health", "healthcare innovation"])
                elif domain1 == "neuroscience" and domain2 == "education":
                    cross_keywords.extend(["cognitive education", "brain-based learning", "neural education", "cognitive training"])
                else:
                    # Combinaison gÃƒÂ©nÃƒÂ©rique
                    cross_keywords.extend([f"{domain1} {domain2}", f"{domain2} {domain1}", f"{domain1} and {domain2}"])
            
            return cross_keywords
            
        except Exception as e:
            logger.error(f"Erreur lors de la gÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s croisÃƒÂ©s: {e}")
            return []
    
    def _close_domains(self, domain_scores: dict, delta: float = 0.25, ratio: float = 0.75, min_conf: float = 0.15) -> list:
        """
        Retourne les domaines 'proches' du max, avec critÃƒÂ¨res permissifs :
          - ÃƒÂ©cart absolu < delta (par dÃƒÂ©faut 0.25)
          - score >= ratio * max_score (par dÃƒÂ©faut 0.75)
          - score >= min_conf (par dÃƒÂ©faut 0.15)
        """
        if not domain_scores:
            return []
        m = max(domain_scores.values())
        return [
            d for d, s in domain_scores.items()
            if s >= min_conf and (m - s) < delta and s >= ratio * m
        ]

    def _select_domains(self, domain_scores: dict, k: int = 2) -> list:
        """
        SÃƒÂ©lectionne jusqu'ÃƒÂ  k domaines avec ÃƒÂ©quilibrage hybride forcÃƒÂ© et diversitÃƒÂ© intelligente :
        - Si plusieurs domaines 'proches', on prend les meilleurs jusqu'ÃƒÂ  k
        - Sinon on prend top-1 et on force le nÃ‚Â°2 si >= 0.25
        - Si on n'a qu'un seul domaine Ã¢â€ â€™ forcer la diversitÃƒÂ© basÃƒÂ©e sur des catÃƒÂ©gories complÃƒÂ©mentaires
        """
        if not domain_scores:
            return []

        candidates = self._close_domains(domain_scores)
        if candidates:
            return sorted(candidates, key=lambda d: domain_scores[d], reverse=True)[:k]

        # Aucun domaine 'proche' Ã¢â€ â€™ prendre le meilleur
        ordered = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [ordered[0][0]]

        # DiversitÃƒÂ© forcÃƒÂ©e si un 2e domaine est raisonnable
        if len(ordered) > 1 and ordered[1][1] >= 0.25:
            selected.append(ordered[1][0])
        
        # DIVERSITÃƒâ€° INTELLIGENTE : si on n'a qu'un seul domaine, forcer un domaine complÃƒÂ©mentaire
        if len(selected) < 2:
            complementary_domain = self._get_complementary_domain(selected[0], domain_scores)
            if complementary_domain:
                selected.append(complementary_domain)
                logger.info(f"DiversitÃƒÂ© intelligente forcÃƒÂ©e: {selected[0]} + {complementary_domain}")
            elif len(ordered) > 1:
                # Fallback : prendre le second meilleur mÃƒÂªme avec un score faible
                second_best = ordered[1][0]
                if second_best not in selected:
                    selected.append(second_best)
                    logger.info(f"DiversitÃƒÂ© fallback forcÃƒÂ©e: {selected}")

        return selected

    def _fallback_domain_confidences_from_tokens(self, tokens: list, analyze_one: callable) -> dict:
        """
        Fallback amÃƒÂ©liorÃƒÂ© qui :
          - analyse mot par mot avec `analyze_one(token)` -> domain | 'general'
          - donne une confiance ÃƒÂ©levÃƒÂ©e (0.75) aux domaines trouvÃƒÂ©s pour forcer l'inclusion
          - si aucun domaine trouvÃƒÂ©, fait une heuristique lexicale pour assigner healthcare/business/tech
        """
        agg = {}
        for t in tokens:
            dom = analyze_one(t)
            if dom and dom != "general":
                # confiance ÃƒÂ©levÃƒÂ©e pour forcer l'inclusion
                agg[dom] = max(agg.get(dom, 0.0), 0.75)

        # Heuristique lexicale si rien n'a ÃƒÂ©tÃƒÂ© dÃƒÂ©tectÃƒÂ©
        if not agg:
            for t in tokens:
                w = t.lower()
                if any(sub in w for sub in ["health", "medical", "healthcare", "medtech"]):
                    agg["healthcare"] = max(agg.get("healthcare", 0.0), 0.75)
                elif any(sub in w for sub in ["business", "entreprise", "entrepreneur", "startup", "innovation"]):
                    agg["business"] = max(agg.get("business", 0.0), 0.75)
                elif any(sub in w for sub in ["ai", "artificial", "machine", "neural", "tech", "technology"]):
                    agg["technology"] = max(agg.get("technology", 0.0), 0.75)

        return agg

    def _assign_default_domains(self, keywords: list) -> dict:
        """
        Assigne des domaines par dÃƒÂ©faut basÃƒÂ©s sur le vocabulaire des mots-clÃƒÂ©s.
        Mapping intelligent pour ÃƒÂ©viter le fallback vers 'general'.
        """
        mapping = {
            "ai": "technology", "software": "technology", "cloud": "technology", "digital": "technology",
            "business": "business", "market": "business", "finance": "business", "startup": "business",
            "health": "healthcare", "medical": "healthcare", "therapy": "healthcare", "patient": "healthcare",
            "education": "education", "learning": "education", "teaching": "education", "school": "education",
            "science": "science", "research": "science", "laboratory": "science", "experiment": "science",
            "neural": "neuroscience", "brain": "neuroscience", "cognitive": "neuroscience", "mental": "neuroscience"
        }
        
        assigned = {}
        for kw in keywords:
            kw_lower = kw.lower()
            for token, domain in mapping.items():
                if token in kw_lower:
                    assigned[domain] = assigned.get(domain, 0) + 0.5
        
        # Normaliser les scores et retourner si des domaines ont ÃƒÂ©tÃƒÂ© assignÃƒÂ©s
        if assigned:
            # Normaliser ÃƒÂ  des confiances exploitables
            for domain in assigned:
                assigned[domain] = min(assigned[domain], 0.75)
            logger.info(f"Domaines par dÃƒÂ©faut assignÃƒÂ©s: {assigned}")
        
        return assigned if assigned else None

    def _get_complementary_domain(self, primary_domain: str, available_domains: dict) -> str:
        """
        Trouve un domaine complÃƒÂ©mentaire intelligent basÃƒÂ© sur des catÃƒÂ©gories logiques.
        Ãƒâ€°vite la sÃƒÂ©lection alÃƒÂ©atoire et favorise les domaines qui apportent une vraie diversitÃƒÂ©.
        """
        # Mapping de domaines complÃƒÂ©mentaires par catÃƒÂ©gorie
        complementary_mapping = {
            # Business + Technology (innovation digitale)
            "business": ["technology", "innovation"],
            "technology": ["business", "innovation"],
            "innovation": ["technology", "business"],
            
            # Healthcare + Technology (medtech)
            "healthcare": ["technology", "science"],
            "medical": ["technology", "science"],
            "science": ["technology", "healthcare"],
            
            # Education + Technology (edtech)
            "education": ["technology", "innovation"],
            "learning": ["technology", "education"],
            
            # Neuroscience + Technology (neurotech)
            "neuroscience": ["technology", "healthcare"],
            "brain": ["technology", "healthcare"],
            
            # Research + Technology (R&D)
            "research": ["technology", "innovation"],
            "development": ["technology", "business"]
        }
        
        # Chercher un domaine complÃƒÂ©mentaire dans le mapping
        if primary_domain in complementary_mapping:
            for complementary in complementary_mapping[primary_domain]:
                if complementary in available_domains:
                    logger.info(f"Domaine complÃƒÂ©mentaire trouvÃƒÂ©: {primary_domain} Ã¢â€ â€™ {complementary}")
                    return complementary
        
        # Si pas de mapping, chercher un domaine avec un score raisonnable
        for domain, score in available_domains.items():
            if domain != primary_domain and score >= 0.15:  # Seuil plus permissif
                logger.info(f"Domaine alternatif sÃƒÂ©lectionnÃƒÂ©: {primary_domain} Ã¢â€ â€™ {domain} (score: {score})")
                return domain
        
        # Dernier recours : prendre le meilleur score restant
        remaining_domains = [(d, s) for d, s in available_domains.items() if d != primary_domain]
        if remaining_domains:
            best_remaining = max(remaining_domains, key=lambda x: x[1])
            logger.info(f"Dernier recours diversitÃƒÂ©: {primary_domain} Ã¢â€ â€™ {best_remaining[0]} (score: {best_remaining[1]})")
            return best_remaining[0]
        
        return None

    def _expansion_factor(self, conf: float, rank: int) -> int:
        """
        DÃƒÂ©termine le nombre d'items d'expansion ÃƒÂ  gÃƒÂ©nÃƒÂ©rer par domaine.
        Rank 0 -> domaine principal, rank 1 -> secondaire, etc.
        """
        if conf >= 0.70:
            return 5 if rank == 0 else 4
        if conf >= 0.50:
            return 4 if rank == 0 else 3
        if conf >= 0.30:
            return 3
        if conf >= 0.20:
            return 2
        return 1

    def _compute_expansion_factor(self, confidence: float, domain_rank: int) -> int:
        """
        Calcule un facteur d'expansion stable basÃƒÂ© sur la confiance et le rang du domaine.
        Garantit une expansion minimale cohÃƒÂ©rente pour ÃƒÂ©viter les cas ÃƒÂ  1 seul mot-clÃƒÂ©.
        """
        if confidence >= 0.6:
            return 4  # Expansion forte
        elif confidence >= 0.4:
            return 3 if domain_rank == 1 else 2  # Expansion modÃƒÂ©rÃƒÂ©e
        elif confidence >= 0.25:
            return 2  # Expansion minimale
        else:
            return 1  # Expansion quasi-nulle
    
    def _filter_keywords(self, keywords: List[str], domain: str) -> List[str]:
        """Filtrage intelligent des mots-clÃƒÂ©s"""
        filtered = []
        seen = set()
        
        # Prioriser les mots-clÃƒÂ©s du domaine
        priority_words = self.priority_words.get(domain, [])
        
        # 1. Ajouter d'abord les mots-clÃƒÂ©s prioritaires
        for keyword in keywords:
            if keyword.lower() in priority_words and keyword.lower() not in seen:
                filtered.append(keyword)
                seen.add(keyword.lower())
        
        # 2. Ajouter les autres mots-clÃƒÂ©s valides
        for keyword in keywords:
            if (keyword.lower() not in seen and 
                keyword.lower() not in self.generic_words and
                len(keyword) > 2):
                filtered.append(keyword)
                seen.add(keyword.lower())
        
        return filtered
    
    def _general_expansion(self, primary_keyword: str) -> List[str]:
        """Expansion gÃƒÂ©nÃƒÂ©rale pour domaines non reconnus avec garantie absolue"""
        # Expansion basique avec synonymes gÃƒÂ©nÃƒÂ©riques
        basic_expansions = {
            "innovation": ["progress", "advancement", "development", "growth"],
            "research": ["study", "investigation", "analysis", "exploration"],
            "technology": ["digital", "modern", "advanced", "future"],
            "business": ["enterprise", "company", "organization", "firm"],
            "health": ["wellness", "fitness", "wellbeing", "vitality"]
        }
        
        expanded = [primary_keyword]
        if primary_keyword.lower() in basic_expansions:
            expanded.extend(basic_expansions[primary_keyword.lower()])
        
        # GARANTIE ABSOLUE : minimum 4 mots-clÃƒÂ©s quoi qu'il arrive
        if len(expanded) < 4:
            generic_pool = ["innovation", "technology", "development", "research", "strategy", "health", "growth", "solution", "platform", "system"]
            i = 0
            while len(expanded) < 4 and i < len(generic_pool):
                g = generic_pool[i]
                if g.lower() not in [kw.lower() for kw in expanded]:
                    expanded.append(g)
                i += 1
            logger.info(f"Garantie absolue activÃƒÂ©e dans _general_expansion: {len(expanded)} mots-clÃƒÂ©s (minimum 4 garanti)")
        
        return expanded[:max(5, len(expanded))]  # Limiter ÃƒÂ  5 ou plus si garantie absolue activÃƒÂ©e
    
    def get_search_queries(self, keywords: List[str], domain: str) -> List[str]:
        """GÃƒÂ©nÃƒÂ¨re des requÃƒÂªtes de recherche optimisÃƒÂ©es"""
        try:
            # Expansion des mots-clÃƒÂ©s
            expanded = self.expand_keywords(keywords[0], domain)
            
            # GÃƒÂ©nÃƒÂ©ration de requÃƒÂªtes de recherche
            queries = []
            
            # 1. RequÃƒÂªte principale avec mots-clÃƒÂ©s prioritaires
            priority_words = [kw for kw in expanded[:3] if kw.lower() in self.priority_words.get(domain, [])]
            if priority_words:
                queries.append(" ".join(priority_words))
            
            # 2. RequÃƒÂªtes avec combinaisons intelligentes
            for i in range(0, len(expanded), 2):
                if i + 1 < len(expanded):
                    query = f"{expanded[i]} {expanded[i+1]}"
                    queries.append(query)
            
            # 3. RequÃƒÂªtes avec mÃƒÂ©taphores visuelles
            if domain in self.domain_keywords:
                visual_words = self.domain_keywords[domain].visual_metaphors[:3]
                if visual_words:
                    queries.append(" ".join(visual_words))
            
            # Limiter ÃƒÂ  5 requÃƒÂªtes optimales
            return queries[:5]
            
        except Exception as e:
            logger.error(f"Erreur lors de la gÃƒÂ©nÃƒÂ©ration des requÃƒÂªtes: {e}")
            return [" ".join(keywords[:3])]  # Fallback basique
    
    def analyze_domain_from_keywords(self, keywords: List[str]) -> str:
        """Analyse automatique du domaine ÃƒÂ  partir des mots-clÃƒÂ©s - VERSION AMÃƒâ€°LIORÃƒâ€°E"""
        try:
            # Mots-clÃƒÂ©s ÃƒÂ©tendus pour une meilleure dÃƒÂ©tection
            extended_domain_keywords = {
                "neuroscience": {
                    "primary": ["brain", "neural", "cognitive", "mental", "neuroscience", "neurology"],
                    "secondary": ["research", "laboratory", "study", "analysis", "experiment", "discovery"],
                    "related": ["intelligence", "thinking", "memory", "learning", "consciousness", "perception"],
                    "technical": ["synapse", "neuron", "neurotransmitter", "cortex", "hippocampus", "amygdala"]
                },
                "technology": {
                    "primary": ["technology", "digital", "innovation", "future", "progress", "development"],
                    "secondary": ["computer", "software", "hardware", "system", "network", "data"],
                    "related": ["artificial", "intelligence", "machine", "learning", "automation", "robotics"],
                    "technical": ["algorithm", "programming", "coding", "database", "cloud", "cybersecurity", "quantum", "blockchain", "virtual", "reality", "augmented", "cybersecurity", "internet", "things", "data", "science"]
                },
                "science": {
                    "primary": ["science", "research", "discovery", "investigation", "exploration", "analysis"],
                    "secondary": ["laboratory", "experiment", "methodology", "hypothesis", "theory", "study"],
                    "related": ["knowledge", "understanding", "explanation", "observation", "measurement", "evidence"],
                    "technical": ["chemistry", "physics", "biology", "mathematics", "statistics", "methodology", "space", "exploration", "medical", "research"]
                },
                "business": {
                    "primary": ["business", "enterprise", "corporation", "company", "organization", "firm"],
                    "secondary": ["growth", "success", "strategy", "planning", "development", "management"],
                    "related": ["profit", "revenue", "market", "customer", "service", "product"],
                    "technical": ["finance", "accounting", "marketing", "sales", "operations", "leadership", "financial", "planning"]
                },
                "education": {
                    "primary": ["education", "learning", "teaching", "knowledge", "instruction", "training"],
                    "secondary": ["school", "university", "college", "classroom", "student", "teacher"],
                    "related": ["development", "growth", "improvement", "skills", "expertise", "understanding"],
                    "technical": ["curriculum", "pedagogy", "assessment", "evaluation", "certification", "accreditation"]
                },
                "healthcare": {
                    "primary": ["health", "medical", "healthcare", "medicine", "treatment", "care"],
                    "secondary": ["patient", "doctor", "hospital", "clinic", "therapy", "recovery"],
                    "related": ["wellness", "fitness", "wellbeing", "vitality", "prevention", "diagnosis"],
                    "technical": ["pharmaceutical", "biotechnology", "diagnostic", "therapeutic", "clinical", "research"]
                }
            }
            
            domain_scores = {}
            
            for domain, domain_data in extended_domain_keywords.items():
                score = 0
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Score pour mots-clÃƒÂ©s primaires (poids ÃƒÂ©levÃƒÂ©)
                    if keyword_lower in [kw.lower() for kw in domain_data["primary"]]:
                        score += 5
                    
                    # Score pour mots-clÃƒÂ©s secondaires
                    if keyword_lower in [kw.lower() for kw in domain_data["secondary"]]:
                        score += 3
                    
                    # Score pour concepts liÃƒÂ©s
                    if keyword_lower in [kw.lower() for kw in domain_data["related"]]:
                        score += 2
                    
                    # Score pour termes techniques
                    if keyword_lower in [kw.lower() for kw in domain_data["technical"]]:
                        score += 4
                    
                    # Score pour correspondances partielles
                    for category_keywords in domain_data.values():
                        for kw in category_keywords:
                            if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
                                score += 1
            
            # Retourner le domaine avec le score le plus ÃƒÂ©levÃƒÂ©
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                if domain_scores[best_domain] > 2:  # Seuil plus bas pour plus de sensibilitÃƒÂ©
                    return best_domain
            
            # Si aucun domaine clair, essayer de dÃƒÂ©tecter par contexte
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # DÃƒÂ©tection par mots-clÃƒÂ©s spÃƒÂ©cifiques (plus sensible)
                if any(word in keyword_lower for word in ["brain", "neural", "cognitive", "neural network"]):
                    return "neuroscience"
                elif any(word in keyword_lower for word in ["computer", "digital", "software", "artificial", "intelligence", "machine", "learning", "quantum", "blockchain", "virtual", "reality", "cybersecurity", "data science", "internet of things"]):
                    return "technology"
                elif any(word in keyword_lower for word in ["research", "laboratory", "experiment", "space", "exploration", "medical research"]):
                    return "science"
                elif any(word in keyword_lower for word in ["company", "business", "enterprise", "growth", "financial", "planning"]):
                    return "business"
                elif any(word in keyword_lower for word in ["school", "learning", "education", "educational"]):
                    return "education"
                elif any(word in keyword_lower for word in ["health", "medical", "patient", "healthcare"]):
                    return "healthcare"
            
            return "general"
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du domaine: {e}")
            return "general"
    
    def analyze_multiple_domains_from_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """Analyse multi-domaines avec scores de confiance - NOUVELLE FONCTIONNALITÃƒâ€° CRITIQUE"""
        try:
            # Mots-clÃƒÂ©s ÃƒÂ©tendus pour une meilleure dÃƒÂ©tection
            extended_domain_keywords = {
                "neuroscience": {
                    "primary": ["brain", "neural", "cognitive", "mental", "neuroscience", "neurology"],
                    "secondary": ["research", "laboratory", "study", "analysis", "experiment", "discovery"],
                    "related": ["intelligence", "thinking", "memory", "learning", "consciousness", "perception"],
                    "technical": ["synapse", "neuron", "neurotransmitter", "cortex", "hippocampus", "amygdala"]
                },
                "technology": {
                    "primary": ["technology", "digital", "innovation", "future", "progress", "development"],
                    "secondary": ["computer", "software", "hardware", "system", "network", "data"],
                    "related": ["artificial", "intelligence", "machine", "learning", "automation", "robotics"],
                    "technical": ["algorithm", "programming", "coding", "database", "cloud", "cybersecurity", "quantum", "blockchain", "virtual", "reality", "augmented", "cybersecurity", "internet", "things", "data", "science", "natural language", "processing", "autonomous", "vehicle", "perception", "big data", "analytics", "platform", "cloud computing", "infrastructure", "mobile app", "development", "web application", "security"],
                    "hybrid": ["business technology", "enterprise tech", "digital transformation", "healthcare technology", "educational technology"]
                },
                "science": {
                    "primary": ["science", "research", "discovery", "investigation", "exploration", "analysis"],
                    "secondary": ["laboratory", "experiment", "methodology", "hypothesis", "theory", "study"],
                    "related": ["knowledge", "understanding", "explanation", "observation", "measurement", "evidence"],
                    "technical": ["chemistry", "physics", "biology", "mathematics", "statistics", "methodology", "space", "exploration", "medical", "research"]
                },
                "business": {
                    "primary": ["business", "enterprise", "corporation", "company", "organization", "firm"],
                    "secondary": ["growth", "success", "strategy", "planning", "development", "management"],
                    "related": ["profit", "revenue", "market", "customer", "service", "product"],
                    "technical": ["finance", "accounting", "marketing", "sales", "operations", "leadership", "financial", "planning"]
                },
                "education": {
                    "primary": ["education", "learning", "teaching", "knowledge", "instruction", "training"],
                    "secondary": ["school", "university", "college", "classroom", "student", "teacher"],
                    "related": ["development", "growth", "improvement", "skills", "expertise", "understanding"],
                    "technical": ["curriculum", "pedagogy", "assessment", "evaluation", "certification", "accreditation"]
                },
                "healthcare": {
                    "primary": ["health", "medical", "healthcare", "medicine", "treatment", "care"],
                    "secondary": ["patient", "doctor", "hospital", "clinic", "therapy", "recovery"],
                    "related": ["wellness", "fitness", "wellbeing", "vitality", "prevention", "diagnosis"],
                    "technical": ["pharmaceutical", "biotechnology", "diagnostic", "therapeutic", "clinical", "research"]
                }
            }
            
            domain_scores = {}
            
            for domain, domain_data in extended_domain_keywords.items():
                score = 0.0
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Score pour mots-clÃƒÂ©s primaires (poids ÃƒÂ©levÃƒÂ©)
                    if keyword_lower in [kw.lower() for kw in domain_data["primary"]]:
                        score += 5.0
                    
                    # Score pour mots-clÃƒÂ©s secondaires
                    if keyword_lower in [kw.lower() for kw in domain_data["secondary"]]:
                        score += 3.0
                    
                    # Score pour concepts liÃƒÂ©s
                    if keyword_lower in [kw.lower() for kw in domain_data["related"]]:
                        score += 2.0
                    
                    # Score pour termes techniques
                    if keyword_lower in [kw.lower() for kw in domain_data["technical"]]:
                        score += 4.0
                    
                    # Score pour correspondances partielles
                    for category_keywords in domain_data.values():
                        for kw in category_keywords:
                            if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
                                score += 1.0
            
            # Nouvelle logique de scoring multi-domaines
            if domain_scores:
                # Calculer le score total pour normalisation
                total_score = sum(domain_scores.values())
                
                if total_score > 0:
                    # Normaliser les scores et calculer la confiance relative
                    for domain in domain_scores:
                        # Score normalisÃƒÂ© par rapport au total (pas seulement au maximum)
                        normalized_score = domain_scores[domain] / total_score
                        # Confiance basÃƒÂ©e sur la contribution relative du domaine
                        confidence = min(normalized_score * 2.0, 1.0)  # Facteur 2 pour ÃƒÂ©tendre la plage
                        domain_scores[domain] = confidence
                
                # Nouveaux seuils adaptatifs basÃƒÂ©s sur la distribution des scores
                if len(domain_scores) > 1:
                    # Pour les cas multi-domaines, ÃƒÂªtre plus permissif
                    min_confidence = 0.2  # Seuil plus bas pour capturer les domaines secondaires
                else:
                    # Pour les cas mono-domaine, maintenir un seuil ÃƒÂ©levÃƒÂ©
                    min_confidence = 0.4
                
                # Filtrer les domaines avec une confiance suffisante
                confident_domains = {domain: score for domain, score in domain_scores.items() if score > min_confidence}
                
                # Si aucun domaine n'atteint le seuil, prendre le top 2
                if not confident_domains and len(domain_scores) >= 2:
                    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
                    confident_domains = {domain: score for domain, score in sorted_domains[:2]}
                    logger.info(f"Fallback: sÃƒÂ©lection des 2 domaines principaux avec scores {confident_domains}")
            else:
                confident_domains = {}
            
            # Si aucun domaine confiant, essayer la dÃƒÂ©tection par contexte
            if not confident_domains:
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # DÃƒÂ©tection par mots-clÃƒÂ©s spÃƒÂ©cifiques (plus sensible)
                    if any(word in keyword_lower for word in ["brain", "neural", "cognitive", "neural network"]):
                        confident_domains["neuroscience"] = 0.8
                    elif any(word in keyword_lower for word in ["computer", "digital", "software", "artificial", "intelligence", "machine", "learning", "quantum", "blockchain", "virtual", "reality", "cybersecurity", "data science", "internet of things", "natural language", "processing", "autonomous", "vehicle", "perception", "big data", "analytics", "platform", "cloud computing", "infrastructure", "mobile app", "development", "web application", "security"]):
                        confident_domains["technology"] = 0.8
                    elif any(word in keyword_lower for word in ["research", "laboratory", "experiment", "space", "exploration", "medical research"]):
                        confident_domains["science"] = 0.8
                    elif any(word in keyword_lower for word in ["company", "business", "enterprise", "growth", "financial", "planning"]):
                        confident_domains["business"] = 0.8
                    elif any(word in keyword_lower for word in ["school", "learning", "education", "educational"]):
                        confident_domains["education"] = 0.8
                    elif any(word in keyword_lower for word in ["health", "medical", "patient", "healthcare"]):
                        confident_domains["healthcare"] = 0.8
            
            return confident_domains
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse multi-domaines: {e}")
            return {}

# Instance globale pour utilisation dans le pipeline
keyword_expander = IntelligentKeywordExpander()

def expand_keywords_with_synonyms(primary_keyword: str, domain: str = "general") -> List[str]:
    """Fonction utilitaire pour l'expansion des mots-clÃƒÂ©s"""
    return keyword_expander.expand_keywords(primary_keyword, domain)

def get_search_queries_for_keywords(keywords: List[str], domain: str = "general") -> List[str]:
    """Fonction utilitaire pour gÃƒÂ©nÃƒÂ©rer des requÃƒÂªtes de recherche"""
    return keyword_expander.get_search_queries(keywords, domain)

def analyze_domain_from_keywords(keywords: List[str]) -> str:
    """Fonction utilitaire pour analyser le domaine"""
    return keyword_expander.analyze_domain_from_keywords(keywords)

def analyze_multiple_domains_from_keywords(keywords: List[str]) -> Dict[str, float]:
    """Fonction utilitaire pour analyser les domaines multiples avec scores de confiance"""
    return keyword_expander.analyze_multiple_domains_from_keywords(keywords)

def expand_keywords_multi_domain(primary_keyword: str, keywords: List[str] = None) -> List[str]:
    """Fonction utilitaire pour l'expansion multi-domaines"""
    return keyword_expander.expand_keywords_multi_domain(primary_keyword, keywords) 

