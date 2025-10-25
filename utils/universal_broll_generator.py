ï»¿# -*- coding: utf-8 -*-
# Ã°Å¸Å¡â‚¬ GÃƒâ€°NÃƒâ€°RATEUR UNIVERSEL B-ROLL + MÃƒâ€°TADONNÃƒâ€°ES TIKTOK/INSTAGRAM
# Fonctionne sur TOUS les domaines : science, sport, finance, lifestyle, tech, etc.

import logging
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class BrollKeyword:
    """Mots-clÃƒÂ©s B-roll avec mÃƒÂ©tadonnÃƒÂ©es"""
    text: str
    category: str
    confidence: float
    search_query: str
    visual_specificity: float

class UniversalBrollGenerator:
    """GÃƒÂ©nÃƒÂ©rateur universel de B-roll et mÃƒÂ©tadonnÃƒÂ©es TikTok/Instagram"""
    
    def __init__(self):
        # CatÃƒÂ©gories universelles pour TOUS les domaines
        self.universal_categories = {
            'people': {
                'weight': 0.25,  # 25% des mots-clÃƒÂ©s
                'max_per_category': 3,
                'examples': {
                    'science': ['scientist', 'researcher', 'student', 'professor'],
                    'sport': ['athlete', 'coach', 'player', 'trainer'],
                    'finance': ['trader', 'analyst', 'entrepreneur', 'investor'],
                    'lifestyle': ['chef', 'traveler', 'artist', 'influencer'],
                    'tech': ['developer', 'designer', 'engineer', 'creator']
                }
            },
            'actions': {
                'weight': 0.25,  # 25% des mots-clÃƒÂ©s
                'max_per_category': 3,
                'examples': {
                    'science': ['experimenting', 'analyzing', 'researching', 'discovering'],
                    'sport': ['running', 'training', 'competing', 'celebrating'],
                    'finance': ['trading', 'analyzing', 'meeting', 'planning'],
                    'lifestyle': ['cooking', 'traveling', 'creating', 'exploring'],
                    'tech': ['coding', 'designing', 'testing', 'innovating']
                }
            },
            'environments': {
                'weight': 0.20,  # 20% des mots-clÃƒÂ©s
                'max_per_category': 2,
                'examples': {
                    'science': ['laboratory', 'research_center', 'university', 'field_study'],
                    'sport': ['stadium', 'gym', 'outdoors', 'training_facility'],
                    'finance': ['office', 'trading_floor', 'conference_room', 'business_center'],
                    'lifestyle': ['kitchen', 'studio', 'outdoors', 'urban_setting'],
                    'tech': ['workspace', 'server_room', 'creative_studio', 'innovation_lab']
                }
            },
            'objects': {
                'weight': 0.20,  # 20% des mots-clÃƒÂ©s
                'max_per_category': 2,
                'examples': {
                    'science': ['microscope', 'test_tubes', 'charts', 'equipment'],
                    'sport': ['equipment', 'trophy', 'uniform', 'gear'],
                    'finance': ['charts', 'documents', 'computer', 'phone'],
                    'lifestyle': ['tools', 'ingredients', 'camera', 'art_supplies'],
                    'tech': ['computer', 'mobile_device', 'prototype', 'software']
                }
            },
            'concepts': {
                'weight': 0.10,  # 10% des mots-clÃƒÂ©s
                'max_per_category': 1,
                'examples': {
                    'science': ['discovery', 'innovation', 'breakthrough', 'research'],
                    'sport': ['achievement', 'teamwork', 'dedication', 'victory'],
                    'finance': ['growth', 'success', 'strategy', 'opportunity'],
                    'lifestyle': ['creativity', 'passion', 'adventure', 'inspiration'],
                    'tech': ['innovation', 'creativity', 'problem_solving', 'future']
                }
            }
        }
        
        # Mots trop gÃƒÂ©nÃƒÂ©riques ÃƒÂ  ÃƒÂ©viter (universels)
        self.generic_keywords = {
            'thing', 'stuff', 'way', 'time', 'place', 'work', 'make', 'do', 'get',
            'go', 'come', 'see', 'look', 'hear', 'feel', 'think', 'know', 'want',
            'need', 'good', 'bad', 'big', 'small', 'new', 'old', 'right', 'wrong'
        }
        
        # Patterns pour identifier la spÃƒÂ©cificitÃƒÂ© visuelle
        self.visual_patterns = [
            r'[a-z]+_[a-z]+',  # doctor_office, therapy_session
            r'[a-z]+\s+[a-z]+',  # medical consultation, brain scan
            r'[a-z]+[A-Z][a-z]+',  # medicalChart, brainScan
            r'[a-z]+ing',  # running, cooking, analyzing
            r'[a-z]+er',  # runner, cooker, analyzer
        ]

    def detect_domain_universal(self, transcript: str) -> Tuple[str, float]:
        """DÃƒÂ©tection de domaine universelle basÃƒÂ©e sur le contenu"""
        transcript_lower = transcript.lower()
        
        # Domaines avec mots-clÃƒÂ©s caractÃƒÂ©ristiques
        domain_keywords = {
            'science': ['research', 'study', 'experiment', 'analysis', 'data', 'scientific', 'discovery', 'theory', 'scientists', 'cognitive', 'brain', 'dopamine', 'norepinephrine'],
            'sport': ['athlete', 'training', 'competition', 'game', 'match', 'team', 'coach', 'performance'],
            'finance': ['money', 'investment', 'trading', 'business', 'market', 'profit', 'financial', 'economy'],
            'lifestyle': ['cooking', 'travel', 'fashion', 'beauty', 'fitness', 'wellness', 'creativity', 'hobby'],
            'tech': ['technology', 'software', 'digital', 'innovation', 'coding', 'design', 'app', 'platform'],
            'education': ['learning', 'teaching', 'education', 'course', 'lesson', 'student', 'knowledge', 'study'],
            'health': ['health', 'medical', 'therapy', 'treatment', 'wellness', 'fitness', 'nutrition', 'medicine'],
            'entertainment': ['entertainment', 'music', 'movie', 'show', 'performance', 'art', 'culture', 'creative']
        }
        
        # Calcul du score par domaine
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in transcript_lower)
            domain_scores[domain] = score / len(keywords)
        
        # Trouver le domaine dominant
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # Si aucun domaine clair, utiliser 'general'
        if best_domain[1] < 0.2:
            return 'general', 0.5
        
        return best_domain[0], best_domain[1]

    def extract_visual_keywords(self, transcript: str, domain: str) -> List[str]:
        """Extraction de mots-clÃƒÂ©s visuels depuis le transcript"""
        words = re.findall(r'\b[a-zA-Z]+\b', transcript.lower())
        
        # Filtrer les mots trop courts et gÃƒÂ©nÃƒÂ©riques
        visual_keywords = []
        for word in words:
            if (len(word) > 3 and 
                word not in self.generic_keywords and
                any(re.match(pattern, word) for pattern in self.visual_patterns)):
                visual_keywords.append(word)
        
        # Ajouter des mots-clÃƒÂ©s spÃƒÂ©cifiques au domaine
        domain_examples = []
        for category in self.universal_categories.values():
            if domain in category['examples']:
                domain_examples.extend(category['examples'][domain])
        
        # Combiner et dÃƒÂ©dupliquer
        all_keywords = list(set(visual_keywords + domain_examples))
        
        return all_keywords[:20]  # Limiter ÃƒÂ  20 mots-clÃƒÂ©s

    def generate_broll_keywords_universal(self, transcript: str, target_count: int = 10) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration universelle de mots-clÃƒÂ©s B-roll"""
        logger.info(f"Ã°Å¸Å¡â‚¬ GÃƒÂ©nÃƒÂ©ration B-roll universelle pour {len(transcript)} caractÃƒÂ¨res")
        
        # 1. DÃƒÂ©tection de domaine
        domain, confidence = self.detect_domain_universal(transcript)
        logger.info(f"Ã°Å¸Å½Â¯ Domaine dÃƒÂ©tectÃƒÂ©: {domain} (confiance: {confidence:.2f})")
        
        # 2. Extraction de mots-clÃƒÂ©s visuels
        visual_keywords = self.extract_visual_keywords(transcript, domain)
        logger.info(f"Ã°Å¸â€Â {len(visual_keywords)} mots-clÃƒÂ©s visuels extraits")
        
        # 3. GÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s diversifiÃƒÂ©s
        diversified_keywords = self._apply_diversity_strategy(visual_keywords, domain, target_count)
        
        # 4. GÃƒÂ©nÃƒÂ©ration de requÃƒÂªtes de recherche
        search_queries = self._generate_search_queries(diversified_keywords, domain)
        
        # 5. Calcul des mÃƒÂ©triques
        metrics = self._calculate_diversity_metrics(diversified_keywords)
        
        result = {
            'keywords': diversified_keywords,
            'search_queries': search_queries,
            'domain': domain,
            'domain_confidence': confidence,
            'diversity_metrics': metrics,
            'total_keywords': len(diversified_keywords)
        }
        
        logger.info(f"Ã¢Å“â€¦ B-roll universel gÃƒÂ©nÃƒÂ©rÃƒÂ©: {len(diversified_keywords)} mots-clÃƒÂ©s")
        return result

    def _apply_diversity_strategy(self, keywords: List[str], domain: str, target_count: int) -> List[str]:
        """Application de la stratÃƒÂ©gie de diversitÃƒÂ© universelle"""
        categorized = defaultdict(list)
        
        # CatÃƒÂ©goriser les mots-clÃƒÂ©s existants
        for keyword in keywords:
            category = self._categorize_keyword(keyword, domain)
            categorized[category].append(keyword)
        
        # Appliquer la pondÃƒÂ©ration par catÃƒÂ©gorie
        final_keywords = []
        for category, config in self.universal_categories.items():
            max_count = min(config['max_per_category'], int(target_count * config['weight']))
            category_keywords = categorized[category][:max_count]
            
            # Si pas assez de mots-clÃƒÂ©s dans cette catÃƒÂ©gorie, en gÃƒÂ©nÃƒÂ©rer
            if len(category_keywords) < max_count:
                needed = max_count - len(category_keywords)
                generated = self._generate_category_keywords(category, domain, needed)
                category_keywords.extend(generated)
            
            final_keywords.extend(category_keywords[:max_count])
        
        # GARANTIR le nombre minimum de mots-clÃƒÂ©s
        if len(final_keywords) < target_count:
            # Ajouter des mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques du domaine si nÃƒÂ©cessaire
            domain_generic = self._get_domain_generic_keywords(domain, target_count - len(final_keywords))
            final_keywords.extend(domain_generic)
        
        # Limiter au nombre cible et mÃƒÂ©langer
        final_keywords = final_keywords[:target_count]
        import random
        random.shuffle(final_keywords)
        
        return final_keywords

    def _categorize_keyword(self, keyword: str, domain: str) -> str:
        """CatÃƒÂ©gorisation d'un mot-clÃƒÂ©"""
        # Logique de catÃƒÂ©gorisation basÃƒÂ©e sur le domaine
        if domain in ['sport', 'fitness']:
            if any(word in keyword for word in ['run', 'train', 'play', 'compete']):
                return 'actions'
            elif any(word in keyword for word in ['athlete', 'player', 'coach']):
                return 'people'
            elif any(word in keyword for word in ['stadium', 'gym', 'field']):
                return 'environments'
        
        # CatÃƒÂ©gorisation par dÃƒÂ©faut
        if keyword.endswith('ing') or keyword.endswith('er'):
            return 'actions'
        elif keyword in ['person', 'people', 'man', 'woman']:
            return 'people'
        elif keyword in ['place', 'room', 'building']:
            return 'environments'
        else:
            return 'objects'

    def _generate_category_keywords(self, category: str, domain: str, count: int) -> List[str]:
        """GÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s pour une catÃƒÂ©gorie spÃƒÂ©cifique"""
        if domain in self.universal_categories[category]['examples']:
            examples = self.universal_categories[category]['examples'][domain]
            return examples[:count]
        return []
    
    def _get_domain_generic_keywords(self, domain: str, count: int) -> List[str]:
        """GÃƒÂ©nÃƒÂ©ration de mots-clÃƒÂ©s gÃƒÂ©nÃƒÂ©riques par domaine pour garantir le nombre minimum"""
        generic_keywords = {
            'science': ['research', 'study', 'analysis', 'discovery', 'experiment', 'data', 'scientific', 'laboratory'],
            'sport': ['training', 'performance', 'athletic', 'fitness', 'competition', 'teamwork', 'dedication'],
            'finance': ['business', 'professional', 'corporate', 'strategy', 'planning', 'success', 'growth'],
            'lifestyle': ['daily', 'modern', 'contemporary', 'creative', 'inspiring', 'motivational', 'wellness'],
            'tech': ['digital', 'modern', 'technology', 'innovation', 'creative', 'problem_solving', 'future'],
            'education': ['learning', 'knowledge', 'growth', 'development', 'skill', 'improvement', 'progress'],
            'health': ['wellness', 'fitness', 'health', 'medical', 'therapeutic', 'healing', 'recovery'],
            'entertainment': ['creative', 'artistic', 'cultural', 'entertaining', 'engaging', 'inspiring', 'fun']
        }
        
        if domain in generic_keywords:
            keywords = generic_keywords[domain]
            # Retourner le nombre demandÃƒÂ©, en rÃƒÂ©pÃƒÂ©tant si nÃƒÂ©cessaire
            result = []
            for i in range(count):
                result.append(keywords[i % len(keywords)])
            return result
        
        # Fallback gÃƒÂ©nÃƒÂ©rique
        return ['content', 'media', 'visual', 'engaging', 'professional'] * count

    def _generate_search_queries(self, keywords: List[str], domain: str) -> List[str]:
        """GÃƒÂ©nÃƒÂ©ration de requÃƒÂªtes de recherche optimisÃƒÂ©es"""
        queries = []
        
        for keyword in keywords:
            # CrÃƒÂ©er des requÃƒÂªtes 2-4 mots
            if '_' in keyword:
                # Mots composÃƒÂ©s
                query = keyword.replace('_', ' ')
            elif len(keyword) > 8:
                # Mots longs
                query = keyword
            else:
                # Ajouter un contexte
                context_words = {
                    'science': ['research', 'study', 'analysis'],
                    'sport': ['training', 'performance', 'athletic'],
                    'finance': ['business', 'professional', 'corporate'],
                    'lifestyle': ['daily', 'modern', 'contemporary'],
                    'tech': ['digital', 'modern', 'technology']
                }
                
                context = context_words.get(domain, ['modern'])[0]
                query = f"{context} {keyword}"
            
            queries.append(query)
        
        return queries

    def _calculate_diversity_metrics(self, keywords: List[str]) -> Dict[str, Any]:
        """Calcul des mÃƒÂ©triques de diversitÃƒÂ©"""
        categories = [self._categorize_keyword(kw, 'general') for kw in keywords]
        category_counts = defaultdict(int)
        
        for cat in categories:
            category_counts[cat] += 1
        
        total = len(keywords)
        diversity_score = len(set(categories)) / len(self.universal_categories)
        
        return {
            'categories_covered': len(set(categories)),
            'diversity_score': diversity_score,
            'category_distribution': dict(category_counts),
            'total_keywords': total
        }

class TikTokInstagramMetadataGenerator:
    """GÃƒÂ©nÃƒÂ©rateur de mÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram"""
    
    def __init__(self):
        # Styles de titres viraux par domaine
        self.viral_titles = {
            'science': [
                "Ã°Å¸â€Â¥ The Science Behind {topic}",
                "Ã°Å¸Â§Â  {topic} Explained Simply",
                "Ã¢Å¡Â¡ {topic} - What You Need to Know",
                "Ã°Å¸â€™Â¡ {topic} - Mind-Blowing Facts",
                "Ã°Å¸â€Â¬ {topic} - The Truth Revealed"
            ],
            'sport': [
                "Ã°Å¸ÂÆ’Ã¢â‚¬ÂÃ¢â„¢â€šÃ¯Â¸Â {topic} - Game Changer",
                "Ã°Å¸â€™Âª {topic} - Next Level",
                "Ã°Å¸â€Â¥ {topic} - Unstoppable",
                "Ã¢Å¡Â¡ {topic} - Peak Performance",
                "Ã°Å¸Ââ€  {topic} - Champion's Guide"
            ],
            'finance': [
                "Ã°Å¸â€™Â° {topic} - Money Moves",
                "Ã°Å¸â€œË† {topic} - Investment Secrets",
                "Ã°Å¸â€™Å½ {topic} - Wealth Building",
                "Ã°Å¸Å¡â‚¬ {topic} - Financial Freedom",
                "Ã°Å¸â€™Â¼ {topic} - Business Success"
            ],
            'lifestyle': [
                "Ã¢Å“Â¨ {topic} - Life Changing",
                "Ã°Å¸Å’Å¸ {topic} - Next Level You",
                "Ã°Å¸â€™Â« {topic} - Transform Your Life",
                "Ã°Å¸â€Â¥ {topic} - Game Changer",
                "Ã°Å¸â€™Å½ {topic} - Premium Tips"
            ],
            'tech': [
                "Ã°Å¸Å¡â‚¬ {topic} - Future Tech",
                "Ã°Å¸â€™Â» {topic} - Innovation Guide",
                "Ã¢Å¡Â¡ {topic} - Tech Revolution",
                "Ã°Å¸â€Â® {topic} - Next Generation",
                "Ã°Å¸â€™Â¡ {topic} - Smart Solutions"
            ]
        }
        
        # Hashtags viraux par domaine
        self.viral_hashtags = {
            'science': ['#science', '#research', '#discovery', '#innovation', '#facts', '#knowledge', '#education', '#learning'],
            'sport': ['#sport', '#fitness', '#training', '#motivation', '#athlete', '#performance', '#goals', '#dedication'],
            'finance': ['#finance', '#money', '#investment', '#business', '#success', '#wealth', '#entrepreneur', '#growth'],
            'lifestyle': ['#lifestyle', '#motivation', '#inspiration', '#selfimprovement', '#goals', '#success', '#mindset', '#growth'],
            'tech': ['#tech', '#innovation', '#technology', '#future', '#digital', '#ai', '#automation', '#progress']
        }

    def generate_viral_metadata(self, transcript: str, domain: str, broll_keywords: List[str]) -> Dict[str, Any]:
        """GÃƒÂ©nÃƒÂ©ration de mÃƒÂ©tadonnÃƒÂ©es virales TikTok/Instagram"""
        logger.info(f"Ã°Å¸Å½Â¬ GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales pour {domain}")
        
        # 1. Titre viral
        title = self._generate_viral_title(transcript, domain)
        
        # 2. Description engageante
        description = self._generate_viral_description(transcript, domain)
        
        # 3. Hashtags viraux
        hashtags = self._generate_viral_hashtags(domain, broll_keywords)
        
        # 4. Call-to-action
        cta = self._generate_call_to_action(domain)
        
        result = {
            'title': title,
            'description': description,
            'hashtags': hashtags,
            'call_to_action': cta,
            'platform_optimized': True,
            'viral_potential': 'high'
        }
        
        logger.info(f"Ã¢Å“â€¦ MÃƒÂ©tadonnÃƒÂ©es virales gÃƒÂ©nÃƒÂ©rÃƒÂ©es: {len(hashtags)} hashtags")
        return result

    def _generate_viral_title(self, transcript: str, domain: str) -> str:
        """GÃƒÂ©nÃƒÂ©ration de titre viral"""
        # Extraire le sujet principal du transcript
        words = transcript.split()[:10]  # Premiers mots
        topic = ' '.join(words[:3]).title()
        
        # SÃƒÂ©lectionner un template viral
        if domain in self.viral_titles:
            import random
            template = random.choice(self.viral_titles[domain])
            return template.format(topic=topic)
        
        # Template par dÃƒÂ©faut
        return f"Ã°Å¸â€Â¥ {topic} - What You Need to Know"

    def _generate_viral_description(self, transcript: str, domain: str) -> str:
        """GÃƒÂ©nÃƒÂ©ration de description virale"""
        # Extraire les points clÃƒÂ©s
        sentences = transcript.split('.')[:2]
        key_points = '. '.join(sentences).strip()
        
        # Ajouter un hook viral
        hooks = {
            'science': "Ã°Å¸Â§Â  Mind-blowing science that will change how you see the world!",
            'sport': "Ã°Å¸â€™Âª This will take your performance to the next level!",
            'finance': "Ã°Å¸â€™Â° Money secrets that successful people know!",
            'lifestyle': "Ã¢Å“Â¨ Transform your life with these game-changing insights!",
            'tech': "Ã°Å¸Å¡â‚¬ Future tech that's happening right now!"
        }
        
        hook = hooks.get(domain, "Ã°Å¸â€Â¥ This will blow your mind!")
        
        return f"{hook}\n\n{key_points}\n\nÃ°Å¸â€™Â¡ Save this for later!"

    def _generate_viral_hashtags(self, domain: str, broll_keywords: List[str]) -> List[str]:
        """GÃƒÂ©nÃƒÂ©ration de hashtags viraux"""
        hashtags = []
        
        # Hashtags de base du domaine
        if domain in self.viral_hashtags:
            hashtags.extend(self.viral_hashtags[domain])
        
        # Hashtags gÃƒÂ©nÃƒÂ©riques viraux
        generic_viral = ['#viral', '#trending', '#fyp', '#foryou', '#mustsee', '#amazing', '#incredible']
        hashtags.extend(generic_viral)
        
        # Hashtags basÃƒÂ©s sur les mots-clÃƒÂ©s B-roll
        for keyword in broll_keywords[:3]:  # Top 3 mots-clÃƒÂ©s
            hashtag = f"#{keyword.replace('_', '')}"
            if len(hashtag) <= 20:  # Limiter la longueur
                hashtags.append(hashtag)
        
        # Limiter ÃƒÂ  15 hashtags maximum
        return hashtags[:15]

    def _generate_call_to_action(self, domain: str) -> str:
        """GÃƒÂ©nÃƒÂ©ration de call-to-action viral"""
        ctas = {
            'science': "Ã°Å¸â€Â¬ Follow for more mind-blowing science!",
            'sport': "Ã°Å¸â€™Âª Follow for peak performance tips!",
            'finance': "Ã°Å¸â€™Â° Follow for financial freedom!",
            'lifestyle': "Ã¢Å“Â¨ Follow for life transformation!",
            'tech': "Ã°Å¸Å¡â‚¬ Follow for future tech insights!"
        }
        
        return ctas.get(domain, "Ã°Å¸â€Â¥ Follow for more amazing content!")

def create_universal_broll_generator() -> UniversalBrollGenerator:
    """Factory pour crÃƒÂ©er le gÃƒÂ©nÃƒÂ©rateur universel"""
    return UniversalBrollGenerator()

def create_tiktok_metadata_generator() -> TikTokInstagramMetadataGenerator:
    """Factory pour crÃƒÂ©er le gÃƒÂ©nÃƒÂ©rateur de mÃƒÂ©tadonnÃƒÂ©es TikTok/Instagram"""
    return TikTokInstagramMetadataGenerator()

# Test rapide
if __name__ == "__main__":
    # Test avec diffÃƒÂ©rents domaines
    generator = create_universal_broll_generator()
    metadata_gen = create_tiktok_metadata_generator()
    
    test_transcripts = {
        'science': "Research shows that cognitive control and effort are linked to dopamine levels in the brain. Scientists discovered that when we exert effort, norepinephrine is released.",
        'sport': "Athletes train hard to improve performance. The coach emphasizes dedication and teamwork. Running and training require consistent effort.",
        'finance': "Investment strategies require careful analysis. Traders study market trends and make informed decisions. Financial planning is crucial for success."
    }
    
    for domain, transcript in test_transcripts.items():
        print(f"\nÃ°Å¸Å½Â¯ TEST DOMAINE: {domain.upper()}")
        print("=" * 50)
        
        # GÃƒÂ©nÃƒÂ©ration B-roll
        broll_result = generator.generate_broll_keywords_universal(transcript)
        print(f"Ã°Å¸â€â€˜ B-roll: {broll_result['keywords']}")
        
        # GÃƒÂ©nÃƒÂ©ration mÃƒÂ©tadonnÃƒÂ©es virales
        metadata_result = metadata_gen.generate_viral_metadata(transcript, domain, broll_result['keywords'])
        print(f"Ã°Å¸â€œÂ Titre: {metadata_result['title']}")
        print(f"#Ã¯Â¸ÂÃ¢Æ’Â£ Hashtags: {metadata_result['hashtags'][:5]}...") 

