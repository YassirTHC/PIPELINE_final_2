# 🚀 GÉNÉRATEUR UNIVERSEL B-ROLL + MÉTADONNÉES TIKTOK/INSTAGRAM
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
    """Mots-clés B-roll avec métadonnées"""
    text: str
    category: str
    confidence: float
    search_query: str
    visual_specificity: float

class UniversalBrollGenerator:
    """Générateur universel de B-roll et métadonnées TikTok/Instagram"""
    
    def __init__(self):
        # Catégories universelles pour TOUS les domaines
        self.universal_categories = {
            'people': {
                'weight': 0.25,  # 25% des mots-clés
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
                'weight': 0.25,  # 25% des mots-clés
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
                'weight': 0.20,  # 20% des mots-clés
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
                'weight': 0.20,  # 20% des mots-clés
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
                'weight': 0.10,  # 10% des mots-clés
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
        
        # Mots trop génériques à éviter (universels)
        self.generic_keywords = {
            'thing', 'stuff', 'way', 'time', 'place', 'work', 'make', 'do', 'get',
            'go', 'come', 'see', 'look', 'hear', 'feel', 'think', 'know', 'want',
            'need', 'good', 'bad', 'big', 'small', 'new', 'old', 'right', 'wrong'
        }
        
        # Patterns pour identifier la spécificité visuelle
        self.visual_patterns = [
            r'[a-z]+_[a-z]+',  # doctor_office, therapy_session
            r'[a-z]+\s+[a-z]+',  # medical consultation, brain scan
            r'[a-z]+[A-Z][a-z]+',  # medicalChart, brainScan
            r'[a-z]+ing',  # running, cooking, analyzing
            r'[a-z]+er',  # runner, cooker, analyzer
        ]

    def detect_domain_universal(self, transcript: str) -> Tuple[str, float]:
        """Détection de domaine universelle basée sur le contenu"""
        transcript_lower = transcript.lower()
        
        # Domaines avec mots-clés caractéristiques
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
        """Extraction de mots-clés visuels depuis le transcript"""
        words = re.findall(r'\b[a-zA-Z]+\b', transcript.lower())
        
        # Filtrer les mots trop courts et génériques
        visual_keywords = []
        for word in words:
            if (len(word) > 3 and 
                word not in self.generic_keywords and
                any(re.match(pattern, word) for pattern in self.visual_patterns)):
                visual_keywords.append(word)
        
        # Ajouter des mots-clés spécifiques au domaine
        domain_examples = []
        for category in self.universal_categories.values():
            if domain in category['examples']:
                domain_examples.extend(category['examples'][domain])
        
        # Combiner et dédupliquer
        all_keywords = list(set(visual_keywords + domain_examples))
        
        return all_keywords[:20]  # Limiter à 20 mots-clés

    def generate_broll_keywords_universal(self, transcript: str, target_count: int = 10) -> Dict[str, Any]:
        """Génération universelle de mots-clés B-roll"""
        logger.info(f"🚀 Génération B-roll universelle pour {len(transcript)} caractères")
        
        # 1. Détection de domaine
        domain, confidence = self.detect_domain_universal(transcript)
        logger.info(f"🎯 Domaine détecté: {domain} (confiance: {confidence:.2f})")
        
        # 2. Extraction de mots-clés visuels
        visual_keywords = self.extract_visual_keywords(transcript, domain)
        logger.info(f"🔍 {len(visual_keywords)} mots-clés visuels extraits")
        
        # 3. Génération de mots-clés diversifiés
        diversified_keywords = self._apply_diversity_strategy(visual_keywords, domain, target_count)
        
        # 4. Génération de requêtes de recherche
        search_queries = self._generate_search_queries(diversified_keywords, domain)
        
        # 5. Calcul des métriques
        metrics = self._calculate_diversity_metrics(diversified_keywords)
        
        result = {
            'keywords': diversified_keywords,
            'search_queries': search_queries,
            'domain': domain,
            'domain_confidence': confidence,
            'diversity_metrics': metrics,
            'total_keywords': len(diversified_keywords)
        }
        
        logger.info(f"✅ B-roll universel généré: {len(diversified_keywords)} mots-clés")
        return result

    def _apply_diversity_strategy(self, keywords: List[str], domain: str, target_count: int) -> List[str]:
        """Application de la stratégie de diversité universelle"""
        categorized = defaultdict(list)
        
        # Catégoriser les mots-clés existants
        for keyword in keywords:
            category = self._categorize_keyword(keyword, domain)
            categorized[category].append(keyword)
        
        # Appliquer la pondération par catégorie
        final_keywords = []
        for category, config in self.universal_categories.items():
            max_count = min(config['max_per_category'], int(target_count * config['weight']))
            category_keywords = categorized[category][:max_count]
            
            # Si pas assez de mots-clés dans cette catégorie, en générer
            if len(category_keywords) < max_count:
                needed = max_count - len(category_keywords)
                generated = self._generate_category_keywords(category, domain, needed)
                category_keywords.extend(generated)
            
            final_keywords.extend(category_keywords[:max_count])
        
        # GARANTIR le nombre minimum de mots-clés
        if len(final_keywords) < target_count:
            # Ajouter des mots-clés génériques du domaine si nécessaire
            domain_generic = self._get_domain_generic_keywords(domain, target_count - len(final_keywords))
            final_keywords.extend(domain_generic)
        
        # Limiter au nombre cible et mélanger
        final_keywords = final_keywords[:target_count]
        import random
        random.shuffle(final_keywords)
        
        return final_keywords

    def _categorize_keyword(self, keyword: str, domain: str) -> str:
        """Catégorisation d'un mot-clé"""
        # Logique de catégorisation basée sur le domaine
        if domain in ['sport', 'fitness']:
            if any(word in keyword for word in ['run', 'train', 'play', 'compete']):
                return 'actions'
            elif any(word in keyword for word in ['athlete', 'player', 'coach']):
                return 'people'
            elif any(word in keyword for word in ['stadium', 'gym', 'field']):
                return 'environments'
        
        # Catégorisation par défaut
        if keyword.endswith('ing') or keyword.endswith('er'):
            return 'actions'
        elif keyword in ['person', 'people', 'man', 'woman']:
            return 'people'
        elif keyword in ['place', 'room', 'building']:
            return 'environments'
        else:
            return 'objects'

    def _generate_category_keywords(self, category: str, domain: str, count: int) -> List[str]:
        """Génération de mots-clés pour une catégorie spécifique"""
        if domain in self.universal_categories[category]['examples']:
            examples = self.universal_categories[category]['examples'][domain]
            return examples[:count]
        return []
    
    def _get_domain_generic_keywords(self, domain: str, count: int) -> List[str]:
        """Génération de mots-clés génériques par domaine pour garantir le nombre minimum"""
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
            # Retourner le nombre demandé, en répétant si nécessaire
            result = []
            for i in range(count):
                result.append(keywords[i % len(keywords)])
            return result
        
        # Fallback générique
        return ['content', 'media', 'visual', 'engaging', 'professional'] * count

    def _generate_search_queries(self, keywords: List[str], domain: str) -> List[str]:
        """Génération de requêtes de recherche optimisées"""
        queries = []
        
        for keyword in keywords:
            # Créer des requêtes 2-4 mots
            if '_' in keyword:
                # Mots composés
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
        """Calcul des métriques de diversité"""
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
    """Générateur de métadonnées virales TikTok/Instagram"""
    
    def __init__(self):
        # Styles de titres viraux par domaine
        self.viral_titles = {
            'science': [
                "🔥 The Science Behind {topic}",
                "🧠 {topic} Explained Simply",
                "⚡ {topic} - What You Need to Know",
                "💡 {topic} - Mind-Blowing Facts",
                "🔬 {topic} - The Truth Revealed"
            ],
            'sport': [
                "🏃‍♂️ {topic} - Game Changer",
                "💪 {topic} - Next Level",
                "🔥 {topic} - Unstoppable",
                "⚡ {topic} - Peak Performance",
                "🏆 {topic} - Champion's Guide"
            ],
            'finance': [
                "💰 {topic} - Money Moves",
                "📈 {topic} - Investment Secrets",
                "💎 {topic} - Wealth Building",
                "🚀 {topic} - Financial Freedom",
                "💼 {topic} - Business Success"
            ],
            'lifestyle': [
                "✨ {topic} - Life Changing",
                "🌟 {topic} - Next Level You",
                "💫 {topic} - Transform Your Life",
                "🔥 {topic} - Game Changer",
                "💎 {topic} - Premium Tips"
            ],
            'tech': [
                "🚀 {topic} - Future Tech",
                "💻 {topic} - Innovation Guide",
                "⚡ {topic} - Tech Revolution",
                "🔮 {topic} - Next Generation",
                "💡 {topic} - Smart Solutions"
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
        """Génération de métadonnées virales TikTok/Instagram"""
        logger.info(f"🎬 Génération métadonnées virales pour {domain}")
        
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
        
        logger.info(f"✅ Métadonnées virales générées: {len(hashtags)} hashtags")
        return result

    def _generate_viral_title(self, transcript: str, domain: str) -> str:
        """Génération de titre viral"""
        # Extraire le sujet principal du transcript
        words = transcript.split()[:10]  # Premiers mots
        topic = ' '.join(words[:3]).title()
        
        # Sélectionner un template viral
        if domain in self.viral_titles:
            import random
            template = random.choice(self.viral_titles[domain])
            return template.format(topic=topic)
        
        # Template par défaut
        return f"🔥 {topic} - What You Need to Know"

    def _generate_viral_description(self, transcript: str, domain: str) -> str:
        """Génération de description virale"""
        # Extraire les points clés
        sentences = transcript.split('.')[:2]
        key_points = '. '.join(sentences).strip()
        
        # Ajouter un hook viral
        hooks = {
            'science': "🧠 Mind-blowing science that will change how you see the world!",
            'sport': "💪 This will take your performance to the next level!",
            'finance': "💰 Money secrets that successful people know!",
            'lifestyle': "✨ Transform your life with these game-changing insights!",
            'tech': "🚀 Future tech that's happening right now!"
        }
        
        hook = hooks.get(domain, "🔥 This will blow your mind!")
        
        return f"{hook}\n\n{key_points}\n\n💡 Save this for later!"

    def _generate_viral_hashtags(self, domain: str, broll_keywords: List[str]) -> List[str]:
        """Génération de hashtags viraux"""
        hashtags = []
        
        # Hashtags de base du domaine
        if domain in self.viral_hashtags:
            hashtags.extend(self.viral_hashtags[domain])
        
        # Hashtags génériques viraux
        generic_viral = ['#viral', '#trending', '#fyp', '#foryou', '#mustsee', '#amazing', '#incredible']
        hashtags.extend(generic_viral)
        
        # Hashtags basés sur les mots-clés B-roll
        for keyword in broll_keywords[:3]:  # Top 3 mots-clés
            hashtag = f"#{keyword.replace('_', '')}"
            if len(hashtag) <= 20:  # Limiter la longueur
                hashtags.append(hashtag)
        
        # Limiter à 15 hashtags maximum
        return hashtags[:15]

    def _generate_call_to_action(self, domain: str) -> str:
        """Génération de call-to-action viral"""
        ctas = {
            'science': "🔬 Follow for more mind-blowing science!",
            'sport': "💪 Follow for peak performance tips!",
            'finance': "💰 Follow for financial freedom!",
            'lifestyle': "✨ Follow for life transformation!",
            'tech': "🚀 Follow for future tech insights!"
        }
        
        return ctas.get(domain, "🔥 Follow for more amazing content!")

def create_universal_broll_generator() -> UniversalBrollGenerator:
    """Factory pour créer le générateur universel"""
    return UniversalBrollGenerator()

def create_tiktok_metadata_generator() -> TikTokInstagramMetadataGenerator:
    """Factory pour créer le générateur de métadonnées TikTok/Instagram"""
    return TikTokInstagramMetadataGenerator()

# Test rapide
if __name__ == "__main__":
    # Test avec différents domaines
    generator = create_universal_broll_generator()
    metadata_gen = create_tiktok_metadata_generator()
    
    test_transcripts = {
        'science': "Research shows that cognitive control and effort are linked to dopamine levels in the brain. Scientists discovered that when we exert effort, norepinephrine is released.",
        'sport': "Athletes train hard to improve performance. The coach emphasizes dedication and teamwork. Running and training require consistent effort.",
        'finance': "Investment strategies require careful analysis. Traders study market trends and make informed decisions. Financial planning is crucial for success."
    }
    
    for domain, transcript in test_transcripts.items():
        print(f"\n🎯 TEST DOMAINE: {domain.upper()}")
        print("=" * 50)
        
        # Génération B-roll
        broll_result = generator.generate_broll_keywords_universal(transcript)
        print(f"🔑 B-roll: {broll_result['keywords']}")
        
        # Génération métadonnées virales
        metadata_result = metadata_gen.generate_viral_metadata(transcript, domain, broll_result['keywords'])
        print(f"📝 Titre: {metadata_result['title']}")
        print(f"#️⃣ Hashtags: {metadata_result['hashtags'][:5]}...") 