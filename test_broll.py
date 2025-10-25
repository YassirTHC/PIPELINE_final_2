ï»¿# -*- coding: utf-8 -*-
from utils.optimized_llm import OptimizedLLM

llm = OptimizedLLM()
transcript = 'The first thing that occurs when people start to realize that rewards are all internal'

print('=== TEST B-ROLL SÃ‰PARÃ‰ ===')
success, broll = llm.generate_broll_keywords_and_queries(transcript, max_keywords=8)
print(f'Success: {success}')
if success:
    print(f'Domain: {broll.get("domain")}')
    print(f'Context: {broll.get("context")}')
    print(f'B-roll keywords: {broll.get("broll_keywords")}')
    print(f'Search queries: {broll.get("search_queries")}')
else:
    print('Ã‰CHEC - Pas de B-roll gÃ©nÃ©rÃ©s')


