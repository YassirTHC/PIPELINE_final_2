from utils.optimized_llm import OptimizedLLM

llm = OptimizedLLM()
transcript = 'The first thing that occurs when people start to realize that rewards are all internal'

# Test avec le prompt SIMPLE
print('=== TEST PROMPT SIMPLE ===')
success, meta = llm.generate_complete_metadata(transcript)
print(f'Success: {success}')
if success:
    print(f'Title: {meta.get("title")}')
    print(f'Description: {meta.get("description")}')
    print(f'Hashtags: {meta.get("hashtags")}')
    print(f'Keywords: {meta.get("keywords")}')
else:
    print('ÉCHEC - Pas de métadonnées générées')

print('\n=== TEST PROMPT COMPLEXE (avec B-roll) ===')
# Test avec le prompt COMPLEXE
success2, meta2 = llm.generate_metadata_with_broll(transcript)
print(f'Success: {success2}')
if success2:
    print(f'Title: {meta2.get("title")}')
    print(f'Broll keywords: {meta2.get("broll_keywords")}')
    print(f'Search queries: {meta2.get("search_queries")}')
else:
    print('ÉCHEC - Pas de métadonnées B-roll générées')
