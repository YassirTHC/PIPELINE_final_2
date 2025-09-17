
with open('temp_function.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modifier le prompt pour inclure emojis viraux
old_pattern = '1. title: short, catchy (<= 60 characters), TikTok/Instagram Reels style'
new_pattern = '1. title: MUST START with viral emoji (fire/lightning/rocket/brain/target), catchy (<= 60 characters), TikTok/Instagram Reels style'

# Modifier description
old_desc_pattern = '2. description: 1-2 punchy sentences with an implicit call-to-action'
new_desc_pattern = '2. description: 1-2 punchy sentences with emojis and STRONG CTA (Watch NOW, BELIEVE this, SHARE if agree)'

content = content.replace(old_pattern, new_pattern)
content = content.replace(old_desc_pattern, new_desc_pattern)

with open('temp_function.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESS: Prompt modified for viral emojis!')
