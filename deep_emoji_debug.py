ï»¿# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Debug approfondi du problÃƒÂ¨me emoji avec PIL
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_font_emoji_support():
    """Tester si la police supporte vraiment les emojis"""
    
    print("Ã°Å¸â€Â TEST SUPPORT EMOJI POLICE")
    print("=" * 35)
    
    sys.path.append('.')
    from tiktok_subtitles import get_emoji_font
    
    # Charger la police
    font = get_emoji_font(60)
    print(f"Ã¢Å“â€¦ Police chargÃƒÂ©e: {font}")
    
    # Tests avec diffÃƒÂ©rents caractÃƒÂ¨res
    test_cases = [
        ("Texte simple", "HELLO"),
        ("Emoji seul", "Ã°Å¸â€™Â¯"),
        ("Emoji fire", "Ã°Å¸â€Â¥"),
        ("Emoji target", "Ã°Å¸Å½Â¯"),
        ("Emoji sparkles", "Ã¢Å“Â¨"),
        ("Texte + emoji", "HELLO Ã°Å¸â€™Â¯"),
        ("Unicode explicit", "\U0001F4AF"),  # Ã°Å¸â€™Â¯ en unicode
    ]
    
    for desc, text in test_cases:
        print(f"\nÃ°Å¸â€œÂ Test: {desc} - '{text}'")
        
        # CrÃƒÂ©er image test
        img = Image.new('RGB', (300, 100), 'black')
        draw = ImageDraw.Draw(img)
        
        try:
            # Tester le rendu
            draw.text((10, 30), text, font=font, fill='white')
            
            # Analyser le rÃƒÂ©sultat
            arr = np.array(img)
            pixels = np.sum(arr > 0)
            
            # Sauvegarder
            filename = f"test_{desc.replace(' ', '_').lower()}.png"
            img.save(filename)
            
            print(f"   Ã°Å¸â€œÅ  Pixels visibles: {pixels}")
            print(f"   Ã°Å¸â€™Â¾ SauvÃƒÂ©: {filename}")
            
            if pixels > 100:
                print("   Ã¢Å“â€¦ Rendu rÃƒÂ©ussi")
            else:
                print("   Ã¢ÂÅ’ Rendu ÃƒÂ©chouÃƒÂ© (trop peu de pixels)")
                
        except Exception as e:
            print(f"   Ã¢ÂÅ’ Erreur: {e}")

def test_different_fonts():
    """Tester diffÃƒÂ©rentes polices pour emojis"""
    
    print("\nÃ°Å¸â€Â TEST DIFFÃƒâ€°RENTES POLICES")
    print("=" * 35)
    
    fonts_to_test = [
        ("Segoe UI", r"C:\Windows\Fonts\segoeui.ttf"),
        ("Segoe UI Emoji", r"C:\Windows\Fonts\seguiemj.ttf"),
        ("Segoe UI Symbol", r"C:\Windows\Fonts\seguisym.ttf"),
        ("Arial Unicode MS", r"C:\Windows\Fonts\arialuni.ttf"),
        ("Noto Color Emoji", r"C:\Windows\Fonts\NotoColorEmoji.ttf"),
        ("Default", None),  # Police par dÃƒÂ©faut PIL
    ]
    
    test_text = "TEST Ã°Å¸â€™Â¯ EMOJI"
    
    for name, path in fonts_to_test:
        print(f"\nÃ°Å¸â€œÂ Test police: {name}")
        
        try:
            # Charger la police
            if path and Path(path).exists():
                font = ImageFont.truetype(path, 60)
                print(f"   Ã¢Å“â€¦ Police chargÃƒÂ©e: {path}")
            elif path is None:
                font = ImageFont.load_default()
                print("   Ã¢Å“â€¦ Police par dÃƒÂ©faut chargÃƒÂ©e")
            else:
                print(f"   Ã¢ÂÅ’ Police introuvable: {path}")
                continue
            
            # CrÃƒÂ©er image test
            img = Image.new('RGB', (400, 100), 'black')
            draw = ImageDraw.Draw(img)
            
            # Rendu
            draw.text((10, 30), test_text, font=font, fill='white')
            
            # Analyser
            arr = np.array(img)
            pixels = np.sum(arr > 0)
            
            # Sauvegarder
            filename = f"font_test_{name.replace(' ', '_').lower()}.png"
            img.save(filename)
            
            print(f"   Ã°Å¸â€œÅ  Pixels: {pixels}")
            print(f"   Ã°Å¸â€™Â¾ SauvÃƒÂ©: {filename}")
            
            if pixels > 1000:
                print("   Ã¢Å“â€¦ Police semble fonctionner")
            else:
                print("   Ã¢ÂÅ’ Police ne rend pas bien")
                
        except Exception as e:
            print(f"   Ã¢ÂÅ’ Erreur: {e}")

def test_unicode_methods():
    """Tester diffÃƒÂ©rentes mÃƒÂ©thodes pour rendre les emojis"""
    
    print("\nÃ°Å¸â€Â TEST MÃƒâ€°THODES UNICODE")
    print("=" * 30)
    
    # DiffÃƒÂ©rentes faÃƒÂ§ons d'encoder l'emoji Ã°Å¸â€™Â¯
    emoji_methods = [
        ("Direct", "Ã°Å¸â€™Â¯"),
        ("Unicode escape", "\U0001F4AF"),
        ("Surrogates", "\ud83d\udcaf"),
        ("Bytes decode", b'\xf0\x9f\x92\xaf'.decode('utf-8')),
        ("HTML entity", "&#128175;"),
    ]
    
    sys.path.append('.')
    from tiktok_subtitles import get_emoji_font
    font = get_emoji_font(60)
    
    for name, emoji_text in emoji_methods:
        print(f"\nÃ°Å¸â€œÂ MÃƒÂ©thode: {name}")
        print(f"   Texte: '{emoji_text}'")
        
        try:
            img = Image.new('RGB', (200, 100), 'black')
            draw = ImageDraw.Draw(img)
            
            draw.text((10, 30), f"TEST {emoji_text}", font=font, fill='white')
            
            arr = np.array(img)
            pixels = np.sum(arr > 0)
            
            filename = f"unicode_{name.lower()}.png"
            img.save(filename)
            
            print(f"   Ã°Å¸â€œÅ  Pixels: {pixels}")
            print(f"   Ã°Å¸â€™Â¾ SauvÃƒÂ©: {filename}")
            
        except Exception as e:
            print(f"   Ã¢ÂÅ’ Erreur: {e}")

def check_system_emoji_support():
    """VÃƒÂ©rifier le support systÃƒÂ¨me des emojis"""
    
    print("\nÃ°Å¸â€Â SUPPORT SYSTÃƒË†ME EMOJI")
    print("=" * 30)
    
    import platform
    print(f"Ã°Å¸â€“Â¥Ã¯Â¸Â OS: {platform.system()} {platform.release()}")
    print(f"Ã°Å¸ÂÂ Python: {platform.python_version()}")
    
    # VÃƒÂ©rifier les polices disponibles
    common_emoji_fonts = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf", 
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
        r"C:\Windows\Fonts\NotoColorEmoji.ttf",
    ]
    
    print("\nÃ°Å¸â€œÂ Polices emoji disponibles:")
    for font_path in common_emoji_fonts:
        if Path(font_path).exists():
            size = Path(font_path).stat().st_size / (1024*1024)
            print(f"   Ã¢Å“â€¦ {Path(font_path).name} ({size:.1f} MB)")
        else:
            print(f"   Ã¢ÂÅ’ {Path(font_path).name} - introuvable")

def main():
    """Debug principal"""
    
    print("Ã°Å¸â€Â DEBUG APPROFONDI EMOJI PIL")
    print("=" * 50)
    
    test_font_emoji_support()
    test_different_fonts()
    test_unicode_methods()
    check_system_emoji_support()
    
    print("\nÃ°Å¸Å½Â¯ ANALYSE FINALE:")
    print("=" * 25)
    print("VÃƒÂ©rifiez les images gÃƒÂ©nÃƒÂ©rÃƒÂ©es:")
    print("Ã¢â‚¬Â¢ font_test_*.png - test polices diffÃƒÂ©rentes")
    print("Ã¢â‚¬Â¢ test_*.png - test caractÃƒÂ¨res diffÃƒÂ©rents") 
    print("Ã¢â‚¬Â¢ unicode_*.png - test mÃƒÂ©thodes unicode")
    
    print("\nÃ°Å¸â€™Â¡ SOLUTIONS POSSIBLES:")
    print("1. Si aucune police ne marche Ã¢â€ â€™ ProblÃƒÂ¨me systÃƒÂ¨me")
    print("2. Si une police marche Ã¢â€ â€™ Changer dans get_emoji_font")
    print("3. Si unicode marche Ã¢â€ â€™ ProblÃƒÂ¨me encodage")
    print("4. Si tout ÃƒÂ©choue Ã¢â€ â€™ Utiliser images emoji externes")

if __name__ == "__main__":
    main() 

