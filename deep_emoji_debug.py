#!/usr/bin/env python3
"""
Debug approfondi du problème emoji avec PIL
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_font_emoji_support():
    """Tester si la police supporte vraiment les emojis"""
    
    print("🔍 TEST SUPPORT EMOJI POLICE")
    print("=" * 35)
    
    sys.path.append('.')
    from tiktok_subtitles import get_emoji_font
    
    # Charger la police
    font = get_emoji_font(60)
    print(f"✅ Police chargée: {font}")
    
    # Tests avec différents caractères
    test_cases = [
        ("Texte simple", "HELLO"),
        ("Emoji seul", "💯"),
        ("Emoji fire", "🔥"),
        ("Emoji target", "🎯"),
        ("Emoji sparkles", "✨"),
        ("Texte + emoji", "HELLO 💯"),
        ("Unicode explicit", "\U0001F4AF"),  # 💯 en unicode
    ]
    
    for desc, text in test_cases:
        print(f"\n📝 Test: {desc} - '{text}'")
        
        # Créer image test
        img = Image.new('RGB', (300, 100), 'black')
        draw = ImageDraw.Draw(img)
        
        try:
            # Tester le rendu
            draw.text((10, 30), text, font=font, fill='white')
            
            # Analyser le résultat
            arr = np.array(img)
            pixels = np.sum(arr > 0)
            
            # Sauvegarder
            filename = f"test_{desc.replace(' ', '_').lower()}.png"
            img.save(filename)
            
            print(f"   📊 Pixels visibles: {pixels}")
            print(f"   💾 Sauvé: {filename}")
            
            if pixels > 100:
                print("   ✅ Rendu réussi")
            else:
                print("   ❌ Rendu échoué (trop peu de pixels)")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

def test_different_fonts():
    """Tester différentes polices pour emojis"""
    
    print("\n🔍 TEST DIFFÉRENTES POLICES")
    print("=" * 35)
    
    fonts_to_test = [
        ("Segoe UI", r"C:\Windows\Fonts\segoeui.ttf"),
        ("Segoe UI Emoji", r"C:\Windows\Fonts\seguiemj.ttf"),
        ("Segoe UI Symbol", r"C:\Windows\Fonts\seguisym.ttf"),
        ("Arial Unicode MS", r"C:\Windows\Fonts\arialuni.ttf"),
        ("Noto Color Emoji", r"C:\Windows\Fonts\NotoColorEmoji.ttf"),
        ("Default", None),  # Police par défaut PIL
    ]
    
    test_text = "TEST 💯 EMOJI"
    
    for name, path in fonts_to_test:
        print(f"\n📝 Test police: {name}")
        
        try:
            # Charger la police
            if path and Path(path).exists():
                font = ImageFont.truetype(path, 60)
                print(f"   ✅ Police chargée: {path}")
            elif path is None:
                font = ImageFont.load_default()
                print("   ✅ Police par défaut chargée")
            else:
                print(f"   ❌ Police introuvable: {path}")
                continue
            
            # Créer image test
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
            
            print(f"   📊 Pixels: {pixels}")
            print(f"   💾 Sauvé: {filename}")
            
            if pixels > 1000:
                print("   ✅ Police semble fonctionner")
            else:
                print("   ❌ Police ne rend pas bien")
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

def test_unicode_methods():
    """Tester différentes méthodes pour rendre les emojis"""
    
    print("\n🔍 TEST MÉTHODES UNICODE")
    print("=" * 30)
    
    # Différentes façons d'encoder l'emoji 💯
    emoji_methods = [
        ("Direct", "💯"),
        ("Unicode escape", "\U0001F4AF"),
        ("Surrogates", "\ud83d\udcaf"),
        ("Bytes decode", b'\xf0\x9f\x92\xaf'.decode('utf-8')),
        ("HTML entity", "&#128175;"),
    ]
    
    sys.path.append('.')
    from tiktok_subtitles import get_emoji_font
    font = get_emoji_font(60)
    
    for name, emoji_text in emoji_methods:
        print(f"\n📝 Méthode: {name}")
        print(f"   Texte: '{emoji_text}'")
        
        try:
            img = Image.new('RGB', (200, 100), 'black')
            draw = ImageDraw.Draw(img)
            
            draw.text((10, 30), f"TEST {emoji_text}", font=font, fill='white')
            
            arr = np.array(img)
            pixels = np.sum(arr > 0)
            
            filename = f"unicode_{name.lower()}.png"
            img.save(filename)
            
            print(f"   📊 Pixels: {pixels}")
            print(f"   💾 Sauvé: {filename}")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

def check_system_emoji_support():
    """Vérifier le support système des emojis"""
    
    print("\n🔍 SUPPORT SYSTÈME EMOJI")
    print("=" * 30)
    
    import platform
    print(f"🖥️ OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    
    # Vérifier les polices disponibles
    common_emoji_fonts = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf", 
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
        r"C:\Windows\Fonts\NotoColorEmoji.ttf",
    ]
    
    print("\n📁 Polices emoji disponibles:")
    for font_path in common_emoji_fonts:
        if Path(font_path).exists():
            size = Path(font_path).stat().st_size / (1024*1024)
            print(f"   ✅ {Path(font_path).name} ({size:.1f} MB)")
        else:
            print(f"   ❌ {Path(font_path).name} - introuvable")

def main():
    """Debug principal"""
    
    print("🔍 DEBUG APPROFONDI EMOJI PIL")
    print("=" * 50)
    
    test_font_emoji_support()
    test_different_fonts()
    test_unicode_methods()
    check_system_emoji_support()
    
    print("\n🎯 ANALYSE FINALE:")
    print("=" * 25)
    print("Vérifiez les images générées:")
    print("• font_test_*.png - test polices différentes")
    print("• test_*.png - test caractères différents") 
    print("• unicode_*.png - test méthodes unicode")
    
    print("\n💡 SOLUTIONS POSSIBLES:")
    print("1. Si aucune police ne marche → Problème système")
    print("2. Si une police marche → Changer dans get_emoji_font")
    print("3. Si unicode marche → Problème encodage")
    print("4. Si tout échoue → Utiliser images emoji externes")

if __name__ == "__main__":
    main() 