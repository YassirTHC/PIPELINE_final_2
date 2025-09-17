# processor_improved.py - Version améliorée avec vos modifications

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import whisper
import requests
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import numpy as np
import cv2
import mediapipe as mp
import random
import time
from tiktok_subtitles import add_tiktok_subtitles


logger = logging.getLogger(__name__)

class Config:
    """Configuration centralisée du pipeline"""
    CLIPS_FOLDER = Path("./clips")
    OUTPUT_FOLDER = Path("./output") 
    TEMP_FOLDER = Path("./temp")
    
    # Résolution cible pour les réseaux sociaux
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1280  # Format 9:16
    
    # Paramètres Whisper
    WHISPER_MODEL = "tiny"
    
    # Paramètres sous-titres améliorés
    SUBTITLE_FONT_SIZE = 70
    SUBTITLE_FONT = 'Segoe UI'  # Police compatible emoji (au lieu d'Impact)
    SUBTITLE_STROKE_WIDTH = 3

class VideoProcessorAI:
    """Processor vidéo avec IA améliorée"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        self._setup_directories()
        
        # Initialisation MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        
        # Cache pour éviter de recalculer les détections
        self.detection_cache = {}
    
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
            folder.mkdir(exist_ok=True)
    
    def reframe_to_vertical(self, clip_path: Path) -> Path:
        """
        Reframe dynamique basé sur la détection IA (MediaPipe)
        Version améliorée avec détection de visage en fallback
        """
        logger.info("🎯 Reframe dynamique avec IA (MediaPipe)")
        
        video = VideoFileClip(str(clip_path))
        fps = int(video.fps)
        duration = video.duration
        
        # Détection des centres d'intérêt
        x_centers = self._detect_focus_points(video, fps, duration)
        
        # Lissage avancé avec fenêtre adaptative
        x_centers_smooth = self._smooth_trajectory(x_centers, window_size=min(8, len(x_centers)//6))
        
        # Application du reframe dynamique
        def crop_frame(get_frame, t):
            frame = get_frame(t)
            h, w, _ = frame.shape
            
            # Index temporel
            frame_idx = int(t * fps)
            frame_idx = min(frame_idx, len(x_centers_smooth) - 1)
            
            x_center = x_centers_smooth[frame_idx] * w
            
            # Calcul du crop avec ratio 9:16
            crop_width = int(Config.TARGET_WIDTH * h / Config.TARGET_HEIGHT)
            crop_width = min(crop_width, w)  # Sécurité
            
            # Centrage avec limites
            x1 = int(max(0, min(w - crop_width, x_center - crop_width / 2)))
            x2 = x1 + crop_width
            
            cropped = frame[:, x1:x2]
            
            # Redimensionnement avec anti-aliasing
            resized = cv2.resize(cropped, (Config.TARGET_WIDTH, Config.TARGET_HEIGHT), 
                               interpolation=cv2.INTER_LANCZOS4)
            
            return resized
        
        reframed = video.fl_image(crop_frame)
        
        output_path = Config.TEMP_FOLDER / f"reframed_{clip_path.name}"
        reframed.write_videofile(
            str(output_path), 
            codec='h264_nvenc', 
            audio_codec='aac', 
            verbose=False, 
            logger=None,
            preset=None,
            ffmpeg_params=['-rc','vbr','-cq','19','-b:v','0','-maxrate','0','-pix_fmt','yuv420p','-movflags','+faststart']
        )
        
        video.close()
        reframed.close()
        
        return output_path
    
    def _detect_focus_points(self, video: VideoFileClip, fps: int, duration: float) -> List[float]:
        """
        Détecte les points d'intérêt (pose + visage) avec fallback intelligent
        """
        x_centers = []
        
        # Initialisation des détecteurs
        with self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose, self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        ) as face_detection:
            
            # Échantillonnage adaptatif (plus dense au début)
            sample_times = self._get_sample_times(duration, fps)
            
            for t in sample_times:
                try:
                    frame = video.get_frame(t)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    x_center = self._detect_single_frame(image_rgb, pose, face_detection)
                    x_centers.append(x_center)
                    
                except Exception as e:
                    logger.warning(f"Erreur détection frame à {t:.2f}s: {e}")
                    x_centers.append(0.5)  # Fallback centre
        
        # Interpolation pour avoir tous les frames
        return self._interpolate_trajectory(x_centers, sample_times, duration, fps)
    
    def _detect_single_frame(self, image_rgb: np.ndarray, pose, face_detection) -> float:
        """Détection sur une frame unique avec fallback hiérarchique"""
        
        h, w = image_rgb.shape[:2]
        
        # 1. Tentative détection pose (priorité haute)
        pose_results = pose.process(image_rgb)
        if pose_results.pose_landmarks:
            # Moyenne des landmarks du torse/tête pour plus de stabilité
            landmarks = pose_results.pose_landmarks.landmark
            
            key_points = [
                landmarks[self.mp_pose.PoseLandmark.NOSE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            ]
            
            valid_points = [p.x for p in key_points if p.visibility > 0.5]
            if valid_points:
                return sum(valid_points) / len(valid_points)
        
        # 2. Fallback : détection visage
        face_results = face_detection.process(image_rgb)
        if face_results.detections:
            detection = face_results.detections[0]  # Plus grand visage
            bbox = detection.location_data.relative_bounding_box
            return bbox.xmin + bbox.width / 2
        
        # 3. Fallback : détection de mouvement (centres de masse)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calcul du centre de masse des contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 100]
            if moments:
                centroids_x = [m['m10']/m['m00'] for m in moments if m['m00'] > 0]
                if centroids_x:
                    return sum(centroids_x) / len(centroids_x) / w
        
        # 4. Dernier fallback : centre de l'image
        return 0.5
    
    def _get_sample_times(self, duration: float, fps: int) -> List[float]:
        """Génère des temps d'échantillonnage adaptatifs"""
        
        if duration <= 10:
            # Vidéo courte : échantillonnage dense
            return list(np.arange(0, duration, 1/fps))
        elif duration <= 30:
            # Vidéo moyenne : tous les 3 frames
            return list(np.arange(0, duration, 3/fps))
        else:
            # Vidéo longue : échantillonnage plus espacé
            return list(np.arange(0, duration, 5/fps))
    
    def _smooth_trajectory(self, x_centers: List[float], window_size: int = 15) -> List[float]:
        """Lissage avancé avec filtre de Savitzky-Golay"""
        
        if len(x_centers) < window_size:
            # Fallback : moyenne mobile simple
            kernel = np.ones(min(5, len(x_centers))) / min(5, len(x_centers))
            return np.convolve(x_centers, kernel, mode='same').tolist()
        
        try:
            from scipy.signal import savgol_filter
            # Filtre Savitzky-Golay pour un lissage plus naturel
            return savgol_filter(x_centers, window_size, 3).tolist()
        except ImportError:
            # Fallback : moyenne mobile
            kernel = np.ones(window_size) / window_size
            return np.convolve(x_centers, kernel, mode='same').tolist()
    
    def _interpolate_trajectory(self, x_centers: List[float], sample_times: List[float], 
                               duration: float, fps: int) -> List[float]:
        """Interpolation pour obtenir une trajectoire complète"""
        
        if not x_centers:
            return [0.5] * int(duration * fps)
        
        target_times = np.arange(0, duration, 1/fps)
        
        if len(x_centers) == 1:
            return [x_centers[0]] * len(target_times)
        
        try:
            interpolated = np.interp(target_times, sample_times, x_centers)
            return interpolated.tolist()
        except:
            # Fallback : répétition du dernier point connu
            return [x_centers[-1]] * len(target_times)
    
    def add_dynamic_subtitles(self, video_path: Path, subtitles: List[Dict]) -> Path:
        """
        Sous-titres stylisés TikTok avec IA contextuelle améliorée
        """
        logger.info("💬 Ajout de sous-titres TikTok stylisés avec IA")
        
        video = VideoFileClip(str(video_path))
        text_clips = []
        
        # Palettes de couleurs thématiques
        color_palettes = {
            'energy': ['#FF6B6B', '#4ECDC4', '#FFE66D'],
            'professional': ['#FFFFFF', '#F7DC6F', '#AED6F1'],
            'viral': ['#FF1744', '#00E676', '#FFD600', '#E91E63']
        }
        
        # Détection du thème général
        all_text = ' '.join([s['text'].lower() for s in subtitles])
        theme = self._detect_content_theme(all_text)
        colors = color_palettes.get(theme, color_palettes['viral'])
        
        for i, subtitle in enumerate(subtitles):
            text = subtitle["text"].strip()
            
            # IA contextuelle pour emojis
            enhanced_text = self._add_contextual_emojis(text)
            
            # Couleur rotative de la palette
            color = colors[i % len(colors)]
            
            # Style dynamique basé sur le contenu
            font_size, stroke_width = self._get_dynamic_style(text)
            
            # Création du clip texte
            txt_clip = TextClip(
                enhanced_text,
                fontsize=font_size,
                color=color,
                stroke_color='black',
                stroke_width=stroke_width,
                font=Config.SUBTITLE_FONT,
                method='caption',
                size=(video.w * 0.9, None)
            ).set_start(subtitle["start"]).set_end(subtitle["end"])
            
            # Position dynamique avec variation
            y_positions = [0.15, 0.75, 0.85]  # Haut, bas, très bas
            y_pos = y_positions[i % len(y_positions)]
            txt_clip = txt_clip.set_position(('center', video.h * y_pos))
            
            # Animations avancées
            txt_clip = self._add_text_animations(txt_clip, i)
            
            text_clips.append(txt_clip)
        
        # Composition finale
        final_video = CompositeVideoClip([video] + text_clips)
        
        output_path = Config.TEMP_FOLDER / f"subtitled_{video_path.name}"
        final_video.write_videofile(
            str(output_path),
            codec='h264_nvenc',
            audio_codec='aac',
            verbose=False,
            logger=None,
            preset=None,
            ffmpeg_params=['-rc','vbr','-cq','19','-b:v','0','-maxrate','0','-pix_fmt','yuv420p','-movflags','+faststart']
        )
        
        # Nettoyage
        video.close()
        for clip in text_clips:
            clip.close()
        final_video.close()
        
        return output_path
    
    def _detect_content_theme(self, text: str) -> str:
        """Détecte le thème du contenu pour adapter le style"""
        
        energy_words = ['wow', 'amazing', 'incredible', 'fire', 'crazy', 'insane']
        professional_words = ['business', 'money', 'success', 'tips', 'advice']
        
        if any(word in text for word in energy_words):
            return 'energy'
        elif any(word in text for word in professional_words):
            return 'professional'
        else:
            return 'viral'
    
    def _add_contextual_emojis(self, text: str) -> str:
        """IA contextuelle pour ajouter des emojis pertinents"""
        
        text_lower = text.lower()
        
        # Dictionnaire contextuel étendu
        emoji_triggers = {
            'money': ' 💰', 'cash': ' 💵', 'rich': ' 🤑', 'expensive': ' 💎',
            'fire': ' 🔥', 'hot': ' 🔥', 'amazing': ' 🤯', 'wow': '🤯 ',
            'love': ' ❤️', 'heart': ' 💖', 'beautiful': ' ✨',
            'food': ' 🍕', 'eat': ' 😋', 'delicious': ' 🤤',
            'fast': ' ⚡', 'quick': ' ⚡', 'speed': ' 🚀',
            'think': ' 🤔', 'question': ' ❓', 'why': ' 🤷‍♂️',
            'win': ' 🏆', 'winner': ' 🎉', 'success': ' ✅',
            'fail': ' ❌', 'wrong': ' ❌', 'mistake': ' 🤦‍♂️'
        }
        
        enhanced_text = text
        
        for trigger, emoji in emoji_triggers.items():
            if trigger in text_lower:
                if emoji.startswith(' '):
                    enhanced_text += emoji
                else:
                    enhanced_text = emoji + enhanced_text
                break  # Un seul emoji par phrase pour éviter la surcharge
        
        return enhanced_text
    
    def _get_dynamic_style(self, text: str) -> tuple:
        """Style dynamique basé sur le contenu"""
        
        if '!' in text or text.isupper():
            # Texte énergique
            return 75, 4
        elif '?' in text:
            # Question
            return 65, 3
        else:
            # Texte normal
            return Config.SUBTITLE_FONT_SIZE, Config.SUBTITLE_STROKE_WIDTH
    
    def _add_text_animations(self, txt_clip, index: int):
        """Ajoute des animations variées aux textes"""
        
        animations = [
            lambda clip: clip.fadein(0.2).fadeout(0.2),  # Fade simple
            lambda clip: clip.fadein(0.1).fadeout(0.1),  # Fade rapide
            lambda clip: clip.crossfadein(0.3),          # Cross fade
        ]
        
        animation = animations[index % len(animations)]
        return animation(txt_clip)
    
    # Méthodes existantes inchangées
    def process_all_clips(self, input_video_path: str):
        """Pipeline principal de traitement"""
        logger.info("🚀 Début du pipeline de traitement avec IA")
        
        # Étape 1: Découpage (votre IA existante)
        self.cut_viral_clips(input_video_path)
        
        # Étape 2: Traitement de chaque clip
        clip_files = list(Config.CLIPS_FOLDER.glob("*.mp4"))
        
        for i, clip_path in enumerate(clip_files):
            logger.info(f"🎬 Traitement du clip {i+1}/{len(clip_files)}: {clip_path.name}")
            try:
                self.process_single_clip(clip_path)
                logger.info(f"✅ Clip {clip_path.name} traité avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {clip_path.name}: {e}")
    
    def cut_viral_clips(self, input_video_path: str):
        """Interface pour votre IA de découpage existante"""
        logger.info("📼 Découpage des clips avec IA...")
        
        # Exemple basique - remplacez par votre IA
        video = VideoFileClip(input_video_path)
        duration = video.duration
        
        # Exemple: découper en segments de 30 secondes
        segment_duration = 30
        segments = int(duration // segment_duration)
        
        for i in range(min(segments, 5)):  # Max 5 clips pour test
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            clip = video.subclip(start_time, end_time)
            output_path = Config.CLIPS_FOLDER / f"clip_{i+1:02d}.mp4"
            clip.write_videofile(str(output_path), verbose=False, logger=None)
        
        video.close()
        logger.info(f"✅ {segments} clips générés")
    
    def process_single_clip(self, clip_path: Path):
        """Traite un clip individuel avec les nouvelles fonctionnalités IA"""
        
        # 1. Reframe et recadrage 9:16 avec IA
        reframed_path = self.reframe_to_vertical(clip_path)
        
        # 2. Génération des sous-titres avec timestamps
        subtitles = self.generate_subtitles_with_timing(reframed_path)
        
        # 3. Ajout des sous-titres stylisés TikTok (méthode compatible emoji)
        final_path = add_tiktok_subtitles(str(reframed_path), subtitles)
        
        # 4. Déplacement vers le dossier output  
        output_path = Config.OUTPUT_FOLDER / f"final_{clip_path.name}"
        Path(final_path).rename(output_path)
        
        return output_path
    
    def transcribe_audio(self, video_path: Path) -> str:
        """Transcription avec Whisper"""
        logger.info("📝 Transcription audio avec Whisper")
        result = self.whisper_model.transcribe(str(video_path))
        return result["text"]
    
    def generate_subtitles_with_timing(self, video_path: Path) -> List[Dict]:
        """Génère des sous-titres avec timestamps précis"""
        logger.info("⏱️ Génération des sous-titres avec timing")
        
        result = self.whisper_model.transcribe(str(video_path), word_timestamps=True)
        
        subtitles = []
        for segment in result["segments"]:
            subtitle = {
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            }
            subtitles.append(subtitle)
        
        return subtitles