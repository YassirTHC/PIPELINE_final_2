import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import logging
import time

# Import de votre pipeline principal
# from video_processor import VideoProcessor, Config
from config import AdvancedConfig



class ClipsPipelineGUI:
	"""Interface graphique pour le pipeline de clips viraux"""
	
	def __init__(self):
		self.root = tk.Tk()
		self.root.title("🎬 Pipeline Clips Viraux - TikTok/Instagram")
		self.root.geometry("900x700")
		
		# Queue pour les messages de log
		self.log_queue = queue.Queue()
		
		# Variables
		self.input_video_path = tk.StringVar()
		self.whisper_model = tk.StringVar(value="base")
		self.subtitle_style = tk.StringVar(value="classic")
		self.target_platform = tk.StringVar(value="tiktok")
		self.export_quality = tk.StringVar(value="medium_quality")
		
		# Processor instance
		self.processor = None
		self.is_processing = False
		
		self.create_widgets()
		self.setup_logging()
		
	def create_widgets(self):
		"""Crée l'interface utilisateur"""
		
		# Header
		header_frame = ttk.Frame(self.root)
		header_frame.pack(fill=tk.X, padx=10, pady=5)
		
		title = ttk.Label(header_frame, text="🎬 Pipeline Clips Viraux", 
					 font=("Arial", 16, "bold"))
		title.pack()
		
		subtitle = ttk.Label(header_frame, 
					   text="Automatisation complète : Découpage IA → Reframe 9:16 → Sous-titres")
		subtitle.pack()
		
		# Section 1: Sélection du fichier
		file_frame = ttk.LabelFrame(self.root, text="📁 Vidéo Source")
		file_frame.pack(fill=tk.X, padx=10, pady=5)
		
		file_select_frame = ttk.Frame(file_frame)
		file_select_frame.pack(fill=tk.X, padx=5, pady=5)
		
		ttk.Entry(file_select_frame, textvariable=self.input_video_path, width=60).pack(side=tk.LEFT, padx=5)
		ttk.Button(file_select_frame, text="Parcourir...", 
			  command=self.select_input_file).pack(side=tk.RIGHT)
		
		# Section 2: Configuration
		config_frame = ttk.LabelFrame(self.root, text="⚙️ Configuration")
		config_frame.pack(fill=tk.X, padx=10, pady=5)
		
		# Ligne 1: Modèle Whisper et Plateforme
		row1 = ttk.Frame(config_frame)
		row1.pack(fill=tk.X, padx=5, pady=2)
		
		ttk.Label(row1, text="Modèle Whisper:").pack(side=tk.LEFT)
		whisper_combo = ttk.Combobox(row1, textvariable=self.whisper_model, 
							   values=list(AdvancedConfig.WHISPER_MODELS.keys()),
							   state="readonly", width=15)
		whisper_combo.pack(side=tk.LEFT, padx=5)
		
		ttk.Label(row1, text="Plateforme:").pack(side=tk.LEFT, padx=(20,0))
		platform_combo = ttk.Combobox(row1, textvariable=self.target_platform,
								   values=list(AdvancedConfig.PLATFORMS.keys()),
								   state="readonly", width=15)
		platform_combo.pack(side=tk.LEFT, padx=5)
		
		# Ligne 2: Style sous-titres et Qualité
		row2 = ttk.Frame(config_frame)
		row2.pack(fill=tk.X, padx=5, pady=2)
		
		ttk.Label(row2, text="Style sous-titres:").pack(side=tk.LEFT)
		style_combo = ttk.Combobox(row2, textvariable=self.subtitle_style,
							  values=list(AdvancedConfig.SUBTITLE_STYLES.keys()),
							  state="readonly", width=15)
		style_combo.pack(side=tk.LEFT, padx=5)
		
		ttk.Label(row2, text="Qualité export:").pack(side=tk.LEFT, padx=(20,0))
		quality_combo = ttk.Combobox(row2, textvariable=self.export_quality,
								 values=list(AdvancedConfig.EXPORT_PRESETS.keys()),
								 state="readonly", width=15)
		quality_combo.pack(side=tk.LEFT, padx=5)
		
		# Section 3: Options avancées
		advanced_frame = ttk.LabelFrame(self.root, text="🔧 Options Avancées")
		advanced_frame.pack(fill=tk.X, padx=10, pady=5)
		
		options_frame = ttk.Frame(advanced_frame)
		options_frame.pack(fill=tk.X, padx=5, pady=5)
		
		self.use_premiere = tk.BooleanVar()
		self.send_webhooks = tk.BooleanVar()
		self.auto_open_output = tk.BooleanVar(value=True)
		
		ttk.Checkbutton(options_frame, text="Utiliser Premiere Pro (ExtendScript)", 
				   variable=self.use_premiere).pack(anchor=tk.W)
		ttk.Checkbutton(options_frame, text="Envoyer webhooks n8n", 
				   variable=self.send_webhooks).pack(anchor=tk.W)
		ttk.Checkbutton(options_frame, text="Ouvrir dossier de sortie automatiquement", 
				   variable=self.auto_open_output).pack(anchor=tk.W)
		
		# Section 4: Contrôles
		control_frame = ttk.Frame(self.root)
		control_frame.pack(fill=tk.X, padx=10, pady=10)
		
		# Boutons principaux
		button_frame = ttk.Frame(control_frame)
		button_frame.pack()
		
		self.start_button = ttk.Button(button_frame, text="🚀 Démarrer le Pipeline", 
								   command=self.start_processing, style="Accent.TButton")
		self.start_button.pack(side=tk.LEFT, padx=5)
		
		self.stop_button = ttk.Button(button_frame, text="⏹️ Arrêter", 
								  command=self.stop_processing, state=tk.DISABLED)
		self.stop_button.pack(side=tk.LEFT, padx=5)
		
		ttk.Button(button_frame, text="📁 Ouvrir Output", 
			  command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
		
		ttk.Button(button_frame, text="🗑️ Nettoyer Temp", 
			  command=self.clean_temp_files).pack(side=tk.LEFT, padx=5)
		
		# Barre de progression
		self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
		self.progress.pack(fill=tk.X, pady=5)
		
		# Section 5: Log et statut
		log_frame = ttk.LabelFrame(self.root, text="📋 Logs et Statut")
		log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		
		# Zone de texte pour les logs
		self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
		self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		
		# Statut en bas
		status_frame = ttk.Frame(self.root)
		status_frame.pack(fill=tk.X, padx=10, pady=5)
		
		self.status_label = ttk.Label(status_frame, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
		self.status_label.pack(fill=tk.X)
	
	def setup_logging(self):
		"""Configure le système de logging pour l'interface"""
		
		class GUILogHandler(logging.Handler):
			def __init__(self, log_queue):
				super().__init__()
				self.log_queue = log_queue
			
			def emit(self, record):
				self.log_queue.put(self.format(record))
		
		# Configuration du logger
		gui_handler = GUILogHandler(self.log_queue)
		gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
		
		logger = logging.getLogger()
		logger.addHandler(gui_handler)
		logger.setLevel(logging.INFO)
		
		# Démarrage du thread de mise à jour des logs
		self.root.after(100, self.update_logs)
	
	def update_logs(self):
		"""Met à jour l'affichage des logs"""
		try:
			while True:
				message = self.log_queue.get_nowait()
				self.log_text.config(state=tk.NORMAL)
				self.log_text.insert(tk.END, message + '\n')
				self.log_text.see(tk.END)
				self.log_text.config(state=tk.DISABLED)
		except queue.Empty:
			pass
		
		# Programmer la prochaine mise à jour
		self.root.after(100, self.update_logs)
	
	def select_input_file(self):
		"""Sélection du fichier vidéo d'entrée"""
		file_path = filedialog.askopenfilename(
			title="Sélectionner la vidéo source",
			filetypes=[
				("Vidéos", "*.mp4 *.avi *.mov *.mkv *.webm"),
				("MP4", "*.mp4"),
				("Tous les fichiers", "*.*")
			]
		)
		if file_path:
			self.input_video_path.set(file_path)
			self.update_status(f"Fichier sélectionné: {Path(file_path).name}")
	
	def start_processing(self):
		"""Démarre le traitement en arrière-plan"""
		
		# Validation
		if not self.input_video_path.get():
			messagebox.showerror("Erreur", "Veuillez sélectionner une vidéo source")
			return
		
		if not Path(self.input_video_path.get()).exists():
			messagebox.showerror("Erreur", "Le fichier sélectionné n'existe pas")
			return
		
		# Configuration de l'interface
		self.is_processing = True
		self.start_button.config(state=tk.DISABLED)
		self.stop_button.config(state=tk.NORMAL)
		self.progress.start()
		
		# Application de la configuration
		self.apply_configuration()
		
		# Démarrage du thread de traitement
		self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
		self.processing_thread.start()
		
		self.update_status("Traitement en cours...")
	
	def apply_configuration(self):
		"""Applique la configuration sélectionnée"""
		
		# Configuration du modèle Whisper
		Config.WHISPER_MODEL = self.whisper_model.get()
		
		# Configuration de la plateforme cible
		platform_config = AdvancedConfig.PLATFORMS[self.target_platform.get()]
		Config.TARGET_WIDTH = platform_config["width"]
		Config.TARGET_HEIGHT = platform_config["height"]
		
		# Configuration du style de sous-titres
		subtitle_config = AdvancedConfig.SUBTITLE_STYLES[self.subtitle_style.get()]
		Config.SUBTITLE_FONT_SIZE = subtitle_config["fontsize"]
		Config.SUBTITLE_COLOR = subtitle_config["color"]
		Config.SUBTITLE_STROKE_COLOR = subtitle_config["stroke_color"]
		Config.SUBTITLE_STROKE_WIDTH = subtitle_config["stroke_width"]
		
		logging.info(f"Configuration appliquée: {self.target_platform.get()}, {self.whisper_model.get()}, {self.subtitle_style.get()}")
	
	def processing_worker(self):
		"""Worker thread pour le traitement vidéo"""
		try:
			# Création du processor
			self.processor = VideoProcessor()
			
			# Traitement
			self.processor.process_all_clips(self.input_video_path.get())
			
			# Succès
			self.root.after(0, self.processing_complete, True)
			
		except Exception as e:
			logging.error(f"Erreur pendant le traitement: {e}")
			self.root.after(0, self.processing_complete, False, str(e))
	
	def processing_complete(self, success, error_msg=None):
		"""Callback appelé à la fin du traitement"""
		
		self.is_processing = False
		self.start_button.config(state=tk.NORMAL)
		self.stop_button.config(state=tk.DISABLED)
		self.progress.stop()
		
		if success:
			self.update_status("✅ Traitement terminé avec succès!")
			messagebox.showinfo("Succès", "Le pipeline s'est terminé avec succès!\nVérifiez le dossier output/")
			
			if self.auto_open_output.get():
				self.open_output_folder()
				
		else:
			self.update_status("❌ Erreur pendant le traitement")
			messagebox.showerror("Erreur", f"Erreur pendant le traitement:\n{error_msg}")
	
	def stop_processing(self):
		"""Arrête le traitement (si possible)"""
		if self.is_processing:
			# Note: Il est difficile d'arrêter proprement le traitement vidéo
			# Cette fonction pourrait être améliorée avec des signaux d'arrêt
			self.update_status("⏹️ Arrêt demandé...")
			messagebox.showinfo("Info", "L'arrêt sera effectif à la fin du clip en cours")
	
	def open_output_folder(self):
		"""Ouvre le dossier de sortie"""
		import os
		import subprocess
		import platform
		
		output_path = Config.OUTPUT_FOLDER
		
		try:
			if platform.system() == "Windows":
				os.startfile(output_path)
			elif platform.system() == "Darwin":  # macOS
				subprocess.run(["open", output_path])
			else:  # Linux
				subprocess.run(["xdg-open", output_path])
		except Exception as e:
			messagebox.showerror("Erreur", f"Impossible d'ouvrir le dossier: {e}")
	
	def clean_temp_files(self):
		"""Nettoie les fichiers temporaires"""
		try:
			import shutil
			temp_folder = Config.TEMP_FOLDER
			
			if temp_folder.exists():
				shutil.rmtree(temp_folder)
				temp_folder.mkdir()
				
			self.update_status("🗑️ Fichiers temporaires nettoyés")
			logging.info("Fichiers temporaires nettoyés")
			
		except Exception as e:
			messagebox.showerror("Erreur", f"Erreur lors du nettoyage: {e}")
	
	def update_status(self, message):
		"""Met à jour la barre de statut"""
		self.status_label.config(text=message)
	
	def run(self):
		"""Démarre l'interface graphique"""
		self.root.mainloop()

class BatchProcessorGUI:
	"""Interface pour le traitement par lots"""
	
	def __init__(self, parent):
		self.window = tk.Toplevel(parent)
		self.window.title("🔄 Traitement par Lots")
		self.window.geometry("600x400")
		
		self.video_files = []
		self.create_batch_widgets()
	
	def create_batch_widgets(self):
		"""Crée l'interface pour le traitement par lots"""
		
		# Header
		ttk.Label(self.window, text="Traitement par Lots", 
				 font=("Arial", 14, "bold")).pack(pady=10)
		
		# Liste des fichiers
		files_frame = ttk.LabelFrame(self.window, text="📁 Fichiers Vidéo")
		files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		
		# Listbox avec scrollbar
		list_frame = ttk.Frame(files_frame)
		list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		
		self.files_listbox = tk.Listbox(list_frame)
		scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
		self.files_listbox.config(yscrollcommand=scrollbar.set)
		
		self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		
		# Boutons de gestion des fichiers
		buttons_frame = ttk.Frame(files_frame)
		buttons_frame.pack(fill=tk.X, padx=5, pady=5)
		
		ttk.Button(buttons_frame, text="➕ Ajouter Fichiers", 
			  command=self.add_files).pack(side=tk.LEFT, padx=2)
		ttk.Button(buttons_frame, text="📁 Ajouter Dossier", 
			  command=self.add_folder).pack(side=tk.LEFT, padx=2)
		ttk.Button(buttons_frame, text="❌ Supprimer", 
			  command=self.remove_selected).pack(side=tk.LEFT, padx=2)
		ttk.Button(buttons_frame, text="🗑️ Tout Supprimer", 
			  command=self.clear_all).pack(side=tk.LEFT, padx=2)
		
		# Contrôles
		control_frame = ttk.Frame(self.window)
		control_frame.pack(fill=tk.X, padx=10, pady=10)
		
		ttk.Button(control_frame, text="🚀 Traiter Tous", 
			  command=self.process_all_batch).pack(side=tk.LEFT, padx=5)
		ttk.Button(control_frame, text="❌ Fermer", 
			  command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
	
	def add_files(self):
		"""Ajoute des fichiers à la liste"""
		files = filedialog.askopenfilenames(
			title="Sélectionner les vidéos",
			filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv *.webm")]
		)
		
		for file_path in files:
			if file_path not in self.video_files:
				self.video_files.append(file_path)
				self.files_listbox.insert(tk.END, Path(file_path).name)
	
	def add_folder(self):
		"""Ajoute tous les fichiers vidéo d'un dossier"""
		folder = filedialog.askdirectory(title="Sélectionner le dossier")
		
		if folder:
			extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
			folder_path = Path(folder)
			
			for ext in extensions:
				for file_path in folder_path.glob(f"*{ext}"):
					if str(file_path) not in self.video_files:
						self.video_files.append(str(file_path))
						self.files_listbox.insert(tk.END, file_path.name)
	
	def remove_selected(self):
		"""Supprime le fichier sélectionné"""
		selection = self.files_listbox.curselection()
		if selection:
			index = selection[0]
			self.files_listbox.delete(index)
			del self.video_files[index]
	
	def clear_all(self):
		"""Supprime tous les fichiers"""
		self.files_listbox.delete(0, tk.END)
		self.video_files.clear()
	
	def process_all_batch(self):
		"""Lance le traitement par lots"""
		if not self.video_files:
			messagebox.showwarning("Attention", "Aucun fichier sélectionné")
			return
		
		# Confirmation
		result = messagebox.askyesno(
			"Confirmation",
			f"Traiter {len(self.video_files)} fichiers?\nCela peut prendre beaucoup de temps."
		)
		
		if result:
			# Démarrer le traitement par lots en arrière-plan
			threading.Thread(target=self.batch_worker, daemon=True).start()
			messagebox.showinfo("Info", "Traitement par lots démarré.\nVérifiez les logs dans la fenêtre principale.")
	
	def batch_worker(self):
		"""Worker pour le traitement par lots"""
		processor = VideoProcessor()
		
		for i, video_file in enumerate(self.video_files, 1):
			try:
				logging.info(f"🔄 Traitement par lots: {i}/{len(self.video_files)} - {Path(video_file).name}")
				processor.process_all_clips(video_file)
				logging.info(f"✅ Terminé: {Path(video_file).name}")
				
			except Exception as e:
				logging.error(f"❌ Erreur sur {Path(video_file).name}: {e}")
		
		logging.info("🎉 Traitement par lots terminé!")


def main():
	"""Fonction principale pour lancer l'interface graphique ou le mode CLI"""
	
	import argparse
	
	# Parsing des arguments de ligne de commande
	parser = argparse.ArgumentParser(description="Pipeline Clips Viraux (GUI/CLI)")
	parser.add_argument("--cli", action="store_true", help="Mode ligne de commande")
	parser.add_argument("--video", type=str, help="Chemin de la vidéo source (mode CLI)")
	parser.add_argument("--json-report", type=str, help="Chemin du rapport JSON global (agrégé)")
	parser.add_argument("--output", type=str, help="Dossier de sortie (optionnel)")
	
	args = parser.parse_args()
	
	# Vérification des dépendances
	try:
		import whisper
		import moviepy
	except ImportError as e:
		if args.cli:
			print(f"❌ Dépendance manquante: {e}")
			print("Installez les dépendances avec: pip install -r requirements.txt")
		else:
			messagebox.showerror(
				"Dépendances manquantes",
				f"Dépendance manquante: {e}\n\nInstallez les dépendances avec:\npip install -r requirements.txt"
			)
		return
	
	# Import tardif du pipeline pour éviter l'échec avant le check de dépendances
	try:
		global VideoProcessor, Config
		from video_processor import VideoProcessor, Config
	except Exception as e:
		if args.cli:
			print(f"❌ Erreur d'import du pipeline: {e}")
		else:
			messagebox.showerror(
				"Erreur de chargement",
				f"Impossible de charger le pipeline (video_processor.py):\n{e}"
			)
		return
	
	# Création des dossiers nécessaires
	for folder in [Config.CLIPS_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER]:
		folder.mkdir(exist_ok=True)
	
	# Mode CLI
	if args.cli:
		if not args.video:
			print("❌ Erreur: --video est requis en mode CLI")
			print("Usage: python main.py --cli --video chemin/vers/video.mp4")
			return
		
		video_path = Path(args.video)
		if not video_path.exists():
			print(f"❌ Erreur: Fichier vidéo introuvable: {video_path}")
			return
		
		print(f"🎬 Démarrage du traitement CLI pour: {video_path.name}", flush=True)
		print("="*50, flush=True)
		
		try:
			processor = VideoProcessor()
			print(f"📐 Étape 1/4: Reframe dynamique IA...", flush=True)
			start_time = time.time()
			
			# Reframe
			reframed_path = processor.reframe_to_vertical(video_path)
			reframe_time = time.time() - start_time
			print(f"    ✅ Reframe terminé ({reframe_time:.1f}s)", flush=True)
			
			print(f"🗣️ Étape 2/4: Transcription Whisper (guide B-roll)...", flush=True)
			transcription_start = time.time()
			
			# Transcription
			subtitles = processor.transcribe_segments(reframed_path)
			transcription_time = time.time() - transcription_start
			print(f"    ✅ {len(subtitles)} segments de sous-titres générés ({transcription_time:.1f}s)", flush=True)
			
			print(f"🎞️ Étape 3/4: Insertion des B-rolls (activée)...", flush=True)
			broll_start = time.time()
			
			# B-rolls avec monitoring temps réel
			broll_path = processor.insert_brolls_if_enabled(reframed_path, subtitles, [])
			broll_time = time.time() - broll_start
			print(f"    ✅ B-roll insérés avec succès ({broll_time:.1f}s)", flush=True)
			
			print(f"✨ Étape 4/4: Ajout des sous-titres Hormozi 1...", flush=True)
			subtitles_start = time.time()
			
			# Sous-titres
			from hormozi_subtitles import add_hormozi_subtitles
			final_path = Path(f"output/final/final_{video_path.stem}.mp4")
			final_path.parent.mkdir(parents=True, exist_ok=True)
			add_hormozi_subtitles(str(broll_path), subtitles, str(final_path))
			subtitles_time = time.time() - subtitles_start
			print(f"    ✅ Sous-titres Hormozi ajoutés : {final_path} ({subtitles_time:.1f}s)", flush=True)
			
			total_time = time.time() - start_time
			print(f"  📤 Export terminé: final_{video_path.stem}.mp4", flush=True)
			print(f"✅ Clip {video_path.name} traité avec succès (TOTAL: {total_time:.1f}s)", flush=True)
			print(f"📊 Détail: Reframe {reframe_time:.1f}s | Transcription {transcription_time:.1f}s | B-roll {broll_time:.1f}s | Sous-titres {subtitles_time:.1f}s", flush=True)
		except Exception as e:
			print(f"❌ Erreur lors du traitement: {e}")
			return
	
	# Mode GUI (par défaut)
	else:
		# Lancement de l'interface
		app = ClipsPipelineGUI()
		
		# Menu pour le traitement par lots
		menubar = tk.Menu(app.root)
		app.root.config(menu=menubar)
		
		tools_menu = tk.Menu(menubar, tearoff=0)
		menubar.add_cascade(label="Outils", menu=tools_menu)
		tools_menu.add_command(label="Traitement par Lots", 
						  command=lambda: BatchProcessorGUI(app.root))
		tools_menu.add_separator()
		tools_menu.add_command(label="Ouvrir Dossier Clips", command=app.open_output_folder)
		tools_menu.add_command(label="Nettoyer Fichiers Temp", command=app.clean_temp_files)
		
		help_menu = tk.Menu(menubar, tearoff=0)
		menubar.add_cascade(label="Aide", menu=help_menu)
		help_menu.add_command(label="À propos", command=lambda: messagebox.showinfo(
			"À propos", 
			"🎬 Pipeline Clips Viraux v1.0\n\nAutomatisation complète pour créer des clips TikTok/Instagram\nà partir de vidéos longues.\n\nUtilise Whisper, MoviePy et optionnellement Premiere Pro."
		))
		
		app.run()

if __name__ == "__main__":
	main()