import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except Exception:
    DND_FILES = None
    TkinterDnD = None
    _HAS_DND = False
import threading
import os
import subprocess
import queue
import time
from pathlib import Path
import logging
import requests
from hormozi_subtitles import add_hormozi_subtitles

# Logging to file for diagnostics
_LOG_PATH = Path(__file__).with_name("gui_debug.log")
try:
    logging.basicConfig(
        level=logging.INFO,
        filename=str(_LOG_PATH),
        filemode='a',
        format='%(asctime)s %(levelname)s %(message)s',
        encoding='utf-8',
    )
except Exception:
    # Fallback if encoding not supported on older Python
    logging.basicConfig(
        level=logging.INFO,
        filename=str(_LOG_PATH),
        filemode='a',
        format='%(asctime)s %(levelname)s %(message)s',
    )

class VideoConverterGUI:
    def __init__(self):
        logging.info("GUI init start; DND available=%s", _HAS_DND)
        # Créer la fenêtre principale avec support du glisser-déposer si dispo
        # Utiliser Tk standard en fallback pour garantir l'ouverture de l'UI
        self.dnd_enabled = _HAS_DND
        if self.dnd_enabled:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title("🎬 Convertisseur Vidéo IA - Pipeline Automatique")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        # Forcer la fenêtre au démarrage
        try:
            self.root.state('normal')
            self.root.deiconify()
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.focus_force()
            self.root.update_idletasks()
            self.root.after(1500, lambda: (self.root.attributes('-topmost', False), self.root.focus_force()))
        except Exception as e:
            logging.warning("Unable to force window on top: %r", e)
        
        # Variables
        self.video_files = []
        self.is_processing = False
        self.progress_queue = queue.Queue()
        self.llm_status_var = tk.StringVar(value="LLM: détection…")
        self._llm_last_status = None
        self._llm_ready = False
        self._llm_probe_running = False
        
        # Créer les dossiers nécessaires
        self.create_directories()
        
        # Interface
        self.create_interface()
        
        # Configurer le glisser-déposer (si dispo)
        self.setup_drag_drop()
        
        # Démarrer le thread de vérification de la progression
        self.root.after(100, self.check_progress_queue)
        # Démarrer le sondage LLM (non bloquant)
        self.root.after(250, self.schedule_llm_probe)
        logging.info("GUI init done; entering mainloop when called")

    def create_directories(self):
        """Créer les dossiers nécessaires"""
        directories = ['clips', 'output', 'temp']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def create_interface(self):
        """Créer l'interface graphique"""
        
        # Titre principal
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="🎬 Convertisseur Vidéo IA",
            font=('Arial', 24, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Glissez-déposez vos vidéos ou utilisez le bouton ci-dessous",
            font=('Arial', 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack()

        # Indicateur LLM
        llm_status_lbl = tk.Label(
            title_frame,
            textvariable=self.llm_status_var,
            font=('Arial', 10, 'bold'),
            fg='#f1c40f',
            bg='#2c3e50'
        )
        llm_status_lbl.pack(pady=(6, 0))
        
        # Zone de glisser-déposer
        self.drop_frame = tk.Frame(self.root, bg='#34495e', relief='solid', bd=2)
        self.drop_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        self.drop_label = tk.Label(
            self.drop_frame,
            text="📁 Glissez-déposez vos fichiers vidéo ici\n(.mp4, .avi, .mov, .mkv)",
            font=('Arial', 14),
            fg='#ecf0f1',
            bg='#34495e',
            justify='center'
        )
        self.drop_label.pack(expand=True)
        
        # Liste des fichiers sélectionnés
        files_frame = tk.Frame(self.root, bg='#2c3e50')
        files_frame.pack(pady=10, padx=40, fill='both')
        
        tk.Label(
            files_frame,
            text="📋 Fichiers sélectionnés :",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        ).pack(anchor='w')
        
        # Listbox avec scrollbar
        list_frame = tk.Frame(files_frame, bg='#2c3e50')
        list_frame.pack(fill='both', expand=True)
        
        self.files_listbox = tk.Listbox(
            list_frame,
            font=('Arial', 10),
            bg='#ecf0f1',
            selectmode='multiple',
            height=6
        )
        self.files_listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)
        
        # Boutons d'action
        buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        buttons_frame.pack(pady=20)

        # Toggle fetchers
        self.fetcher_var = tk.BooleanVar(value=False)
        fetcher_check = tk.Checkbutton(
            buttons_frame,
            text="Activer fetchers B‑roll (Pexels/Pixabay)",
            variable=self.fetcher_var,
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50',
            activebackground='#2c3e50',
            activeforeground='#ecf0f1',
            selectcolor='#2c3e50'
        )
        fetcher_check.grid(row=0, column=0, padx=10, pady=5, sticky='w')

        self.select_btn = tk.Button(
            buttons_frame,
            text="📂 Sélectionner des fichiers",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.select_files
        )
        self.select_btn.grid(row=0, column=1, padx=10)

        self.process_btn = tk.Button(
            buttons_frame,
            text="🚀 Lancer le traitement IA",
            font=('Arial', 12, 'bold'),
            bg='#2ecc71',
            fg='white',
            command=self.start_processing
        )
        self.process_btn.grid(row=0, column=2, padx=10)
        
        self.clean_btn = tk.Button(
            buttons_frame,
            text="🧹 Nettoyer caches",
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white',
            command=self.clean_caches
        )
        self.clean_btn.grid(row=0, column=3, padx=10)
        
        # Barre de progression
        progress_frame = tk.Frame(self.root, bg='#2c3e50')
        progress_frame.pack(pady=10, padx=40, fill='x')
        
        self.progress_label = tk.Label(
            progress_frame,
            text="📊 Prêt à traiter vos vidéos",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=700
        )
        self.progress_bar.pack(pady=5)
        
        # Zone de logs
        logs_frame = tk.Frame(self.root, bg='#2c3e50')
        logs_frame.pack(pady=10, padx=40, fill='both', expand=True)
        
        tk.Label(
            logs_frame,
            text="📝 Logs de traitement :",
            font=('Arial', 10, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        ).pack(anchor='w')
        
        self.logs_text = tk.Text(
            logs_frame,
            height=8,
            font=('Courier', 9),
            bg='#1a1a1a',
            fg='#00ff00',
            insertbackground='white'
        )
        self.logs_text.pack(fill='both', expand=True)
        
        logs_scrollbar = ttk.Scrollbar(logs_frame, orient='vertical')
        logs_scrollbar.pack(side='right', fill='y')
        self.logs_text.config(yscrollcommand=logs_scrollbar.set)

    def setup_drag_drop(self):
        """Configurer le glisser-déposer"""
        if not getattr(self, 'dnd_enabled', False):
            # Mettre à jour le label pour indiquer l'absence de DnD
            try:
                self.drop_label.config(text="📁 Sélectionnez vos fichiers avec le bouton ci‑dessous\n(Drag & Drop indisponible)")
            except Exception:
                pass
            logging.info("Drag&Drop disabled; running without DnD")
            return
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        """Gérer le glisser-déposer de fichiers"""
        files = self.root.tk.splitlist(event.data)
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        for file_path in files:
            if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                if file_path not in self.video_files:
                    self.video_files.append(file_path)
                    self.files_listbox.insert(tk.END, os.path.basename(file_path))
                    self.log_message(f"✅ Ajouté: {os.path.basename(file_path)}")
            else:
                self.log_message(f"❌ Format non supporté: {os.path.basename(file_path)}")

    def select_files(self):
        """Sélectionner des fichiers via dialogue"""
        files = filedialog.askopenfilenames(
            title="Sélectionnez vos vidéos",
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        for file_path in files:
            if file_path not in self.video_files:
                self.video_files.append(file_path)
                self.files_listbox.insert(tk.END, os.path.basename(file_path))
                self.log_message(f"✅ Ajouté: {os.path.basename(file_path)}")

    def clear_files(self):
        """Vider la liste des fichiers"""
        self.video_files.clear()
        self.files_listbox.delete(0, tk.END)
        self.log_message("🗑️ Liste vidée")

    def log_message(self, message):
        """Ajouter un message aux logs"""
        timestamp = time.strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        """Démarrer le traitement en arrière-plan"""
        if not self.video_files:
            messagebox.showwarning("Attention", "Aucune vidéo sélectionnée !")
            return
            
        if self.is_processing:
            messagebox.showinfo("Info", "Un traitement est déjà en cours...")
            return
        
        self.is_processing = True
        self.process_btn.config(state='disabled', text="🔄 Traitement en cours...")
        self.progress_bar['value'] = 0
        
        # Lancer le traitement dans un thread séparé
        thread = threading.Thread(target=self.process_videos)
        thread.daemon = True
        thread.start()

    def process_videos(self):
        """Traiter les vidéos (fonction principale)"""
        total_videos = len(self.video_files)
        
        try:
            for i, video_path in enumerate(self.video_files):
                self.progress_queue.put(('progress', f"📹 Traitement de {os.path.basename(video_path)}..."))
                self.progress_queue.put(('progress_bar', (i / total_videos) * 100))
                
                # Copier la vidéo dans le dossier clips
                video_name = os.path.basename(video_path)
                dest_path = os.path.join('clips', video_name)
                
                self.progress_queue.put(('log', f"📁 Copie vers clips/{video_name}"))
                
                # NOUVEAU: copie réelle du fichier dans clips/
                try:
                    from pathlib import Path
                    import shutil
                    Path('clips').mkdir(exist_ok=True)
                    shutil.copy2(video_path, dest_path)
                    self.progress_queue.put(('log', f"✅ Copié: {video_name}"))
                except Exception as e:
                    self.progress_queue.put(('log', f"❌ Erreur de copie: {e}"))
                    continue
                
                # Traitement avec gestion d'erreurs améliorée
                try:
                    # Utiliser le Python de l'environnement virtuel
                    python_path = os.path.join(os.getcwd(), 'venv311', 'Scripts', 'python.exe')
                    env = os.environ.copy()
                    env.setdefault('PYTHONIOENCODING', 'utf-8')
                    env['ENABLE_PIPELINE_CORE_FETCHER'] = 'true'
                    env['BROLL_FETCH_ALLOW_IMAGES'] = 'false'
                    env['CONTEXTUAL_BROLL_YML'] = 'config\\contextual_broll.yml'
                    if self.fetcher_var.get():
                        env['AI_BROLL_ENABLE_FETCHER'] = '1'
                        # Provider par défaut: pexels,pixabay (modifiable via .env)
                        env.setdefault('AI_BROLL_FETCH_PROVIDER', 'pexels,pixabay')
                    result = subprocess.run([
                        python_path, 'main.py',
                        '--cli',
                        '--video', dest_path
                    ], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True, env=env)

                    self.progress_queue.put(('log', f"✅ Sortie: {result.stdout}"))
                    self.progress_queue.put(('log', f"✅ {video_name} traité avec succès"))

                    # Enchaîner AI-B-roll si demandé (désactivé: pipeline principal gère déjà l'insertion)
                    if False and self.fetcher_var.get():
                        try:
                            self.progress_queue.put(('log', f"🎬 Insertion B‑roll pour {video_name}..."))
                            # Déterminer le meilleur input: chercher un mp4 récent dans output/ correspondant au nom
                            stem = os.path.splitext(video_name)[0]
                            output_dir = Path('output').resolve()
                            temp_dir = Path('temp').resolve()
                            candidate = None
                            
                            # Préférer la vidéo reframée sans sous-titres
                            temp_candidate = temp_dir / f"reframed_{stem}.mp4"
                            if temp_candidate.exists():
                                candidate = temp_candidate
                            elif output_dir.exists():
                                matches = sorted(output_dir.glob(f"*{stem}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                                if matches:
                                    candidate = matches[0]
                            
                                                        # PRIORITÉ ABSOLUE: Vidéo avec sous-titres TikTok v2
                            subtitled_candidates = [
                                output_dir / "subtitled" / f"{stem}_hormozi_perfect.mp4",  # Priorité: Hormozi 1
                                output_dir / "subtitled" / f"reframed_{stem}_submagic.mp4",  # Submagic
                                output_dir / "subtitled" / f"reframed_{stem}_tiktok_subs_v2.mp4",  # Ancien TikTok v2
                                output_dir / "subtitled" / f"reframed_{stem}_tiktok_subs.mp4",  # Ancien TikTok v1
                                output_dir / f"final_{stem}_submagic.mp4",  # Autre localisation Submagic
                            ]
                            
                            # Pour éviter d'écraser les sous-titres, on conserve la vidéo reframée/non-sous-titrée comme base.
                            # Aucun remplacement par la version sous-titrée ici.
                            self.progress_queue.put(('log', f"✅ Base pour AI B‑roll: {Path(candidate if candidate else dest_path).name}"))
                            
                            # FALLBACK seulement si aucune vidéo avec sous-titres TikTok trouvée
                            if candidate is None:
                                self.progress_queue.put(('log', f"⚠️ Aucune vidéo avec sous-titres TikTok trouvée, utilisation fallback"))
                                # Préférer la vidéo reframée sans sous-titres
                                temp_candidate = temp_dir / f"reframed_{stem}.mp4"
                                if temp_candidate.exists():
                                    candidate = temp_candidate
                                elif output_dir.exists():
                                    matches = sorted(output_dir.glob(f"*{stem}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                                    if matches:
                                        candidate = matches[0]
                            ai_input = str(Path(candidate if candidate else dest_path).resolve())

                            # Préparer chemins AI-B-roll
                            ai_repo = Path('AI-B-roll').resolve()
                            broll_lib = (ai_repo / 'broll_library').resolve()
                            broll_lib.mkdir(parents=True, exist_ok=True)
                            out_path = (output_dir / f"final_{stem}_with_broll.mp4").resolve()

                            # SRT: privilégier celui généré dans output/
                            srt_candidates = [
                                (output_dir / f"final_{stem}.srt").resolve(),
                                Path(ai_input).with_suffix('.srt').resolve(),
                            ]
                            srt_path = next((p for p in srt_candidates if p.exists()), None)

                            args = [
                                python_path, '-m', 'src.pipeline.cli',
                                ai_input,
                                str(broll_lib),
                                str(out_path),
                            ]
                            if srt_path and srt_path.exists():
                                args += ['--srt', str(srt_path)]
                                # Nouveau pipeline contextuel + metadata
                                args += [
                                    '--contextual-fetcher',
                                    '--contextual-config', str((ai_repo / 'contextual_broll.yml').resolve()),
                                    '--use-visual-scoring', 'true',
                                    '--min-final-score', '0.55',
                                    '--max-broll-per-minute', '25',
                                    '--min-broll', '3.5',
                                    '--max-broll', '5.0',
                                    '--generate-metadata',
                                    '--fetch-max', '8',
                                    '--no-fetch-images',
                                ]
                                # Providers - n'utiliser que les sources vidéo
                                fetch_list = []
                                if os.environ.get('PEXELS_API_KEY'):
                                    fetch_list.append('pexels')
                                if os.environ.get('PIXABAY_API_KEY'):
                                    fetch_list.append('pixabay')
                                fetch_list.append('local')
                                args += ['--fetch-providers', ','.join(fetch_list)]
                                if pex:
                                    args += ['--pexels-key', pex]
                                if pxb:
                                    args += ['--pixabay-key', pxb]
                                # Tune contextual thresholds: keep defaults for count
                                try:
                                    idx = args.index('--min-final-score')
                                    args[idx+1] = '0.60'
                                except Exception:
                                    pass
                                try:
                                    idx = args.index('--max-broll-per-minute')
                                    args[idx+1] = '10'
                                except Exception:
                                    pass
                                # Encourage LLM assistance in selector
                                env['AI_BROLL_LLM_ASSIST'] = '1'
                                # Activer les emojis dans les sous-titres B-roll
                                env['AI_BROLL_ENABLE_EMOJI_SUBS'] = '1'
                                # LLM auto-détection
                                llm_provider = None
                                llm_base = None
                                try:
                                    r = requests.get('http://localhost:11434/api/tags', timeout=1)
                                    if r.status_code == 200:
                                        llm_provider = 'ollama'
                                        llm_base = 'http://localhost:11434'
                                except Exception:
                                    pass
                                if not llm_provider:
                                    try:
                                        r = requests.get('http://localhost:1234/v1/models', timeout=1)
                                        if r.status_code == 200:
                                            llm_provider = 'lmstudio'
                                            llm_base = 'http://localhost:1234/v1'
                                    except Exception:
                                        pass
                                if llm_provider and llm_base:
                                    # Charger la configuration LLM depuis le fichier config
                                    try:
                                        import yaml
                                        config_path = Path('config/llm_config.yaml')
                                        if config_path.exists():
                                            with open(config_path, 'r', encoding='utf-8') as f:
                                                config = yaml.safe_load(f)
                                                llm_model = config.get('llm', {}).get('model', 'qwen3:8b')
                                        else:
                                            llm_model = 'qwen3:8b'  # Modèle par défaut
                                    except Exception:
                                        llm_model = 'qwen3:8b'  # Fallback en cas d'erreur
                                    
                                    args += ['--llm-provider', llm_provider, '--llm-base-url', llm_base, '--llm-model', llm_model]
                            else:
                                # Pas de SRT: ignorer l'étape B‑roll contextuel pour cette vidéo
                                self.progress_queue.put(('log', f"⚠️ SRT non trouvé pour {os.path.basename(ai_input)}. B‑roll contextuel ignoré."))
                                continue
                            # Emojis dans les sous-titres activés par défaut
                            env['AI_BROLL_ENABLE_EMOJI_SUBS'] = '1'

                            # Exécuter depuis le dossier AI-B-roll pour résoudre le package src/
                            proc = subprocess.Popen(
                                args,
                                cwd=str(ai_repo),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                encoding='utf-8',
                                errors='replace',
                                env=env,
                            )
                            assert proc.stdout is not None
                            for line in proc.stdout:
                                line = line.rstrip('\n')
                                if line:
                                    self.progress_queue.put(('log', line))
                            ret = proc.wait()
                            if ret != 0:
                                raise subprocess.CalledProcessError(ret, args)
                            self.progress_queue.put(('log', f"✅ B‑roll inséré: {out_path}"))

                            # Post-process: overlay Hormozi subtitles on B‑roll output
                            try:
                                # Reuse SRT already generated; if missing, skip overlay
                                if Path(out_path).exists():
                                    # Load SRT back to segments if needed — minimal parser
                                    def _parse_srt_minimal(srt_file: Path):
                                        segs = []
                                        try:
                                            text = Path(srt_file).read_text(encoding='utf-8', errors='ignore')
                                        except Exception:
                                            return segs
                                        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
                                        for b in blocks:
                                            lines = b.splitlines()
                                            if len(lines) < 2:
                                                continue
                                            # timestamp on line 2 typically
                                            ts_line = next((ln for ln in lines if '-->' in ln), None)
                                            if not ts_line:
                                                continue
                                            def _to_seconds(ts: str):
                                                ts = ts.replace(',', '.')
                                                hh, mm, rest = ts.split(':')
                                                ss = float(rest)
                                                return int(hh)*3600 + int(mm)*60 + ss
                                            try:
                                                l, r = [p.strip() for p in ts_line.split('-->')]
                                                start = _to_seconds(l)
                                                end = _to_seconds(r)
                                            except Exception:
                                                continue
                                            content_lines = [ln for ln in lines if '-->' not in ln and not ln.strip().isdigit()]
                                            content = ' '.join(content_lines).strip()
                                            if content:
                                                segs.append({'start': start, 'end': end, 'text': content})
                                        return segs
                                    # Préférer le JSON exact si présent
                                    seg_json = (output_dir / f"final_{stem}_segments.json").resolve()
                                    segs = []
                                    if seg_json.exists():
                                        try:
                                            import json as _json
                                            segs = _json.loads(seg_json.read_text(encoding='utf-8'))
                                        except Exception:
                                            segs = []
                                    if not segs and srt_path and srt_path.exists():
                                        segs = _parse_srt_minimal(srt_path)
                                    if segs:
                                        subtitled_out = (output_dir / 'subtitled' / f"{stem}_hormozi_perfect_broll.mp4").resolve()
                                        subtitled_out.parent.mkdir(parents=True, exist_ok=True)
                                        add_hormozi_subtitles(str(out_path), segs, str(subtitled_out))
                                        self.progress_queue.put(('log', f"✅ Sous‑titres Hormozi superposés sur B‑roll: {subtitled_out}"))
                                        # Export final unifié avec anti-collision
                                        try:
                                            final_target = (output_dir / f"final_{stem}.mp4").resolve()
                                            dst = final_target
                                            if dst.exists():
                                                import datetime
                                                ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                                                dst = (output_dir / f"final_{stem}_{ts}.mp4").resolve()
                                            import shutil
                                            shutil.copy2(str(subtitled_out), str(dst))
                                            self.progress_queue.put(('log', f"✅ Export final unifié: {dst.name}"))
                                            # Mettre à jour output/latest avec la version B‑roll
                                            try:
                                                latest_root = (output_dir / 'latest').resolve()
                                                latest_root.mkdir(parents=True, exist_ok=True)
                                                # Sous‑dossier par clip
                                                per_clip_dir = (latest_root / stem).resolve()
                                                per_clip_dir.mkdir(parents=True, exist_ok=True)
                                                # Nettoyer uniquement ce sous‑dossier
                                                for _p in per_clip_dir.glob('*'):
                                                    try:
                                                        _p.unlink()
                                                    except Exception:
                                                        pass
                                                # Copier artefacts dans le sous‑dossier
                                                import shutil
                                                shutil.copy2(str(dst), str(per_clip_dir / 'latest.mp4'))
                                                if srt_path and srt_path.exists():
                                                    shutil.copy2(str(srt_path), str(per_clip_dir / 'latest.srt'))
                                                meta_src = (output_dir / f"final_{stem}_meta.txt").resolve()
                                                if meta_src.exists():
                                                    shutil.copy2(str(meta_src), str(per_clip_dir / 'latest_meta.txt'))
                                                # Pointer pratique: écraser latest.mp4 top‑level
                                                shutil.copy2(str(dst), str(latest_root / 'latest.mp4'))
                                            except Exception as e_latest:
                                                self.progress_queue.put(('log', f"⚠️ Latest non mis à jour: {e_latest}"))
                                        except Exception as e_copy:
                                            self.progress_queue.put(('log', f"⚠️ Copie finale échouée: {e_copy}"))
                                    else:
                                        self.progress_queue.put(('log', f"⚠️ Overlay Hormozi post B‑roll ignoré (segments introuvables)"))
                            except Exception as e_overlay:
                                self.progress_queue.put(('log', f"⚠️ Échec overlay Hormozi post B‑roll: {e_overlay}"))
                        except subprocess.CalledProcessError as e2:
                            err = e2.stderr if e2.stderr else str(e2)
                            self.progress_queue.put(('log', f"❌ Échec B‑roll: {err}"))

                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr if e.stderr else str(e)
                    self.progress_queue.put(('log', f"❌ Erreur: {error_msg}"))
                    self.progress_queue.put(('log', f"❌ {video_name} échoué"))
                    # Continuer avec la vidéo suivante au lieu d'arrêter tout
            
            self.progress_queue.put(('progress', "🎉 Tous les traitements terminés !"))
            self.progress_queue.put(('progress_bar', 100))
            self.progress_queue.put(('complete', None))
            
        except Exception as e:
            self.progress_queue.put(('error', f"❌ Erreur: {str(e)}"))
            self.progress_queue.put(('complete', None))

    def schedule_llm_probe(self):
        """Planifie un sondage non bloquant de l'état du LLM."""
        if not self._llm_probe_running:
            t = threading.Thread(target=self.probe_llm_once, daemon=True)
            self._llm_probe_running = True
            t.start()
        # Re-planifier toutes les 3 secondes
        self.root.after(3000, self.schedule_llm_probe)

    def probe_llm_once(self):
        """Teste Ollama puis LM Studio; met à jour le statut via la queue UI."""
        status = "LLM: non détecté"
        ready = False
        try:
            # Essai Ollama
            r = requests.get('http://localhost:11434/api/tags', timeout=1.0)
            if r.ok:
                data = r.json()
                names = []
                if isinstance(data, dict) and 'models' in data:
                    # LM Studio style accidental? keep generic
                    names = [m.get('name','') for m in data.get('models', [])]
                elif isinstance(data, dict) and 'models' not in data:
                    # Ollama returns {"models":[{"name":...}]} in newer builds too; tolerate variations
                    names = [m.get('name','') for m in data.get('models', [])]
                else:
                    # Fallback try list
                    try:
                        names = [m.get('name','') for m in data]
                    except Exception:
                        names = []
                model_hint = ''
                if names:
                    # Charger la configuration LLM pour afficher le bon modèle
                    try:
                        import yaml
                        config_path = Path('config/llm_config.yaml')
                        if config_path.exists():
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = yaml.safe_load(f)
                                configured_model = config.get('llm', {}).get('model', 'qwen3:8b')
                                if any(configured_model in n for n in names):
                                    model_hint = f' ({configured_model})'
                        else:
                            # Fallback si pas de config
                            if any('qwen3:8b' in n for n in names):
                                model_hint = ' (qwen3:8b)'
                    except Exception:
                        # Fallback en cas d'erreur
                        if any('qwen3:8b' in n for n in names):
                            model_hint = ' (qwen3:8b)'
                
                status = f"LLM: Ollama PRÊT{model_hint}"
                ready = True
        except Exception:
            pass
        if not ready:
            try:
                r = requests.get('http://localhost:1234/v1/models', timeout=1.0)
                if r.ok:
                    data = r.json()
                    names = []
                    if isinstance(data, dict) and 'data' in data:
                        names = [m.get('id','') or m.get('name','') for m in data.get('data', [])]
                    status = f"LLM: LM Studio PRÊT ({len(names)} modèles)"
                    ready = True
            except Exception:
                pass
        # Poster le résultat au thread UI
        self.progress_queue.put(('llm_status', {'text': status, 'ready': ready}))
        self._llm_probe_running = False

    def check_progress_queue(self):
        """Vérifier la queue de progression"""
        try:
            while True:
                item_type, data = self.progress_queue.get_nowait()
                
                if item_type == 'log':
                    self.log_message(data)
                elif item_type == 'progress':
                    self.progress_label.config(text=data)
                elif item_type == 'progress_bar':
                    self.progress_bar['value'] = data
                elif item_type == 'complete':
                    self.is_processing = False
                    self.process_btn.config(state='normal', text="🚀 Lancer le traitement IA")
                elif item_type == 'error':
                    self.log_message(data)
                    messagebox.showerror("Erreur", data)
                elif item_type == 'llm_status':
                    # Mettre à jour l'indicateur et logger les transitions vers PRÊT
                    try:
                        text = data.get('text', 'LLM: statut inconnu')
                        ready = bool(data.get('ready', False))
                        self.llm_status_var.set(text)
                        if ready and not self._llm_ready:
                            self.log_message(f"✅ {text}")
                        self._llm_ready = ready
                    except Exception:
                        pass
         
        except queue.Empty:
            pass
        
        # Programmer la prochaine vérification
        self.root.after(100, self.check_progress_queue)

    def clean_caches(self):
        """Nettoie intelligemment les caches en préservant la diversité B-roll récente"""
        try:
            total_before = 0
            total_after = 0
            
            # Fonction pour calculer la taille d'un dossier
            def get_folder_size(path):
                if not path.exists():
                    return 0
                total = 0
                try:
                    for item in path.rglob('*'):
                        if item.is_file():
                            total += item.stat().st_size
                except:
                    pass
                return total
            
            # Nettoyage intelligent du B-roll: garder les 30 jours récents
            broll_fetched = Path('AI-B-roll/broll_library/fetched')
            if broll_fetched.exists():
                size_before = get_folder_size(broll_fetched)
                total_before += size_before
                
                import time
                cutoff_time = time.time() - (30 * 24 * 3600)  # 30 jours
                
                # Supprimer les anciens B-rolls seulement
                for provider_dir in broll_fetched.iterdir():
                    if provider_dir.is_dir():
                        for media_file in provider_dir.rglob('*'):
                            if media_file.is_file():
                                try:
                                    if media_file.stat().st_mtime < cutoff_time:
                                        media_file.unlink(missing_ok=True)
                                except:
                                    pass
                
                size_after = get_folder_size(broll_fetched)
                total_after += size_after
                self.progress_queue.put(('log', f"🧹 B-roll (gardé 30j récents): {size_before/1e9:.2f} GB -> {size_after/1e9:.2f} GB"))
            
            # Nettoyage standard pour les autres caches
            other_paths = [
                Path('temp'),
                Path('output/subtitled'),
                Path('.cache'),  # Cache Whisper/embeddings/modèles
            ]
            
            for path in other_paths:
                if path.exists():
                    size_before = get_folder_size(path)
                    total_before += size_before
                    
                    # Supprimer le contenu
                    import shutil
                    if path.name == '.cache':
                        # Pour .cache, vider le contenu mais garder le dossier
                        for item in path.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            else:
                                item.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(path, ignore_errors=True)
                        path.mkdir(parents=True, exist_ok=True)
                    
                    size_after = get_folder_size(path)
                    total_after += size_after
                    
                    self.progress_queue.put(('log', f"🧹 {path}: {size_before/1e9:.2f} GB -> {size_after/1e9:.2f} GB"))
            
            saved = (total_before - total_after) / 1e9
            self.progress_queue.put(('log', f"✅ Nettoyage intelligent terminé: {saved:.2f} GB libérés, diversité B-roll préservée"))
            
        except Exception as e:
            self.progress_queue.put(('log', f"❌ Erreur nettoyage: {str(e)}"))

def main():
    """Fonction principale"""
    try:
        logging.info("GUI main() start")
        app = VideoConverterGUI()
        logging.info("GUI before mainloop")
        app.root.mainloop()
        logging.info("GUI exited mainloop")
    except ImportError:
        logging.exception("tkinterdnd2 import error")
        print("❌ Erreur: tkinterdnd2 n'est pas installé")
        print("📦 Installez-le avec: pip install tkinterdnd2")
        input("Appuyez sur Entrée pour fermer...")
    except Exception as e:
        logging.exception("Unhandled exception in GUI")
        messagebox.showerror("Erreur", f"Erreur critique GUI: {e}")

if __name__ == "__main__":
    main()
