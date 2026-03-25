import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import logging
import os
import json
from pathlib import Path
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

from player import HFPACPlayer
from hfpac_format import display_version

# Player version — first two numbers track the HFPAC format version
PLAYER_VERSION = "6.2.2.0"
# Versions of the HFPAC format this player can read
COMPATIBLE_VERSIONS = "v2, v3, v4, v4.5, v5, v5.1, v6, v6.1, v6.2"
COPYRIGHT = "© 2026 HFPAC Project"

# ---------------------------------------------------------------------------
# Debug logging — writes to hfpac_player.log next to gui.py
# ---------------------------------------------------------------------------
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hfpac_player.log")

logging.basicConfig(
    level    = logging.DEBUG,
    format   = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt  = "%H:%M:%S.%f"[:-3],   # HH:MM:SS.mmm
    handlers = [
        logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("hfpac.gui")
log.info("=" * 60)
log.info(f"HFPAC Player GUI  v{PLAYER_VERSION}  starting up")
log.info(f"Log file: {_LOG_PATH}")
log.info("=" * 60)


class GuiLogHandler(logging.Handler):
    def __init__(self, text_widget, root):
        super().__init__()
        self.text_widget = text_widget
        self.root = root
        self.log_queue = queue.Queue()
        self._poll_log_queue()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.config(state=tk.DISABLED)
                self.text_widget.yview(tk.END)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_log_queue)


class HFPACGUI:
    def __init__(self, root):
        log.info("HFPACGUI.__init__: building window")
        self.root = root
        self.root.title(f"HFPAC Player v{PLAYER_VERSION}")
        self.root.geometry("450x640")
        self.root.resizable(False, False)

        self.player = None
        self.player_thread = None
        self.msg_queue = queue.Queue()
        
        # Preferences
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        self.pref_autoplay = tk.BooleanVar(value=True)
        self.pref_show_metadata = tk.BooleanVar(value=False)
        self.pref_allow_v2 = tk.BooleanVar(value=False)
        self.pref_advanced_logging = tk.BooleanVar(value=False)
        
        self.eq_bands = [tk.DoubleVar(value=0.0) for _ in range(10)]
        self.eq_enabled = tk.BooleanVar(value=False)
        
        self.load_settings()

        self.pref_autoplay.trace_add("write", lambda *args: self.save_settings())
        self.pref_show_metadata.trace_add("write", lambda *args: self.save_settings())
        self.pref_allow_v2.trace_add("write", lambda *args: self.save_settings())
        self.pref_advanced_logging.trace_add("write", lambda *args: self._on_advanced_logging_changed())
        
        self.eq_enabled.trace_add("write", lambda *args: self._on_eq_changed())
        for var in self.eq_bands:
            var.trace_add("write", lambda *args: self._on_eq_changed())

        # Queue System State
        self.playlist = []
        self.current_track_idx = -1
        self.autoplay_next = False

        # Track last-seen player state for change-detecting logging in update_timer
        self._last_stopped  = None
        self._last_paused   = None
        self._last_underrun = False

        self._build_menubar()
        self.setup_ui()
        self.apply_log_level()
        self.update_timer()
        self._poll_progress_queue()

        # Keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_pause() if self.player else None)
        self.root.bind('<Left>', lambda e: self._seek_relative(-5))
        self.root.bind('<Right>', lambda e: self._seek_relative(5))
        self.root.bind('<Up>', lambda e: self._change_volume_relative(0.05))
        self.root.bind('<Down>', lambda e: self._change_volume_relative(-0.05))
        self.root.bind('<n>', lambda e: self._play_next_in_queue())
        self.root.bind('<p>', lambda e: self._play_prev_in_queue())
        self.root.bind('<Delete>', lambda e: self.remove_from_queue())
        # Media key bindings (OS level integration where supported natively by Tk/OS)
        try:
            self.root.bind('<MediaNextTrack>', lambda e: self._play_next_in_queue())
            self.root.bind('<MediaPrevTrack>', lambda e: self._play_prev_in_queue())
            self.root.bind('<MediaPlayPause>', lambda e: self.toggle_pause() if self.player else None)
            self.root.bind('<MediaStop>', lambda e: self.stop_audio())
        except tk.TclError:
            log.warning("Media key bindings are not supported natively on this Tkinter OS implementation. Skipping.")

        messagebox.showinfo("v2 Playback Information", "HFPAC files encoded in the v2 format version are not playable by default. To play v2-encoded files, go to 'Settings -> General -> Legacy' and enable 'Allow v2 files to be played'.")

        log.info("HFPACGUI.__init__: window ready")

    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open & Play…", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Add to Queue…", command=self.add_to_queue)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="General", command=self.open_general_settings)
        settings_menu.add_command(label="Advanced", command=self.open_advanced_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="Equalizer (10-Band)", command=self.open_eq_window)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About HFPAC Player…",
                              command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind_all("<Control-o>", lambda e: self.open_file())

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.pref_autoplay.set(settings.get("autoplay", True))
                    self.pref_show_metadata.set(settings.get("show_metadata", False))
                    self.pref_allow_v2.set(settings.get("allow_v2", False))
                    self.pref_advanced_logging.set(settings.get("advanced_logging", False))
                    
                    self.eq_enabled.set(settings.get("eq_enabled", False))
                    gains = settings.get("eq_bands", [0.0]*10)
                    for i, var in enumerate(self.eq_bands):
                        if i < len(gains):
                            var.set(gains[i])
            except Exception as e:
                log.error(f"Failed to load settings: {e}")

    def save_settings(self):
        settings = {
            "autoplay": self.pref_autoplay.get(),
            "show_metadata": self.pref_show_metadata.get(),
            "allow_v2": self.pref_allow_v2.get(),
            "advanced_logging": self.pref_advanced_logging.get(),
            "eq_enabled": self.eq_enabled.get(),
            "eq_bands": [var.get() for var in self.eq_bands]
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            log.error(f"Failed to save settings: {e}")

    def apply_log_level(self):
        level = logging.DEBUG if self.pref_advanced_logging.get() else logging.INFO
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)
        log.info(f"Log level set to {'DEBUG' if self.pref_advanced_logging.get() else 'INFO'}")

        if hasattr(self, 'log_frame'):
            if self.pref_advanced_logging.get():
                self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                # Expand window when logging is shown
                self.root.geometry("450x860")
            else:
                self.log_frame.pack_forget()
                # Restore base window size
                self.root.geometry("450x660")

    def _on_advanced_logging_changed(self):
        self.save_settings()
        self.apply_log_level()

    def open_advanced_settings(self):
        adv_win = tk.Toplevel(self.root)
        adv_win.title("Advanced Settings")
        adv_win.geometry("320x150")
        adv_win.resizable(False, False)
        adv_win.grab_set()

        adv_frame = tk.LabelFrame(adv_win, text="Advanced")
        adv_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Checkbutton(adv_frame, text="Show advanced logging", variable=self.pref_advanced_logging).pack(anchor=tk.W, padx=5, pady=2)
        
        tk.Button(adv_win, text="Close", command=adv_win.destroy, width=10).pack(pady=10)

    def open_general_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("General Settings")
        settings_win.geometry("320x280")
        settings_win.resizable(False, False)
        settings_win.grab_set()

        # Playback section
        play_frame = tk.LabelFrame(settings_win, text="Playback")
        play_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Checkbutton(play_frame, text="Enable autoplay", variable=self.pref_autoplay).pack(anchor=tk.W, padx=5, pady=2)
        tk.Checkbutton(play_frame, text="Always show metadata (even if blank)", variable=self.pref_show_metadata).pack(anchor=tk.W, padx=5, pady=2)

        # Legacy section
        leg_frame = tk.LabelFrame(settings_win, text="Legacy")
        leg_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Checkbutton(leg_frame, text="Allow v2 HFPAC files to be played", variable=self.pref_allow_v2).pack(anchor=tk.W, padx=5, pady=2)
        
        # Close button
        def on_close():
            self.save_settings()
            settings_win.destroy()
            
        tk.Button(settings_win, text="Close", command=on_close, width=10).pack(pady=10)
        settings_win.protocol("WM_DELETE_WINDOW", on_close)

    def _on_eq_changed(self):
        self.save_settings()
        if self.player:
            self.player.eq_enabled = self.eq_enabled.get()
            self.player.set_eq_gains([var.get() for var in self.eq_bands])

    def open_eq_window(self):
        # Prevent multiple EQ windows
        if hasattr(self, "eq_win") and self.eq_win.winfo_exists():
            self.eq_win.lift()
            return

        self.eq_win = tk.Toplevel(self.root)
        self.eq_win.title("10-Band Equalizer")
        self.eq_win.resizable(False, False)
        
        main_frame = tk.Frame(self.eq_win, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Toggle checkbox
        toggle_cb = tk.Checkbutton(main_frame, text="Enable Equalizer", variable=self.eq_enabled, font=("", 10, "bold"))
        toggle_cb.pack(anchor=tk.W, pady=(0, 10))
        
        # EQ sliders frame
        sliders_frame = tk.Frame(main_frame)
        sliders_frame.pack(fill=tk.BOTH, expand=True)

        freqs = ["31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
        
        for i, (var, freq) in enumerate(zip(self.eq_bands, freqs)):
            col = tk.Frame(sliders_frame)
            col.pack(side=tk.LEFT, padx=5)
            
            # Value label
            val_lbl = tk.Label(col, text=f"{var.get():+0.1f} dB", width=6, font=("", 8))
            val_lbl.pack(side=tk.TOP)
            
            # Update label when slider moves
            def make_updater(lbl):
                def update(*args):
                    lbl.config(text=f"{float(args[0]):+0.1f} dB")
                return update
            
            # Slider
            scale = ttk.Scale(
                col, from_=12.0, to=-12.0, value=var.get(), variable=var, 
                orient=tk.VERTICAL, length=150, command=make_updater(val_lbl)
            )
            scale.pack(side=tk.TOP, pady=5)
            
            # Center Zero Binding
            def make_z_bind(v):
                return lambda e: v.set(0.0)
            scale.bind("<Double-Button-1>", make_z_bind(var))
            
            # Freq label
            tk.Label(col, text=freq, font=("", 8)).pack(side=tk.BOTTOM)
            
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        def reset_eq():
            for var in self.eq_bands:
                var.set(0.0)
                
        tk.Button(btn_frame, text="Reset All to Zero", command=reset_eq).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Close", command=self.eq_win.destroy, width=10).pack(side=tk.RIGHT)

    def _show_about(self):
        """About dialog styled to match the main window."""
        dlg = tk.Toplevel(self.root)
        dlg.title("About HFPAC Player")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.focus_set()

        # Outer padding frame
        outer = tk.Frame(dlg, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # App name — slightly larger, bold
        tk.Label(outer, text="HFPAC Player",
                 font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        tk.Label(outer, text=f"Version {PLAYER_VERSION}").pack(anchor="w", pady=(2, 10))

        # Separator
        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=(0, 10))

        # Info rows
        def _row(label, value):
            row = tk.Frame(outer)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label,
                     font=("TkDefaultFont", 9, "bold"),
                     width=20, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=value,
                     anchor="w").pack(side=tk.LEFT)

        _row("Compatible formats:", COMPATIBLE_VERSIONS)

        # Separator
        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=(10, 8))

        # Copyright
        tk.Label(outer, text=COPYRIGHT,
                 fg="grey").pack(anchor="w")

        # OK button — right-aligned, same style as main window buttons
        btn_frame = tk.Frame(outer)
        btn_frame.pack(fill=tk.X, pady=(12, 0))
        ok_btn = tk.Button(btn_frame, text="OK", width=8,
                           command=dlg.destroy)
        ok_btn.pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda e: dlg.destroy())
        dlg.bind("<Escape>", lambda e: dlg.destroy())

        # Centre over parent
        dlg.update_idletasks()
        pw = self.root.winfo_x() + self.root.winfo_width()  // 2
        ph = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.geometry(f"+{pw - dlg.winfo_width() // 2}+{ph - dlg.winfo_height() // 2}")

    def _poll_progress_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                log.debug(f"_poll_progress_queue: msg_type={msg_type!r}  "
                          f"data={repr(data)[:80]}")
                if msg_type == "progress":
                    self.info_text.set(data)
                elif msg_type == "success":
                    self._on_file_loaded_success(data)
                elif msg_type == "error":
                    self._on_file_loaded_error(data)
                else:
                    log.warning(f"_poll_progress_queue: unknown msg_type={msg_type!r}")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_progress_queue)

    def setup_ui(self):
        # File Frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.file_label = tk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        btn_open = tk.Button(file_frame, text="Open...", command=self.open_file)
        btn_open.pack(side=tk.RIGHT)

        # Info Frame
        self.info_frame = tk.LabelFrame(self.root, text="Info")
        self.info_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Content frame inside Info Frame to place text and image side-by-side
        info_content_frame = tk.Frame(self.info_frame)
        info_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.info_text = tk.StringVar()
        self.info_text.set("Sample rate: --\nChannels: --\nBit depth: --\nDuration: --")
        self.info_label = tk.Label(info_content_frame, textvariable=self.info_text, justify=tk.LEFT)
        self.info_label.pack(side=tk.LEFT, anchor="nw")

        # Album Art Label
        self.art_label = tk.Label(info_content_frame)  # Will resize when image is loaded
        self.art_label.pack(side=tk.RIGHT, anchor="ne", padx=(10, 0))
        self.current_art_image = None # Hold reference to prevent garbage collection

        # Controls Frame
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)

        self.btn_prev = tk.Button(ctrl_frame, text="Prev", width=8, command=self._play_prev_in_queue, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(ctrl_frame, text="Play", width=8, command=self.play_audio, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_pause = tk.Button(ctrl_frame, text="Pause", width=8, command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(ctrl_frame, text="Stop", width=8, command=self.stop_audio, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = tk.Button(ctrl_frame, text="Next", width=8, command=self._play_next_in_queue, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Progress Frame
        prog_frame = tk.Frame(self.root)
        prog_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.time_label = tk.Label(prog_frame, text="0:00 / 0:00")
        self.time_label.pack(side=tk.TOP)
        
        self.progress = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X)

        # Make the progress bar seekable — click anywhere to jump to that position
        self.progress.bind("<Button-1>",        self._on_progress_click)
        self.progress.bind("<B1-Motion>",       self._on_progress_click)
        self.progress.config(cursor="hand2")

        # Volume Frame
        vol_frame = tk.Frame(self.root)
        vol_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(vol_frame, text="Volume:").pack(side=tk.LEFT)
        self.vol_scale = ttk.Scale(vol_frame, from_=0.0, to=1.0, value=1.0, command=self.change_volume)
        self.vol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Queue / Playlist Frame
        queue_frame = tk.LabelFrame(self.root, text="Queue")
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.playlist_listbox = tk.Listbox(queue_frame, selectmode=tk.SINGLE, height=6)
        scrollbar = ttk.Scrollbar(queue_frame, command=self.playlist_listbox.yview)
        self.playlist_listbox.config(yscrollcommand=scrollbar.set)
        
        self.playlist_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)
        
        self.playlist_listbox.bind("<Double-Button-1>", self._on_queue_double_click)

        # Drag and drop support
        if HAS_DND and hasattr(self.root, 'drop_target_register'):
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_file_drop)
            self.playlist_listbox.drop_target_register(DND_FILES)
            self.playlist_listbox.dnd_bind('<<Drop>>', self._on_file_drop)

        queue_btn_frame = tk.Frame(queue_frame)
        queue_btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Button(queue_btn_frame, text="Add...", width=10, command=self.add_to_queue).pack(pady=2)
        tk.Button(queue_btn_frame, text="Remove", width=10, command=self.remove_from_queue).pack(pady=2)
        tk.Button(queue_btn_frame, text="Clear", width=10, command=self.clear_queue).pack(pady=2)
        tk.Button(queue_btn_frame, text="Move Up", width=10, command=self.move_queue_up).pack(pady=2)
        tk.Button(queue_btn_frame, text="Move Down", width=10, command=self.move_queue_down).pack(pady=2)

        # Log Frame
        self.log_frame = tk.Frame(self.root)
        self.log_text = tk.Text(self.log_frame, height=8, state=tk.DISABLED, bg='black', fg='lightgrey', font=("Consolas", 9))
        self.log_scroll = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.gui_handler = GuiLogHandler(self.log_text, self.root)
        self.gui_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(self.gui_handler)
        self.gui_handler._poll_log_queue()

    def add_to_queue(self):
        paths = filedialog.askopenfilenames(filetypes=[("HFPAC Files", "*.hfpac")])
        if paths:
            for path in paths:
                self.playlist.append(path)
                self.playlist_listbox.insert(tk.END, Path(path).name)

    def _on_file_drop(self, event):
        # tkinterdnd2 handles multiple dropped files with curly braces if there are spaces.
        # splitlist handles this nicely
        files = self.root.tk.splitlist(event.data)
        for f in files:
            path = str(Path(f).resolve())
            if path.lower().endswith(".hfpac"):
                self.playlist.append(path)
                self.playlist_listbox.insert(tk.END, Path(path).name)

    def move_queue_up(self):
        selected = self.playlist_listbox.curselection()
        if not selected:
            return
            
        idx = selected[0]
        if idx == 0:
            return # Already at top
            
        # Swap in internal list
        self.playlist[idx], self.playlist[idx-1] = self.playlist[idx-1], self.playlist[idx]
        
        # Swap in GUI
        text = self.playlist_listbox.get(idx)
        self.playlist_listbox.delete(idx)
        self.playlist_listbox.insert(idx - 1, text)
        
        # Adjust current playing index tracker
        if self.current_track_idx == idx:
            self.current_track_idx -= 1
        elif self.current_track_idx == idx - 1:
            self.current_track_idx += 1
            
        # Re-select the moved item and update colours
        self.playlist_listbox.selection_set(idx - 1)
        self._update_queue_selection()

    def move_queue_down(self):
        selected = self.playlist_listbox.curselection()
        if not selected:
            return
            
        idx = selected[0]
        if idx == len(self.playlist) - 1:
            return # Already at bottom
            
        # Swap in internal list
        self.playlist[idx], self.playlist[idx+1] = self.playlist[idx+1], self.playlist[idx]
        
        # Swap in GUI
        text = self.playlist_listbox.get(idx)
        self.playlist_listbox.delete(idx)
        self.playlist_listbox.insert(idx + 1, text)
        
        # Adjust current playing index tracker
        if self.current_track_idx == idx:
            self.current_track_idx += 1
        elif self.current_track_idx == idx + 1:
            self.current_track_idx -= 1
            
        # Re-select the moved item and update colours
        self.playlist_listbox.selection_set(idx + 1)
        self._update_queue_selection()

    def remove_from_queue(self):
        selected = self.playlist_listbox.curselection()
        if selected:
            idx = selected[0]
            del self.playlist[idx]
            self.playlist_listbox.delete(idx)
            # Adjust current track index if necessary
            if self.current_track_idx == idx:
                self.current_track_idx = -1
                self.stop_audio()
                self.info_text.set("Track removed from queue.")
                self.art_label.config(image='')
                self.current_art_image = None
            elif self.current_track_idx > idx:
                self.current_track_idx -= 1
            self._update_queue_selection()

    def clear_queue(self):
        self.playlist.clear()
        self.playlist_listbox.delete(0, tk.END)
        self.current_track_idx = -1
        self.stop_audio()
        self.file_label.config(text="No file selected")
        self.info_text.set("Sample rate: --\nChannels: --\nBit depth: --\nDuration: --")
        self.art_label.config(image='')
        self.current_art_image = None
        
    def _on_queue_double_click(self, event):
        selected = self.playlist_listbox.curselection()
        if selected:
            self._play_queue_index(selected[0])

    def _play_queue_index(self, index):
        if 0 <= index < len(self.playlist):
            self.current_track_idx = index
            path = self.playlist[index]
            self.stop_audio()
            self.file_label.config(text=Path(path).name)
            self.info_text.set(f"Loading {Path(path).name}, please wait...")
            self._update_queue_selection()
            self.root.update_idletasks()
            
            self.autoplay_next = True
            current_vol = self.vol_scale.get()
            threading.Thread(
                target=self._load_file_background,
                args=(path, current_vol),
                daemon=True,
            ).start()
            
    def _update_queue_selection(self):
        self.playlist_listbox.selection_clear(0, tk.END)
        for i in range(self.playlist_listbox.size()):
            self.playlist_listbox.itemconfig(i, {'bg': 'white', 'fg': 'black'})
            
        if len(self.playlist) > 0:
            self.btn_next.config(state=tk.NORMAL if self.current_track_idx + 1 < len(self.playlist) else tk.DISABLED)
            self.btn_prev.config(state=tk.NORMAL if self.current_track_idx > 0 or (self.player and self.player._elapsed() > 3.0) else tk.DISABLED)
        else:
            self.btn_next.config(state=tk.DISABLED)
            self.btn_prev.config(state=tk.DISABLED)

    def open_file(self):
        log.info("open_file: opening file dialog")
        path = filedialog.askopenfilename(filetypes=[("HFPAC Files", "*.hfpac")])
        if not path:
            log.info("open_file: dialog cancelled — no file selected")
            return

        log.info(f"open_file: selected path={path!r}")
        
        # Clear existing queue and add this file
        self.clear_queue()
        self.playlist.append(path)
        self.playlist_listbox.insert(tk.END, Path(path).name)
        
        self._play_queue_index(0)

    def _load_file_background(self, path, vol):
        log.info(f"_load_file_background: thread started  path={path!r}  vol={vol:.3f}")
        try:
            frames_seen = [0]

            def update_progress(current, total):
                frames_seen[0] = current
                if total > 0:
                    pct = int((current / total) * 100)
                    if current % max(1, total // 10) == 0 or current == total:
                        log.debug(f"_load_file_background: progress {current}/{total} ({pct}%)")
                    
                    if current % max(1, total // 100) == 0 or current == total:
                        msg = f"Loading file, please wait...\nRead {current}/{total} frames ({pct}%)"
                        self.msg_queue.put(("progress", msg))

            log.debug("_load_file_background: calling HFPACPlayer()")
            player = HFPACPlayer(
                path, gui_mode=True, volume=vol,
                progress_callback=update_progress,
            )
            h = player._header
            log.info(
                f"_load_file_background: load complete  "
                f"version={display_version(h.version)}  "
                f"sr={h.sample_rate}  ch={h.channels}  "
                f"bit_depth={h.bit_depth}  "
                f"frames={h.num_frames}  "
                f"duration={player._duration:.2f}s  "
                f"file_version_int={h.version}"
            )
            log.debug(
                f"_load_file_background: player internals  "
                f"entropy={'Rice' if player._entropy_mode == 1 else 'Huffman'}  "
                f"lpc={'Integer' if player._lpc_mode == 1 else 'Float32'}  "
                f"mid_side={player._mid_side}  "
                f"file_version={player._file_version}  "
                f"seek_table_entries={len(h.seek_table)}"
            )
            
            if h.version == 2 and not self.pref_allow_v2.get():
                raise ValueError("HFPAC files encoded in the v2 format version are not playable by default. To play v2-encoded files, go to 'Settings -> General -> Legacy' and enable 'Allow v2 files to be played'.")
                
            self.msg_queue.put(("success", player))
            log.debug("_load_file_background: 'success' sent to queue")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error(f"_load_file_background: EXCEPTION  {type(e).__name__}: {e}\n{tb}")
            self.msg_queue.put(("error", e))

    def _on_file_loaded_success(self, player):
        log.info("_on_file_loaded_success: updating UI with loaded player")
        self.player = player
        
        # Apply initial EQ settings
        self.player.eq_enabled = self.eq_enabled.get()
        self.player.set_eq_gains([var.get() for var in self.eq_bands])
        
        h = self.player._header
        version_map = {2: "v2", 3: "v3", 4: "v4", 5: "v4.5", 6: "v5", 7: "v5.1", 8: "v6"}
        version_str = version_map.get(
            getattr(h, 'version', 0),
            display_version(getattr(h, 'version', 0)),
        )

        from hfpac_format import Metadata
        meta = getattr(h, 'metadata', None) or Metadata()

        lines = [f"Format: HFPAC {version_str}"]
        if meta.title:        lines.append(f"Title: {meta.title}")
        if meta.artist:       lines.append(f"Artist: {meta.artist}")
        if meta.album:        lines.append(f"Album: {meta.album}")
        if meta.track_number:
            t = str(meta.track_number)
            if meta.year: t += f"  ({meta.year})"
            lines.append(f"Track: {t}")
        elif meta.year:       lines.append(f"Year: {meta.year}")
        if getattr(meta, 'pcm_md5', ""): lines.append(f"PCM MD5: {meta.pcm_md5[:8]}... (Valid)")

        # Load Album Art
        self.art_label.config(image='')
        self.current_art_image = None
        if hasattr(meta, 'cover_art') and meta.cover_art:
            try:
                import io
                from PIL import Image, ImageTk
                img_data = io.BytesIO(meta.cover_art)
                pil_img = Image.open(img_data)
                pil_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                self.current_art_image = ImageTk.PhotoImage(pil_img)
                self.art_label.config(image=self.current_art_image)
            except Exception as e:
                log.warning(f"Failed to load embedded cover art: {e}")
                
        lines += [
            f"Sample rate: {h.sample_rate} Hz",
            f"Channels: {h.channels}",
            f"Bit depth: {h.bit_depth}-bit",
            f"Frame size: {h.frame_size}",
            f"Duration: {self.player._fmt_time(self.player._duration)} "
            f"({self.player._duration:.1f}s)",
        ]
        self.info_text.set("\n".join(lines))
        self.progress['maximum'] = self.player._duration
        self.time_label.config(
            text=f"0:00 / {self.player._fmt_time(self.player._duration)}")
        self.btn_play.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.root.update_idletasks()
        self.btn_pause.config(text="Pause")

        # Reset change-detection state for the new player
        self._last_stopped  = False
        self._last_paused   = False
        self._last_underrun = False
        log.info(f"_on_file_loaded_success: UI ready  duration={self.player._duration:.2f}s  "
                 f"version_str={version_str!r}")
                 
        if self.autoplay_next:
            self.autoplay_next = False
            self.play_audio()

    def _on_file_loaded_error(self, e):
        log.error(f"_on_file_loaded_error: {type(e).__name__}: {e}")
        messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")
        self.player = None

    def play_audio(self):
        if not self.player:
            log.warning("play_audio: called with no player loaded — ignoring")
            return

        log.info(f"play_audio: called  _paused={self.player._paused}  "
                 f"_stopped={self.player._stopped}")

        # If paused, resume
        if self.player._paused:
            log.info("play_audio: resuming from pause")
            self.player.toggle_pause()
            self.btn_pause.config(text="Pause")
            return

        # If stopped or finished, recreate to start from beginning
        if self.player._stopped:
            log.info("play_audio: player was stopped — recreating from preloaded data")
            path     = self.player.path
            vol      = self.vol_scale.get()
            preloaded = (self.player._header, self.player._frames)
            self.player = HFPACPlayer(
                path, gui_mode=True, volume=vol, preloaded_data=preloaded)
            log.debug(f"play_audio: new player created  vol={vol:.3f}")

        self.btn_play.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_pause.config(text="Pause")

        log.info("play_audio: starting player thread")
        self.player_thread = threading.Thread(
            target=self.player.play, daemon=True)
        self.player_thread.start()
        log.debug(f"play_audio: player thread started  id={self.player_thread.ident}")

    def toggle_pause(self):
        if self.player and not self.player._stopped:
            prev = self.player._paused
            self.player.toggle_pause()
            new  = self.player._paused
            log.info(f"toggle_pause: paused {prev} → {new}")
            if new:
                self.btn_pause.config(text="Resume")
            else:
                self.btn_pause.config(text="Pause")
        else:
            log.debug("toggle_pause: no active player — ignored")

    def stop_audio(self):
        if self.player and not self.player._stopped:
            log.info("stop_audio: stopping player")
            self.player.stop()
            if self.player_thread:
                log.debug("stop_audio: joining player thread (timeout=1.0s)")
                self.player_thread.join(timeout=1.0)
                if self.player_thread.is_alive():
                    log.warning("stop_audio: player thread did not stop within 1.0s")
                else:
                    log.debug("stop_audio: player thread joined cleanly")
        else:
            log.debug("stop_audio: no active player — UI reset only")

        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_pause.config(text="Pause")
        self.progress['value'] = 0
        if self.player:
            self.time_label.config(
                text=f"0:00 / {self.player._fmt_time(self.player._duration)}")
        
    def _on_progress_click(self, event):
        """Seek to the position the user clicked on the progress bar."""
        if not self.player or self.player._stopped:
            log.debug("_on_progress_click: no active player — ignoring")
            return
        if not self.player._duration:
            log.debug("_on_progress_click: duration is 0 — ignoring")
            return

        bar_width = self.progress.winfo_width()
        if bar_width <= 0:
            log.warning(f"_on_progress_click: bar_width={bar_width} — ignoring")
            return

        fraction = max(0.0, min(1.0, event.x / bar_width))
        target_s = fraction * self.player._duration
        frame_before = self.player._current_frame

        log.info(f"_on_progress_click: click x={event.x}  bar_width={bar_width}  "
                 f"fraction={fraction:.4f}  target={target_s:.3f}s  "
                 f"frame_before={frame_before}")

        self.player.seek(target_s)

        frame_after  = self.player._current_frame
        landed_s     = frame_after * self.player._frame_size / self.player._sample_rate
        log.info(f"_on_progress_click: seek done  "
                 f"frame_after={frame_after}  landed={landed_s:.3f}s  "
                 f"delta={landed_s - target_s:+.3f}s")

    def change_volume(self, val):
        fval = float(val)
        if self.player:
            self.player.volume = fval
            log.debug(f"change_volume: volume → {fval:.3f}")

    def _change_volume_relative(self, delta):
        current = self.vol_scale.get()
        new_vol = max(0.0, min(1.0, current + delta))
        self.vol_scale.set(new_vol)
        self.change_volume(new_vol)

    def _seek_relative(self, delta):
        if not self.player or self.player._stopped:
            return
        current_time = self.player._elapsed()
        target_time = max(0.0, min(self.player._duration, current_time + delta))
        self.player.seek(target_time)

    def _play_next_in_queue(self):
        if self.current_track_idx + 1 < len(self.playlist):
            self._play_queue_index(self.current_track_idx + 1)

    def _play_prev_in_queue(self):
        if self.current_track_idx - 1 >= 0:
            self._play_queue_index(self.current_track_idx - 1)
        elif self.player:
            # If at the very first track, just restart it.
            self.player.seek(0)
            
    def update_timer(self):
        if self.player and not self.player._stopped:
            elapsed  = self.player._elapsed()
            duration = self.player._duration
            self.progress['value'] = min(elapsed, duration)
            self.time_label.config(
                text=f"{self.player._fmt_time(elapsed)} / "
                     f"{self.player._fmt_time(duration)}")
                     
            if self.current_track_idx == 0:
                if elapsed > 3.0 and self.btn_prev['state'] == tk.DISABLED:
                    self.btn_prev.config(state=tk.NORMAL)
                elif elapsed <= 3.0 and self.btn_prev['state'] == tk.NORMAL:
                    self.btn_prev.config(state=tk.DISABLED)

            # Log state changes only (not every tick)
            cur_paused   = self.player._paused
            cur_underrun = self.player._underrun
            if cur_paused != self._last_paused:
                log.info(f"update_timer: pause state changed → {cur_paused}  "
                         f"at {elapsed:.2f}s")
                self._last_paused = cur_paused
            if cur_underrun and not self._last_underrun:
                log.warning(f"update_timer: BUFFER UNDERRUN at {elapsed:.2f}s  "
                            f"queue_size={self.player._q.qsize()}")
            self._last_underrun = cur_underrun

        elif self.player and self.player._stopped:
            if self._last_stopped is False:
                elapsed = self.player._elapsed()
                log.info(f"update_timer: playback STOPPED  "
                         f"last_pos={elapsed:.2f}s  "
                         f"duration={self.player._duration:.2f}s")
                self._last_stopped = True

            if self.btn_play['state'] == tk.DISABLED:
                # Reached natural end of file
                log.info("update_timer: end-of-file detected")
                if self.current_track_idx + 1 < len(self.playlist):
                    # Play next track in queue
                    log.info("update_timer: moving to next track in queue")
                    self._play_queue_index(self.current_track_idx + 1)
                else:
                    log.info("update_timer: end of queue — re-enabling Play button")
                    self.btn_play.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)

        self.root.after(100, self.update_timer)


def main():
    import sys
    import traceback

    log.info("main(): entered")

    def global_exception_handler(exctype, value, tb):
        msg = "".join(traceback.format_exception(exctype, value, tb))
        log.critical(f"UNHANDLED EXCEPTION:\n{msg}")
        with open("crash.log", "w") as f:
            traceback.print_exception(exctype, value, tb, file=f)
        try:
            from tkinter import messagebox
            import tkinter as tk
            err_root = tk.Tk()
            err_root.withdraw()
            messagebox.showerror(
                "Fatal Error",
                f"A fatal error occurred:\n{value}\n"
                f"See hfpac_player.log and crash.log for details.",
            )
            err_root.destroy()
        except Exception:
            pass

    sys.excepthook = global_exception_handler
    log.info("main(): global exception hook installed")

    if HAS_DND:
        root = TkinterDnD.Tk()
        log.info("main(): TkinterDnD root created")
    else:
        import tkinter as tk
        root = tk.Tk()
        log.info("main(): Standard Tk root created (DND not available)")
    app = HFPACGUI(root)
    log.info("main(): entering mainloop")
    root.mainloop()
    log.info("main(): mainloop exited — shutting down")

if __name__ == "__main__":
    main()