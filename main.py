import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
from pathlib import Path

from player import HFPACPlayer
from hfpac_format import display_version

# Player version — first two numbers track the HFPAC format version
PLAYER_VERSION = "6.0.0.0"
# Versions of the HFPAC format this player can read
COMPATIBLE_VERSIONS = "v2, v3, v4, v4.5, v5, v5.1, v6"
COPYRIGHT = "© 2026 HFPAC Project"


class HFPACGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HFPAC Player")
        self.root.geometry("450x420")
        self.root.resizable(False, False)

        self.player = None
        self.player_thread = None
        self.msg_queue = queue.Queue()

        self._build_menubar()
        self.setup_ui()
        self.update_timer()
        self._poll_progress_queue()

    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…",       command=self.open_file,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit",         command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About HFPAC Player…",
                              command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind_all("<Control-o>", lambda e: self.open_file())

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
            # Process all pending messages from background threads
            while True:
                msg_type, data = self.msg_queue.get_nowait()
                if msg_type == "progress":
                    self.info_text.set(data)
                elif msg_type == "success":
                    self._on_file_loaded_success(data)
                elif msg_type == "error":
                    self._on_file_loaded_error(data)
        except queue.Empty:
            pass
        # Check again every 50ms safely on the main thread
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
        
        self.info_text = tk.StringVar()
        self.info_text.set("Sample rate: --\nChannels: --\nBit depth: --\nDuration: --")
        tk.Label(self.info_frame, textvariable=self.info_text, justify=tk.LEFT).pack(anchor="w", padx=5, pady=5)

        # Controls Frame
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        
        self.btn_play = tk.Button(ctrl_frame, text="Play", width=8, command=self.play_audio, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_pause = tk.Button(ctrl_frame, text="Pause", width=8, command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(ctrl_frame, text="Stop", width=8, command=self.stop_audio, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

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

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("HFPAC Files", "*.hfpac")])
        if not path:
            return
        
        self.stop_audio()
        
        self.file_label.config(text=Path(path).name)
        self.info_text.set("Loading file, please wait...")
        self.root.update_idletasks()
        
        # Read the scale value in the main thread (thread-safe)
        current_vol = self.vol_scale.get()
        
        # Start processing in a separate thread so it doesn't freeze the Tkinter event loop
        threading.Thread(target=self._load_file_background, args=(path, current_vol), daemon=True).start()

    def _load_file_background(self, path, vol):
        try:
            def update_progress(current, total):
                if total > 0:
                    pct = int((current / total) * 100)
                    msg = f"Loading file, please wait...\nRead {current}/{total} frames ({pct}%)"
                    
                    # Strictly thread-safe: push updating data to queue
                    self.msg_queue.put(("progress", msg))

            player = HFPACPlayer(path, gui_mode=True, volume=vol, progress_callback=update_progress)
            
            # Send the final player object over the queue
            self.msg_queue.put(("success", player))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(("error", e))

    def _on_file_loaded_success(self, player):
        self.player = player
        h = self.player._header
        version_map = {2: "v2", 3: "v3", 4: "v4", 5: "v4.5", 6: "v5", 7: "v5.1", 8: "v6"}
        version_str = version_map.get(getattr(h, 'version', 0), display_version(getattr(h, 'version', 0)))

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
        lines += [
            f"Sample rate: {h.sample_rate} Hz",
            f"Channels: {h.channels}",
            f"Bit depth: {h.bit_depth}-bit",
            f"Duration: {self.player._fmt_time(self.player._duration)} "
            f"({self.player._duration:.1f}s)",
        ]
        self.info_text.set("\n".join(lines))
        self.progress['maximum'] = self.player._duration
        self.time_label.config(text=f"0:00 / {self.player._fmt_time(self.player._duration)}")
        self.btn_play.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
    
        self.root.update_idletasks()
        self.btn_pause.config(text="Pause")

    def _on_file_loaded_error(self, e):
        messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")
        self.player = None

    def play_audio(self):
        if not self.player:
            return
        
        # If paused, resume
        if self.player._paused:
            self.player.toggle_pause()
            self.btn_pause.config(text="Pause")
            return
            
        # If stopped or finished, recreate to start from beginning
        if self.player._stopped:
            path = self.player.path
            vol = self.vol_scale.get()
            preloaded = (self.player._header, self.player._frames)
            self.player = HFPACPlayer(path, gui_mode=True, volume=vol, preloaded_data=preloaded)
        
        self.btn_play.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_pause.config(text="Pause")
        
        self.player_thread = threading.Thread(target=self.player.play, daemon=True)
        self.player_thread.start()

    def toggle_pause(self):
        if self.player and not self.player._stopped:
            self.player.toggle_pause()
            if self.player._paused:
                self.btn_pause.config(text="Resume")
            else:
                self.btn_pause.config(text="Pause")

    def stop_audio(self):
        if self.player and not self.player._stopped:
            self.player.stop()
            if self.player_thread:
                self.player_thread.join(timeout=1.0)
                
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_pause.config(text="Pause")
        self.progress['value'] = 0
        if self.player:
             self.time_label.config(text=f"0:00 / {self.player._fmt_time(self.player._duration)}")
        
    def _on_progress_click(self, event):
        """Seek to the position the user clicked on the progress bar."""
        if not self.player or self.player._stopped:
            return
        if not self.player._duration:
            return

        # Translate click x-position to a fraction of the bar width,
        # then convert to seconds and seek
        bar_width = self.progress.winfo_width()
        if bar_width <= 0:
            return
        fraction = max(0.0, min(1.0, event.x / bar_width))
        target_s = fraction * self.player._duration
        self.player.seek(target_s)

    def change_volume(self, val):
        if self.player:
            self.player.volume = float(val)

    def update_timer(self):
        if self.player and not self.player._stopped:
            elapsed = self.player._elapsed()
            duration = self.player._duration
            self.progress['value'] = min(elapsed, duration)
            self.time_label.config(text=f"{self.player._fmt_time(elapsed)} / {self.player._fmt_time(duration)}")
        elif self.player and self.player._stopped and self.btn_play['state'] == tk.DISABLED:
            # Reached end of file
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            
        self.root.after(100, self.update_timer)


def main():
    import sys
    import traceback
    
    # Catch any fatal exceptions that cause the GUI to crash and log them to a file
    def global_exception_handler(exctype, value, tb):
        with open("crash.log", "w") as f:
            traceback.print_exception(exctype, value, tb, file=f)
        # Also try to show a popup before dying if Tkinter is still alive
        try:
            from tkinter import messagebox
            import tkinter as tk
            err_root = tk.Tk()
            err_root.withdraw()
            messagebox.showerror("Fatal Error", f"A fatal error occurred:\n{value}\nSee crash.log for details.")
            err_root.destroy()
        except:
            pass

    sys.excepthook = global_exception_handler

    root = tk.Tk()
    app = HFPACGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()