"""
encoder_gui.py — HFPAC Encoder GUI
====================================
A standalone tkinter interface for encoding WAV files to HFPAC format.
Provides access to all encoding options and metadata fields.

Launch via:
    python main.py encode-gui
or directly:
    python encoder_gui.py
"""

import threading
import tkinter as tk
import json
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from codec import encode_wav
from hfpac_format import (
    ENTROPY_HUFFMAN, ENTROPY_RICE,
    LPC_FLOAT, LPC_INTEGER,
    Metadata,
    STEREO_INDEPENDENT, STEREO_MID_SIDE,
)
from lpc import DEFAULT_LPC_ORDER, FRAME_SIZE

ENCODER_VERSION = "6.1.2.0"


class EncoderGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"HFPAC Encoder v{ENCODER_VERSION}")
        self.root.resizable(False, False)

        self._encoding = False   # True while an encode is in progress

        self._build_menubar()
        self._build_ui()

        # Auto-size window after widgets are placed
        self.root.update_idletasks()

    # ------------------------------------------------------------------
    # Menubar
    # ------------------------------------------------------------------

    def _build_menubar(self):
        menubar  = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Profile...", command=self._load_profile)
        file_menu.add_command(label="Save Profile...", command=self._save_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About HFPAC Encoder…",
                              command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _show_about(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("About HFPAC Encoder")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.focus_set()

        outer = tk.Frame(dlg, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(outer, text="HFPAC Encoder",
                 font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        tk.Label(outer, text=f"Version {ENCODER_VERSION}").pack(anchor="w", pady=(2, 10))

        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=(0, 10))

        def _row(label, value):
            row = tk.Frame(outer)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, font=("TkDefaultFont", 9, "bold"),
                     width=20, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=value, anchor="w").pack(side=tk.LEFT)

        _row("Output format:", "HFPAC v2, v3, v4, v4.5, v5, v5.1, v6, v6.1")

        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=(10, 8))
        tk.Label(outer, text="© 2026 HFPAC Project", fg="grey").pack(anchor="w")

        btn_frame = tk.Frame(outer)
        btn_frame.pack(fill=tk.X, pady=(12, 0))
        tk.Button(btn_frame, text="OK", width=8,
                  command=dlg.destroy).pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda e: dlg.destroy())
        dlg.bind("<Escape>", lambda e: dlg.destroy())

        dlg.update_idletasks()
        pw = self.root.winfo_x() + self.root.winfo_width()  // 2
        ph = self.root.winfo_y() + self.root.winfo_height() // 2
        dlg.geometry(f"+{pw - dlg.winfo_width() // 2}+{ph - dlg.winfo_height() // 2}")

    def _save_profile(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Profile", "*.json")],
            title="Save Encoding Profile"
        )
        if not path:
            return
            
        profile = {
            "version": self._version_var.get(),
            "preset": self._preset_var.get(),
            "stereo": self._stereo_var.get(),
            "lpc_mode": self._lpc_mode_var.get(),
            "entropy": self._entropy_var.get(),
            "lpc_order": self._lpc_order_var.get(),
            "adaptive": self._adaptive_var.get(),
            "sync": self._sync_var.get(),
            "frame_size": self._frame_size_var.get()
        }
        try:
            with open(path, "w") as f:
                json.dump(profile, f, indent=4)
            messagebox.showinfo("Profile Saved", f"Encoding profile saved to:\n{Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save profile:\n{e}")

    def _load_profile(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON Profile", "*.json")],
            title="Load Encoding Profile"
        )
        if not path:
            return
            
        try:
            with open(path, "r") as f:
                profile = json.load(f)
                
            # Validation: Make sure the profile matches the currently selected format version
            # Or we can just set the version to whatever the profile is! 
            # The prompt requested: "Make sure the Profile will only load when the correct 
            # HFPAC format version is selected (as to not activate Encoding Options that 
            # can't be changed on older versions)"
            current_version = self._version_var.get()
            profile_version = profile.get("version", current_version)
            
            if current_version != profile_version:
                messagebox.showwarning(
                    "Version Mismatch", 
                    f"This profile was created for '{profile_version}', "
                    f"but your currently selected format is '{current_version}'.\n\n"
                    f"Please change your Format version to '{profile_version}' to load this profile."
                )
                return

            self._applying_preset = True
            
            # Since version is correct, it's safe to just apply the options
            if "stereo" in profile: self._stereo_var.set(profile["stereo"])
            if "lpc_mode" in profile: self._lpc_mode_var.set(profile["lpc_mode"])
            if "entropy" in profile: self._entropy_var.set(profile["entropy"])
            if "lpc_order" in profile: self._lpc_order_var.set(profile["lpc_order"])
            if "adaptive" in profile: self._adaptive_var.set(profile["adaptive"])
            if "sync" in profile: self._sync_var.set(profile["sync"])
            if "frame_size" in profile: self._frame_size_var.set(profile["frame_size"])
            if "preset" in profile: self._preset_var.set(profile["preset"])
            
            self._on_adaptive_change()  # Update grayed-out states based on profile
            
            messagebox.showinfo("Profile Loaded", f"Successfully loaded profile:\n{Path(path).name}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load profile:\n{e}")
        finally:
            self._applying_preset = False

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=4)

        # ── Source ────────────────────────────────────────────────────
        src_frame = tk.LabelFrame(self.root, text="Source")
        src_frame.pack(fill=tk.X, **pad)

        self._src_var = tk.StringVar()
        tk.Entry(src_frame, textvariable=self._src_var,
                 width=48, state="readonly").pack(side=tk.LEFT, padx=(6, 4), pady=4)
        tk.Button(src_frame, text="Browse…",
                  command=self._browse_src).pack(side=tk.LEFT, padx=(0, 6), pady=4)

        # ── Output ────────────────────────────────────────────────────
        out_frame = tk.LabelFrame(self.root, text="Output")
        out_frame.pack(fill=tk.X, **pad)

        self._out_var = tk.StringVar()
        tk.Entry(out_frame, textvariable=self._out_var,
                 width=48).pack(side=tk.LEFT, padx=(6, 4), pady=4)
        tk.Button(out_frame, text="Browse…",
                  command=self._browse_out).pack(side=tk.LEFT, padx=(0, 6), pady=4)

        # ── Metadata ──────────────────────────────────────────────────
        meta_frame = tk.LabelFrame(self.root, text="Metadata  (optional)")
        meta_frame.pack(fill=tk.X, **pad)

        def _meta_row(parent, label, row, width=28):
            tk.Label(parent, text=label, anchor="w",
                     width=12).grid(row=row, column=0, sticky="w",
                                    padx=(6, 2), pady=2)
            var = tk.StringVar()
            tk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=1, sticky="ew", padx=(0, 6), pady=2)
            return var

        self._title_var  = _meta_row(meta_frame, "Title",  0)
        self._artist_var = _meta_row(meta_frame, "Artist", 1)
        self._album_var  = _meta_row(meta_frame, "Album",  2)

        # Track number and year on same row
        num_row = tk.Frame(meta_frame)
        num_row.grid(row=3, column=0, columnspan=2, sticky="ew",
                     padx=6, pady=(2, 4))

        tk.Label(num_row, text="Track #", anchor="w", width=12).pack(side=tk.LEFT)
        self._track_var = tk.StringVar()
        tk.Entry(num_row, textvariable=self._track_var, width=6).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(num_row, text="Year", anchor="w", width=6).pack(side=tk.LEFT)
        self._year_var = tk.StringVar()
        tk.Entry(num_row, textvariable=self._year_var, width=6).pack(side=tk.LEFT)

        # ── Encoding options ──────────────────────────────────────────
        opt_frame = tk.LabelFrame(self.root, text="Encoding Options")
        opt_frame.pack(fill=tk.X, **pad)

        def _info_btn(parent, title, message):
            """Small ⓘ button that shows a plain info dialog when clicked."""
            def _show():
                dlg = tk.Toplevel(self.root)
                dlg.title(title)
                dlg.resizable(False, False)
                dlg.grab_set()
                dlg.focus_set()
                outer = tk.Frame(dlg, padx=16, pady=14)
                outer.pack()
                tk.Label(outer, text=title,
                         font=("TkDefaultFont", 9, "bold"),
                         anchor="w").pack(anchor="w")
                ttk.Separator(outer, orient="horizontal").pack(
                    fill=tk.X, pady=(6, 8))
                tk.Label(outer, text=message, justify=tk.LEFT,
                         wraplength=340, anchor="w").pack(anchor="w")
                bf = tk.Frame(outer)
                bf.pack(fill=tk.X, pady=(12, 0))
                tk.Button(bf, text="OK", width=8,
                          command=dlg.destroy).pack(side=tk.RIGHT)
                dlg.bind("<Return>", lambda e: dlg.destroy())
                dlg.bind("<Escape>", lambda e: dlg.destroy())
                dlg.update_idletasks()
                pw = self.root.winfo_x() + self.root.winfo_width()  // 2
                ph = self.root.winfo_y() + self.root.winfo_height() // 2
                dlg.geometry(
                    f"+{pw - dlg.winfo_width() // 2}"
                    f"+{ph - dlg.winfo_height() // 2}"
                )
            return tk.Button(parent, text="ⓘ", font=("TkDefaultFont", 8),
                             relief=tk.FLAT, cursor="hand2",
                             padx=2, pady=0, command=_show)

        # Row 0: Format version
        row_ver = tk.Frame(opt_frame)
        row_ver.pack(fill=tk.X, padx=6, pady=(4, 2))

        tk.Label(row_ver, text="Format version", anchor="w",
                 width=14).pack(side=tk.LEFT)
        self._version_var = tk.StringVar(value="Latest (v6.1)")
        self._version_cb  = ttk.Combobox(
            row_ver, textvariable=self._version_var,
            values=["Latest (v6.1)", "v6", "v5.1", "v5", "v4.5", "v4", "v3", "v2"],
            state="readonly", width=14,
        )
        self._version_cb.pack(side=tk.LEFT)
        _info_btn(row_ver, "Format version",
            "Selects which version of the HFPAC format to write.\n\n"
            "Latest (v6.1) uses all current features: true gapless playback metadata, "
            "adaptive LPC order per frame, silence detection, history carry-over "
            "between frames, integer LPC, Rice coding, mid-side stereo, seek table, "
            "and metadata.\n\n"
            "v6 uses all features of v6.1 but lacks gapless metadata.\n\n"
            "v5.1 is the previous stable format with fixed LPC order, "
            "integer LPC, Rice, metadata, and per-frame CRC.\n\n"
            "Older versions produce files compatible with older players but "
            "disable some encoding options. The options below are locked "
            "to valid settings automatically when you choose an older version."
        ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(row_ver, text="  Older versions lock the options below.",
                 fg="grey", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(opt_frame, orient="horizontal").pack(fill=tk.X, padx=6, pady=(4, 2))

        # Row 0.5: Compression Presets
        row_preset = tk.Frame(opt_frame)
        row_preset.pack(fill=tk.X, padx=6, pady=(2, 2))

        tk.Label(row_preset, text="Preset", anchor="w", width=14).pack(side=tk.LEFT)
        self._preset_var = tk.StringVar(value="Custom")
        self._preset_cb  = ttk.Combobox(
            row_preset, textvariable=self._preset_var,
            values=["Ultra Fast", "Fast", "Standard", "High", "Ultra", "Custom"],
            state="readonly", width=14
        )
        self._preset_cb.pack(side=tk.LEFT)
        self._preset_cb.bind("<<ComboboxSelected>>", self._apply_preset)
        
        _info_btn(row_preset, "Compression Preset",
            "Select a predefined compression preset.\n"
            "Note: Presets are only available with the latest codec format (v6).\n\n"
            "• Ultra Fast: Fastest encoding, larger file size.\n"
            "• Fast: Good speed, slightly larger files.\n"
            "• Standard: Balanced speed and compression (Default).\n"
            "• High: Better compression, slightly slower.\n"
            "• Ultra: Maximum compression, slowest to encode."
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Separator(opt_frame, orient="horizontal").pack(fill=tk.X, padx=6, pady=(4, 2))

        # Row 1: Stereo mode + LPC mode
        row0 = tk.Frame(opt_frame)
        row0.pack(fill=tk.X, padx=6, pady=(2, 2))

        tk.Label(row0, text="Stereo mode", anchor="w", width=14).pack(side=tk.LEFT)
        self._stereo_var = tk.StringVar(value="Mid-Side")
        self._stereo_cb  = ttk.Combobox(row0, textvariable=self._stereo_var,
                                         values=["Mid-Side", "Independent"],
                                         state="readonly", width=14)
        self._stereo_cb.pack(side=tk.LEFT)
        _info_btn(row0, "Stereo mode",
            "Controls how the two audio channels are stored.\n\n"
            "Mid-Side converts L/R into a sum channel (M = (L+R)/2) and a "
            "difference channel (S = (L-R)/2). On most music the difference "
            "channel is nearly silent, so it compresses much harder than raw "
            "L and R. Recommended for music.\n\n"
            "Independent encodes L and R separately with no conversion. "
            "Use this for audio where the two channels are genuinely unrelated, "
            "or when encoding for an older format version. "
            "Mono files ignore this setting entirely."
        ).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(row0, text="LPC mode", anchor="w", width=10).pack(side=tk.LEFT)
        self._lpc_mode_var = tk.StringVar(value="Integer")
        self._lpc_mode_cb  = ttk.Combobox(row0, textvariable=self._lpc_mode_var,
                                            values=["Integer", "Float32"],
                                            state="readonly", width=10)
        self._lpc_mode_cb.pack(side=tk.LEFT)
        _info_btn(row0, "LPC mode",
            "Controls the arithmetic used inside the predictor.\n\n"
            "Integer uses int16 coefficients and int64 accumulators — no "
            "floating point anywhere in the filter. Reconstruction is "
            "mathematically exact (bit-perfect). This is the default and "
            "what FLAC also uses.\n\n"
            "Float32 uses 32-bit floating point coefficients, which introduces "
            "tiny rounding errors that make roughly half of decoded samples "
            "off by 1 LSB (~1/65536 of full scale). Inaudible, but technically "
            "not lossless. Only useful when targeting format v3 or v4."
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Row 2: Entropy + LPC order
        row1 = tk.Frame(opt_frame)
        row1.pack(fill=tk.X, padx=6, pady=(2, 4))

        tk.Label(row1, text="Entropy", anchor="w", width=14).pack(side=tk.LEFT)
        self._entropy_var = tk.StringVar(value="Rice")
        self._entropy_cb  = ttk.Combobox(row1, textvariable=self._entropy_var,
                                          values=["Rice", "Huffman"],
                                          state="readonly", width=14)
        self._entropy_cb.pack(side=tk.LEFT)
        _info_btn(row1, "Entropy coding",
            "The algorithm used to compress the residuals (prediction errors) "
            "after LPC.\n\n"
            "Rice is tuned for the distribution LPC residuals follow — most "
            "values cluster near zero. It stores each value as a short prefix "
            "plus fixed-width bits, with no tree to store (just 1 byte per "
            "frame). Faster and typically smaller on real music.\n\n"
            "Huffman builds an optimal variable-length code tree from the "
            "actual residual frequencies across each 64-frame block. Adapts "
            "better to unusual distributions but costs ~100 bytes per block "
            "to store the tree. May win on highly irregular audio."
        ).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(row1, text="LPC order", anchor="w", width=10).pack(side=tk.LEFT)
        self._lpc_order_var = tk.StringVar(value=str(DEFAULT_LPC_ORDER))
        self._lpc_order_cb  = ttk.Combobox(row1, textvariable=self._lpc_order_var,
                                             values=[str(i) for i in range(4, 25, 2)],
                                             width=10)
        self._lpc_order_cb.pack(side=tk.LEFT)
        _info_btn(row1, "LPC order",
            "How many previous samples the predictor uses to estimate the "
            "next one. Order 12 means 'look back 12 samples and fit a linear "
            "combination of them.'\n\n"
            "Higher orders model the signal's frequency content more precisely "
            "— complex orchestral music may benefit from 16 or 18 — but "
            "returns diminish quickly past 12 and encode time increases.\n\n"
            "Lower orders (4–8) work well for simple signals like speech or a "
            "solo instrument.\n\n"
            "12 is the recommended default for most music."
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Row 3: Adaptive order + Sync interval  (v6 only)
        row2 = tk.Frame(opt_frame)
        row2.pack(fill=tk.X, padx=6, pady=(2, 4))

        self._adaptive_var = tk.BooleanVar(value=False)
        self._adaptive_cb  = tk.Checkbutton(
            row2, text="Adaptive LPC order", variable=self._adaptive_var,
            command=self._on_adaptive_change,
        )
        self._adaptive_cb.pack(side=tk.LEFT)
        _info_btn(row2, "Adaptive LPC order",
            "When enabled, HFPAC tries every even LPC order from 2 to 20 for "
            "each frame and picks the one with the lowest total bit cost "
            "(Rice bits + 16 bits per coefficient).\n\n"
            "Best suited for broadband or noisy material (percussion, room "
            "ambience, speech) where different parts of the track benefit from "
            "different orders. On such material adaptive typically matches or "
            "beats fixed-12.\n\n"
            "On simple tonal content (pure sine waves, solo instruments) the "
            "cost model may legitimately choose low orders (2–4) that produce "
            "smaller individual frames but larger Rice payloads overall. For "
            "those signals, fixed-12 or fixed-16 usually compresses better.\n\n"
            "Disabled by default. Enable the LPC Order field below to set a "
            "fixed order when adaptive is off.\n\n"
            "Only available for v6 format."
        ).pack(side=tk.LEFT, padx=(4, 24))

        tk.Label(row2, text="Sync interval", anchor="w",
                 width=13).pack(side=tk.LEFT)
        self._sync_var = tk.StringVar(value="64")
        self._sync_cb  = ttk.Combobox(
            row2, textvariable=self._sync_var,
            values=["16", "32", "64", "128", "256"],
            state="readonly", width=6,
        )
        self._sync_cb.pack(side=tk.LEFT)
        _info_btn(row2, "Sync interval",
            "How often to insert a SYNC frame — a frame that resets the LPC "
            "history to zero and marks a seek point.\n\n"
            "Every seek (clicking in the progress bar) jumps to the nearest "
            "SYNC frame before the target time, so a smaller interval means "
            "more accurate seeking at the cost of slightly lower compression.\n\n"
            "64 frames ≈ 1.5 seconds at 44100 Hz / 1024 samples per frame. "
            "This is the recommended default.\n\n"
            "Only available for v6 format."
        ).pack(side=tk.LEFT, padx=(4, 0))

        row3 = tk.Frame(opt_frame)
        row3.pack(fill=tk.X, padx=6, pady=(2, 4))

        tk.Label(row3, text="Frame size", anchor="w",
                 width=13).pack(side=tk.LEFT)
        self._frame_size_var = tk.StringVar(value=str(FRAME_SIZE))
        self._frame_size_cb  = ttk.Combobox(
            row3, textvariable=self._frame_size_var,
            values=["256", "512", "1024", "2048", "4096", "8192"],
            state="readonly", width=8,
        )
        self._frame_size_cb.pack(side=tk.LEFT)
        _info_btn(row3, "Frame size",
            "The number of audio samples processed in a single LPC frame.\n\n"
            "Smaller frames track rapidly changing audio better but cost more "
            "framing overhead. Larger frames give better compression on steady "
            "signals but may cause artifacts on transients.\n\n"
            "1024 is the recommended default for most music."
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Wire up change handlers
        self._version_var.trace_add("write", lambda *_: self._on_version_change())
        
        self._applying_preset = False
        
        # Share a single change handler for other encoding options
        def _on_options_change(*args):
            if not getattr(self, "_applying_preset", False):
                self._preset_var.set("Custom")
            
        for var in (self._stereo_var, self._lpc_mode_var, self._entropy_var, 
                    self._lpc_order_var, self._sync_var, self._adaptive_var, self._frame_size_var):
            var.trace_add("write", _on_options_change)

        # ── Progress ──────────────────────────────────────────────────
        prog_frame = tk.Frame(self.root)
        prog_frame.pack(fill=tk.X, padx=10, pady=(6, 2))

        self._progress = ttk.Progressbar(prog_frame, mode="determinate")
        self._progress.pack(fill=tk.X)

        self._status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self._status_var,
                 anchor="w", fg="grey").pack(fill=tk.X, padx=10)

        # ── Encode button ─────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=(4, 8))

        self._encode_btn = tk.Button(btn_frame, text="Encode",
                                     width=14, font=("TkDefaultFont", 9, "bold"),
                                     command=self._start_encode)
        self._encode_btn.pack()

        # ── Results ───────────────────────────────────────────────────
        self._result_frame = tk.LabelFrame(self.root, text="Results")
        # Not packed yet — shown only after a successful encode

        self._result_text = tk.StringVar()
        tk.Label(self._result_frame, textvariable=self._result_text,
                 justify=tk.LEFT, anchor="w").pack(anchor="w", padx=8, pady=6)

    # ------------------------------------------------------------------
    # Version-locking logic
    # ------------------------------------------------------------------

    # Per-version constraints:
    # (stereo_options, lpc_mode_options, entropy_options,
    #  stereo_default, lpc_mode_default, entropy_default,
    #  adaptive_available, sync_available)
    _VERSION_CONSTRAINTS = {
        "v2":          (["Independent"],              ["Float"],              ["Huffman"],
                        "Independent", "Float",   "Huffman", False, False),
        "v3":          (["Independent"],              ["Float"],              ["Huffman"],
                        "Independent", "Float",   "Huffman", False, False),
        "v4":          (["Mid-Side", "Independent"],  ["Float"],              ["Huffman"],
                        "Mid-Side",    "Float",   "Huffman", False, False),
        "v4.5":        (["Mid-Side", "Independent"],  ["Float"],              ["Rice", "Huffman"],
                        "Mid-Side",    "Float",   "Rice",    False, False),
        "v5":          (["Mid-Side", "Independent"],  ["Integer", "Float32"], ["Rice", "Huffman"],
                        "Mid-Side",    "Integer", "Rice",    False, False),
        "v5.1":        (["Mid-Side", "Independent"],  ["Integer", "Float32"], ["Rice", "Huffman"],
                        "Mid-Side",    "Integer", "Rice",    False, False),
        "v6":          (["Mid-Side", "Independent"],  ["Integer", "Float32"], ["Rice", "Huffman"],
                        "Mid-Side",    "Integer", "Rice",    True,  True),
        "Latest (v6.1)": (["Mid-Side", "Independent"],  ["Integer", "Float32"], ["Rice", "Huffman"],
                        "Mid-Side",    "Integer", "Rice",    True,  True),
    }

    def _on_adaptive_change(self):
        """Enable/disable LPC order when adaptive order is toggled."""
        if self._adaptive_var.get():
            self._lpc_order_cb.config(state="disabled")
        else:
            self._lpc_order_cb.config(state="normal")
            
        self._preset_var.set("Custom")

    def _apply_preset(self, event=None):
        preset = self._preset_var.get()
        if preset == "Custom":
            return
            
        self._applying_preset = True
        try:
            # Lock format to Latest if applying a modern preset
            self._version_var.set("Latest (v6.1)")
            self._on_version_change()
            
            # Defaults for High/Ultra
            self._stereo_var.set("Mid-Side")
            self._lpc_mode_var.set("Integer")
            self._entropy_var.set("Rice")
            self._sync_var.set("64")
            
            if preset == "Ultra Fast":
                self._stereo_var.set("Independent")
                self._lpc_order_var.set("4")
                self._adaptive_var.set(False)
            elif preset == "Fast":
                self._lpc_order_var.set("8")
                self._adaptive_var.set(False)
            elif preset == "Standard":
                self._lpc_order_var.set("12")
                self._adaptive_var.set(False)
            elif preset == "High":
                self._lpc_order_var.set("16")
                self._adaptive_var.set(True)
            elif preset == "Ultra":
                self._lpc_order_var.set("24")
                self._adaptive_var.set(True)
                self._sync_var.set("128")
                
            self._on_adaptive_change()
            # Reset the preset name because _on_adaptive_change/etc might reset it tracking manual changes
            self._preset_var.set(preset)
        finally:
            self._applying_preset = False

    def _on_version_change(self):
        ver = self._version_var.get()
        if ver != "Latest (v6.1)":
            self._preset_var.set("Custom")
            self._preset_cb.config(state="disabled")
        else:
            self._preset_cb.config(state="readonly")
            
        c   = self._VERSION_CONSTRAINTS.get(ver)
        if c is None:
            return

        stereo_vals, lpc_vals, ent_vals, s_def, lpc_def, ent_def, \
            adaptive_avail, sync_avail = c

        # Update dropdown values and defaults
        self._stereo_cb.config(values=stereo_vals)
        self._lpc_mode_cb.config(values=lpc_vals)
        self._entropy_cb.config(values=ent_vals)
        self._stereo_var.set(s_def)
        self._lpc_mode_var.set(lpc_def)
        self._entropy_var.set(ent_def)

        # Lock/unlock dropdowns based on how many choices exist
        self._stereo_cb.config(
            state="readonly" if len(stereo_vals) > 1 else "disabled")
        self._lpc_mode_cb.config(
            state="readonly" if len(lpc_vals)    > 1 else "disabled")
        self._entropy_cb.config(
            state="readonly" if len(ent_vals)    > 1 else "disabled")

        # Adaptive order and sync interval — v6 only
        if adaptive_avail:
            self._adaptive_cb.config(state="normal")
            self._adaptive_var.set(False)      # off by default; user opts in
            self._sync_cb.config(state="readonly")
            self._lpc_order_cb.config(state="normal")  # order active when adaptive=False
        else:
            self._adaptive_cb.config(state="disabled")
            self._adaptive_var.set(False)
            self._sync_cb.config(state="disabled")
            self._lpc_order_cb.config(state="normal")     # manual order only

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    def _browse_src(self):
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        self._src_var.set(path)
        # Auto-fill output path if it's empty
        if not self._out_var.get():
            self._out_var.set(str(Path(path).with_suffix(".hfpac")))

    def _browse_out(self):
        initial = self._out_var.get() or self._src_var.get()
        path = filedialog.asksaveasfilename(
            title="Save HFPAC file as",
            defaultextension=".hfpac",
            filetypes=[("HFPAC files", "*.hfpac"), ("All files", "*.*")],
            initialfile=str(Path(initial).with_suffix(".hfpac")) if initial else "",
        )
        if path:
            self._out_var.set(path)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    # Map display label → internal format version integer
    _VERSION_MAP = {
        "v2":            2,
        "v3":            3,
        "v4":            4,
        "v4.5":          5,
        "v5":            6,
        "v5.1":          7,
        "v6":            8,
        "Latest (v6.1)": 9,
    }

    def _collect_options(self):
        """Read all widget values and return kwargs dict, or raise ValueError."""
        src = self._src_var.get().strip()
        out = self._out_var.get().strip()
        if not src:
            raise ValueError("Please select a source WAV file.")
        if not out:
            raise ValueError("Please specify an output path.")
        if not Path(src).exists():
            raise ValueError(f"Source file not found:\n{src}")

        ver_label  = self._version_var.get()
        target_ver = self._VERSION_MAP.get(ver_label, 9)

        stereo_mode  = STEREO_MID_SIDE if self._stereo_var.get() == "Mid-Side" \
                       else STEREO_INDEPENDENT
        lpc_mode     = LPC_INTEGER if self._lpc_mode_var.get() == "Integer" \
                       else LPC_FLOAT
        entropy_mode = ENTROPY_RICE if self._entropy_var.get() == "Rice" \
                       else ENTROPY_HUFFMAN

        try:
            lpc_order = int(self._lpc_order_var.get())
            if not 2 <= lpc_order <= 32:
                raise ValueError
        except ValueError:
            raise ValueError("LPC order must be an integer between 2 and 32.")

        try:
            sync_interval = int(self._sync_var.get())
            if sync_interval <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Sync interval must be a positive integer.")

        try:
            frame_size = int(self._frame_size_var.get())
            if frame_size <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Frame size must be a positive integer.")

        adaptive_order = self._adaptive_var.get() and (target_ver >= 8)

        def _int(s):
            try:    return int(s.strip())
            except: return 0

        # Metadata only meaningful for v5.1+ (target_ver ≥ 7)
        if target_ver >= 7:
            meta = Metadata(
                title        = self._title_var.get().strip(),
                artist       = self._artist_var.get().strip(),
                album        = self._album_var.get().strip(),
                track_number = _int(self._track_var.get()),
                year         = _int(self._year_var.get()),
            )
        else:
            meta = Metadata()

        kwargs = dict(
            input_wav      = src,
            output_hfpac   = out,
            lpc_order      = lpc_order,
            stereo_mode    = stereo_mode,
            entropy_mode   = entropy_mode,
            lpc_mode       = lpc_mode,
            sync_interval  = sync_interval,
            frame_size     = frame_size,
            adaptive_order = adaptive_order,
            metadata       = meta,
            _target_ver    = target_ver,   # handled in _encode_worker
        )
        return kwargs

    def _start_encode(self):
        if self._encoding:
            return
        try:
            kwargs = self._collect_options()
        except ValueError as e:
            messagebox.showerror("Input error", str(e), parent=self.root)
            return

        self._encoding = True
        self._encode_btn.config(state=tk.DISABLED)
        self._result_frame.pack_forget()
        self._status_var.set("Encoding…")
        self._progress["value"] = 0

        threading.Thread(
            target=self._encode_worker,
            args=(kwargs,),
            daemon=True,
        ).start()

    def _update_progress(self, current, total):
        pct = (current / total) * 100 if total > 0 else 0
        self._progress["value"] = pct

    def _encode_worker(self, kwargs):
        import hfpac_format as hfmt

        target_ver     = kwargs.pop("_target_ver", 9)
        adaptive_order = kwargs.pop("adaptive_order", False)
        sync_interval  = kwargs.pop("sync_interval", 64)
        orig_ver       = hfmt.FORMAT_VERSION

        def prog_cb(c, t): self.root.after(0, self._update_progress, c, t)

        # v6 (target_ver 8) is the current format — no patching needed.
        # For legacy targets (2–7) we temporarily patch FORMAT_VERSION.
        try:
            if target_ver != hfmt.FORMAT_VERSION:
                hfmt.FORMAT_VERSION = target_ver
            stats = encode_wav(
                **kwargs,
                adaptive_order   = adaptive_order,
                sync_interval    = sync_interval,
                progress_callback= prog_cb,
            )
            self.root.after(0, self._on_encode_done, stats, kwargs['output_hfpac'])
        except Exception as e:
            self.root.after(0, self._on_encode_error, e)
        finally:
            hfmt.FORMAT_VERSION = orig_ver

    def _on_encode_done(self, stats, out_path):
        self._progress.stop()
        self._encoding = False
        self._encode_btn.config(state=tk.NORMAL)
        self._status_var.set("Done.")

        ratio   = stats.get('ratio', 0)
        in_sz   = stats.get('input_size', 0)
        out_sz  = stats.get('output_size', 0)
        elapsed = stats.get('encode_time', 0)
        dur     = stats.get('duration', 0)
        ver_lbl = self._version_var.get().replace("Latest (", "").rstrip(")")

        self._result_text.set(
            f"Format:   {'HFPAC ' + ver_lbl:>14}\n"
            f"Input:    {in_sz:>12,} bytes\n"
            f"Output:   {out_sz:>12,} bytes\n"
            f"Ratio:    {ratio:>11.2f}x\n"
            f"Duration: {dur:>10.1f} s\n"
            f"Time:     {elapsed:>10.2f} s\n"
            f"Speed:    {dur/elapsed:>10.1f}x realtime"
        )
        self._result_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.root.update_idletasks()

    def _on_encode_error(self, exc):
        self._progress.stop()
        self._encoding = False
        self._encode_btn.config(state=tk.NORMAL)
        self._status_var.set("Encoding failed.")
        messagebox.showerror("Encoding error", str(exc), parent=self.root)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    EncoderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()