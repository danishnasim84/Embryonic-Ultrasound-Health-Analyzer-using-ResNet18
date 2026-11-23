"""
app.py  -  FetalScan AI  |  Fetal Ultrasound Analyzer
======================================================
HOW TO RUN:
    pip install torch torchvision pillow grad-cam opencv-python
    python app.py

REQUIREMENTS:
    model.pth must exist in the same folder (run train.py first)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#f5f9f5"   # light green-white background
CARD     = "#ffffff"   # white card surfaces
CARD2    = "#e8f5e9"   # light mint (bar tracks)
BORDER   = "#c8e6c9"   # soft green borders
ACCENT   = "#2e7d32"   # deep green (accent, buttons)
GREEN    = "#2e7d32"   # normal / success — deep green
RED      = "#c62828"   # malignant / danger — red
AMBER    = "#e65100"   # benign / warning — burnt orange
TEXT     = "#1b2e1b"   # near-black dark forest green text
DIM      = "#4a6a4a"   # muted green secondary text
DIMMER   = "#7a9a7a"   # soft green placeholder
IMGBG    = "#eef5ee"   # very light green behind images

FONT     = "Times New Roman"   # crisp serif — user requested
MONO     = "Courier New"       # classic monospace to match
MODEL_PATH = "model.pth"

# Fixed size for BOTH image canvases
CANV_W = 393
CANV_H = 240


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt        = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]
    model       = models.resnet18(weights=None)
    model.fc    = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names


def run_model(model, class_names, image_path):
    """Run ResNet prediction + Grad-CAM heatmap on one image."""
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_pil = Image.open(image_path).convert("RGB")
    tensor  = tfm(img_pil).unsqueeze(0)        # shape: [1, 3, 224, 224]

    # Prediction
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx   = int(torch.argmax(probs))

    pred = class_names[idx]
    conf = float(probs[idx]) * 100

    # Grad-CAM
    cam         = GradCAM(model=model, target_layers=[model.layer4[-1]])
    gcam        = cam(input_tensor=tensor, targets=None)[0]          # (224,224)
    arr         = np.array(img_pil.resize((224, 224))) / 255.0       # (224,224,3)
    heatmap_pil = Image.fromarray(
        show_cam_on_image(arr, gcam, use_rgb=True)
    )

    return pred, conf, probs, class_names, heatmap_pil


def make_report(label, confidence):
    sep = "─" * 46
    if label.lower() == "normal":
        obs = ("· Sonographic architecture within standard ranges.\n"
               "  · No major structural anomalies detected.")
        rec = "· Routine follow-up per standard obstetric schedule."
    elif label.lower() == "benign":
        obs = ("· Non-malignant variance or benign fluid accumulation observed.\n"
               "  · Structure appears contained with regular margins.")
        rec = "· Periodic monitoring and clinical correlation advised."
    else:
        obs = ("· Potential structural irregularity or atypical density detected.\n"
               "  · High-risk pattern — secondary evaluation strongly advised.")
        rec = "· Comprehensive targeted ultrasound and diagnostic review required."

    return (
        f"  CLASSIFICATION  :  {label.upper()}\n"
        f"  CONFIDENCE      :  {confidence:.2f}%\n"
        f"  {sep}\n\n"
        f"  Clinical Observations:\n"
        f"  {obs}\n\n"
        f"  Recommendation:\n"
        f"  {rec}\n\n"
        f"  {sep}\n"
    
    )


# ── Rounded canvas button ─────────────────────────────────────────────────────
class RoundBtn(tk.Canvas):
    """A pill-shaped button drawn on a Canvas (no platform chrome)."""

    def __init__(self, parent, text, command=None,
                 w=200, h=36, fill=ACCENT, fg="#ffffff", **kw):
        super().__init__(parent, width=w, height=h,
                         bg=parent["bg"], bd=0, highlightthickness=0, **kw)
        self._text    = text
        self._cmd     = command
        self._fill    = fill
        self._fg      = fg
        self._enabled = True
        self._redraw(fill)
        self.bind("<ButtonRelease-1>",
                  lambda _e: self._enabled and self._cmd and self._cmd())
        self.bind("<Enter>",  lambda _e: self._on_enter())
        self.bind("<Leave>",  lambda _e: self._on_leave())

    def _redraw(self, fill):
        self.delete("all")
        w  = int(self["width"])
        h  = int(self["height"])
        r  = 8                          # corner radius
        fg = self._fg if self._enabled else DIMMER

        # Build rounded rect from arcs + rectangles
        for x1, y1, x2, y2, start in (
            (0,     0,     r*2, r*2, 90),
            (w-r*2, 0,     w,   r*2, 0),
            (0,     h-r*2, r*2, h,   180),
            (w-r*2, h-r*2, w,   h,   270),
        ):
            self.create_arc(x1, y1, x2, y2,
                            start=start, extent=90,
                            fill=fill, outline=fill)
        self.create_rectangle(r,   0,   w-r, h,   fill=fill, outline=fill)
        self.create_rectangle(0,   r,   w,   h-r, fill=fill, outline=fill)
        self.create_text(w // 2, h // 2,
                         text=self._text, fill=fg,
                         font=(FONT, 10, "bold"))

    def _on_enter(self):
        if self._enabled:
            self._redraw(self._brighten(self._fill))
            self.config(cursor="hand2")

    def _on_leave(self):
        self._redraw(self._fill if self._enabled else BORDER)
        self.config(cursor="")

    def enable(self, val=True):
        self._enabled = val
        self._redraw(self._fill if val else BORDER)

    @staticmethod
    def _brighten(h, f=1.18):
        r = min(255, int(int(h[1:3], 16) * f))
        g = min(255, int(int(h[3:5], 16) * f))
        b = min(255, int(int(h[5:7], 16) * f))
        return f"#{r:02x}{g:02x}{b:02x}"


# ── Main application ──────────────────────────────────────────────────────────
class UltrasoundApp:

    def __init__(self, root):
        self.root        = root
        self.root.title("EUHA  ·  Analyzer")
        self.root.geometry("870x980")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        # Keep PhotoImage refs alive (Python GC will destroy them otherwise)
        self._orig_photo = None
        self._heat_photo = None

        self.image_path  = None
        self.model       = None
        self.class_names = None
        self.bar_data    = {}   # { cls_name: (canvas, pct_label, color) }

        self.model, self.class_names = load_model()
        self._build_ui()

    # ── Top-level layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._divider(self.root)
        self._build_body()
        self._divider(self.root)
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg="#2e7d32", height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  EUHA",
                 font=(FONT, 19, "bold"), bg="#2e7d32", fg="#ffffff"
                 ).pack(side="left", padx=18, pady=12)

        tk.Label(hdr,
                 text="Embryonic Ultrasound Health Analyzer System  ·  Final Year Project",
                 font=(FONT, 9), bg="#2e7d32", fg="#c8e6c9"
                 ).pack(side="left", pady=20)

        if self.model:
            ptxt, pfg, pbg = "● Model Ready",     GREEN, "#e8f5e9"
        else:
            ptxt, pfg, pbg = "● No Model Found",  RED,   "#ffebee"

        tk.Label(hdr, text=ptxt, font=(FONT, 9, "bold"),
                 fg=pfg, bg=pbg, padx=12, pady=5
                 ).pack(side="right", padx=16, pady=14)

    # ── Body (all the main content) ───────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        P = 18   # standard horizontal padding

        # ── Row 1 : Upload controls ──────────────────────────────────────────
        row1 = tk.Frame(body, bg=BG)
        row1.pack(fill="x", padx=P, pady=(14, 0))

        tk.Label(row1, text="SELECT IMAGE",
                 font=(FONT, 9, "bold"), bg=BG, fg=DIMMER
                 ).pack(side="left", pady=4)

        self.upload_btn = RoundBtn(row1, "⬆   Upload Ultrasound Image",
                                   command=self.upload_image,
                                   w=242, h=34, fill=ACCENT)
        self.upload_btn.pack(side="right")

        self.path_label = tk.Label(body, text="No file selected.",
                                   font=(FONT, 9), bg=BG, fg=DIMMER,
                                   anchor="w", justify="left")
        self.path_label.pack(fill="x", padx=P, pady=(4, 8))

        # ── Row 2 : Image canvases side-by-side ─────────────────────────────
        #
        #   BUG IN PREVIOUS VERSION: _image_panel returned only the canvas,
        #   but the outer frame was never packed.  Now we return (frame, canvas)
        #   and pack the frame here.
        #
        img_row = tk.Frame(body, bg=BG)
        img_row.pack(fill="x", padx=P, pady=(0, 10))

        orig_frame, self.orig_canvas = self._make_image_panel("  Original Scan")
        orig_frame.pack(in_=img_row, side="left", padx=(0, 6))

        heat_frame, self.heat_canvas = self._make_image_panel(
            "  AI Attention Heatmap  (Grad-CAM)"
        )
        heat_frame.pack(in_=img_row, side="left", padx=(6, 0))

        # Draw placeholder text on both canvases
        self._canvas_text(self.orig_canvas, "Upload an image to preview")
        self._canvas_text(self.heat_canvas, "Heatmap will appear after analysis")

        # ── Row 3 : Analyze button ───────────────────────────────────────────
        self.analyze_btn = RoundBtn(body, "⚡   Analyze Image",
                                    command=self.analyze,
                                    w=834, h=42, fill=GREEN)
        self.analyze_btn.enable(False)
        self.analyze_btn.pack(padx=P, pady=(0, 12))

        # ── Row 4 : Result card ──────────────────────────────────────────────
        res_card, res_body = self._make_card(body, "ANALYSIS RESULT")
        res_card.pack(fill="x", padx=P, pady=(0, 10))

        self.result_label = tk.Label(
            res_body,
            text="Run an analysis to see the prediction here.",
            font=(FONT, 14, "bold"), bg=CARD, fg=DIMMER, pady=10
        )
        self.result_label.pack()

        # Confidence bars (one per class)
        bars_frame = tk.Frame(res_body, bg=CARD)
        bars_frame.pack(fill="x", padx=16, pady=(4, 12))

        cls_colors = {"benign": AMBER, "malignant": RED, "normal": GREEN}
        cls_list   = sorted(self.class_names or ["benign", "malignant", "normal"])

        for cls in cls_list:
            row = tk.Frame(bars_frame, bg=CARD)
            row.pack(fill="x", pady=4)

            tk.Label(row, text=f"  {cls.capitalize()}",
                     font=(MONO, 10), bg=CARD, fg=DIM,
                     width=12, anchor="w"
                     ).pack(side="left")

            track = tk.Canvas(row, height=16,
                              bg=CARD2, bd=0,
                              highlightthickness=1,
                              highlightbackground=BORDER)
            track.pack(side="left", fill="x", expand=True, padx=6)

            pct = tk.Label(row, text=" 0.0%",
                           font=(MONO, 10), bg=CARD, fg=DIM, width=7)
            pct.pack(side="right")

            self.bar_data[cls] = (track, pct, cls_colors.get(cls, ACCENT))

        # ── Row 5 : Clinical report card ────────────────────────────────────
        rep_card, rep_body = self._make_card(body, "CLINICAL REPORT")
        rep_card.pack(fill="x", padx=P, pady=(0, 16))

        self.report_text = tk.Text(
            rep_body,
            height=8, bg=CARD, fg=DIM,
            font=(MONO, 10), bd=0, relief="flat",
            wrap=tk.WORD, padx=14, pady=10,
            insertbackground=TEXT,
            selectbackground=BORDER
        )
        self.report_text.pack(fill="x", padx=2, pady=(0, 8))
        self.report_text.insert(
            tk.END,
            "  Clinical report will be generated automatically after analysis."
        )
        self.report_text.config(state="disabled")

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg="#1b5e20", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        stxt = ("●  model.pth loaded successfully" if self.model
                else "●  model.pth not found — run train.py first")
        scol = GREEN if self.model else RED

        tk.Label(bar, text=stxt, font=(FONT, 9, "bold"),
                 bg="#1b5e20", fg=scol).pack(side="left",  padx=12, pady=5)
        tk.Label(bar, text="⚠  Research use only — not a clinical tool",
                 font=(FONT, 9, "bold"), bg="#1b5e20", fg="#81c784").pack(side="right", padx=12)

    # ── Widget factories ──────────────────────────────────────────────────────
    def _make_image_panel(self, title):
        """
        Build a styled image panel.
        Returns (outer_frame, canvas).
        The CALLER must pack outer_frame.  The canvas is already packed inside.
        """
        outer = tk.Frame(self.root, bg=BORDER, padx=1, pady=1)

        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True)

        # Title bar
        tk.Label(inner, text=title,
                 font=(FONT, 9, "bold"), bg=CARD, fg=DIMMER,
                 anchor="w", padx=8, pady=5
                 ).pack(fill="x")
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        # The image canvas — explicit width/height so it ALWAYS renders
        canvas = tk.Canvas(inner,
                           width=CANV_W, height=CANV_H,
                           bg=IMGBG, bd=0, highlightthickness=0)
        canvas.pack()        # pack inside inner; outer_frame packed by caller

        return outer, canvas

    def _make_card(self, parent, title):
        """Card with 1 px blue top accent border.  Returns (outer, body_frame)."""
        outer = tk.Frame(parent, bg=ACCENT, pady=1)
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(fill="both", expand=True, padx=1, pady=(0, 1))

        tk.Label(inner, text=title,
                 font=(FONT, 9, "bold"), bg=CARD, fg=DIMMER,
                 padx=14, pady=8, anchor="w"
                 ).pack(fill="x")
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(inner, bg=CARD)
        body.pack(fill="both", expand=True)
        return outer, body

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")

    def _canvas_text(self, canvas, text):
        """Draw centered placeholder text on a canvas."""
        canvas.delete("all")
        canvas.create_text(CANV_W // 2, CANV_H // 2,
                           text=text, fill=DIMMER, font=(FONT, 10))

    def _draw_bar(self, cls, pct):
        """Fill a confidence bar canvas to the given percentage."""
        track, lbl, color = self.bar_data[cls]
        track.update_idletasks()
        track.delete("all")
        w = track.winfo_width() or 500
        fill_w = max(0, int(w * pct / 100))
        if fill_w:
            track.create_rectangle(0, 0, fill_w, 16,
                                    fill=color, outline=color)
        lbl.config(text=f" {pct:4.1f}%")

    # ── Show image on canvas ──────────────────────────────────────────────────
    def _show_on_canvas(self, canvas, pil_image, ref_attr):
        """
        Resize pil_image to fit inside the canvas, draw it centered,
        and store the PhotoImage on self.<ref_attr> to prevent GC.
        """
        img = pil_image.copy()
        img.thumbnail((CANV_W, CANV_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        setattr(self, ref_attr, photo)          # ← keep reference alive!

        canvas.delete("all")
        canvas.create_image(CANV_W // 2, CANV_H // 2,
                             image=photo, anchor="center")

    # ── Actions ───────────────────────────────────────────────────────────────
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Ultrasound Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files",   "*.*")]
        )
        if not path:
            return

        self.image_path = path

        # Display original image on orig_canvas
        pil_img = Image.open(path)
        self._show_on_canvas(self.orig_canvas, pil_img, "_orig_photo")

        # Reset heatmap canvas
        self._canvas_text(self.heat_canvas, "Heatmap will appear after analysis")

        # Reset bars
        for cls in self.bar_data:
            self._draw_bar(cls, 0.0)

        self.path_label.config(text=f"📂  {path}", fg=DIM)
        self.analyze_btn.enable(True)
        self.result_label.config(
            text="Image loaded — click  ⚡ Analyze Image  to run the model.",
            fg=DIM, font=(FONT, 11)
        )

        self.report_text.config(state="normal")
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END,
            "  Ready — report will be generated after analysis.")
        self.report_text.config(state="disabled")

    def analyze(self):
        if not self.model:
            messagebox.showerror(
                "No Model",
                "model.pth was not found.\nRun train.py first to create it."
            )
            return
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        self.result_label.config(
            text="⏳  Analyzing — please wait…",
            fg=AMBER, font=(FONT, 12, "bold")
        )
        self.root.update()

        try:
            pred, conf, probs, cls_names, heatmap_pil = run_model(
                self.model, self.class_names, self.image_path
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Prediction failed:\n{exc}")
            self.result_label.config(text="Analysis failed.", fg=RED)
            return

        # 1. Display Grad-CAM heatmap
        self._show_on_canvas(self.heat_canvas, heatmap_pil, "_heat_photo")

        # 2. Result label
        col_map  = {"normal": GREEN, "benign": AMBER, "malignant": RED}
        icon_map = {"normal": "✔", "benign": "◆", "malignant": "⚠"}
        col  = col_map.get(pred.lower(),  TEXT)
        icon = icon_map.get(pred.lower(), "●")

        self.result_label.config(
            text=f"{icon}   {pred.upper()}   —   {conf:.1f}% Confidence",
            fg=col, font=(FONT, 14, "bold")
        )

        # 3. Confidence bars
        for i, name in enumerate(cls_names):
            if name in self.bar_data:
                self._draw_bar(name, float(probs[i]) * 100)

        # 4. Clinical report
        report = make_report(pred, conf)
        self.report_text.config(state="normal", fg=DIM)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, report)

        self.report_text.tag_configure("hl", foreground=col,
                                        font=(MONO, 10, "bold"))
        hit = self.report_text.search(pred.upper(), "1.0", tk.END)
        if hit:
            self.report_text.tag_add("hl", hit, f"{hit}+{len(pred)}c")

        self.report_text.config(state="disabled")

        # Console
        print(f"\n[RESULT]  {self.image_path}")
        print(f"          Prediction : {pred.upper()}  ({conf:.1f}%)")
        for i, name in enumerate(cls_names):
            print(f"          {name:<12}: {float(probs[i])*100:.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = UltrasoundApp(root)
    root.mainloop()
