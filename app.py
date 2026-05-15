import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import os
import glob
import cv2
import shutil
import yaml
import torch
import numpy as np
import threading
import queue
import sys
import json
from tkinter import filedialog, messagebox
import time
import datetime

class DummyWriter:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None: sys.stdout = DummyWriter()
if sys.stderr is None: sys.stderr = DummyWriter()

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging

# ==========================================
# CONFIGURAÇÕES E PASTAS LOCAIS
# ==========================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
VERSIONS_DIR = os.path.join(MODEL_DIR, "versions")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

UNLAB = os.path.join(DATASET_DIR, "unlabeled")
TR_IM = os.path.join(DATASET_DIR, "images", "train")
TR_LB = os.path.join(DATASET_DIR, "labels", "train")
VAL_IM = os.path.join(DATASET_DIR, "images", "val")
VAL_LB = os.path.join(DATASET_DIR, "labels", "val")
RV_IM = os.path.join(DATASET_DIR, "images", "review")
RV_META = os.path.join(DATASET_DIR, "review_meta")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
ACCURACY_FILE = os.path.join(BASE_DIR, "accuracy.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

for p in [MODEL_DIR, VERSIONS_DIR, UNLAB, TR_IM, TR_LB, VAL_IM, VAL_LB, RV_IM, RV_META, RUNS_DIR]:
    os.makedirs(p, exist_ok=True)

DATA_YAML = os.path.join(DATASET_DIR, "dataset.yaml")
ds = {
    "path": DATASET_DIR,
    "train": "images/train",
    "val": "images/train",
    "nc": 1,
    "names": {0: "third_molar"}
}
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(ds, f, sort_keys=False)

# ==========================================
# FUNÇÕES AUXILIARES YOLO E HEURÍSTICA
# ==========================================
def check_device(force_cpu=False):
    if force_cpu: return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_latest_model():
    models = glob.glob(os.path.join(VERSIONS_DIR, "*.pt"))
    if not models:
        return os.path.join(MODEL_DIR, "best.pt")
    latest = max(models, key=os.path.getmtime)
    return latest

def load_model(model_path, device="cpu"):
    if not os.path.exists(model_path):
        raise ValueError(f"Modelo não encontrado: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    third_id = 0
    for k, v in model.names.items():
        if str(v).lower().replace("-", "_") in ["third_molar", "terceiro_molar"]:
            third_id = int(k)
            break
    return model, third_id

def _predict_boxes(model, img_bgr, conf, imgsz, max_det=200):
    rs = model(img_bgr[..., ::-1], conf=conf, imgsz=imgsz, max_det=max_det, verbose=False)
    dets = []
    r = rs[0]
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        
        for b, c, s in zip(boxes, classes, scores):
            x1, y1, x2, y2 = b.tolist()
            dets.append((int(c), x1, y1, x2, y2, float(s)))
    return dets

def detect_tiled(model, img_bgr, conf, imgsz, max_det=200, overlap=0.2, tta_manual=False):
    dets_all_scored = _predict_boxes(model, img_bgr, conf, imgsz=640, max_det=max_det)
    
    dets_all_scored = sorted(dets_all_scored, key=lambda d: d[5], reverse=True)
    dets_keep = []
    
    for d in dets_all_scored:
        keep = True
        c, x1, y1, x2, y2, s = d
        area = (x2 - x1) * (y2 - y1)
        for dk in dets_keep:
            kc, kx1, ky1, kx2, ky2, ks = dk
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_k = (kx2 - kx1) * (ky2 - ky1)
            iou = inter / (area + area_k - inter + 1e-9)
            if iou > 0.5 and c == kc:
                keep = False
                break
        if keep:
            dets_keep.append(d)
            
    return dets_all_scored, dets_keep

def draw_boxes(img_bgr, dets_keep, model_names, highlight_cls=None):
    vis = img_bgr.copy()
    for (cls, x1, y1, x2, y2, sc) in dets_keep:
        color = (0, 200, 0) if (highlight_cls is not None and cls == highlight_cls) else (160, 160, 160)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{model_names.get(cls, str(cls))} {sc*100:.0f}%"
        cv2.putText(vis, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis

def is_out_of_quadrant(x_center, W):
    return (0.30 * W) < x_center < (0.70 * W)

# ==========================================
# SALVAMENTO
# ==========================================
def salvar_yolo(img_path, dets, W, H, out_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_txt = os.path.join(out_dir, stem + ".txt")
    with open(out_txt, "w") as f:
        for b in dets:
            cls = b.get('cls', 0)
            x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
            xc = (x1+x2)/2.0 / W
            yc = (y1+y2)/2.0 / H
            w  = (x2-x1) / W
            h  = (y2-y1) / H
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return out_txt

def salvar_vazio(img_path, out_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_txt = os.path.join(out_dir, stem + ".txt")
    with open(out_txt, "w") as f:
        f.write("")
    return out_txt

def mover_para_pasta(img_path, destino):
    dst = os.path.join(destino, os.path.basename(img_path))
    shutil.move(img_path, dst)
    return dst

def get_unlabeled_images():
    return sorted(glob.glob(os.path.join(UNLAB, "*.jpg")) + glob.glob(os.path.join(UNLAB, "*.jpeg")) + glob.glob(os.path.join(UNLAB, "*.png")))

# ==========================================
# REDIRECIONAMENTO DE LOGS
# ==========================================
class StreamRedirector:
    def __init__(self, queue):
        self.queue = queue
    def write(self, text):
        self.queue.put(text)
    def flush(self): pass

# ==========================================
# INTERFACE GRÁFICA (UI) E CANVAS INTERATIVO
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class InteractiveCanvas(tk.Canvas):
    def __init__(self, master, on_boxes_change=None, on_alter=None, on_summary_change=None, **kwargs):
        super().__init__(master, bg="#2b2b2b", highlightthickness=0, **kwargs)
        self.on_boxes_change = on_boxes_change
        self.on_alter = on_alter # Called when user modifies something
        self.on_summary_change = on_summary_change
        
        self.orig_img = None
        self.orig_w = 0
        self.orig_h = 0
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.boxes = [] 
        
        self.action = None # 'draw', 'move', 'resize'
        self.active_box_idx = None
        self.start_x = 0
        self.start_y = 0
        self.curr_x = 0
        self.curr_y = 0
        
        self.bind("<ButtonPress-1>", self.on_press_left)
        self.bind("<B1-Motion>", self.on_drag_left)
        self.bind("<ButtonRelease-1>", self.on_release_left)
        self.bind("<ButtonPress-3>", self.on_press_right)

    def load_image(self, pil_img, initial_boxes=None):
        self.orig_img = pil_img
        self.orig_w, self.orig_h = pil_img.size
        self.boxes = initial_boxes if initial_boxes else []
        self.update_view()

    def update_view(self):
        if not self.orig_img: return
        self.delete("all")
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1: return
        
        self.scale = min(cw / self.orig_w, ch / self.orig_h)
        new_w, new_h = int(self.orig_w * self.scale), int(self.orig_h * self.scale)
        
        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2
        
        img_resized = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)
        
        self.draw_boxes()

    def process_boxes_logic(self):
        if not self.orig_img: return [], {18: False, 28: False, 38: False, 48: False}
        W, H = self.orig_w, self.orig_h
        
        processed = []
        counts_per_quadrant = {1:0, 2:0, 3:0, 4:0}
        
        for b in self.boxes:
            x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            
            is_center = (0.35 * W) < xc < (0.65 * W)
            
            quadrant = None
            if not is_center:
                if xc < W / 2: # Lado esquerdo da imagem (Direita do Paciente)
                    if yc < H / 2: quadrant = 1 # Q1 (18)
                    else: quadrant = 4 # Q4 (48)
                else: # Lado direito da imagem (Esquerda do Paciente)
                    if yc < H / 2: quadrant = 2 # Q2 (28)
                    else: quadrant = 3 # Q3 (38)
                    
            if quadrant:
                counts_per_quadrant[quadrant] += 1
                
            processed.append({
                'box_ref': b,
                'is_center': is_center,
                'quadrant': quadrant
            })
            
        results = []
        summary = {18: False, 28: False, 38: False, 48: False}
        
        for p in processed:
            q = p['quadrant']
            b = p['box_ref']
            
            if p['is_center']:
                text = "Fora do Padrão"
                color = "yellow"
            elif q is not None:
                tooth_map = {1: 18, 2: 28, 3: 38, 4: 48}
                tooth = tooth_map[q]
                
                if counts_per_quadrant[q] > 1:
                    text = f"Dente {tooth} (Supranumerário?)"
                    color = "yellow"
                    summary[tooth] = True
                else:
                    text = f"Dente {tooth}"
                    color = "#00FF00"
                    summary[tooth] = True
            else:
                text = "Desconhecido"
                color = "yellow"
                
            results.append({
                'box': b,
                'text': text,
                'color': color
            })
            
        return results, summary

    def draw_boxes(self):
        self.delete("box")
        processed_boxes, summary = self.process_boxes_logic()
        
        if self.on_summary_change:
            self.on_summary_change(summary)
            
        for i, pb in enumerate(processed_boxes):
            b = pb['box']
            x1, y1 = self.to_canvas(b['x1'], b['y1'])
            x2, y2 = self.to_canvas(b['x2'], b['y2'])
            
            color = pb['color']
            outline_width = 3 if self.active_box_idx == i else 2
            
            self.create_rectangle(x1, y1, x2, y2, outline=color, width=outline_width, tags="box")
            
            # Puxador de redimensionamento
            s = 4
            self.create_rectangle(x2-s, y2-s, x2+s, y2+s, fill="#00A2FF", outline="white", tags="box")
            
            self.create_text(x1, y1-10, text=pb['text'], fill="white", font=("Arial", 10, "bold"), anchor="sw", tags="box")
            
            self.create_text(x1, y1-10, text=pb['text'], fill=color, font=("Arial", 10, "bold"), anchor="sw", tags="box")

    def to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
        
    def to_img(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def find_clicked_box(self, ix, iy):
        for i, b in enumerate(self.boxes):
            if b['x1'] <= ix <= b['x2'] and b['y1'] <= iy <= b['y2']:
                return i
        return None

    def on_press_left(self, event):
        if not self.orig_img: return
        ix, iy = self.to_img(event.x, event.y)
        
        handle_size_img = 12 / self.scale
        for i, b in enumerate(self.boxes):
            if abs(ix - b['x2']) <= handle_size_img and abs(iy - b['y2']) <= handle_size_img:
                self.action = 'resize'
                self.active_box_idx = i
                return
        
        clicked_idx = self.find_clicked_box(ix, iy)
        if clicked_idx is not None:
            self.action = 'move'
            self.active_box_idx = clicked_idx
            self.start_x, self.start_y = ix, iy
        else:
            self.action = 'draw'
            self.active_box_idx = None
            self.start_x, self.start_y = ix, iy
            self.curr_x, self.curr_y = ix, iy

    def on_drag_left(self, event):
        if not self.orig_img or not self.action: return
        ix, iy = self.to_img(event.x, event.y)
        
        if self.action == 'move':
            dx, dy = ix - self.start_x, iy - self.start_y
            b = self.boxes[self.active_box_idx]
            b['x1'] += dx; b['y1'] += dy
            b['x2'] += dx; b['y2'] += dy
            self.start_x, self.start_y = ix, iy
            self.draw_boxes()
            if self.on_alter: self.on_alter()
            
        elif self.action == 'resize':
            b = self.boxes[self.active_box_idx]
            b['x2'] = max(b['x1'] + 10, ix)
            b['y2'] = max(b['y1'] + 10, iy)
            self.draw_boxes()
            if self.on_alter: self.on_alter()
            
        elif self.action == 'draw':
            self.curr_x, self.curr_y = ix, iy
            self.draw_boxes()
            cx1, cy1 = self.to_canvas(self.start_x, self.start_y)
            cx2, cy2 = self.to_canvas(self.curr_x, self.curr_y)
            self.create_rectangle(cx1, cy1, cx2, cy2, outline="#00A2FF", width=2, dash=(4,4), tags="box")

    def on_release_left(self, event):
        if not self.orig_img or not self.action: return
        if self.action == 'draw':
            ix, iy = self.to_img(event.x, event.y)
            x1, x2 = min(self.start_x, ix), max(self.start_x, ix)
            y1, y2 = min(self.start_y, iy), max(self.start_y, iy)
            if x2 - x1 > 5 and y2 - y1 > 5:
                xc = (x1+x2)/2
                warn = is_out_of_quadrant(xc, self.orig_w)
                self.boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cls': 0, 'warn': warn})
                if self.on_alter: self.on_alter()
        
        self.action = None
        self.active_box_idx = None
        self.draw_boxes()
        if self.on_boxes_change: self.on_boxes_change()

    def on_press_right(self, event):
        if not self.orig_img: return
        ix, iy = self.to_img(event.x, event.y)
        clicked_idx = self.find_clicked_box(ix, iy)
        if clicked_idx is not None:
            del self.boxes[clicked_idx]
            self.draw_boxes()
            if self.on_alter: self.on_alter()
            if self.on_boxes_change: self.on_boxes_change()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3molarAI - Sistema de Inteligência v3.0")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        self.settings = self.load_settings()
        self.unlabeled_dir = self.settings.get("unlabeled_dir", UNLAB)
        
        self.log_queue = queue.Queue()
        self.current_review_img_path = None
        self.t1_playlist = []
        self.t1_play_idx = 0
        self.model_path = get_latest_model()
        self.use_gpu_inference = True
        
        # Variáveis de Estado de Assertividade e Undo
        self.user_altered_boxes = False
        self.original_ai_box_count = 0
        self.accuracy = self.load_accuracy()
        self.last_action = None # {'img_path':, 'dest_img':, 'dest_lbl':, 'acc_delta': (c, i)}
        
        self.setup_ui()
        self.update_logs()
        self.update_accuracy_label()
        
        self.bind("<KeyPress>", self.on_key_press)
        
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_settings(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f)

    def load_accuracy(self):
        if os.path.exists(ACCURACY_FILE):
            with open(ACCURACY_FILE, "r") as f:
                return json.load(f)
        return {"correct": 0, "incorrect": 0}
        
    def save_accuracy(self):
        with open(ACCURACY_FILE, "w") as f:
            json.dump(self.accuracy, f)
            
    def update_accuracy_label(self):
        c = self.accuracy["correct"]
        i = self.accuracy["incorrect"]
        total = c + i
        pct = (c / total * 100) if total > 0 else 0
        text = f"Assertividade do Modelo: {pct:.1f}%  ({c} Acertos / {total} Total)"
        self.lbl_accuracy.configure(text=text)

    def register_action_score(self, action_type):
        """Calcula e aplica pontuação, retorna tupla de delta (delta_c, delta_i) para o Undo"""
        dc, di = 0, 0
        if action_type == "save":
            if not self.user_altered_boxes:
                dc = 1 # Acertou
            else:
                di = 1 # Errou (usuário alterou)
        elif action_type == "negative":
            if self.original_ai_box_count == 0:
                dc = 1 # Verdadeiro Negativo (IA não achou, user confirmou)
            else:
                di = 1 # Falso Positivo (IA achou, mas era vazio)
                
        self.accuracy["correct"] += dc
        self.accuracy["incorrect"] += di
        self.save_accuracy()
        self.update_accuracy_label()
        return (dc, di)

    def setup_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        self.tab1 = self.tabview.add("🔍 Detecção")
        self.tab2 = self.tabview.add("✍️ Revisão Interativa")
        self.tab3 = self.tabview.add("⚙️ Treinamento")
        self.tab4 = self.tabview.add("🛠️ Configurações")
        
        self.build_tab_simple()
        self.build_tab_review()
        self.build_tab_train()
        self.build_tab_config()
        
        footer = ctk.CTkLabel(self, text="Desenvolvido por Thiago José Domingues de Andrade", font=("Arial", 11, "italic"), text_color="gray")
        footer.pack(side="bottom", pady=5)

    # --- ABA 1: DETECÇÃO ---
    def build_tab_simple(self):
        self.tab1.grid_columnconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(1, weight=1)
        
        control_frame = ctk.CTkFrame(self.tab1)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkButton(control_frame, text="Carregar Imagens", command=self.t1_upload).pack(side="left", padx=5)
        
        self.btn_t1_prev = ctk.CTkButton(control_frame, text="⬅️", width=40, command=self.t1_prev, state="disabled")
        self.btn_t1_prev.pack(side="left", padx=2)
        
        self.lbl_t1_counter = ctk.CTkLabel(control_frame, text="0/0")
        self.lbl_t1_counter.pack(side="left", padx=2)
        
        self.btn_t1_next = ctk.CTkButton(control_frame, text="➡️", width=40, command=self.t1_next, state="disabled")
        self.btn_t1_next.pack(side="left", padx=2)
        
        self.t1_conf = ctk.CTkSlider(control_frame, from_=1, to=100, number_of_steps=99, width=150, command=self.t1_conf_update)
        self.t1_conf.set(15)
        self.t1_conf.pack(side="left", padx=10)
        self.t1_conf_lbl = ctk.CTkLabel(control_frame, text="Confiança: 15%")
        self.t1_conf_lbl.pack(side="left")
        
        ctk.CTkButton(control_frame, text="Analisar Imagem", command=self.t1_analyze, fg_color="green").pack(side="left", padx=20)
        self.t1_status = ctk.CTkLabel(control_frame, text="")
        self.t1_status.pack(side="left", padx=5)
        
        self.t1_summary_lbl = ctk.CTkLabel(control_frame, text="Dentes: 18(❌) 28(❌) 38(❌) 48(❌)", font=("Arial", 12, "bold"))
        self.t1_summary_lbl.pack(side="right", padx=20)
        
        self.t1_canvas = InteractiveCanvas(self.tab1, on_summary_change=self.update_summary_t1)
        self.t1_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.t1_canvas.bind("<Configure>", lambda e: self.t1_canvas.update_view())
        self.t1_current_img_path = None

    def t1_conf_update(self, val):
        self.t1_conf_lbl.configure(text=f"Confiança: {int(val)}%")

    def t1_upload(self):
        paths = filedialog.askopenfilenames(parent=self, filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if paths:
            self.t1_playlist = list(paths)
            self.t1_play_idx = 0
            self.t1_load_current()
            
    def t1_load_current(self):
        if not self.t1_playlist: return
        p = self.t1_playlist[self.t1_play_idx]
        self.t1_current_img_path = p
        pil_img = Image.open(p)
        self.t1_canvas.load_image(pil_img, [])
        self.t1_status.configure(text=f"Carregada: {os.path.basename(p)}")
        self.lbl_t1_counter.configure(text=f"{self.t1_play_idx + 1}/{len(self.t1_playlist)}")
        
        self.btn_t1_prev.configure(state="normal" if self.t1_play_idx > 0 else "disabled")
        self.btn_t1_next.configure(state="normal" if self.t1_play_idx < len(self.t1_playlist) - 1 else "disabled")

    def t1_prev(self):
        if self.t1_play_idx > 0:
            self.t1_play_idx -= 1
            self.t1_load_current()

    def t1_next(self):
        if self.t1_play_idx < len(self.t1_playlist) - 1:
            self.t1_play_idx += 1
            self.t1_load_current()

    def t1_analyze(self):
        if not self.t1_current_img_path: return
        self.t1_status.configure(text="Analisando...")
        self.update()
        try:
            device = check_device(not self.use_gpu_inference)
            model, third_id = load_model(self.model_path, device)
            img_bgr = cv2.imread(self.t1_current_img_path)
            conf = self.t1_conf.get() / 100.0
            
            dets_all, dets_keep = detect_tiled(model, img_bgr, conf, imgsz=640)
            
            W = img_bgr.shape[1]
            boxes_data = []
            for (cls, x1, y1, x2, y2, sc) in dets_keep:
                if cls == third_id:
                    boxes_data.append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cls':cls})
                    
            pil_img = Image.open(self.t1_current_img_path)
            self.t1_canvas.load_image(pil_img, boxes_data)
            self.t1_status.configure(text=f"Concluído! {len(boxes_data)} molares.")
        except Exception as e:
            self.t1_status.configure(text=f"Erro: {str(e)}")

    # --- ABA 2: REVISÃO INTERATIVA ---
    def build_tab_review(self):
        self.tab2.grid_columnconfigure(0, weight=3)
        self.tab2.grid_columnconfigure(1, weight=1)
        self.tab2.grid_rowconfigure(1, weight=1)
        
        # Banner Assertividade
        banner = ctk.CTkFrame(self.tab2, fg_color="#1f538d")
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        self.lbl_accuracy = ctk.CTkLabel(banner, text="", font=("Arial", 14, "bold"), text_color="white")
        self.lbl_accuracy.pack(pady=5)
        
        self.t2_canvas = InteractiveCanvas(self.tab2, on_boxes_change=self.t2_update_counter, on_alter=self.t2_mark_altered, on_summary_change=self.update_summary_t2)
        self.t2_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.t2_canvas.bind("<Configure>", lambda e: self.t2_canvas.update_view())
        
        right_frame = ctk.CTkFrame(self.tab2)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkButton(right_frame, text="Carregar Próxima", command=self.t2_load_next).pack(pady=10, fill="x", padx=10)
        self.t2_status = ctk.CTkLabel(right_frame, text="", text_color="yellow")
        self.t2_status.pack(pady=5)
        
        instrucoes = ("Mouse e Atalhos:\n"
                      "• Clicar Vazio + Arrastar:\n  Nova caixa manual\n"
                      "• Clicar Caixa + Arrastar:\n  Mover\n"
                      "• Puxador Inferior Direito:\n  Redimensionar\n"
                      "• Clique Direito:\n  Excluir")
        ctk.CTkLabel(right_frame, text=instrucoes, justify="left", fg_color="#333333", corner_radius=5).pack(pady=5, padx=10, fill="x", ipady=5)
        
        self.t2_count_lbl = ctk.CTkLabel(right_frame, text="Caixas na tela: 0", font=("Arial", 12, "bold"))
        self.t2_count_lbl.pack(pady=10)
        
        ctk.CTkButton(right_frame, text="[Z] ↩️ Desfazer Última Ação", fg_color="#34495E", hover_color="#2C3E50", command=self.t2_undo).pack(pady=15, fill="x", padx=10)
        
        ctk.CTkLabel(right_frame, text="Atalhos Rápidos:", font=("Arial", 12, "bold")).pack(pady=(10,5), anchor="w", padx=10)
        ctk.CTkButton(right_frame, text="[S] ✅ Salvar", fg_color="green", command=self.t2_save).pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(right_frame, text="[N] ❌ Negativo", fg_color="#C0392B", hover_color="#922B21", command=self.t2_negative).pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(right_frame, text="[I] 🕘 Inconclusivo", fg_color="#D68910", hover_color="#B9770E", command=self.t2_inconclusive).pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(right_frame, text="[P] ⏭️ Pular", fg_color="gray", command=self.t2_skip).pack(pady=5, fill="x", padx=10)

        # Painel de Resumo
        summary_frame = ctk.CTkFrame(right_frame, fg_color="#2b2b2b")
        summary_frame.pack(pady=10, fill="x", padx=10)
        self.t2_summary_lbl = ctk.CTkLabel(summary_frame, text="Dentes: 18(❌) 28(❌)\n38(❌) 48(❌)", font=("Arial", 12, "bold"))
        self.t2_summary_lbl.pack(pady=10)

    def update_summary_t1(self, summary):
        text = "Dentes Encontrados: " + " ".join([f"{k}({'✅' if v else '❌'})" for k, v in summary.items()])
        self.t1_summary_lbl.configure(text=text)

    def update_summary_t2(self, summary):
        text = f"Resumo Quadrantes:\n18({'✅' if summary[18] else '❌'})   28({'✅' if summary[28] else '❌'})\n48({'✅' if summary[48] else '❌'})   38({'✅' if summary[38] else '❌'})"
        self.t2_summary_lbl.configure(text=text)

    def t2_update_counter(self):
        self.t2_count_lbl.configure(text=f"Caixas na tela: {len(self.t2_canvas.boxes)}")
        
    def t2_mark_altered(self):
        self.user_altered_boxes = True

    def get_unlabeled_images(self):
        return sorted(glob.glob(os.path.join(self.unlabeled_dir, "*.jpg")) + 
                      glob.glob(os.path.join(self.unlabeled_dir, "*.jpeg")) + 
                      glob.glob(os.path.join(self.unlabeled_dir, "*.png")))

    def t2_load_next(self):
        imgs = self.get_unlabeled_images()
        if not imgs:
            self.t2_status.configure(text="✅ Nenhuma imagem restante!")
            self.t2_canvas.delete("all")
            self.current_review_img_path = None
            return
            
        self.t2_status.configure(text="Analisando...")
        self.update()
        
        path = imgs[0]
        boxes_data = []
        
        try:
            if os.path.exists(self.model_path):
                device = check_device(not self.use_gpu_inference)
                model, third_id = load_model(self.model_path, device)
                img_bgr = cv2.imread(path)
                if img_bgr is not None:
                    conf = 0.15
                    dets_all, _ = detect_tiled(model, img_bgr, conf, imgsz=640)
                    for d in dets_all:
                        if d[0] == third_id:
                            boxes_data.append({
                                'x1': d[1], 
                                'y1': d[2], 
                                'x2': d[3], 
                                'y2': d[4], 
                                'cls': third_id
                            })
        except Exception as e:
            print(f"Aviso: Não foi possível usar a IA. {e}")
            
        try:
            pil_img = Image.open(path)
            self.current_review_img_path = path
            
            # Reset trackers
            self.original_ai_box_count = len(boxes_data)
            self.user_altered_boxes = False
            
            self.t2_canvas.load_image(pil_img, boxes_data)
            self.t2_update_counter()
            msg = f"Pronto: {os.path.basename(path)}"
            if not os.path.exists(self.model_path):
                msg += " (Modo 100% Manual)"
            self.t2_status.configure(text=msg)
            
        except Exception as e:
            self.t2_status.configure(text=f"Erro ao abrir imagem: {str(e)}")

    def store_undo_state(self, source_img, dest_img, dest_lbl, acc_delta):
        self.last_action = {
            'img_source': source_img,
            'dest_img': dest_img,
            'dest_lbl': dest_lbl,
            'acc_delta': acc_delta
        }

    def t2_undo(self):
        if not self.last_action:
            self.t2_status.configure(text="Nenhuma ação para desfazer.")
            return
            
        try:
            # Reverter pontuação
            dc, di = self.last_action['acc_delta']
            self.accuracy['correct'] = max(0, self.accuracy['correct'] - dc)
            self.accuracy['incorrect'] = max(0, self.accuracy['incorrect'] - di)
            self.save_accuracy()
            self.update_accuracy_label()
            
            # Reverter arquivos
            if os.path.exists(self.last_action['dest_img']):
                shutil.move(self.last_action['dest_img'], self.last_action['img_source'])
            if self.last_action['dest_lbl'] and os.path.exists(self.last_action['dest_lbl']):
                os.remove(self.last_action['dest_lbl'])
                
            self.last_action = None # Limpa para não desfazer 2x
            self.t2_load_next() # Recarrega
            self.t2_status.configure(text="↩️ Última ação desfeita.")
        except Exception as e:
            self.t2_status.configure(text=f"Erro ao desfazer: {str(e)}")

    def _execute_review_action(self, action_name):
        if not self.current_review_img_path: return
        src = self.current_review_img_path
        img_bgr = cv2.imread(src)
        H, W = img_bgr.shape[:2]
        
        dest_img = None
        dest_lbl = None
        acc_delta = (0, 0)
        
        if action_name == "save":
            boxes = self.t2_canvas.boxes
            if not boxes:
                self.t2_status.configure(text="⚠ Adicione dentes para Salvar ou use Negativo.")
                return
            dest_lbl = salvar_yolo(src, boxes, W, H, TR_LB)
            dest_img = mover_para_pasta(src, TR_IM)
            acc_delta = self.register_action_score("save")
            
        elif action_name == "negative":
            dest_lbl = salvar_vazio(src, TR_LB)
            dest_img = mover_para_pasta(src, TR_IM)
            acc_delta = self.register_action_score("negative")
            
        elif action_name == "inconclusive":
            dest_img = mover_para_pasta(src, RV_IM)
            
        elif action_name == "skip":
            dest_img = mover_para_pasta(src, RV_IM)

        self.store_undo_state(src, dest_img, dest_lbl, acc_delta)
        self.t2_load_next()

    def t2_save(self): self._execute_review_action("save")
    def t2_negative(self): self._execute_review_action("negative")
    def t2_inconclusive(self): self._execute_review_action("inconclusive")
    def t2_skip(self): self._execute_review_action("skip")

    # --- ABA 3: TREINAMENTO ---
    def build_tab_train(self):
        self.tab3.grid_columnconfigure(0, weight=1)
        self.tab3.grid_columnconfigure(1, weight=2)
        self.tab3.grid_rowconfigure(0, weight=1)
        
        cfg_frame = ctk.CTkFrame(self.tab3)
        cfg_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(cfg_frame, text="Configurações de Treino", font=("Arial", 16, "bold")).pack(pady=10)
        self.tr_mode = ctk.CTkSegmentedButton(cfg_frame, values=["Incremental", "Do Zero"])
        self.tr_mode.set("Incremental")
        self.tr_mode.pack(pady=10, fill="x", padx=10)
        
        ctk.CTkLabel(cfg_frame, text="Dispositivo:").pack(anchor="w", padx=10)
        self.tr_device = ctk.CTkSegmentedButton(cfg_frame, values=["Auto (GPU)", "Forçar CPU"])
        self.tr_device.set("Auto (GPU)")
        self.tr_device.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(cfg_frame, text="Épocas (Epochs):").pack(anchor="w", padx=10)
        self.tr_epochs = ctk.CTkEntry(cfg_frame)
        self.tr_epochs.insert(0, "30")
        self.tr_epochs.pack(fill="x", padx=10, pady=(0, 10))
        
        self.btn_train = ctk.CTkButton(cfg_frame, text="🚀 Iniciar Treino", fg_color="green", command=self.start_training_thread)
        self.btn_train.pack(pady=20, fill="x", padx=10)
        
        self.tr_progress_lbl = ctk.CTkLabel(cfg_frame, text="Progresso: 0%", font=("Arial", 12))
        self.tr_progress_lbl.pack(pady=(10, 0))
        self.tr_progress = ctk.CTkProgressBar(cfg_frame)
        self.tr_progress.set(0)
        self.tr_progress.pack(pady=5, fill="x", padx=10)
        self.tr_eta_lbl = ctk.CTkLabel(cfg_frame, text="Tempo restante: --", font=("Arial", 12))
        self.tr_eta_lbl.pack(pady=(0, 10))
        
        console_frame = ctk.CTkFrame(self.tab3)
        console_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(console_frame, text="Console Ao Vivo", font=("Arial", 14, "bold")).pack(pady=5)
        self.console = ctk.CTkTextbox(console_frame, font=("Consolas", 12))
        self.console.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start_training_thread(self):
        self.btn_train.configure(state="disabled", text="Treinando...")
        self.tr_progress.set(0)
        self.tr_progress_lbl.configure(text="Progresso: 0%")
        self.tr_eta_lbl.configure(text="Tempo restante: Calculando...")
        self.train_start_time = time.time()
        
        sys.stdout = StreamRedirector(self.log_queue)
        sys.stderr = StreamRedirector(self.log_queue)
        
        for h in LOGGER.handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = sys.stdout
                
        threading.Thread(target=self.run_training, args=(self.tr_mode.get(), int(self.tr_epochs.get()), self.tr_device.get() == "Forçar CPU"), daemon=True).start()

    def run_training(self, mode, epochs, force_cpu):
        try:
            device = check_device(force_cpu)
            if mode == "Incremental":
                model = YOLO(self.model_path)
                lr = 0.005
            else:
                model = YOLO("yolov8s.pt")
                lr = 0.01

            def on_train_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                pct = current / total
                elapsed = time.time() - self.train_start_time
                if current > 0:
                    eta = (elapsed / current) * (total - current)
                    eta_mins = int(eta // 60)
                    eta_secs = int(eta % 60)
                    eta_str = f"{eta_mins}m {eta_secs}s"
                else:
                    eta_str = "Calculando..."
                    
                self.after(0, self._update_progress_ui, pct, current, total, eta_str)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Adicionado workers=0 para evitar o engasgo/lentidão inicial de multiprocessing no Windows
            model.train(data=DATA_YAML, epochs=epochs, imgsz=640, batch=8, device=device, project=RUNS_DIR, name="trainFinal", exist_ok=True, lr0=lr, mosaic=0.7, fliplr=0.5, workers=0)
            
            new_best = os.path.join(RUNS_DIR, "trainFinal", "weights", "best.pt")
            self.after(0, self.finish_training, new_best)
        except Exception as e:
            print(f"\n❌ ERRO: {str(e)}")
            self.after(0, self.finish_training, None)
        finally:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            for h in LOGGER.handlers:
                if isinstance(h, logging.StreamHandler):
                    h.stream = sys.stdout

    def _update_progress_ui(self, pct, current, total, eta_str):
        self.tr_progress.set(pct)
        self.tr_progress_lbl.configure(text=f"Progresso: {int(pct*100)}% ({current}/{total} épocas)")
        self.tr_eta_lbl.configure(text=f"Tempo restante estimado: {eta_str}")

    def finish_training(self, new_best_path):
        self.btn_train.configure(state="normal", text="🚀 Iniciar Treino")
        self.tr_progress.set(1.0)
        self.tr_progress_lbl.configure(text="Progresso: Concluído!")
        self.tr_eta_lbl.configure(text="Tempo restante: 0m 0s")
        if new_best_path and os.path.exists(new_best_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = os.path.join(VERSIONS_DIR, f"modelo_v_{timestamp}.pt")
            shutil.copy2(new_best_path, versioned_path)
            
            self.model_path = versioned_path
            self.lbl_model_path.delete(0, "end")
            self.lbl_model_path.insert(0, versioned_path)
            messagebox.showinfo("Treinamento Concluído", f"Treinamento finalizado com sucesso!\nNovo modelo salvo em:\n{versioned_path}")
        else:
            messagebox.showerror("Erro no Treinamento", "O treinamento falhou ou o modelo não foi salvo corretamente. Verifique os logs.")

    def update_logs(self):
        while not self.log_queue.empty():
            self.console.insert("end", self.log_queue.get())
            self.console.see("end")
        self.after(100, self.update_logs)

    # --- ABA 4: CONFIGURAÇÕES ---
    def build_tab_config(self):
        f = ctk.CTkFrame(self.tab4)
        f.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(f, text="Modelo Ativo:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=5)
        
        self.lbl_model_path = ctk.CTkEntry(row, width=500)
        self.lbl_model_path.insert(0, self.model_path)
        self.lbl_model_path.pack(side="left")
        ctk.CTkButton(row, text="Trocar", command=self.select_model, width=80).pack(side="left", padx=10)
        ctk.CTkButton(row, text="Exportar Modelo Ativo", command=self.export_model, fg_color="#D68910", hover_color="#B9770E", width=150).pack(side="left", padx=10)
        
        ctk.CTkLabel(f, text="Usar GPU na Inferência (Aba 1 e 2)?").pack(anchor="w", padx=10, pady=(20,5))
        self.sw_gpu = ctk.CTkSwitch(f, text="Ligado", command=self.toggle_gpu)
        self.sw_gpu.select()
        self.sw_gpu.pack(anchor="w", padx=10)
        
        ctk.CTkLabel(f, text="Pasta de Imagens para Rotular (Aba 2):", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        row2 = ctk.CTkFrame(f, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)
        
        self.lbl_unlab_path = ctk.CTkEntry(row2, width=500)
        self.lbl_unlab_path.insert(0, self.unlabeled_dir)
        self.lbl_unlab_path.pack(side="left")
        ctk.CTkButton(row2, text="Trocar", command=self.select_unlab_dir, width=80).pack(side="left", padx=10)
        
        ctk.CTkLabel(f, text="Controle de Estatísticas:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(30, 5))
        ctk.CTkButton(f, text="Zerar Assertividade do Modelo", command=self.reset_accuracy, fg_color="#C0392B", hover_color="#922B21", width=250).pack(anchor="w", padx=10)

    def reset_accuracy(self):
        if messagebox.askyesno("Zerar Assertividade", "Tem certeza que deseja zerar a pontuação de assertividade da IA?"):
            self.accuracy = {"correct": 0, "incorrect": 0}
            self.save_accuracy()
            self.update_accuracy_label()
            messagebox.showinfo("Sucesso", "Estatísticas de assertividade zeradas com sucesso!")

    def export_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Erro", "O modelo ativo não foi encontrado!")
            return
            
        default_name = os.path.basename(self.model_path)
        dest = filedialog.asksaveasfilename(parent=self, defaultextension=".pt", filetypes=[("PyTorch Model", "*.pt")], initialfile=default_name, title="Exportar Modelo")
        if dest:
            try:
                shutil.copy2(self.model_path, dest)
                messagebox.showinfo("Sucesso", f"Modelo exportado com sucesso para:\n{dest}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao exportar modelo:\n{e}")

    def select_model(self):
        p = filedialog.askopenfilename(parent=self, filetypes=[("PyTorch Model", "*.pt")])
        if p:
            self.model_path = p
            self.lbl_model_path.delete(0, "end")
            self.lbl_model_path.insert(0, p)
            
    def select_unlab_dir(self):
        d = filedialog.askdirectory(parent=self, title="Selecione a pasta de radiografias novas")
        if d:
            self.unlabeled_dir = d
            self.settings["unlabeled_dir"] = d
            self.lbl_unlab_path.delete(0, "end")
            self.lbl_unlab_path.insert(0, d)
            self.save_settings()
            messagebox.showinfo("Sucesso", f"Pasta atualizada para:\n{d}")
            
    def toggle_gpu(self):
        self.use_gpu_inference = bool(self.sw_gpu.get())

    # --- ATALHOS GLOBAIS ---
    def on_key_press(self, event):
        if isinstance(self.focus_get(), (tk.Entry, ctk.CTkEntry, tk.Text, ctk.CTkTextbox)): return
        k = event.keysym.lower()
        char = event.char.lower() if event.char else ""
        if "Revis" in self.tabview.get():
            if k == 's' or char == 's': self.t2_save()
            elif k == 'n' or char == 'n': self.t2_negative()
            elif k == 'i' or char == 'i': self.t2_inconclusive()
            elif k in ['p', 'space'] or char in ['p', ' ']: self.t2_skip()
            elif k == 'z' or char == 'z': self.t2_undo()

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        import traceback
        with open("crash.txt", "w") as f:
            f.write(traceback.format_exc())
