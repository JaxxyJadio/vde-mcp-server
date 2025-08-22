#!/usr/bin/env python3
"""
Vision Diffusion Encoder++ (VDE++)
Purpose: parse raster UI mockups into a hierarchical layout tree via
"diffuse → edge-aware encode → vectorize → solve → JSON".
Dependencies: Python 3.9+, PyTorch. Optional: OpenCV for better vectorization.
CLI example:
  python vde.py --image /path/to/mock.png --snr_db -4.5 --out_json layout.json --viz out.png
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2 
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)
def load_image(path: str) -> torch.Tensor:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t
def save_image(path: str, tensor_chw: torch.Tensor) -> None:
    from PIL import Image
    t = tensor_chw.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)

class CosineSchedule:
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        self.s = s
        t = torch.linspace(0, T, T + 1)
        f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
        f = f / f[0]
        self.alpha_bar = f  
        self.snr = self.alpha_bar / (1 - self.alpha_bar + 1e-12)
    def t_from_snr_db(self, snr_db: float) -> int:
        target = 10 ** (snr_db / 10.0)
        idx = torch.argmin(torch.abs(self.snr[1:-1] - target)).item() + 1
        return int(idx)
    def alpha_bar_t(self, t: int) -> float:
        t = int(max(0, min(self.T, t)))
        return float(self.alpha_bar[t].item())

class Pos2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    def forward(self, H: int, W: int, device=None, dtype=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        gy = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        gx = torch.arange(W, device=device, dtype=dtype).view(1, W)
        dim = self.d_model // 4
        freq = torch.exp(torch.arange(0, dim, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, dim)))
        y = gy * freq.view(1, 1, -1)
        x = gx * freq.view(1, 1, -1)
        pe = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1) 
        pe = pe.reshape(H * W, -1)
        if pe.shape[-1] < self.d_model:
            pe = F.pad(pe, (0, self.d_model - pe.shape[-1]))
        else:
            pe = pe[:, : self.d_model]
        return pe[None, :, :]  

class VDE(nn.Module):
    def __init__(self, d_model: int = 512, patch_sizes: Tuple[int, ...] = (8, 16, 32), T: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.schedule = CosineSchedule(T=T)
        self.pos2d = Pos2D(d_model)
        self.proj: nn.ModuleDict = nn.ModuleDict()
        for ps in patch_sizes:
            self.proj[str(ps)] = nn.Linear(3 * ps * ps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=4 * d_model, batch_first=True)
        self.fuse = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ln = nn.LayerNorm(d_model)
        self.edge_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))  
    @torch.no_grad()
    def diffuse(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        alpha_bar = self.schedule.alpha_bar_t(t)
        noise = torch.randn_like(x0)
        return math.sqrt(alpha_bar) * x0 + math.sqrt(max(1e-12, 1 - alpha_bar)) * noise
    def patchify(self, x: torch.Tensor, ps: int) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            H, W = x.shape[-2:]
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)  
        Hn, Wn = x.size(2), x.size(3)
        N = Hn * Wn
        x = x.contiguous().view(B, 3, Hn, Wn, ps * ps).permute(0, 2, 3, 1, 4).reshape(B, N, 3 * ps * ps)
        return x, Hn, Wn
    def forward(self, images: torch.Tensor, snr_db: Optional[float] = -4.5, t: Optional[int] = None,
                add_noise: bool = True, return_edges: bool = True) -> Dict[str, torch.Tensor]:
        B, C, H, W = images.shape
        x0 = (images + 1) * 0.5 if images.min() < 0 else images
        if t is None:
            t = self.schedule.t_from_snr_db(snr_db if snr_db is not None else -4.5)
        xt = self.diffuse(x0, t) if add_noise else x0
        edges = self.edge_head(xt)
        edge_pool = F.avg_pool2d(torch.sigmoid(edges), kernel_size=4, stride=4)  
        tok_list = []
        for ps in self.patch_sizes:
            patches, Hn, Wn = self.patchify(xt, ps)
            tok = self.proj[str(ps)](patches) 
            pos = self.pos2d(Hn, Wn, tok.device, tok.dtype) 
            tok = tok + pos
            t_feat = torch.tensor([
                math.sin(math.pi * t / self.schedule.T),
                math.cos(math.pi * t / self.schedule.T),
                10 ** ((snr_db if snr_db is not None else 0.0) / 10.0)
            ], device=tok.device, dtype=tok.dtype).view(1, 1, 3).expand(B, tok.size(1), 3)
            tok = torch.cat([tok, F.pad(t_feat, (0, self.d_model - 3))], dim=-1)
            tok = self.ln(tok)
            tok_list.append(tok)
        tokens = torch.cat(tok_list, dim=1) 
        tokens = self.fuse(tokens)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = {"tokens": tokens}
        if return_edges:
            out["edges"] = edges
        return out

@dataclass
class Node:
    type: str  
    orient: Optional[str] = None  
    ratio: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    role: Optional[str] = None  
    tabs: Optional[List[str]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  
    def to_json(self) -> Dict:
        d = {"type": self.type}
        if self.type == "split":
            d.update({"orient": self.orient, "ratio": round(self.ratio or 0.5, 4),
                      "left": self.left.to_json() if self.left else None,
                      "right": self.right.to_json() if self.right else None})
        elif self.type == "panel":
            d.update({"role": self.role or "panel"})
        elif self.type == "tabs":
            d.update({"tabs": self.tabs or []})
        return d

class Vectorizer:
    def __init__(self, min_gutter: int = 6, min_region: int = 48, score_thresh: float = 0.65):
        self.min_gutter = min_gutter
        self.min_region = min_region
        self.score_thresh = score_thresh
    def edges_from_tensor(self, edges_logits: torch.Tensor, image: torch.Tensor) -> np.ndarray:
        B, _, H, W = edges_logits.shape
        e = torch.sigmoid(edges_logits[0]).detach().cpu().numpy()[0]
        e = (e - e.min()) / max(1e-6, e.max() - e.min())
        e = (e > 0.4).astype(np.uint8) * 255
        if HAS_CV2:
            img = (image[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            can = cv2.Canny(gray, 50, 150)
            e = cv2.bitwise_or(e, can)
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return e
    def infer_split_tree(self, edges: np.ndarray, bbox: Tuple[int, int, int, int], depth: int = 0) -> Node:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < self.min_region or h < self.min_region or depth > 6:
            return Node(type="panel", role=self._guess_role(bbox), bbox=bbox)
        sub = edges[y0:y1, x0:x1]
        col = sub.sum(axis=0) / 255.0
        row = sub.sum(axis=1) / 255.0
        v_idx, v_score = self._best_split(col, h)
        h_idx, h_score = self._best_split(row, w)
        if max(v_score, h_score) < self.score_thresh:
            return Node(type="panel", role=self._guess_role(bbox), bbox=bbox)
        if v_score >= h_score:  
            split_x = x0 + v_idx
            left_bbox = (x0, y0, split_x, y1)
            right_bbox = (split_x, y0, x1, y1)
            ratio = (split_x - x0) / max(1, w)
            return Node(
                type="split", orient="v", ratio=float(ratio),
                left=self.infer_split_tree(edges, left_bbox, depth + 1),
                right=self.infer_split_tree(edges, right_bbox, depth + 1),
                bbox=bbox,
            )
        else: 
            split_y = y0 + h_idx
            top_bbox = (x0, y0, x1, split_y)
            bot_bbox = (x0, split_y, x1, y1)
            ratio = (split_y - y0) / max(1, h)
            return Node(
                type="split", orient="h", ratio=float(ratio),
                left=self.infer_split_tree(edges, top_bbox, depth + 1),
                right=self.infer_split_tree(edges, bot_bbox, depth + 1),
                bbox=bbox,
            )
    def _best_split(self, proj: np.ndarray, span: int) -> Tuple[int, float]:
        sm = cv2.GaussianBlur(proj.astype(np.float32), (0, 0), 3) if HAS_CV2 else proj
        idx = int(np.argmax(sm))
        score = float(sm[idx] / max(1.0, span))
        return idx, score
    def _guess_role(self, bbox: Tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < 0.22 * 1600:  
            return "project-tree"
        if h < 0.2 * 900:  
            return "console"
        return "editor-or-sidebar"

class VDEPipeline:
    def __init__(self, d_model: int = 512, patch_sizes: Tuple[int, ...] = (8, 16, 32)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = VDE(d_model=d_model, patch_sizes=patch_sizes).to(self.device)
        self.vectorizer = Vectorizer()
    @torch.no_grad()
    def encode(self, image_chw: torch.Tensor, snr_db: float = -4.5, add_noise: bool = True) -> Dict:
        img = image_chw.unsqueeze(0).to(self.device)
        out = self.encoder(img, snr_db=snr_db, add_noise=add_noise, return_edges=True)
        edges = self.vectorizer.edges_from_tensor(out["edges"], img)
        H, W = edges.shape
        root = self.vectorizer.infer_split_tree(edges, (0, 0, W, H), depth=0)
        return {"layout": root.to_json(), "edges": edges}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--snr_db", type=float, default=-4.5)
    ap.add_argument("--no_noise", action="store_true")
    ap.add_argument("--out_json", type=str, default="layout.json")
    ap.add_argument("--viz", type=str, default=None)
    args = ap.parse_args()
    img = load_image(args.image)  
    pipe = VDEPipeline()
    result = pipe.encode(img, snr_db=args.snr_db, add_noise=not args.no_noise)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result["layout"], f, indent=2)
    print(f"wrote {args.out_json}")
    if args.viz is not None:
        from PIL import Image
        edges = result["edges"]
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        Image.fromarray(edges_rgb).save(args.viz)
        print(f"wrote {args.viz}")
if __name__ == "__main__":
    main()
