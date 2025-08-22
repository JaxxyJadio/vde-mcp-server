import base64
import io
import json
import sys
import traceback
from typing import Optional
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vde import VDEPipeline

app = FastAPI(title="VDE MCP Tool", version="0.1.0")

# Initialize pipeline
try:
    pipe = VDEPipeline()
    print("VDE Pipeline initialized successfully", file=sys.stderr)
except Exception as e:
    print(f"Failed to initialize VDE Pipeline: {e}", file=sys.stderr)
    pipe = None

class ParseLayoutIn(BaseModel):
    image_b64: str
    snr_db: float = -4.5
    add_noise: bool = True
    return_edges: bool = True

class ParseLayoutOut(BaseModel):
    layout: dict
    edges_b64: Optional[str] = None
    snr_db: float

def pil_from_b64(s: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        data = base64.b64decode(s)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

def tensor_from_pil(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor"""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def b64_png(arr_uint8_hwc: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string"""
    buf = io.BytesIO()
    Image.fromarray(arr_uint8_hwc).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.post("/tools/parse_layout", response_model=ParseLayoutOut)
def parse_layout(req: ParseLayoutIn):
    """Parse UI layout from image using VDE++"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="VDE Pipeline not initialized")
    
    try:
        # Convert base64 to tensor
        pil = pil_from_b64(req.image_b64)
        tensor_img = tensor_from_pil(pil)
        
        # Run VDE pipeline
        result = pipe.encode(tensor_img, snr_db=req.snr_db, add_noise=req.add_noise)
        
        # Encode edges if requested
        edges_b64 = None
        if req.return_edges and "edges" in result:
            edges = result["edges"]  # Should be numpy array [H,W] uint8
            if len(edges.shape) == 2:
                # Convert grayscale to RGB
                edges_rgb = np.stack([edges, edges, edges], axis=-1)
            else:
                edges_rgb = edges
            edges_b64 = b64_png(edges_rgb)
        
        return ParseLayoutOut(
            layout=result["layout"], 
            edges_b64=edges_b64, 
            snr_db=req.snr_db
        )
    
    except Exception as e:
        print(f"Error in parse_layout: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Layout parsing failed: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": pipe is not None,
        "version": "0.1.0"
    }