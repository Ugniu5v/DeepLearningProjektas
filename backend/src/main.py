import io
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/templates", StaticFiles(directory=str(BASE_DIR / "templates")), name="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})



MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()


@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected image/*.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Unable to decode image file.")

    try:
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().tolist()
        return {"embedding": embedding}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to run DINOv3 inference.")


@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...), x: float = Form(...), y: float = Form(...), ):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected image/*.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Unable to decode image file.")

    try:
        inputs = processor(images=pil_image, return_tensors="pt", do_resize=False, do_center_crop=False)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        patch_size = int(getattr(model.config, "patch_size", 16))
        pixel_values = inputs["pixel_values"]
        input_h = int(pixel_values.shape[-2])
        input_w = int(pixel_values.shape[-1])
        grid_h = input_h // patch_size
        grid_w = input_w // patch_size

        num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)
        token_start = 1 + num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, token_start:, :].squeeze(0)

        expected_tokens = grid_h * grid_w
        if expected_tokens <= 0 or patch_tokens.shape[0] < expected_tokens:
            raise HTTPException(status_code=500, detail="Unexpected model output shape.")
        patch_tokens = patch_tokens[:expected_tokens, :]

        patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)

        click_x = min(max(float(x), 0.0), float(max(pil_image.width - 1, 0)))
        click_y = min(max(float(y), 0.0), float(max(pil_image.height - 1, 0)))
        patch_x = min(int((click_x / max(pil_image.width, 1)) * grid_w), grid_w - 1)
        patch_y = min(int((click_y / max(pil_image.height, 1)) * grid_h), grid_h - 1)
        query_index = patch_y * grid_w + patch_x

        query_vec = patch_tokens[query_index]
        similarities = patch_tokens @ query_vec

        min_val = float(similarities.min().item())
        max_val = float(similarities.max().item())
        if max_val - min_val < 1e-8:
            normalized = torch.zeros_like(similarities)
        else:
            normalized = (similarities - min_val) / (max_val - min_val)

        heatmap = normalized.reshape(grid_h, grid_w).cpu().tolist()
        return {
            "heatmap": heatmap,
            "patch_size": patch_size,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "query_patch": {"x": patch_x, "y": patch_y},
            "image_size": {"width": pil_image.width, "height": pil_image.height},
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to run DINOv3 similarity analysis.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)