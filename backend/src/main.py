import io
import os
from pathlib import Path

import torch
import torch.nn as nn
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

ok = load_dotenv(BASE_DIR / ".env", verbose=True)
if not ok:
    raise RuntimeError(f"Failed to load .env file from {BASE_DIR / '.env'}")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/templates", StaticFiles(directory=str(BASE_DIR / "templates")), name="templates")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("ACCESS_TOKEN environment variable is not set.")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

print(f"Loading checkpoint ...")
checkpoint = torch.load(
    BASE_DIR / "checkpoints" / "dinov3_food101_best.pt",
    map_location=DEVICE,
    weights_only=False,
)

processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)
model = AutoModel.from_pretrained(MODEL_ID, token=ACCESS_TOKEN).to(DEVICE)
model.load_state_dict(checkpoint["backbone_state"])
model.eval()

classifier_head = nn.Linear(model.config.hidden_size, len(FOOD101_CLASSES)).to(DEVICE)
classifier_head.load_state_dict(checkpoint["head_state"])
classifier_head.eval()

val_acc_pct = f"{float(checkpoint.get('val_acc', 0)) * 100:.1f}"
print(f"Ready — epoch {checkpoint.get('epoch')}, val_acc {val_acc_pct}%")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"val_acc": val_acc_pct})


@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected image/*.")
    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Unable to decode image file.")
    try:
        inputs = {k: v.to(DEVICE) for k, v in processor(images=pil_image, return_tensors="pt").items()}
        with torch.no_grad():
            cls_token = model(**inputs).last_hidden_state[:, 0, :]
            probs = torch.softmax(classifier_head(cls_token), dim=-1).squeeze(0)
        top = probs.topk(5)
        return {"predictions": [
            {"class": FOOD101_CLASSES[int(idx)], "score": float(score)}
            for score, idx in zip(top.values, top.indices)
        ]}
    except Exception:
        raise HTTPException(status_code=500, detail="Classification failed.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
