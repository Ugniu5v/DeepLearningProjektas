const fileInput = document.getElementById("fileInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const coordsEl = document.getElementById("coords");
const rawEl = document.getElementById("raw");

let currentFile = null;
let imageEl = null;
let selectedPoint = null;
let lastHeatmapPayload = null;

function getColor(v) {
  const value = Math.max(0, Math.min(1, v));
  const r = Math.floor(255 * value);
  const g = Math.floor(180 * (1 - Math.abs(value - 0.5) * 2));
  const b = Math.floor(255 * (1 - value));
  return "rgba(" + r + "," + g + "," + b + ",0.40)";
}

function drawCrosshair(x, y) {
  ctx.save();
  ctx.strokeStyle = "cyan";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x - 12, y);
  ctx.lineTo(x + 12, y);
  ctx.moveTo(x, y - 12);
  ctx.lineTo(x, y + 12);
  ctx.stroke();
  ctx.restore();
}

function paintHeatmapSmooth(payload) {
  const { heatmap, grid_w, grid_h } = payload;
  if (!grid_w || !grid_h) return;

  const lowResCanvas = document.createElement("canvas");
  lowResCanvas.width = grid_w;
  lowResCanvas.height = grid_h;
  const lowResCtx = lowResCanvas.getContext("2d");
  const imageData = lowResCtx.createImageData(grid_w, grid_h);
  const pixels = imageData.data;

  for (let py = 0; py < grid_h; py++) {
    for (let px = 0; px < grid_w; px++) {
      const v = Math.max(0, Math.min(1, Number(heatmap[py][px]) || 0));
      const idx = (py * grid_w + px) * 4;
      pixels[idx] = Math.floor(255 * v);
      pixels[idx + 1] = Math.floor(180 * (1 - Math.abs(v - 0.5) * 2));
      pixels[idx + 2] = Math.floor(255 * (1 - v));
      pixels[idx + 3] = 140;
    }
  }

  lowResCtx.putImageData(imageData, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(lowResCanvas, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function redrawCanvas() {
  if (!imageEl) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imageEl, 0, 0, canvas.width, canvas.height);

  if (lastHeatmapPayload && Array.isArray(lastHeatmapPayload.heatmap)) {
    paintHeatmapSmooth(lastHeatmapPayload);
  }

  if (selectedPoint) {
    drawCrosshair(selectedPoint.x, selectedPoint.y);
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  currentFile = file;
  lastHeatmapPayload = null;
  selectedPoint = null;
  rawEl.textContent = "-";
  statusEl.textContent = "Image loaded. Click on image to analyze.";

  const url = URL.createObjectURL(file);
  imageEl = new Image();
  imageEl.onload = () => {
    const maxWidth = 860;
    const scale = Math.min(1, maxWidth / imageEl.width);
    canvas.width = Math.max(1, Math.round(imageEl.width * scale));
    canvas.height = Math.max(1, Math.round(imageEl.height * scale));
    redrawCanvas();
    URL.revokeObjectURL(url);
  };
  imageEl.src = url;
});

canvas.addEventListener("click", async (event) => {
  if (!currentFile || !imageEl) {
    statusEl.textContent = "Please upload an image first.";
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  selectedPoint = { x, y };
  redrawCanvas();

  const imgX = (x / canvas.width) * imageEl.width;
  const imgY = (y / canvas.height) * imageEl.height;
  coordsEl.textContent = Math.round(imgX) + ", " + Math.round(imgY);
  statusEl.textContent = "Analyzing...";

  try {
    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("x", String(imgX));
    formData.append("y", String(imgY));

    const res = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();
    rawEl.textContent = JSON.stringify(data, null, 2);

    if (!res.ok) {
      statusEl.textContent = "Error " + res.status;
      return;
    }

    lastHeatmapPayload = data;
    redrawCanvas();
    statusEl.textContent = "Done. Click another point to update heatmap.";
  } catch (err) {
    statusEl.textContent = "Request failed";
    rawEl.textContent = String(err);
  }
});
