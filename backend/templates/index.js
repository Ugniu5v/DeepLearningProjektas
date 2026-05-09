const fileInput   = document.getElementById("fileInput");
const preview     = document.getElementById("preview");
const previewHint = document.querySelector(".preview-hint");
const statusEl    = document.getElementById("status");
const placeholder = document.getElementById("placeholder");
const resultsEl   = document.getElementById("results");

function fmt(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function render(predictions) {
  placeholder.style.display = "none";

  const top = predictions[0];
  const rest = predictions.slice(1);

  resultsEl.innerHTML = `
    <div class="top-result">
      <div class="label">Geriausias atitikimas</div>
      <div class="name">${fmt(top.class)}</div>
      <div class="score">${(top.score * 100).toFixed(1)}%</div>
    </div>
    <div class="pred-list">
      ${rest.map(p => `
        <div class="pred-item">
          <div class="pred-row">
            <span>${fmt(p.class)}</span>
            <span class="pred-pct">${(p.score * 100).toFixed(1)}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${(p.score * 100).toFixed(1)}%"></div>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  // Show preview
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.onload = () => URL.revokeObjectURL(url);
  preview.classList.add("visible");
  previewHint.style.display = "none";

  // Classify
  statusEl.textContent = "Klasifikuojama...";
  resultsEl.innerHTML = "";
  placeholder.style.display = "none";

  const fd = new FormData();
  fd.append("image", file);

  try {
    const res = await fetch("/classify", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    render(data.predictions);
    statusEl.textContent = "Atlikta";
  } catch (err) {
    statusEl.textContent = "Klaida: " + err.message;
    placeholder.style.display = "block";
    placeholder.textContent = "Nepavyko klasifikuoti.";
  }
});
