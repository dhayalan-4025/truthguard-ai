// ---------- panel switching ----------
const picker = document.getElementById('picker');
const panels = document.querySelectorAll('.panel');

picker.querySelectorAll('.glass-card').forEach(card => {
  card.addEventListener('click', () => {
    const targetId = card.dataset.target;
    picker.hidden = true;
    panels.forEach(p => p.hidden = (p.id !== targetId));
  });
});

document.querySelectorAll('[data-back]').forEach(btn => {
  btn.addEventListener('click', () => {
    panels.forEach(p => p.hidden = true);
    picker.hidden = false;
  });
});

// ---------- helpers ----------
function labelClass(label) {
  if (label === 'suspicious') return 'suspicious';
  if (label === 'caution') return 'caution';
  return 'normal';
}

function labelText(label) {
  if (label === 'suspicious') return '🚨 Suspicious';
  if (label === 'caution') return '⚠️ Use caution';
  return '✅ Looks normal';
}

function showError(container, message) {
  container.hidden = false;
  container.innerHTML = `<p class="result-error">${message}</p>`;
}

// ---------- TEXT CHECK ----------
const textBtn = document.getElementById('text-scan-btn');
const textInput = document.getElementById('text-input');
const textResult = document.getElementById('text-result');

textBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) { showError(textResult, 'Please paste some text first.'); return; }

  textBtn.disabled = true;
  textBtn.textContent = 'Scanning...';
  try {
    const res = await fetch('/check-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) { showError(textResult, data.error || 'Something went wrong.'); return; }

    let html = `<span class="result-label ${labelClass(data.label)}">${labelText(data.label)}</span>`;
    html += `<div class="result-confidence">Confidence: ${data.confidence}%</div>`;
    if (data.top_words && data.top_words.length) {
      html += `<div class="result-words">${data.top_words.map(w => `<span class="chip">${w}</span>`).join('')}</div>`;
    }
    textResult.hidden = false;
    textResult.innerHTML = html;
  } catch (err) {
    showError(textResult, 'Could not reach the server. Is app.py running?');
  } finally {
    textBtn.disabled = false;
    textBtn.textContent = 'Scan text';
  }
});

// ---------- IMAGE CHECK ----------
const imageInput = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const dropZoneLabel = document.getElementById('drop-zone-label');
const imagePreview = document.getElementById('image-preview');
const imageBtn = document.getElementById('image-scan-btn');
const imageResult = document.getElementById('image-result');

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    imagePreview.src = e.target.result;
    imagePreview.hidden = false;
    dropZoneLabel.hidden = true;
  };
  reader.readAsDataURL(file);
});

dropZone.addEventListener('dragover', e => e.preventDefault());
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) {
    imageInput.files = e.dataTransfer.files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

imageBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) { showError(imageResult, 'Please choose a screenshot first.'); return; }

  const formData = new FormData();
  formData.append('image', file);

  imageBtn.disabled = true;
  imageBtn.textContent = 'Reading image...';
  try {
    const res = await fetch('/check-image', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) { showError(imageResult, data.error || 'Could not read the image.'); return; }

    let html = `<span class="result-label ${labelClass(data.label)}">${labelText(data.label)}</span>`;
    html += `<div class="result-confidence">Confidence: ${data.confidence}%</div>`;
    if (data.top_words && data.top_words.length) {
      html += `<div class="result-words">${data.top_words.map(w => `<span class="chip">${w}</span>`).join('')}</div>`;
    }
    if (data.extracted_text) {
      html += `<div class="result-extracted"><strong>Extracted text:</strong><br>${data.extracted_text}</div>`;
    }
    imageResult.hidden = false;
    imageResult.innerHTML = html;
  } catch (err) {
    showError(imageResult, 'Could not reach the server. Is app.py running?');
  } finally {
    imageBtn.disabled = false;
    imageBtn.textContent = 'Extract & scan';
  }
});

// ---------- URL CHECK ----------
const urlBtn = document.getElementById('url-scan-btn');
const urlInput = document.getElementById('url-input');
const urlResult = document.getElementById('url-result');

urlBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) { showError(urlResult, 'Please paste a URL first.'); return; }

  urlBtn.disabled = true;
  urlBtn.textContent = 'Scanning...';
  try {
    const res = await fetch('/check-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (!res.ok) { showError(urlResult, data.error || 'Something went wrong.'); return; }

    let html = `<span class="result-label ${labelClass(data.label)}">${labelText(data.label)}</span>`;
    html += `<div class="result-confidence">Risk score: ${data.confidence}/100 &middot; host: ${data.host}</div>`;
    if (data.reasons && data.reasons.length) {
      html += `<ul class="result-reasons">${data.reasons.map(r => `<li>${r}</li>`).join('')}</ul>`;
    }
    urlResult.hidden = false;
    urlResult.innerHTML = html;
  } catch (err) {
    showError(urlResult, 'Could not reach the server. Is app.py running?');
  } finally {
    urlBtn.disabled = false;
    urlBtn.textContent = 'Scan URL';
  }
});
