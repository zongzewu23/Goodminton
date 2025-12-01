// Handle predict button click
const submitBtn = document.getElementById('submit-btn');
const videoInput = document.getElementById('video-input');
const modelSelect = document.getElementById('model-select');
const resultEl = document.getElementById('result');

submitBtn.addEventListener('click', async () => {
  // Require a video file
  const file = videoInput.files[0];
  if (!file) {
    alert('Select video');
    return;
  }

  // Show evaluating state (clear stale result)
  resultEl.className = 'result-evaluating';
  resultEl.textContent = 'Evaluating…';

  // Build form data for upload
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', modelSelect.value);

  // Call backend API
  try {
    const res = await fetch('/api/predict-clear', { method: 'POST', body: formData });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Request failed');
    }
    const data = await res.json();

    // Decide display color by label
    const cls = data.label === 'correct' ? 'result-correct' : 'result-incorrect';
    resultEl.className = cls;
    resultEl.textContent = `${data.model.toUpperCase()}: ${data.label.toUpperCase()} (P=${Number(data.prob_correct).toFixed(2)})`;
  } catch (err) {
    resultEl.className = 'result-incorrect';
    resultEl.textContent = `ERROR: ${String(err.message || err)}`;
  }
});