const API_BASE = `${window.location.origin}/api`;

const state = {
  modelCSamples: [],
};

function setActivePanel(targetId) {
  document.querySelectorAll('.nav-chip').forEach((chip) => {
    chip.classList.toggle('is-active', chip.dataset.target === targetId);
  });
  document.querySelectorAll('.model-panel').forEach((panel) => {
    panel.classList.toggle('is-active', panel.id === targetId);
  });
}

function currency(value) {
  const num = Number(value || 0);
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(num);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatConfidence(confidence) {
  if (!confidence || typeof confidence !== 'object') return 'n/a';
  const score = confidence.score !== undefined ? confidence.score.toFixed(2) : 'n/a';
  const mode = confidence.fallback_to_rule_range ? 'fallback' : 'model';
  return `${score} (${mode})`;
}

function createResultBox(label, value) {
  return `<div class="result-box"><span>${label}</span><strong>${value}</strong></div>`;
}

async function request(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: payload ? 'POST' : 'GET',
    headers: payload ? { 'Content-Type': 'application/json' } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || 'Request failed');
  }
  return data;
}

function renderModelA(data) {
  const root = document.getElementById('result-model-a');
  const reasons = (data.confidence?.reasons || []).map((reason) => `<span class="reason-tag">${reason}</span>`).join('');
  root.innerHTML = `
    <div class="result-banner approve">
      <h3>Predicted range: ${currency(data.final_price_range.min_price)} - ${currency(data.final_price_range.max_price)}</h3>
      <p>Model route: <span class="mono">${data.model_route}</span></p>
    </div>
    <div class="result-grid">
      ${createResultBox('Predicted Midpoint', currency(data.predicted_price_mid))}
      ${createResultBox('Confidence', formatConfidence(data.confidence))}
      ${createResultBox('Range Min', currency(data.final_price_range.min_price))}
      ${createResultBox('Range Max', currency(data.final_price_range.max_price))}
    </div>
    ${reasons ? `<div class="reason-tags">${reasons}</div>` : ''}
    <p class="inline-note">Model A is a supervised regression model. It predicts a safe rental range rather than one rigid price, and in the full platform flow lender-entered price must stay inside this range.</p>
  `;
}

function renderModelB(data) {
  const root = document.getElementById('result-model-b');
  const prediction = data.prediction || {};
  const lifecycle = data.lifecycle || {};
  const derived = data.derived_features || {};
  const ageContext = data.age_context || {};
  const decision = prediction.predicted_decision || 'No decision';
  const finalStatus = lifecycle.next_status || prediction.suggested_listing_status || 'No status';
  const bannerClass = String(decision).toLowerCase();
  const popupMessage = lifecycle.frontend_popup_message || '';
  const probabilityBoxes = prediction.class_probabilities
    ? Object.entries(prediction.class_probabilities)
        .map(([label, value]) => createResultBox(`${label} probability`, percent(value)))
        .join('')
    : '';

  root.innerHTML = `
    <div class="result-banner ${bannerClass}">
      <h3>${decision} -> ${finalStatus}</h3>
      <p>Model B is a supervised multi-class classification model with lifecycle rules layered on top of the classifier output.</p>
    </div>
    <div class="result-grid">
      ${createResultBox('Gender Used', `${derived.gender || 'n/a'} (${derived.gender_source || 'n/a'})`)}
      ${createResultBox('Gender Conflict', derived.gender_conflict_flag ? 'Yes' : 'No')}
      ${createResultBox('Rule Quality Score', derived.rule_quality_score ?? 'n/a')}
      ${createResultBox('Rule Decision', derived.rule_decision || 'n/a')}
      ${createResultBox('Listing Age (months)', ageContext.listing_age_months ?? lifecycle.listing_age_months ?? 'n/a')}
      ${createResultBox('Age Source', ageContext.listing_age_source || lifecycle.listing_age_source || 'n/a')}
      ${createResultBox('Review Required', lifecycle.review_required ? 'Yes' : 'No')}
      ${createResultBox('Visible To Renters', lifecycle.visible_to_renters ? 'Yes' : 'No')}
      ${createResultBox('Removal Recommended', lifecycle.removal_recommended ? 'Yes' : 'No')}
      ${createResultBox('Auto Removed', lifecycle.auto_removed ? 'Yes' : 'No')}
      ${probabilityBoxes}
    </div>
    ${popupMessage ? `<p class="inline-note"><strong>Popup guidance:</strong> ${popupMessage}</p>` : ''}
    ${lifecycle.review_reason ? `<p class="inline-note"><strong>Review reason:</strong> ${lifecycle.review_reason}</p>` : ''}
  `;
}

function renderModelC(data) {
  const root = document.getElementById('result-model-c');
  const items = data.items || data.recommendations || [];
  if (!items.length) {
    root.innerHTML = '<p class="result-placeholder">No recommendations were returned for this query.</p>';
    return;
  }

  const cards = items.map((item, index) => {
    const reasons = (item.explanation_reasons || item.reason_tags || item.reasonTags || [])
      .map((tag) => `<span class="reason-tag">${tag}</span>`)
      .join('');
    const title = item.product_name || item.name || item.listing_id || `Recommendation ${index + 1}`;
    return `
      <article class="rec-card">
        <h4>${index + 1}. ${title}</h4>
        <p><strong>ID:</strong> <span class="mono">${item.listing_id || item.id || 'n/a'}</span></p>
        <p><strong>Brand / Category:</strong> ${item.brand || 'n/a'} · ${item.category || 'n/a'}</p>
        <p><strong>Size / Gender:</strong> ${item.size || 'n/a'} · ${item.gender || 'n/a'}</p>
        <p><strong>Price:</strong> ${currency(item.provider_price || item.price || 0)}</p>
        <p><strong>Final Score:</strong> ${item.final_score !== undefined ? Number(item.final_score).toFixed(3) : 'n/a'}</p>
        ${(reasons || item.pool_status || item.recommendation_pool_status) ? `<div class="reason-tags">${reasons}${item.pool_status || item.recommendation_pool_status ? `<span class="reason-tag">${item.pool_status || item.recommendation_pool_status}</span>` : ''}</div>` : ''}
      </article>
    `;
  }).join('');

  root.innerHTML = `
    <div class="result-banner approve">
      <h3>${items.length} recommendation(s) returned</h3>
      <p>${data.query_mode || 'policy-aware recommendation output'} · ${(data.policy_summary && data.policy_summary.budget_source) || 'category_average_budget'}</p>
    </div>
    <div class="result-grid">
      ${createResultBox('Review fallback used', data.policy_summary ? data.policy_summary.review_items_used ?? 0 : 'n/a')}
      ${createResultBox('Budget reference', data.policy_summary && data.policy_summary.budget_reference_price !== undefined ? currency(data.policy_summary.budget_reference_price) : 'n/a')}
      ${createResultBox('Gender rule', data.policy_summary?.query_gender || 'n/a')}
      ${createResultBox('Size rule', data.policy_summary?.query_size || 'n/a')}
    </div>
    <div class="recommendation-list">${cards}</div>
    <p class="inline-note">Model C is a recommendation and ranking system. It retrieves similar candidates first and then re-ranks them with policy-aware signals like size, budget, quality, and safety.</p>
  `;
}

function fillExample(type) {
  if (type === 'model-a-luxury') {
    const form = document.getElementById('form-model-a');
    form.brand.value = 'Anita Dongre';
    form.category.value = 'Lehenga';
    form.material.value = 'Silk';
    form.size.value = 'L';
    form.condition.value = 'New';
    form.age_months.value = '2';
    form.original_price.value = '22000';
  }
  if (type === 'model-b-stale') {
    const form = document.getElementById('form-model-b');
    form.brand.value = 'Biba';
    form.category.value = 'Kurta';
    form.gender.value = 'Women';
    form.material.value = 'Silk';
    form.size.value = 'M';
    form.condition.value = 'Like New';
    form.garment_age_months.value = '8';
    form.original_price.value = '4000';
    form.provider_price.value = '320';
    form.current_status.value = 'ACTIVE';
    form.listing_created_at.value = '2025-01-01';
    form.as_of_date.value = '2026-03-18';
  }
  if (type === 'model-c-budget') {
    const form = document.getElementById('form-model-c');
    document.getElementById('model-c-mode').value = 'seed';
    toggleModelCMode();
    form.seed_item_id.value = 'L0007';
    form.top_k.value = '5';
    form.category_filter.value = 'Dress';
    form.max_provider_price.value = '2000';
  }
}

function toggleModelCMode() {
  const mode = document.getElementById('model-c-mode').value;
  document.getElementById('liked-items-wrap').classList.toggle('hidden', mode !== 'profile');
  document.getElementById('seed-item-id').parentElement.classList.toggle('hidden', mode !== 'seed');
}

async function loadSamples() {
  const wrap = document.getElementById('sample-items');
  try {
    const data = await request('/model-c/samples');
    state.modelCSamples = data.items || [];
    wrap.innerHTML = state.modelCSamples.map((item) => `<button type="button" class="sample-pill" data-sample-id="${item.listing_id}">${item.listing_id} · ${item.brand} · ${item.category}</button>`).join('');
  } catch (error) {
    wrap.textContent = `Could not load samples: ${error.message}`;
  }
}

async function onSubmitModelA(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.age_months = Number(payload.age_months);
  payload.original_price = Number(payload.original_price);
  renderLoading('result-model-a');
  try {
    const data = await request('/predict-price', payload);
    renderModelA(data);
  } catch (error) {
    renderError('result-model-a', error.message);
  }
}

async function onSubmitModelB(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.garment_age_months = Number(payload.garment_age_months);
  payload.original_price = Number(payload.original_price);
  payload.provider_price = Number(payload.provider_price);
  renderLoading('result-model-b');
  try {
    const data = await request('/model-b/predict', payload);
    renderModelB(data);
  } catch (error) {
    renderError('result-model-b', error.message);
  }
}

async function onSubmitModelC(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.top_k = Number(payload.top_k || 5);
  if (!payload.category_filter) delete payload.category_filter;
  if (!payload.max_provider_price) delete payload.max_provider_price;
  else payload.max_provider_price = Number(payload.max_provider_price);

  if (payload.query_mode === 'profile') {
    payload.liked_item_ids = String(payload.liked_item_ids || '').split(',').map((v) => v.trim()).filter(Boolean);
    delete payload.seed_item_id;
  } else {
    delete payload.liked_item_ids;
  }
  delete payload.query_mode;

  renderLoading('result-model-c');
  try {
    const data = await request('/model-c/recommend', payload);
    renderModelC(data);
  } catch (error) {
    renderError('result-model-c', error.message);
  }
}

function renderLoading(targetId) {
  document.getElementById(targetId).innerHTML = '<p class="result-placeholder">Running model...</p>';
}

function renderError(targetId, message) {
  document.getElementById(targetId).innerHTML = `<div class="result-banner reject"><h3>Request failed</h3><p>${message}</p></div>`;
}

document.querySelectorAll('.nav-chip').forEach((chip) => {
  chip.addEventListener('click', () => setActivePanel(chip.dataset.target));
});

document.querySelectorAll('[data-fill]').forEach((button) => {
  button.addEventListener('click', () => fillExample(button.dataset.fill));
});

document.getElementById('form-model-a').addEventListener('submit', onSubmitModelA);
document.getElementById('form-model-b').addEventListener('submit', onSubmitModelB);
document.getElementById('form-model-c').addEventListener('submit', onSubmitModelC);
document.getElementById('model-c-mode').addEventListener('change', toggleModelCMode);

document.getElementById('sample-items').addEventListener('click', (event) => {
  const button = event.target.closest('[data-sample-id]');
  if (!button) return;
  document.querySelector('#form-model-c [name="seed_item_id"]').value = button.dataset.sampleId;
  document.getElementById('model-c-mode').value = 'seed';
  toggleModelCMode();
  setActivePanel('model-c');
});

toggleModelCMode();
loadSamples();
