function renderResults(d) {
  var style = VERDICT_STYLES[d.verdict] || VERDICT_STYLES.UNVERIFIABLE;
  var stamp = document.getElementById('verdict-stamp');
  stamp.textContent = style.label;
  stamp.style.background   = style.bg;
  stamp.style.borderColor  = style.border;
  stamp.style.color        = style.text;

  var pct = Math.round(d.confidence * 100);
  document.getElementById('confidence-pct').textContent = pct + '%';
  document.getElementById('confidence-pct').style.color = style.border;
  var fill = document.getElementById('confidence-fill');
  fill.style.background = style.border;
  requestAnimationFrame(function() { fill.style.width = pct + '%'; });

  var badge = document.getElementById('manipulation-badge');
  badge.textContent = d.manipulation_type.replace(/_/g, ' ');
  badge.className = 'manipulation-badge' + (d.manipulation_type !== 'NONE' ? ' flagged' : '');

  document.getElementById('summary-box').textContent = d.summary;

  var claimsList = document.getElementById('claims-list');
  claimsList.innerHTML = '';
  (d.main_claims || []).forEach(function(c, i) {
    var li = document.createElement('li');
    li.innerHTML = '<span class="claim-num">0' + (i + 1) + '</span><span>' + esc(c) + '</span>';
    claimsList.appendChild(li);
  });
  if (!d.main_claims || !d.main_claims.length) {
    claimsList.innerHTML = '<li style="color:var(--text-muted);font-size:.85rem">No specific claims extracted.</li>';
  }

  var dateCard = document.getElementById('date-card');
  if (d.date_context) {
    document.getElementById('date-context').textContent = d.date_context;
  } else {
    dateCard.classList.add('hidden');
  }

  var psCard = document.getElementById('primary-source-card');
  if (d.primary_source) {
    var ps = document.getElementById('primary-source');
    if (d.primary_source.startsWith('http')) {
      ps.innerHTML = '<a href="' + esc(d.primary_source) + '" target="_blank" rel="noopener">' + esc(d.primary_source) + '</a>';
    } else {
      ps.textContent = d.primary_source;
    }
  } else {
    psCard.classList.add('hidden');
  }

  document.getElementById('for-count').textContent     = (d.evidence_for     || []).length;
  document.getElementById('against-count').textContent = (d.evidence_against || []).length;
  document.getElementById('evidence-for').innerHTML     = renderEvidenceList(d.evidence_for     || []);
  document.getElementById('evidence-against').innerHTML = renderEvidenceList(d.evidence_against || []);

  var fcList = document.getElementById('fact-checkers');
  fcList.innerHTML = '';
  (d.fact_checker_results || []).forEach(function(fc) {
    var li = document.createElement('li');
    li.innerHTML = '<span class="checker-dot"></span><span>' + esc(fc) + '</span>';
    fcList.appendChild(li);
  });
  if (!d.fact_checker_results || !d.fact_checker_results.length) {
    fcList.innerHTML = '<li style="color:var(--text-muted);font-size:.85rem">No professional fact-checker results found.</li>';
  }

  var sourcesList = document.getElementById('sources-list');
  var srcs = d.sources_checked || [];
  document.getElementById('sources-count').textContent = srcs.length;
  sourcesList.innerHTML = '';
  srcs.forEach(function(url, i) {
    var li = document.createElement('li');
    var display = url.length > 65 ? url.substring(0, 62) + '…' : url;
    li.innerHTML =
      '<span class="source-num">' + String(i + 1).padStart(2, '0') + '</span>' +
      '<span class="source-url"><a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(display) + '</a></span>';
    sourcesList.appendChild(li);
  });

  document.getElementById('reasoning-text').textContent = d.reasoning || '';
}

function renderEvidenceList(items) {
  if (!items.length) {
    return '<p style="color:var(--text-muted);font-size:.85rem;padding:8px 0">No evidence items found.</p>';
  }
  return items.map(function(item) {
    var sourceHtml = item.url
      ? '<a href="' + esc(item.url) + '" target="_blank" rel="noopener">' + esc(item.source) + '</a>'
      : esc(item.source);
    var credHtml = item.credibility
      ? '<span class="evidence-cred">' + esc(item.credibility) + '</span>'
      : '';
    return (
      '<div class="evidence-item">' +
        '<div class="evidence-item-header">' +
          '<span class="evidence-source">' + sourceHtml + '</span>' +
          credHtml +
        '</div>' +
        '<p class="evidence-summary">' + esc(item.summary) + '</p>' +
      '</div>'
    );
  }).join('');
}
