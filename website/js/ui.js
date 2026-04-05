function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function updateCharCount() {
  var len = document.getElementById('news-text').value.length;
  document.getElementById('char-count').textContent = len.toLocaleString() + ' characters';
}

function showTab(tab) {
  var forEl      = document.getElementById('evidence-for');
  var againstEl  = document.getElementById('evidence-against');
  var tabFor     = document.getElementById('tab-for');
  var tabAgainst = document.getElementById('tab-against');
  if (tab === 'for') {
    forEl.classList.remove('hidden');
    againstEl.classList.add('hidden');
    tabFor.classList.add('active');
    tabAgainst.classList.remove('active');
  } else {
    againstEl.classList.remove('hidden');
    forEl.classList.add('hidden');
    tabAgainst.classList.add('active');
    tabFor.classList.remove('active');
  }
}
