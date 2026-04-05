var _mode = 'fast';

function setMode(mode) {
  _mode = mode;
  document.getElementById('mode-fast').classList.toggle('active', mode === 'fast');
  document.getElementById('mode-precise').classList.toggle('active', mode === 'precise');
}

document.addEventListener('DOMContentLoaded', function() {
  var d = new Date();
  document.getElementById('today-date').textContent =
    d.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }).toUpperCase();
});

async function handleAnalyze() {
  var text = document.getElementById('news-text').value.trim();
  if (text.length < 10) {
    alert('Please enter at least 10 characters of news text.');
    return;
  }

  hide('error-banner');
  hide('results');
  show('loading-section');
  document.getElementById('analyze-btn').disabled = true;
  startLoadingAnimation();

  try {
    var data = await fetchWithStream(text);
    hide('loading-section');
    renderResults(data);
    show('results');
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    hide('loading-section');
    document.getElementById('error-msg').textContent = err.message;
    show('error-banner');
  } finally {
    stopLoadingAnimation();
    document.getElementById('analyze-btn').disabled = false;
  }
}

function fetchWithStream(text) {
  return new Promise(function(resolve, reject) {
    fetch('/api/analyze/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, mode: _mode }),
    }).then(function(response) {
      if (!response.ok) {
        response.json().catch(function() { return { detail: 'HTTP ' + response.status }; })
          .then(function(err) { reject(new Error(err.detail || 'HTTP ' + response.status)); });
        return;
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      function read() {
        reader.read().then(function(chunk) {
          if (chunk.done) { reject(new Error('Stream ended without result')); return; }

          buffer += decoder.decode(chunk.value, { stream: true });
          var lines = buffer.split('\n');
          buffer = lines.pop();

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line.startsWith('data:')) continue;
            var raw = line.slice(5).trim();
            if (!raw) continue;
            try {
              var event = JSON.parse(raw);
              if (event.type === 'result') { resolve(event.data); return; }
              if (event.type === 'error')  { reject(new Error(event.message)); return; }
              onAgentEvent(event);
            } catch (e) {}
          }
          read();
        }).catch(reject);
      }
      read();
    }).catch(reject);
  });
}
