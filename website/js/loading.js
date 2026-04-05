var _timerInterval = null;
var _elapsed = 0;

function startLoadingAnimation() {
  _elapsed = 0;
  document.getElementById('loading-timer').textContent = '0s';
  setStatus('Thinking');
  clearInterval(_timerInterval);
  _timerInterval = setInterval(function() {
    _elapsed++;
    document.getElementById('loading-timer').textContent = _elapsed + 's';
  }, 1000);
}

function stopLoadingAnimation() {
  clearInterval(_timerInterval);
}

function setStatus(text) {
  var el = document.getElementById('loading-status');
  if (!text) {
    el.classList.add('hidden');
    el.innerHTML = '';
  } else {
    el.innerHTML = text + '<span class="investigating-dots"></span>';
    el.classList.remove('hidden');
  }
}

function onAgentEvent(event) {
  if (event.type === 'thinking') {
    setStatus('Thinking');
  } else if (event.type === 'tool_call') {
    var label = event.label || event.tool;
    setStatus(label + (event.query ? ' — ' + event.query : ''));
  } else if (event.type === 'tool_result') {
    setStatus('Thinking');
  } else if (event.type === 'extracting') {
    setStatus('Structuring verdict');
  }
}
