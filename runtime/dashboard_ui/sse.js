// sse.js — SSE stream helper (plain global script, no ES modules)

window.SDN = window.SDN || {};

window.SDN.connectStream = function (runId, handlers) {
  handlers = handlers || {};
  var source = new EventSource('/stream?run_id=' + encodeURIComponent(runId));

  source.addEventListener('open',      function ()   { if (handlers.open)      handlers.open(); });
  source.addEventListener('error',     function ()   { if (handlers.error)     handlers.error(); });
  source.addEventListener('heartbeat', function ()   { if (handlers.heartbeat) handlers.heartbeat(); });

  source.addEventListener('snapshot', function (ev) {
    if (handlers.snapshot) handlers.snapshot(JSON.parse(ev.data || '{}'));
  });
  source.addEventListener('state', function (ev) {
    if (handlers.state) handlers.state(JSON.parse(ev.data || '{}'));
  });
  source.addEventListener('alerts', function (ev) {
    if (handlers.alerts) handlers.alerts(JSON.parse(ev.data || '{}'));
  });

  return source;
};
