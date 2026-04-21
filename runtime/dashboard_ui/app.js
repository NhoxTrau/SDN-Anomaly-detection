// app.js — SDN NIDS Dashboard (plain script, no ES modules)
// Depends on: charts.js (window.SDN.*), sse.js (window.SDN.connectStream)

(function () {
  'use strict';

  // ── CONSTANTS ────────────────────────────────────────────────────────────
  var WINDOW_SIZE      = 90;
  var SIDE_REFRESH_MS  = 5000;
  var RUNS_REFRESH_MS  = 12000;
  var HEAVY_REFRESH_MS = 2500;
  var STATIC_REFRESH_MS = 60000;
  var FLOW_FETCH_LIMIT = 120;
  var TOPO_HOST_LIMIT  = 60;
  var TOPO_LINK_LIMIT  = 60;

  var PAGE_TITLES = {
    overview:    'Overview',
    incidents:   'Incidents',
    network:     'Network Map',
    flows:       'Flow Monitor',
    models:      'Model Info',
    scalability: 'Scalability',
    settings:    'Settings',
  };


  // ── STATE ────────────────────────────────────────────────────────────────
  var S = {
    runId:        '',
    latestState:  null,
    alerts:       [],
    incidents:    [],
    flows:        [],
    flowMeta:     { total: 0, returned: 0 },
    attackSources:{},
    topology:     {},    // loaded from /api/topology — dynamic!
    polling:      {},
    latency:      {},
    modelInfo:    {},
    scalability:  {},
    threshold:    0,
    scoreHistory: [],
    thrHistory:   [],
    timeLabels:   [],
    lastLabelMs:  null,
    lastScore:    NaN,
    lastState:    '',
    openIncident: null,
    evtSrc:       null,
    reconnTimer:  null,
  };
  S.inflight = { flows:false, topology:false, side:false, static:false };
  S.lastStaticRefreshAt = 0;

  // ── DOM HELPERS ──────────────────────────────────────────────────────────
  function $(id)        { return document.getElementById(id); }
  function setText(id, v) { var e=$(id); if(e) e.textContent = (v != null ? v : '—'); }
  function setHTML(id, v) { var e=$(id); if(e) e.innerHTML   = v; }

  // ── FORMATTING ───────────────────────────────────────────────────────────
  function fmt(v, d) {
    d = d !== undefined ? d : 4;
    var n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : '—';
  }

  function fmtCompact(v) {
    var n = Number(v);
    if (!Number.isFinite(n)) return '—';
    if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
    if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
    return n.toFixed(1);
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso).slice(11,19) || '—';
    return d.toLocaleTimeString('en-GB', {hour12:false});
  }

  function fmtMs(v) { return Number.isFinite(Number(v)) ? Number(v).toFixed(2)+' ms' : '—'; }

  function fmtUptime(s) {
    s = Number(s);
    if (!s || s < 0) return '—';
    var h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
    return h>0 ? h+'h '+m+'m' : m>0 ? m+'m '+sec+'s' : sec+'s';
  }

  function protoName(p) {
    var n = Number(p);
    if (n===6)  return 'TCP';
    if (n===17) return 'UDP';
    if (n===1)  return 'ICMP';
    return n>=0 ? String(n) : '—';
  }

  function attackLabel(cat) {
    var v = String(cat||'').toLowerCase();
    if (!v || v==='normal')    return 'Anomaly';
    if (v.includes('bfa') || v.includes('brute'))  return 'Brute Force';
    if (v.includes('probe') || v.includes('scan')) return 'Probe / Scan';
    if (v.includes('ddos'))    return 'DDoS';
    if (v.includes('dos'))     return 'DoS';
    if (v.includes('udp'))     return 'UDP Flood';
    if (v.includes('tcp'))     return 'TCP Flood';
    if (v.includes('icmp'))    return 'ICMP Flood';
    if (v.includes('bot'))     return 'Botnet';
    if (v.includes('u2r'))     return 'Priv. Escalation';
    if (v.includes('web'))     return 'Web Attack';
    if (v.includes('flood') || v.includes('traffic')) return 'Traffic Flood';
    return cat || 'Model-Attack';
  }

  function sevBadge(sev) {
    var s   = String(sev||'low').toLowerCase();
    var cls = s==='high' ? 'sev-high' : s==='medium' ? 'sev-medium' : 'sev-low';
    var lbl = s==='high' ? 'High'     : s==='medium' ? 'Medium'     : 'Low';
    return '<span class="sev '+cls+'"><span class="sev-dot"></span>'+lbl+'</span>';
  }

  function scoreColor(score, thr) {
    if (thr <= 0)            return 'var(--blue)';
    if (score >= thr)        return 'var(--red)';
    if (score >= thr * 0.82) return 'var(--yellow)';
    return 'var(--green)';
  }

  function kvRows(obj, keys) {
    var entries = keys
      ? keys.map(function(k){ return [k, obj&&obj[k]]; }).filter(function(p){ return p[1]!=null && p[1]!==''; })
      : Object.entries(obj || {});
    if (!entries.length) return '<div class="empty">No data</div>';
    return entries.map(function(p) {
      var k=p[0], v=p[1];
      var display = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return '<div class="kv-row"><div class="kv-k">'+k+'</div><div class="kv-v">'+display+'</div></div>';
    }).join('');
  }

  // ── CHARTS ───────────────────────────────────────────────────────────────
  var scoreChart = SDN.createScoreChart($('scoreChart'));
  var gaugeChart = SDN.createGaugeChart($('gaugeChart'));

  // ── CLOCK ────────────────────────────────────────────────────────────────
  setInterval(function () {
    setText('topClock', new Date().toLocaleTimeString('en-GB', {hour12:false}));
  }, 1000);


  function currentTab() {
    var active = document.querySelector('.tab-pane.active');
    return active ? String(active.id || '').replace(/^tab-/, '') : 'overview';
  }

  // ── NAVIGATION ───────────────────────────────────────────────────────────
  function setActiveTab(tab) {
    document.querySelectorAll('.nav button').forEach(function(b) {
      b.classList.toggle('active', b.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-pane').forEach(function(p) {
      p.classList.toggle('active', p.id === 'tab-'+tab);
    });
    setText('pageTitle', PAGE_TITLES[tab] || tab);
    if (tab === 'flows')       loadFlows(true);
    if (tab === 'network')     loadTopology(true);
    if (tab === 'scalability') renderScalability();
    if (tab === 'models')      renderModelInfo();
  }

  document.querySelectorAll('.nav button').forEach(function(btn) {
    btn.addEventListener('click', function() { setActiveTab(btn.dataset.tab); });
  });

  var vab = $('viewAllBtn');
  if (vab) vab.addEventListener('click', function() { setActiveTab('incidents'); });
  var irb = $('incidentRefreshBtn');
  if (irb) irb.addEventListener('click', function() { renderIncidents(S.incidents); });
  var frb = $('flowRefreshBtn');
  if (frb) frb.addEventListener('click', loadFlows);

  // ── CONNECTION ───────────────────────────────────────────────────────────
  function setConn(ok, label) {
    var dot = $('connDot');
    dot.className = 'conn-dot' + (ok ? ' live' : '');
    setText('connLabel', label);
  }

  // ── FETCH HELPER ─────────────────────────────────────────────────────────
  function fetchJson(path, cb) {
    var ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function(){ ctrl.abort(); }, 5000) : null;
    fetch(path, { cache: 'no-store', signal: ctrl ? ctrl.signal : undefined })
      .then(function(r){ return r.ok ? r.json() : {}; })
      .then(function(d){ if (timer) clearTimeout(timer); cb(null, d); })
      .catch(function(e){ if (timer) clearTimeout(timer); cb(e, {}); });
  }

  // ── RUNS ─────────────────────────────────────────────────────────────────
  function loadRuns() {
    fetchJson('/api/runs', function(err, data) {
      if (err) return;
      var runs = data.runs || [];
      var sel  = $('runSelect');
      var prev = S.runId;
      sel.innerHTML = runs.map(function(r) {
        return '<option value="'+r+'">'+r+'</option>';
      }).join('');
      if (!S.runId && runs.length) S.runId = runs[0];
      if (S.runId && runs.indexOf(S.runId) >= 0) sel.value = S.runId;
      if (prev && S.runId !== prev) resetView();
      if (S.runId && !S.evtSrc) {
        openStream();
        refreshSideData();
      }
    });
  }

  $('runSelect').addEventListener('change', function(e) {
    S.runId = e.target.value;
    resetView();
    openStream();
    refreshSideData();
  });

  // ── STATUS BANNER ────────────────────────────────────────────────────────
  function applyStatus(p) {
    var raw    = String(p.status || 'NORMAL').toUpperCase();
    var hasInc = ((p.active_signal_count || p.recent_alert_count || 0) > 0);
    var st     = (raw === 'ATTACK' || hasInc) ? 'attack'
               : raw === 'SUSPECT' ? 'suspect' : 'normal';

    $('statusBanner').className = 'status-banner ' + st;

    var icon = $('bannerIcon');
    icon.className   = 'banner-icon ' + st;
    icon.textContent = st === 'attack' ? '⚠' : st === 'suspect' ? '◈' : '✓';

    var lbl = $('bannerLabel');
    lbl.className   = 'banner-label ' + st;
    lbl.textContent = st === 'attack' ? 'ATTACK DETECTED'
                    : st === 'suspect' ? 'SUSPICIOUS ACTIVITY' : 'NORMAL';

    setText('bannerReason', p.reason || (st === 'normal' ? 'All flows within normal parameters' : ''));
    setText('bannerModel', p.model_name);
    setText('bannerTs', fmtTime(p.server_timestamp || p.poll_timestamp));
  }

  // ── KPIs ─────────────────────────────────────────────────────────────────
  function applyKPIs(p) {
    var score  = Number(p.display_score != null ? p.display_score : (p.recent_peak_score != null ? p.recent_peak_score : p.max_score)) || 0;
    var thr    = Number(p.threshold) || 0;
    var lat    = Number(p.avg_latency_ms || p.latency_ms) || 0;
    var seqBuf = Number(p.sequence_buffer_count) || 0;
    var alerts = S.incidents.length;
    S.threshold = thr;

    // Threats
    var kpA = $('kpiAlerts');
    kpA.textContent = alerts;
    kpA.className = 'kpi-value ' + (alerts > 0 ? 'c-red' : 'c-green');
    setText('kpiAlertsSub', alerts > 0 ? alerts+' active incident(s)' : 'no active incidents');
    $('navBadge').textContent = alerts;

    // Score
    var col = scoreColor(score, thr);
    var kpS = $('kpiScore');
    kpS.textContent  = fmt(score, 5);
    kpS.style.color  = col;
    setText('kpiScoreSub', thr > 0
      ? ((score/thr)*100).toFixed(0)+'% of threshold ('+fmt(thr,4)+')'
      : 'vs threshold');

    // Latency
    var kpL = $('kpiLatency');
    kpL.textContent = lat > 0 ? lat.toFixed(2)+' ms' : '—';
    kpL.className   = 'kpi-value '+ (lat>50?'c-red':lat>20?'c-yellow':'c-cyan');

    // Flows
    setText('kpiFlows',    seqBuf > 0 ? seqBuf : (p.total_rows_read ? fmtCompact(p.total_rows_read) : '—'));
    setText('kpiFlowsSub', seqBuf > 0 ? 'active seq buffers' : 'total obs processed');
    setText('topUptime',   p.uptime_s ? 'up '+fmtUptime(p.uptime_s) : '');

    // Gauge
    var g = SDN.updateGauge(gaugeChart, score, thr);
    var gs = $('gaugeScore'); gs.textContent = fmt(score,5); gs.style.color = g.col;
    var gp = $('gaugePill');
    gp.className   = 'pill '+(score>=thr?'pill-red':score>=thr*0.82?'pill-yellow':'pill-green');
    gp.textContent = score>=thr ? 'Alert' : score>=thr*0.82 ? 'High' : 'Normal';
    setText('gaugeThrLabel', 'thr: '+fmt(thr,4));
    setText('gaugeMax',      fmt(g.max,3));

    pushTimeline(score, thr, p.server_timestamp || p.poll_timestamp, p);
  }

  // ── SCORE TIMELINE ────────────────────────────────────────────────────────
  function pushTimeline(score, thr, ts, p) {
    var stepS  = Math.max(Number((p && p.dashboard_timeline_interval_s) || (p && p.poll_interval_s)) || 1.0, 0.25);
    var baseMs = ts ? (new Date(ts).getTime() || Date.now()) : Date.now();
    var stateKey = (p&&p.status||'NORMAL')+'|'+fmt(score,5)+'|'+fmt(thr,4);

    var byTime  = !S.lastLabelMs || (baseMs - S.lastLabelMs >= stepS * 900);
    var byState = stateKey !== S.lastState;
    var byScore = !Number.isFinite(S.lastScore) || Math.abs(score - S.lastScore) >= 0.012;
    if (!byTime && !byState && !byScore) return;

    S.lastLabelMs = baseMs; S.lastState = stateKey; S.lastScore = score;
    var label = new Date(baseMs).toLocaleTimeString('en-GB', {hour12:false});

    if (S.scoreHistory.length >= WINDOW_SIZE) {
      S.scoreHistory.shift(); S.thrHistory.shift(); S.timeLabels.shift();
    }
    S.scoreHistory.push(score); S.thrHistory.push(thr); S.timeLabels.push(label);
    SDN.updateScoreChart(scoreChart, S.timeLabels, S.scoreHistory, S.thrHistory);
  }

  // ── ALERTS / INCIDENTS ────────────────────────────────────────────────────
  function aggregateAlerts(rows) {
    var map = {};
    (rows || []).forEach(function(row) {
      var type  = attackLabel(row.category);
      var src   = row.src_ip  || '—';
      var dst   = row.dst_ip ? row.dst_ip+':'+(row.dst_port||'?') : '—';
      var key   = type+'|'+src+'|'+dst;
      var score = Number(row.anomaly_score) || 0;
      var sev   = String(row.severity || 'low').toLowerCase();
      var ts    = row.poll_timestamp || row.server_timestamp || '';
      if (!map[key]) {
        map[key] = { type:type, src:src, dst:dst, score:0, sev:'low', ts:'', count:0, hits:0, reason:'', raw:row };
      }
      var cur = map[key];
      cur.score = Math.max(cur.score, score);
      cur.hits  = Math.max(cur.hits,  Number(row.hit_count) || 0);
      if (ts > cur.ts) cur.ts = ts;
      cur.count += 1;
      cur.reason = row.reason || cur.reason;
      var so = {high:0, medium:1, low:2};
      if ((so[sev]||2) < (so[cur.sev]||2)) cur.sev = sev;
    });
    return Object.values(map).sort(function(a,b) {
      var so={high:0,medium:1,low:2};
      var d = (so[a.sev]||2) - (so[b.sev]||2);
      return d !== 0 ? d : b.score - a.score;
    });
  }

  function applyAlerts(rows) {
    S.alerts    = Array.isArray(rows) ? rows : [];
    S.incidents = aggregateAlerts(S.alerts);
    renderOverviewAlerts();
    renderSources(S.incidents);
    if ($('tab-incidents').classList.contains('active')) renderIncidents(S.incidents);
  }

  // ── OVERVIEW ─────────────────────────────────────────────────────────────
  function renderOverviewAlerts() {
    var rows = S.incidents.slice(0, 6);
    setText('overviewIncidentPill', S.incidents.length);
    $('navBadge').textContent = S.incidents.length;
    if (!rows.length) {
      setHTML('overviewIncidents', '<tr><td colspan="5"><div class="empty"><div class="empty-icon">🛡</div>No incidents</div></td></tr>');
      return;
    }
    setHTML('overviewIncidents', rows.map(function(r) {
      var col = scoreColor(r.score, S.threshold);
      return '<tr onclick="document.querySelector(\'[data-tab=incidents]\').click()">'
        +'<td style="color:var(--text3);font-size:11px">'+fmtTime(r.ts)+'</td>'
        +'<td>'+sevBadge(r.sev)+'</td>'
        +'<td><span class="ip ip-src">'+r.src+'</span></td>'
        +'<td style="font-size:11px;color:var(--text2)">'+r.type+'</td>'
        +'<td style="font-family:var(--mono);font-size:11px;color:'+col+'">'+r.score.toFixed(5)+'</td>'
        +'</tr>';
    }).join(''));
  }

  function renderSources(incidents) {
    S.attackSources = {};
    incidents.forEach(function(r) {
      if (!r.src || r.src === '—') return;
      if (!S.attackSources[r.src]) S.attackSources[r.src] = {count:0, lastTs:'', type:r.type};
      S.attackSources[r.src].count += r.count;
      if (r.ts > (S.attackSources[r.src].lastTs||'')) S.attackSources[r.src].lastTs = r.ts;
    });
    var entries = Object.entries(S.attackSources).sort(function(a,b){ return b[1].count - a[1].count; });
    setText('sourcePill', entries.length+' IP'+(entries.length!==1?'s':''));

    ['sourceList', 'mapSourceList'].forEach(function(id) {
      var el = $(id); if (!el) return;
      if (!entries.length) {
        el.innerHTML = '<div class="empty"><div class="empty-icon">🔍</div>No active sources</div>';
        return;
      }
      el.innerHTML = entries.slice(0,8).map(function(p) {
        var ip=p[0], info=p[1];
        return '<div class="source-item">'
          +'<div><div class="source-ip">'+ip+'</div>'
          +'<div class="source-meta">'+info.type+' · '+fmtTime(info.lastTs)+'</div></div>'
          +'<span class="source-count">'+info.count+'</span>'
          +'</div>';
      }).join('');
    });
  }

  // ── INCIDENTS PAGE ────────────────────────────────────────────────────────
  function filterIncidents() {
    var q   = (($('incidentSearch')  && $('incidentSearch').value)   || '').toLowerCase();
    var sev = (($('incidentSevFilter')&& $('incidentSevFilter').value)|| '').toLowerCase();
    var typ = (($('incidentTypeFilter')&&$('incidentTypeFilter').value)||'').toLowerCase();
    var filtered = S.incidents.filter(function(r) {
      if (sev && (r.sev||'').toLowerCase() !== sev) return false;
      if (typ && r.type.toLowerCase().indexOf(typ) < 0) return false;
      if (q) {
        var txt = (r.src+' '+r.dst+' '+r.type+' '+r.reason+' '+(r.raw&&r.raw.category||'')).toLowerCase();
        if (txt.indexOf(q) < 0) return false;
      }
      return true;
    });
    renderIncidents(filtered);
  }

  var isr = $('incidentSearch');    if(isr) isr.addEventListener('input', filterIncidents);
  var isf = $('incidentSevFilter'); if(isf) isf.addEventListener('change', filterIncidents);
  var itf = $('incidentTypeFilter');if(itf) itf.addEventListener('change', filterIncidents);

  function renderIncidents(incidents) {
    S.openIncident = null;
    setText('incidentTotalPill', incidents.length+' record'+(incidents.length!==1?'s':''));
    if (!incidents.length) {
      setHTML('incidentBody', '<tr><td colspan="9"><div class="empty"><div class="empty-icon">🛡</div>No incidents match filters</div></td></tr>');
      return;
    }
    var maxScore = Math.max.apply(null, incidents.map(function(r){ return r.score; }).concat([0.001]));
    setHTML('incidentBody', incidents.map(function(r, i) {
      var pct = Math.min((r.score/maxScore)*100, 100).toFixed(0);
      var col = scoreColor(r.score, S.threshold);
      return '<tr id="inc-'+i+'" onclick="window._sdn_toggleInc('+i+')">'
        +'<td style="color:var(--text3);font-size:13px;cursor:pointer">▸</td>'
        +'<td style="font-size:11px;color:var(--text3);white-space:nowrap">'+fmtTime(r.ts)+'</td>'
        +'<td>'+sevBadge(r.sev)+'</td>'
        +'<td><span class="ip ip-src">'+r.src+'</span></td>'
        +'<td><span class="ip ip-dst">'+r.dst+'</span></td>'
        +'<td style="font-size:11px;color:var(--text2)">'+r.type+'</td>'
        +'<td>'
          +'<div class="sbar-wrap">'
            +'<div class="sbar"><div class="sbar-fill" style="width:'+pct+'%;background:'+col+'"></div></div>'
            +'<span class="sbar-txt" style="color:'+col+'">'+r.score.toFixed(5)+'</span>'
          +'</div>'
        +'</td>'
        +'<td style="font-family:var(--mono);font-size:11px;color:var(--yellow);text-align:center">'+(r.hits||r.count)+'</td>'
        +'<td><span class="pill pill-gray" style="font-size:10px">'+((r.raw&&r.raw.decision_source)||'model').slice(0,14)+'</span></td>'
        +'</tr>'
        // Detail expand row
        +'<tr id="inc-det-'+i+'" class="detail-row">'
        +'<td colspan="9" class="detail-cell">'
          +'<div class="detail-grid">'
            +'<div><div class="d-label">Packet Rate</div><div class="d-value">'+fmtCompact((r.raw&&r.raw.packet_rate)||0)+'/s</div></div>'
            +'<div><div class="d-label">Byte Rate</div><div class="d-value">'+fmtCompact((r.raw&&r.raw.byte_rate)||0)+' B/s</div></div>'
            +'<div><div class="d-label">Δ Packets</div><div class="d-value">'+fmtCompact((r.raw&&r.raw.packet_delta)||0)+'</div></div>'
            +'<div><div class="d-label">Δ Bytes</div><div class="d-value">'+fmtCompact((r.raw&&r.raw.byte_delta)||0)+' B</div></div>'
            +'<div><div class="d-label">Protocol</div><div class="d-value">'+protoName(r.raw&&r.raw.protocol)+'</div></div>'
            +'<div><div class="d-label">Score</div><div class="d-value" style="color:'+col+'">'+r.score.toFixed(6)+'</div></div>'
            +'<div><div class="d-label">Threshold</div><div class="d-value">'+fmt((r.raw&&r.raw.threshold)||S.threshold,5)+'</div></div>'
            +'<div><div class="d-label">Model Hit</div><div class="d-value" style="color:'+((r.raw&&r.raw.model_hit)==='True'?'var(--red)':'var(--green)')+'">'+((r.raw&&r.raw.model_hit)||'—')+'</div></div>'
          +'</div>'
          +'<div class="rule-trace">'
            +'<div class="rule-trace-hd">Detection Trace</div>'
            +'<div class="rule-trace-body">'+(r.reason||(r.raw&&r.raw.reason)||'No trace available')+'</div>'
            +((r.raw&&r.raw.support_reasons)?'<div class="rule-trace-body" style="margin-top:6px;opacity:.65">'+String(r.raw.support_reasons).replace(/\|/g,' → ')+'</div>':'')
          +'</div>'
        +'</td>'
        +'</tr>';
    }).join(''));
  }

  window._sdn_toggleInc = function(idx) {
    var row    = $('inc-'+idx);
    var detail = $('inc-det-'+idx);
    if (!row || !detail) return;
    var isOpen = detail.classList.contains('open');
    document.querySelectorAll('.detail-row.open').forEach(function(r){ r.classList.remove('open'); });
    document.querySelectorAll('#incidentBody tr td:first-child').forEach(function(td){
      if (td.textContent === '▾') td.textContent = '▸';
    });
    if (!isOpen) {
      detail.classList.add('open');
      row.querySelector('td:first-child').textContent = '▾';
      S.openIncident = idx;
    } else {
      S.openIncident = null;
    }
  };

  // ── NETWORK MAP (DYNAMIC) ─────────────────────────────────────────────────
  // Topology is loaded from /api/topology → S.topology
  // Switches: S.topology.switches = [{dpid, id, label?}, ...]
  // Hosts:    inferred from active alerts & flows (src_ip/dst_ip)
  // Fallback: show plain card list if no SVG layout available

  function computeTopoLayout(switches, hostIPs, linkPairs) {
    // Auto-layout: switches row at top, hosts at bottom
    // hostIPs: array of ip strings
    // linkPairs: [{src_dpid, dst_ip}, ...] or we just show all hosts under closest switch
    var W = 700, H = 300;
    var swCount = Math.max(switches.length, 1);
    var hCount  = hostIPs.length;

    var nodes = {};

    // Position switches evenly in a row
    switches.forEach(function(sw, i) {
      var x = (i + 1) * W / (swCount + 1);
      nodes[sw.id || sw.dpid] = { type:'switch', label: sw.label || sw.id || 'sw'+sw.dpid, x:x, y:80, sw:sw };
    });

    // Position hosts evenly in bottom row
    hostIPs.forEach(function(ip, i) {
      var x = (i + 1) * W / (hCount + 1);
      nodes['h_'+ip] = { type:'host', label:ip, x:x, y:220, ip:ip };
    });

    return nodes;
  }

  function renderTopology() {
    var svg   = $('topoSvg');
    if (!svg) return;

    var topology = S.topology || {};
    var switches = topology.switches || [];

    // Collect unique IPs from alerts + flows
    var ipSet = {};
    S.alerts.forEach(function(a) {
      if (a.src_ip) ipSet[a.src_ip] = true;
      if (a.dst_ip) ipSet[a.dst_ip] = true;
    });
    S.flows.forEach(function(f) {
      if (f.src_ip) ipSet[f.src_ip] = true;
      if (f.dst_ip) ipSet[f.dst_ip] = true;
    });
    // Also add IPs from topology hosts if provided
    (topology.hosts || []).forEach(function(h) {
      if (h.ip) ipSet[h.ip] = true;
    });
    var hostIPs = Object.keys(ipSet).sort();

    var attackIPs = new Set(Object.keys(S.attackSources));
    var attackDpids = new Set();
    // Mark switches that have alert flows
    S.alerts.forEach(function(a) {
      if (a.dpid != null) attackDpids.add(String(a.dpid));
    });
    (topology.links || []).forEach(function(l) {
      if (attackIPs.has(l.src_ip) || attackIPs.has(l.dst_ip)) {
        if (l.dpid != null) attackDpids.add(String(l.dpid));
      }
    });

    // If no switches and no hosts: show waiting message
    if (!switches.length && !hostIPs.length) {
      var hasState = S.latestState;
      svg.style.display = 'none';
      setHTML('topoPlaceholder', hasState
        ? '<div class="empty"><div class="empty-icon">⬡</div>No topology data from controller yet.<br>Topology populates as switches connect and flows are observed.</div>'
        : '<div class="empty"><div class="empty-icon">⬡</div>Waiting for controller connection…</div>');
      var ph = $('topoPlaceholder');
      if (ph) ph.style.display = '';
      return;
    }

    // Show SVG
    svg.style.display = '';
    var ph2 = $('topoPlaceholder');
    if (ph2) ph2.style.display = 'none';

    var W = 700, SWY = 80, HOSTY = 220;
    var swCount = switches.length || 1;
    var hCount  = hostIPs.length  || 1;

    // Position switches
    var swPos = {};
    switches.forEach(function(sw, i) {
      var id = String(sw.id || sw.dpid || i);
      swPos[id] = {
        x: (i+1) * W/(swCount+1),
        y: SWY,
        label: sw.label || sw.id || ('s'+sw.dpid),
        isAtt: attackDpids.has(String(sw.dpid || sw.id || id)),
      };
    });

    // Position hosts
    var hPos = {};
    hostIPs.forEach(function(ip, i) {
      hPos[ip] = {
        x: (i+1) * W/(hCount+1),
        y: HOSTY,
        isAtt: attackIPs.has(ip),
      };
    });

    // Find nearest switch for each host (by flow links or fallback)
    var hostToSwitch = {};
    (topology.links || []).forEach(function(l) {
      var dpid = String(l.dpid || '');
      if (l.src_ip && swPos[dpid]) hostToSwitch[l.src_ip] = dpid;
      if (l.dst_ip && swPos[dpid]) hostToSwitch[l.dst_ip] = dpid;
    });
    // For hosts without a known switch, assign to nearest switch by x-position
    hostIPs.forEach(function(ip) {
      if (!hostToSwitch[ip] && switches.length > 0) {
        var hx = hPos[ip].x;
        var best = null, bestDist = Infinity;
        Object.keys(swPos).forEach(function(sid) {
          var d = Math.abs(swPos[sid].x - hx);
          if (d < bestDist) { bestDist = d; best = sid; }
        });
        if (best) hostToSwitch[ip] = best;
      }
    });

    var html = '<defs>'
      +'<filter id="glow-b"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
      +'<filter id="glow-r"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
      +'</defs>';

    // Switch–switch links (from topology.links if available, else skip or draw all pairs)
    var swIds = Object.keys(swPos);
    if (switches.length > 1 && (topology.switch_links || []).length > 0) {
      (topology.switch_links || []).forEach(function(lk) {
        var a = swPos[String(lk.a)], b = swPos[String(lk.b)];
        if (!a || !b) return;
        var isAtt = a.isAtt || b.isAtt;
        html += '<line x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y
          +'" stroke="'+(isAtt?'#f05252':'#4f8ef7')+'" stroke-width="2.5" opacity="0.6"/>';
      });
    } else if (switches.length > 1) {
      // Connect consecutive switches as a chain
      for (var si=0; si<swIds.length-1; si++) {
        var a = swPos[swIds[si]], b = swPos[swIds[si+1]];
        var isAtt = a.isAtt || b.isAtt;
        html += '<line x1="'+a.x+'" y1="'+a.y+'" x2="'+b.x+'" y2="'+b.y
          +'" stroke="'+(isAtt?'#f05252':'rgba(79,142,247,.5)')+'" stroke-width="2"/>';
      }
    }

    // Host–switch links
    hostIPs.forEach(function(ip) {
      var h  = hPos[ip];
      var sw = swPos[hostToSwitch[ip] || swIds[0]];
      if (!sw) return;
      var col = h.isAtt ? '#f05252' : 'rgba(99,140,200,.2)';
      html += '<line x1="'+sw.x+'" y1="'+sw.y+'" x2="'+h.x+'" y2="'+h.y
        +'" stroke="'+col+'" stroke-width="1.5"/>';
    });

    // Draw switches
    Object.keys(swPos).forEach(function(sid) {
      var s    = swPos[sid];
      var fill = s.isAtt ? 'rgba(240,82,82,.18)' : 'rgba(99,140,200,.10)';
      var strk = s.isAtt ? '#f05252' : '#4f8ef7';
      var tcol = s.isAtt ? '#fca5a5' : '#93c5fd';
      var filt = s.isAtt ? 'filter="url(#glow-r)"' : 'filter="url(#glow-b)"';
      html += '<g style="cursor:pointer" onclick="window._sdn_topoTip(\''+sid+'\',\'sw\',event)">'
        +'<rect x="'+(s.x-30)+'" y="'+(s.y-18)+'" width="60" height="36" rx="9"'
          +' fill="'+fill+'" stroke="'+strk+'" stroke-width="1.5" '+filt+'/>'
        +'<text x="'+s.x+'" y="'+(s.y+6)+'" text-anchor="middle"'
          +' fill="'+tcol+'" font-size="11" font-family="Inter,sans-serif" font-weight="700">'+s.label+'</text>'
        +'</g>';
    });

    // Draw hosts
    hostIPs.forEach(function(ip) {
      var h    = hPos[ip];
      var fill = h.isAtt ? 'rgba(240,82,82,.16)' : 'rgba(16,200,138,.09)';
      var strk = h.isAtt ? '#f05252' : '#10c88a';
      var tcol = h.isAtt ? '#fca5a5' : '#6ee7b7';
      var icol = h.isAtt ? 'rgba(252,165,165,.65)' : 'rgba(110,231,183,.55)';
      var filt = h.isAtt ? 'filter="url(#glow-r)"' : '';
      // Short label: last octet or hostname alias
      var shortLabel = ip.split('.').pop();
      html += '<g style="cursor:pointer" onclick="window._sdn_topoTip(\''+ip+'\',\'host\',event)">'
        +'<circle cx="'+h.x+'" cy="'+(h.y-4)+'" r="19"'
          +' fill="'+fill+'" stroke="'+strk+'" stroke-width="1.5" '+filt+'/>'
        +'<text x="'+h.x+'" y="'+(h.y-8)+'" text-anchor="middle"'
          +' fill="'+tcol+'" font-size="11" font-family="Inter,sans-serif" font-weight="700">'+shortLabel+'</text>'
        +'<text x="'+h.x+'" y="'+(h.y+4)+'" text-anchor="middle"'
          +' fill="'+icol+'" font-size="7.5" font-family="JetBrains Mono,monospace">'+ip+'</text>'
        +'</g>';
    });

    svg.innerHTML = html;

    // Update topology badge
    var hasAttSw = Object.values(swPos).some(function(s){ return s.isAtt; });
    var hasAttH  = hostIPs.some(function(ip){ return attackIPs.has(ip); });
    var tb = $('topoBadge');
    tb.className   = 'pill '+(hasAttSw||hasAttH?'pill-red':'pill-green');
    tb.textContent = (hasAttSw||hasAttH) ? '⚠ Attack Detected' : '✓ Normal';

    // Switch table
    var polStats = (S.polling&&S.polling.polling_stats) || {};
    setHTML('switchBody', switches.length ? switches.map(function(sw) {
      var sid  = String(sw.id || sw.dpid || '');
      var ps   = polStats[sid] || {};
      var isAt = attackDpids.has(String(sw.dpid||sw.id||''));
      return '<tr>'
        +'<td style="font-weight:600">'+(sw.label||sw.id||'dpid:'+sw.dpid)+'</td>'
        +'<td style="font-family:var(--mono)">'+(ps.polls_sent!=null?ps.polls_sent:'—')+'</td>'
        +'<td style="font-family:var(--mono)">'+(ps.replies_received!=null?ps.replies_received:'—')+'</td>'
        +'<td style="font-family:var(--mono);color:var(--yellow)">'+(ps.timeouts||0)+'</td>'
        +'<td style="font-family:var(--mono)">'+(ps.avg_reply_delay_ms!=null?(+ps.avg_reply_delay_ms).toFixed(1)+' ms':'—')+'</td>'
        +'<td>'+(isAt?'<span class="sev sev-high"><span class="sev-dot"></span>Attack</span>'
                    :'<span class="sev sev-low"><span class="sev-dot"></span>Normal</span>')+'</td>'
        +'</tr>';
    }).join('')
    : '<tr><td colspan="6"><div class="empty">No switch data — waiting for controller</div></td></tr>');
  }

  window._sdn_topoTip = function(id, type, event) {
    var tip = $('topoTip');
    if (!tip) return;
    var content = '';
    if (type === 'sw') {
      var sw = (S.topology.switches||[]).find(function(s){ return String(s.id||s.dpid)===id; });
      content = '<strong style="color:var(--text)">'+(sw ? (sw.label||sw.id||'dpid:'+sw.dpid) : id)+'</strong>';
      content += '<div style="color:var(--text3);font-size:11px;margin-top:4px">DPID: '+(sw&&sw.dpid||id)+'</div>';
    } else {
      var info = S.attackSources[id];
      content = '<strong style="color:var(--text)">'+id+'</strong>';
      content += info
        ? '<div style="color:var(--red);font-size:11px;margin-top:5px">⚠ '+info.count+' alert(s) — '+info.type+'</div>'
        : '<div style="color:var(--green);font-size:11px;margin-top:5px">✓ No active alerts</div>';
    }
    tip.innerHTML = content;
    tip.className = 'topo-tooltip on';
    // Position near click
    var rect = $('topoSvg').getBoundingClientRect();
    tip.style.left = Math.min(event.clientX - rect.left + 14, rect.width - 200) + 'px';
    tip.style.top  = Math.max(event.clientY - rect.top  - 10, 4) + 'px';
    clearTimeout(tip._timer);
    tip._timer = setTimeout(function(){ tip.className='topo-tooltip'; }, 3500);
  };

  // ── FLOW MONITOR ──────────────────────────────────────────────────────────
  function loadFlows(force) {
    if (!S.runId) return;
    if (!force && currentTab() !== 'flows') return;
    if (S.inflight.flows) return;
    S.inflight.flows = true;
    setText('flowLastTs', 'Refreshing…');
    var rid = encodeURIComponent(S.runId);
    fetchJson('/api/flows?run_id='+rid+'&limit='+FLOW_FETCH_LIMIT, function(err, d) {
      S.inflight.flows = false;
      if (err) {
        setText('flowLastTs', 'Refresh failed');
        return;
      }
      S.flows = d.rows || [];
      S.flowMeta = { total: Number(d.total) || S.flows.length, returned: Number(d.returned) || S.flows.length };
      setText('flowLastTs', 'Refreshed '+new Date().toLocaleTimeString());
      if (S.latestState) {
        var qd  = Number(S.latestState.queue_depth||0);
        var qEl = $('psQueue'); qEl.textContent = qd;
        qEl.className = 'poll-stat-val '+(qd>1000?'c-red':qd>200?'c-yellow':'c-green');
        setText('psThroughput', S.latestState.throughput_obs_s ? S.latestState.throughput_obs_s.toFixed(1)+'/s' : '—');
        setText('psPoll',       S.latestState.poll_interval_s  ? S.latestState.poll_interval_s.toFixed(2)+' s'  : '—');
      }
      renderFlows(null);
    });
  }

  function loadTopology(force) {
    if (!S.runId) return;
    if (!force && currentTab() !== 'network') return;
    if (S.inflight.topology) return;
    S.inflight.topology = true;
    var rid = encodeURIComponent(S.runId);
    fetchJson('/api/topology?run_id='+rid+'&host_limit='+TOPO_HOST_LIMIT+'&link_limit='+TOPO_LINK_LIMIT, function(err, d) {
      S.inflight.topology = false;
      if (err) return;
      S.topology = d.topology || {};
      renderTopology();
    });
  }

  function filterFlows() {
    var q     = (($('flowSearch')       && $('flowSearch').value)       || '').toLowerCase();
    var proto = ($('flowProtoFilter')   && $('flowProtoFilter').value)  || '';
    renderFlows(S.flows.filter(function(r) {
      if (proto && String(r.protocol)!==proto) return false;
      if (q) {
        var hay = [r.src_ip||'', r.dst_ip||'', r.entity_key||r.key||'', String(r.dst_port||''), protoName(r.protocol)].join(' ').toLowerCase();
        if (hay.indexOf(q) < 0) return false;
      }
      return true;
    }));
  }

  function renderFlows(data) {
    data = data || S.flows;
    var totalFlows = Number((S.flowMeta && S.flowMeta.total) || data.length || 0);
    var shownFlows = Number((S.flowMeta && S.flowMeta.returned) || data.length || 0);
    var pill = shownFlows < totalFlows ? ('Top '+shownFlows+' / '+totalFlows+' flows') : (data.length+' flow'+(data.length!==1?'s':''));
    setText('flowCountPill', pill);
    if (!data.length) {
      setHTML('flowBody', '<tr><td colspan="10"><div class="empty"><div class="empty-icon">≋</div>'
        +(S.latestState?'No active flows in monitoring window':'No data — ensure controller is running')+'</div></td></tr>');
      return;
    }
    setHTML('flowBody', data.slice(0,200).map(function(r) {
      var isAtt = String(r.status||'').toLowerCase() === 'attack';
      var age   = r.flow_age_s != null ? Number(r.flow_age_s).toFixed(1)+'s' : '—';
      return '<tr>'
        +'<td><span class="ip ip-src">'+(r.src_ip||'—')+'</span></td>'
        +'<td><span class="ip ip-dst">'+(r.dst_ip||'—')+(r.dst_port>=0?':'+r.dst_port:'')+'</span></td>'
        +'<td><span class="pill pill-gray" style="font-size:10px">'+protoName(r.protocol)+'</span></td>'
        +'<td style="font-family:var(--mono);font-size:11px">'+fmtCompact(r.last_packet_rate||r.packet_rate||0)+'/s</td>'
        +'<td style="font-family:var(--mono);font-size:11px">'+fmtCompact(r.last_byte_rate||r.byte_rate||0)+' B/s</td>'
        +'<td style="font-family:var(--mono);font-size:11px">'+fmtCompact(r.packet_delta||0)+'</td>'
        +'<td style="font-family:var(--mono);font-size:11px">'+fmtCompact(r.byte_delta||0)+' B</td>'
        +'<td style="font-family:var(--mono);font-size:11px">'+age+'</td>'
        +'<td style="font-family:var(--mono);font-size:11px;text-align:center;color:var(--cyan)">'+(r.seen_polls!=null?r.seen_polls:'—')+'</td>'
        +'<td>'+(isAtt
          ?'<span class="sev sev-high"><span class="sev-dot"></span>Attack</span>'
          :'<span class="sev sev-low"><span class="sev-dot"></span>Normal</span>')+'</td>'
        +'</tr>';
    }).join(''));
  }

  var fsr = $('flowSearch');       if(fsr) fsr.addEventListener('input', filterFlows);
  var fpf = $('flowProtoFilter');  if(fpf) fpf.addEventListener('change', filterFlows);

  // ── MODEL INFO ────────────────────────────────────────────────────────────
  function renderModelInfo() {
    var info    = S.modelInfo || {};
    var bundle  = info.bundle_info   || {};
    var metrics = info.model_metrics || {};
    setHTML('bundleInfo', kvRows(bundle, ['model_name','task_type','seq_len','feature_scheme','feature_count','bundle_dir']));
    setHTML('valMetrics',  kvRows(metrics.val_metrics  ||{}, ['threshold','precision','recall','f1','roc_auc','pr_auc']) || '<div class="empty">Validation metrics not loaded for this bundle yet</div>');
    setHTML('testMetrics', kvRows(metrics.test_metrics ||{}, ['threshold','precision','recall','f1','roc_auc','pr_auc']) || '<div class="empty">Test metrics not loaded for this bundle yet</div>');
    var bd = (metrics.test_metrics&&metrics.test_metrics.attack_breakdown)
          || (metrics.val_metrics &&metrics.val_metrics.attack_breakdown) || {};
    var bdE = Object.entries(bd);
    setHTML('attackBreakdown', bdE.length
      ? bdE.map(function(p) {
          var atk=p[0], row=p[1];
          return '<tr><td style="font-weight:600">'+atk+'</td>'
            +'<td style="font-family:var(--mono)">'+(row.count!=null?row.count:'—')+'</td>'
            +'<td style="font-family:var(--mono)">'+fmt(row.mean_score,5)+'</td>'
            +'<td style="font-family:var(--mono)">'+fmt(row.recall,4)+'</td>'
            +'<td style="font-family:var(--mono)">'+fmt(row.precision,4)+'</td></tr>';
        }).join('')
      : '<tr><td colspan="5"><div class="empty">No breakdown data</div></td></tr>');
  }

  // ── SCALABILITY ───────────────────────────────────────────────────────────
  function renderScalability() {
    var ps   = (S.polling&&S.polling.polling_stats) || {};
    var ctrl = (S.polling&&S.polling.controller_metrics) || {};
    var rows = Object.entries(ps);
    setHTML('pollingTable', rows.length
      ? rows.map(function(p) {
          var sw=p[0], r=p[1];
          return '<tr><td style="font-weight:600">'+sw+'</td>'
            +'<td style="font-family:var(--mono)">'+(r.polls_sent!=null?r.polls_sent:'—')+'</td>'
            +'<td style="font-family:var(--mono)">'+(r.replies_received!=null?r.replies_received:'—')+'</td>'
            +'<td style="font-family:var(--mono);color:var(--yellow)">'+(r.timeouts||0)+'</td>'
            +'<td style="font-family:var(--mono)">'+fmtMs(r.avg_reply_delay_ms)+'</td>'
            +'<td style="font-family:var(--mono)">'+fmtMs(r.max_reply_delay_ms)+'</td>'
            +'<td style="font-family:var(--mono)">'+fmt(r.avg_flows_per_reply,1)+'</td>'
            +'<td style="font-family:var(--mono)">'+(r.trimmed_flows!=null?r.trimmed_flows:'—')+'</td>'
            +'<td><span class="pill '+(r.pressure_state==='high'?'pill-red':r.pressure_state==='medium'?'pill-yellow':'pill-green')+'">'+(r.pressure_state||'normal')+'</span></td></tr>';
        }).join('')
      : '<tr><td colspan="9"><div class="empty">No polling data yet</div></td></tr>');
    setHTML('latencyBreakdownList', kvRows((S.latency&&S.latency.average)||{}));
    setHTML('controllerMetrics', kvRows({
      raw_stats_queue_depth: ctrl.raw_stats_queue_depth,
      raw_stats_drop_count:  ctrl.raw_stats_drop_count,
      feature_state_count:   ctrl.feature_state_count,
      poll_interval_s:       ctrl.poll_interval_s,
      recommended_poll_interval_s: ctrl.recommended_poll_interval_s,
      pressure_state:        ctrl.pressure_state,
      adaptive_polling:      ctrl.adaptive_polling,
      avg_reply_delay_ms:    ctrl.polling_summary && ctrl.polling_summary.avg_reply_delay_ms,
      timeout_total:         ctrl.polling_summary && ctrl.polling_summary.timeout_total,
      trimmed_flows_total:   ctrl.polling_summary && ctrl.polling_summary.trimmed_flows_total,
    }));
    var rep = S.scalability||{};
    setHTML('scalabilitySummary', rep.runs
      ? kvRows({total_runs:rep.runs.length, recommended_poll_interval_s: rep.recommended_poll_interval_s, best_run: rep.best_run && rep.best_run.run_id, output_path:rep.output_path, generated_at:rep.generated_at})
      : '<div class="empty">No scalability report found</div>');
  }

  // ── SETTINGS ──────────────────────────────────────────────────────────────
  function renderSettings(p) {
    if (!p) return;
    setHTML('configList', kvRows({
      run_id:           p.run_id,
      model_name:       p.model_name,
      task_type:        p.task_type,
      feature_scheme:   p.feature_scheme,
      seq_len:          p.seq_len,
      threshold:        p.threshold!=null?Number(p.threshold).toFixed(6):undefined,
      score_direction:  p.score_direction,
      poll_interval_s:  p.poll_interval_s,
      inference_batch_max: p.inference_batch_max,
      uptime:           fmtUptime(p.uptime_s),
    }));
    setHTML('healthList', kvRows({
      queue_depth:           p.queue_depth,
      csv_queue_depth:       p.csv_queue_depth,
      dropped_rows:          p.dropped_rows,
      csv_drop_count:        p.csv_drop_count,
      stream_drop_count:     p.stream_drop_count,
      total_rows_read:       p.total_rows_read,
      total_inferences:      p.total_inferences,
      avg_latency_ms:        p.avg_latency_ms!=null?Number(p.avg_latency_ms).toFixed(3)+' ms':undefined,
      throughput_obs_s:      p.throughput_obs_s!=null?Number(p.throughput_obs_s).toFixed(1)+' obs/s':undefined,
      sequence_buffer_count: p.sequence_buffer_count,
    }));
    var fp = $('featureSchemePill');
    if (fp) fp.textContent = p.feature_scheme || 'bundle';
    setHTML('featureList', (p.feature_names||[]).map(function(f,i){
      return '<span class="feature-tag">'+i+'. '+f+'</span>';
    }).join(''));
  }

  // ── MAIN STATE ────────────────────────────────────────────────────────────
  function applyState(p) {
    if (!p) return;
    S.latestState = p;
    applyStatus(p);
    applyKPIs(p);
    renderSettings(p);
    if ($('tab-scalability').classList.contains('active')) renderScalability();
  }

  // ── SIDE-DATA (separate API calls) ────────────────────────────────────────
  function refreshSideData() {
    if (!S.runId || S.inflight.side) return;
    S.inflight.side = true;
    var rid = encodeURIComponent(S.runId);
    var reqs = [
      ['/api/polling-stats?run_id='+rid,    function(d){ S.polling   = d || {}; if($('tab-scalability').classList.contains('active')) renderScalability(); }],
      ['/api/latency-breakdown?run_id='+rid,function(d){ S.latency   = d || {}; if($('tab-scalability').classList.contains('active')) renderScalability(); }],
    ];
    var alertPath = null;
    var tab = currentTab();
    if (tab === 'overview' || tab === 'incidents') {
      alertPath = '/api/alerts?run_id='+rid+'&limit=50';
    }
    if (alertPath) reqs.push([alertPath, function(d){ applyAlerts(d.rows || []); }]);
    var pending = reqs.length;
    if (!pending) { S.inflight.side = false; return; }
    reqs.forEach(function(pair) {
      fetchJson(pair[0], function(err, d){
        if (!err) pair[1](d);
        pending -= 1;
        if (pending <= 0) S.inflight.side = false;
      });
    });
    if ((Date.now() - S.lastStaticRefreshAt) >= STATIC_REFRESH_MS && !S.inflight.static) {
      S.inflight.static = true;
      var staticReqs = [
        ['/api/model-info?run_id='+rid, function(d){ S.modelInfo = d || {}; if($('tab-models').classList.contains('active')) renderModelInfo(); }],
        ['/api/scalability-report', function(d){ S.scalability = d || {}; if($('tab-scalability').classList.contains('active')) renderScalability(); }],
      ];
      var staticPending = staticReqs.length;
      staticReqs.forEach(function(pair){
        fetchJson(pair[0], function(err, d){
          if (!err) pair[1](d);
          staticPending -= 1;
          if (staticPending <= 0) {
            S.inflight.static = false;
            S.lastStaticRefreshAt = Date.now();
          }
        });
      });
    }
    if (currentTab() === 'flows') loadFlows();
    if (currentTab() === 'network') loadTopology();
  }

  function refreshActiveHeavyData() {
    var tab = currentTab();
    if (tab === 'flows') loadFlows();
    if (tab === 'network') loadTopology();
  }

  // ── STREAM ────────────────────────────────────────────────────────────────
  function closeStream() {
    if (S.reconnTimer) { clearTimeout(S.reconnTimer); S.reconnTimer = null; }
    if (S.evtSrc)      { S.evtSrc.close();            S.evtSrc = null; }
  }

  function openStream() {
    if (!S.runId) return;
    closeStream();
    setConn(false, 'connecting…');
    S.evtSrc = SDN.connectStream(S.runId, {
      open:     function()  { setConn(true, 'live stream'); },
      error:    function()  { setConn(false,'reconnecting…'); closeStream(); S.reconnTimer = setTimeout(openStream, 2500); },
      snapshot: function(p) { if(p.state) applyState(p.state); if(p.alerts&&p.alerts.rows) applyAlerts(p.alerts.rows); },
      state:    function(p) { applyState(p); },
      alerts:   function(p) { applyAlerts(p.rows||[]); },
    });
  }

  // ── RESET ────────────────────────────────────────────────────────────────
  function resetView() {
    S.alerts=[]; S.incidents=[]; S.flows=[]; S.flowMeta={ total: 0, returned: 0 }; S.attackSources={};
    S.scoreHistory=[]; S.thrHistory=[]; S.timeLabels=[];
    S.lastLabelMs=null; S.lastScore=NaN; S.lastState='';
    S.latestState=null; S.openIncident=null;
    SDN.updateScoreChart(scoreChart, [], [], []);
    applyAlerts([]);
  }

  // ── BOOTSTRAP ────────────────────────────────────────────────────────────
  function bootstrap() {
    loadRuns();
    setTimeout(function() {
      if (S.runId) { openStream(); refreshSideData(); }
      else setConn(false, 'offline');
    }, 400);
    setInterval(refreshSideData, SIDE_REFRESH_MS);
    setInterval(refreshActiveHeavyData, HEAVY_REFRESH_MS);
    setInterval(loadRuns,        RUNS_REFRESH_MS);
  }

  // Start after DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }

})();
