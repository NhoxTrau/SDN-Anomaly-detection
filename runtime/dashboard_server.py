from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import queue
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dashboard_stream import DashboardEventHub

# ---------------------------------------------------------------------------
# Dashboard HTML — single-file, self-contained
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SDN NIDS — Realtime Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <style>
    :root {
      --bg: #070d1a;
      --surface: #0d1627;
      --surface2: #111e35;
      --border: rgba(99,130,200,.14);
      --border2: rgba(99,130,200,.24);
      --text: #e2eaf8;
      --muted: #6b7fa8;
      --ok: #22d3a0;
      --warn: #f5a623;
      --bad: #f04b4b;
      --info: #38b2f8;
      --purple: #a78bfa;
      --glow-ok: 0 0 18px rgba(34,211,160,.25);
      --glow-bad: 0 0 18px rgba(240,75,75,.3);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { font-size: 14px; }
    body {
      font-family: ui-monospace, "Cascadia Code", "Fira Code", "Consolas", monospace;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ---- Layout ---- */
    .app { max-width: 1560px; margin: 0 auto; padding: 20px 24px; }

    /* ---- Top bar ---- */
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 20px; padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
    }
    .topbar-left { display: flex; align-items: center; gap: 14px; }
    .logo { font-size: 18px; font-weight: 700; letter-spacing: -.02em; color: var(--text); }
    .logo span { color: var(--info); }
    .run-select {
      background: var(--surface2); color: var(--text);
      border: 1px solid var(--border2); border-radius: 8px;
      padding: 6px 12px; font-size: 12px; font-family: inherit;
      outline: none; cursor: pointer;
    }
    .topbar-right { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--muted); }
    .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
      animation: pulse 2s ease-in-out infinite; }
    .pulse.dead { background: var(--muted); animation: none; }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }

    /* ---- Status banner ---- */
    .status-banner {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 20px; border-radius: 14px; margin-bottom: 20px;
      border: 1px solid var(--border); background: var(--surface);
      transition: border-color .3s, box-shadow .3s;
    }
    .status-banner.attack {
      border-color: rgba(240,75,75,.5);
      box-shadow: var(--glow-bad);
      background: rgba(240,75,75,.06);
    }
    .status-banner.normal {
      border-color: rgba(34,211,160,.3);
      box-shadow: var(--glow-ok);
    }
    .status-banner.suspect {
      border-color: rgba(245,166,35,.4);
      box-shadow: 0 0 18px rgba(245,166,35,.18);
      background: rgba(245,166,35,.05);
    }
    .status-left { display: flex; align-items: center; gap: 16px; }
    .status-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
    .status-dot.normal { background: var(--ok); box-shadow: 0 0 10px var(--ok); }
    .status-dot.attack { background: var(--bad); box-shadow: 0 0 12px var(--bad);
      animation: blinkdot 1s steps(1) infinite; }
    .status-dot.idle, .status-dot.warming_up, .status-dot.suspect { background: var(--warn); }
    @keyframes blinkdot { 0%,100%{opacity:1} 50%{opacity:.3} }
    .status-label { font-size: 22px; font-weight: 700; letter-spacing: -.01em; }
    .status-label.normal { color: var(--ok); }
    .status-label.attack { color: var(--bad); }
    .status-label.idle, .status-label.warming_up, .status-label.suspect { color: var(--warn); }
    .status-reason { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .status-right { text-align: right; font-size: 12px; color: var(--muted); }
    .status-right strong { color: var(--text); }

    /* ---- KPI row ---- */
    .kpi-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 20px; }
    .kpi {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px 16px;
    }
    .kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
    .kpi-value { font-size: 26px; font-weight: 700; margin-top: 8px; line-height: 1; }
    .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 5px; }
    .kpi-value.ok { color: var(--ok); }
    .kpi-value.bad { color: var(--bad); }
    .kpi-value.warn { color: var(--warn); }
    .kpi-value.info { color: var(--info); }

    /* ---- Main grid ---- */
    .main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 16px; margin-bottom: 16px; }
    .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

    /* ---- Card ---- */
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 14px; padding: 18px 20px;
    }
    .card-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 16px;
    }
    .card-title { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--info); font-weight: 700; }
    .badge {
      font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
      padding: 3px 10px; border-radius: 999px;
    }
    .badge.ok { background: rgba(34,211,160,.12); color: var(--ok); }
    .badge.bad { background: rgba(240,75,75,.12); color: var(--bad); }
    .badge.warn { background: rgba(245,166,35,.12); color: var(--warn); }
    .badge.info { background: rgba(56,178,248,.12); color: var(--info); }

    /* ---- Score gauge ---- */
    .gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: 12px; }
    #gaugeCanvas { max-width: 240px; }
    .gauge-score { font-size: 36px; font-weight: 800; }
    .gauge-label { font-size: 11px; color: var(--muted); }
    .threshold-row { display: flex; justify-content: space-between; font-size: 11px;
      color: var(--muted); width: 100%; padding: 0 8px; margin-top: 4px; }

    /* ---- Chart containers ---- */
    .chart-wrap { position: relative; height: 200px; }
    .chart-wrap.tall { height: 240px; }

    /* ---- Alerts table ---- */
    .alert-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .alert-table th {
      text-align: left; padding: 7px 10px;
      border-bottom: 1px solid var(--border);
      color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
      font-weight: 500;
    }
    .alert-table td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(99,130,200,.07);
      vertical-align: middle;
    }
    .alert-table tr:last-child td { border-bottom: none; }
    .alert-table tr:hover td { background: rgba(255,255,255,.02); }
    .ip-tag {
      display: inline-block; padding: 2px 8px; border-radius: 6px;
      background: rgba(240,75,75,.12); color: #f9a0a0; font-size: 11px;
    }
    .ip-tag.dst { background: rgba(56,178,248,.1); color: #7dd3fc; }
    .score-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .score-bar { height: 5px; border-radius: 3px; background: var(--border); flex: 1; min-width: 60px; }
    .score-bar-fill { height: 100%; border-radius: 3px; background: var(--bad); transition: width .3s; }
    .empty-state { text-align: center; color: var(--muted); padding: 32px 0; font-size: 12px; }

    /* ---- Model info table ---- */
    .info-table { width: 100%; font-size: 12px; border-collapse: collapse; }
    .info-table td { padding: 7px 8px; }
    .info-table tr td:first-child { color: var(--muted); width: 40%; }
    .info-table tr td:last-child { color: var(--text); font-weight: 500; word-break: break-all; }
    .info-table tr { border-bottom: 1px solid rgba(99,130,200,.07); }
    .info-table tr:last-child { border-bottom: none; }

    /* ---- Attack source tracker ---- */
    .source-list { display: flex; flex-direction: column; gap: 8px; max-height: 200px; overflow-y: auto; }
    .source-item {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 12px; background: rgba(240,75,75,.06);
      border: 1px solid rgba(240,75,75,.15); border-radius: 8px;
    }
    .source-ip { color: #f9a0a0; font-size: 12px; }
    .source-hits { font-size: 11px; color: var(--muted); }
    .source-badge { font-size: 10px; padding: 2px 8px; border-radius: 999px;
      background: rgba(240,75,75,.15); color: var(--bad); font-weight: 700; }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 999px; }

    @media (max-width: 1200px) {
      .kpi-row { grid-template-columns: repeat(3, 1fr); }
      .main-grid { grid-template-columns: 1fr; }
      .bottom-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
      .kpi-row { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
<div class="app">

  <!-- Top bar -->
  <div class="topbar">
    <div class="topbar-left">
      <div class="logo">SDN <span>NIDS</span></div>
      <select id="runSelect" class="run-select"></select>
    </div>
    <div class="topbar-right">
      <div id="connDot" class="pulse dead"></div>
      <span id="connLabel">connecting…</span>
      <span style="color:var(--border2)">|</span>
      <span>live stream (SSE)</span>
    </div>
  </div>

  <!-- Status banner -->
  <div id="statusBanner" class="status-banner">
    <div class="status-left">
      <div id="statusDot" class="status-dot idle"></div>
      <div>
        <div id="statusLabel" class="status-label idle">NORMAL</div>
        <div id="statusReason" class="status-reason"></div>
      </div>
    </div>
    <div class="status-right">
      <div><strong id="statusTs">—</strong></div>
      <div style="margin-top:4px">model: <strong id="statusModel">—</strong></div>
    </div>
  </div>

  <!-- KPI row -->
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Anomaly Score</div>
      <div id="kpiScore" class="kpi-value info">—</div>
      <div id="kpiScoreSub" class="kpi-sub">vs threshold</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Threshold</div>
      <div id="kpiThreshold" class="kpi-value">—</div>
      <div class="kpi-sub">decision boundary</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Active Incidents</div>
      <div id="kpiAlerts" class="kpi-value">0</div>
      <div class="kpi-sub">within hold window</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Rows Processed</div>
      <div id="kpiRows" class="kpi-value">0</div>
      <div class="kpi-sub">total observations</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Queue Depth</div>
      <div id="kpiQueue" class="kpi-value">0</div>
      <div class="kpi-sub">pending inference</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Latency</div>
      <div id="kpiLatency" class="kpi-value">—</div>
      <div class="kpi-sub">per observation</div>
    </div>
  </div>

  <!-- Main grid: score timeline + gauge -->
  <div class="main-grid" style="margin-bottom:16px">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Score Timeline</div>
        <span id="timelineBadge" class="badge info">live</span>
      </div>
      <div class="chart-wrap tall">
        <canvas id="timelineChart"></canvas>
      </div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;align-items:center;justify-content:center">
      <div class="card-header" style="width:100%">
        <div class="card-title">Score Gauge</div>
        <span id="gaugeBadge" class="badge info">realtime</span>
      </div>
      <div class="gauge-wrap" style="width:100%">
        <canvas id="gaugeChart" style="max-width:220px;max-height:130px"></canvas>
        <div id="gaugeScore" class="gauge-score" style="color:var(--info)">—</div>
        <div class="gauge-label">anomaly score</div>
        <div class="threshold-row">
          <span>0</span>
          <span id="gaugeThreshLabel" style="color:var(--warn)">threshold: —</span>
          <span id="gaugeMax">1</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom grid -->
  <div class="bottom-grid">
    <!-- Alerts table -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">Recent Incidents</div>
        <span id="alertCountBadge" class="badge bad">0</span>
      </div>
      <div style="overflow-x:auto">
        <table class="alert-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Source IP</th>
              <th>Destination</th>
              <th>Score</th>
              <th>Hits</th>
            </tr>
          </thead>
          <tbody id="alertBody">
            <tr><td colspan="5"><div class="empty-state">No alerts yet</div></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Right column: attack sources + model info -->
    <div style="display:flex;flex-direction:column;gap:16px">
      <!-- Attack sources -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Active Sources</div>
          <span id="sourcesBadge" class="badge bad">0 IPs</span>
        </div>
        <div id="sourceList" class="source-list">
          <div class="empty-state">No active sources</div>
        </div>
      </div>

      <!-- Model info -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Runtime Info</div>
          <span id="uptimeBadge" class="badge info">—</span>
        </div>
        <table class="info-table">
          <tr><td>Run ID</td><td id="infoRunId">—</td></tr>
          <tr><td>Model</td><td id="infoModel">—</td></tr>
          <tr><td>Task type</td><td id="infoTask">—</td></tr>
          <tr><td>Score direction</td><td id="infoDir">—</td></tr>
          <tr><td>Seq buffers</td><td id="infoBuffers">—</td></tr>
          <tr><td>Dropped rows</td><td id="infoDropped">—</td></tr>
          <tr><td>Feature scheme</td><td id="infoScheme">—</td></tr>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
/* ====================================================================
   State
==================================================================== */
let currentRun = '';
const WINDOW = 80;   // number of points in timeline
const scoreHistory = [];
const timeLabels = [];
let threshold = 0;
let attackSources = {};   // ip → {count, lastTs}
let isConnected = false;
let eventSource = null;
let latestAlerts = [];
let latestState = null;
let runsRefreshTimer = null;
let reconnectTimer = null;
let timelineStartMs = null;
let timelineTick = -1;
let lastTimelineAcceptedMs = null;
let lastTimelineState = '';
let lastAcceptedScore = NaN;


function normalizeAttackType(value) {
  const v = String(value || '').toLowerCase();
  if (!v) return 'Attack';
  if (v.includes('bfa')) return 'Brute Force';
  if (v.includes('probe') || v.includes('scan')) return 'Scanning';
  if (v.includes('udp') || v.includes('tcp') || v.includes('icmp') || v.includes('flood') || v.includes('traffic')) return 'Flood Attack';
  if (v.includes('botnet')) return 'Botnet';
  if (v.includes('u2r')) return 'U2R-like';
  return 'Attack';
}

function aggregateIncidents(rows) {
  const map = new Map();
  (rows || []).forEach(row => {
    const type = normalizeAttackType(row.category);
    const src = row.src_ip || '—';
    const dst = row.dst_ip ? `${row.dst_ip}:${row.dst_port || '?'}` : '—';
    const key = `${type}|${src}|${dst}`;
    const score = Number(row.anomaly_score) || 0;
    const hits = Number(row.hit_count) || 0;
    const ts = row.poll_timestamp || row.server_timestamp || '';
    const cur = map.get(key) || { type, src, dst, score: 0, hits: 0, ts: ts, count: 0 };
    cur.score = Math.max(cur.score, score);
    cur.hits = Math.max(cur.hits, hits);
    cur.ts = ts > cur.ts ? ts : cur.ts;
    cur.count += 1;
    map.set(key, cur);
  });
  return Array.from(map.values()).sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return (b.ts || '').localeCompare(a.ts || '');
  });
}

function computeDisplayState(p, incidents) {
  const hasIncidents = (incidents || []).length > 0;
  const statusRaw = String(p?.status || 'NORMAL').toUpperCase();
  const score = Number(p?.display_score ?? p?.recent_peak_score ?? p?.max_score) || 0;
  const thr = Number(p?.threshold) || 0;
  const attack = hasIncidents || statusRaw === 'ATTACK';
  if (!attack) return { label: 'NORMAL', state: 'normal', reason: '' };
  return { label: 'ATTACK', state: 'attack', reason: '' };
}

/* ====================================================================
   Chart.js setup
==================================================================== */
const COLORS = {
  ok: '#22d3a0', bad: '#f04b4b', warn: '#f5a623',
  info: '#38b2f8', purple: '#a78bfa', grid: 'rgba(99,130,200,.1)',
  text: '#6b7fa8',
};

// --- Timeline chart ---
const tlCtx = document.getElementById('timelineChart').getContext('2d');
const timelineChart = new Chart(tlCtx, {
  type: 'line',
  data: {
    labels: timeLabels,
    datasets: [
      {
        label: 'Anomaly score',
        data: scoreHistory,
        borderColor: COLORS.info,
        backgroundColor: 'rgba(56,178,248,.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.35,
      },
      {
        label: 'Threshold',
        data: [],
        borderColor: COLORS.warn,
        borderWidth: 1,
        borderDash: [5, 4],
        pointRadius: 0,
        fill: false,
        tension: 0,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111e35',
        borderColor: 'rgba(99,130,200,.25)',
        borderWidth: 1,
        titleColor: COLORS.text,
        bodyColor: '#e2eaf8',
        callbacks: {
          label: ctx => `${ctx.dataset.label}: ${(+ctx.raw).toFixed(5)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: COLORS.text, font: { size: 10 }, maxTicksLimit: 8 },
        grid: { color: COLORS.grid },
      },
      y: {
        ticks: { color: COLORS.text, font: { size: 10 }, maxTicksLimit: 6 },
        grid: { color: COLORS.grid },
        min: 0,
      },
    },
  },
});

// --- Gauge chart (doughnut half) ---
const gaugeCtx = document.getElementById('gaugeChart').getContext('2d');
const gaugeChart = new Chart(gaugeCtx, {
  type: 'doughnut',
  data: {
    datasets: [{
      data: [0, 1],
      backgroundColor: [COLORS.info, 'rgba(99,130,200,.1)'],
      borderWidth: 0,
      circumference: 180,
      rotation: 270,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    cutout: '72%',
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
  },
});

/* ====================================================================
   Helpers
==================================================================== */
function fmt(v, d = 4) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : '—';
}
function fmtTime(isoStr) {
  if (!isoStr) return '—';
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString('en-GB', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3,'0');
  } catch { return isoStr.slice(11, 19) || '—'; }
}
function fmtUptime(s) {
  if (!s || s < 0) return '—';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function setClass(el, cls) {
  el.className = el.className.replace(/\b(normal|attack|suspect|idle|warming_up|info|ok|bad|warn)\b/g, '');
  if (cls) el.classList.add(cls);
}

/* ====================================================================
   Update gauge
==================================================================== */
function updateGauge(score, thr) {
  const max = Math.max(score * 1.5, thr * 2, 0.001);
  const ratio = Math.min(score / max, 1);
  const color = score >= thr ? COLORS.bad : score >= thr * 0.8 ? COLORS.warn : COLORS.ok;
  gaugeChart.data.datasets[0].data = [ratio, 1 - ratio];
  gaugeChart.data.datasets[0].backgroundColor[0] = color;
  gaugeChart.update();

  const gaugeEl = document.getElementById('gaugeScore');
  gaugeEl.textContent = fmt(score, 5);
  gaugeEl.style.color = color;
  document.getElementById('gaugeBadge').className = 'badge ' +
    (score >= thr ? 'bad' : score >= thr * 0.8 ? 'warn' : 'ok');
  document.getElementById('gaugeBadge').textContent = score >= thr ? 'ALERT' : score >= thr * 0.8 ? 'HIGH' : 'normal';
  document.getElementById('gaugeThreshLabel').textContent = `threshold: ${fmt(thr, 4)}`;
  document.getElementById('gaugeMax').textContent = fmt(max, 3);
}

/* ====================================================================
   Update timeline
==================================================================== */
function pushTimeline(score, ts, thr, state = {}) {
  const nominalStepS = Math.max(Number(state.dashboard_timeline_interval_s ?? state.poll_interval_s) || 1.0, 0.25);
  const baseMs = (() => {
    if (!ts) return Date.now();
    const parsed = new Date(ts).getTime();
    return Number.isFinite(parsed) ? parsed : Date.now();
  })();

  if (timelineStartMs === null) {
    timelineStartMs = baseMs;
    timelineTick = -1;
    lastTimelineAcceptedMs = null;
    lastTimelineState = '';
  }

  const stateKey = `${state.status || 'NORMAL'}|${fmt(score,5)}|${fmt(thr,4)}`;
  const dueByCadence = (lastTimelineAcceptedMs === null) || ((baseMs - lastTimelineAcceptedMs) >= (nominalStepS * 950));
  const dueByStateChange = stateKey !== lastTimelineState;
  const dueByScoreMove = !Number.isFinite(lastAcceptedScore) || Math.abs(score - lastAcceptedScore) >= 0.01;
  if (!dueByCadence && !dueByStateChange && !dueByScoreMove) return;

  timelineTick += 1;
  lastTimelineAcceptedMs = baseMs;
  lastTimelineState = stateKey;
  lastAcceptedScore = score;
  const labelDate = new Date(timelineStartMs + (timelineTick * nominalStepS * 1000));
  const label = labelDate.toLocaleTimeString('en-GB', { hour12: false }) + '.' + String(labelDate.getMilliseconds()).padStart(3,'0');

  if (scoreHistory.length >= WINDOW) {
    scoreHistory.shift(); timeLabels.shift();
    timelineChart.data.datasets[1].data.shift();
  }
  scoreHistory.push(score);
  timeLabels.push(label);
  timelineChart.data.datasets[1].data.push(thr);

  const hasAttack = scoreHistory.some(s => s >= thr);
  timelineChart.data.datasets[0].borderColor = hasAttack ? COLORS.bad : COLORS.info;
  timelineChart.data.datasets[0].backgroundColor = hasAttack
    ? 'rgba(240,75,75,.07)' : 'rgba(56,178,248,.07)';
  timelineChart.update('none');
}

/* ====================================================================
   Update KPIs
==================================================================== */
function updateKPIs(p, incidents = null) {
  const score = Number(p.display_score ?? p.recent_peak_score ?? p.max_score) || 0;
  const thr = Number(p.threshold) || 0;
  const alerts = incidents ? incidents.length : (Number(p.active_signal_count ?? p.recent_alert_count) || 0);
  const queue = Number(p.queue_depth) || 0;
  const lat = Number(p.latency_ms);

  const scoreEl = document.getElementById('kpiScore');
  scoreEl.textContent = fmt(score, 5);
  setClass(scoreEl, 'kpi-value');
  scoreEl.classList.add(score >= thr ? 'bad' : score >= thr * 0.8 ? 'warn' : 'info');

  document.getElementById('kpiScoreSub').textContent =
    thr > 0 ? `${((score / thr) * 100).toFixed(0)}% of threshold` : 'vs threshold';

  document.getElementById('kpiThreshold').textContent = fmt(thr, 4);

  const alertEl = document.getElementById('kpiAlerts');
  alertEl.textContent = alerts;
  setClass(alertEl, 'kpi-value');
  alertEl.classList.add(alerts > 0 ? 'bad' : 'ok');

  document.getElementById('kpiRows').textContent =
    (Number(p.total_rows_read) || 0).toLocaleString();

  const qEl = document.getElementById('kpiQueue');
  qEl.textContent = queue;
  setClass(qEl, 'kpi-value');
  qEl.classList.add(queue > 1000 ? 'bad' : queue > 200 ? 'warn' : 'ok');

  document.getElementById('kpiLatency').textContent =
    Number.isFinite(lat) ? lat.toFixed(2) + ' ms' : '—';
}

/* ====================================================================
   Update status banner
==================================================================== */
function updateStatus(p, incidents = null) {
  const display = computeDisplayState(p, incidents || aggregateIncidents(latestAlerts));
  const banner = document.getElementById('statusBanner');
  banner.className = 'status-banner ' + display.state;

  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot ' + display.state;

  const label = document.getElementById('statusLabel');
  label.textContent = display.label;
  label.className = 'status-label ' + display.state;

  document.getElementById('statusReason').textContent = display.reason || '';
  document.getElementById('statusTs').textContent = fmtTime(p.server_timestamp || p.poll_timestamp);
  document.getElementById('statusModel').textContent = p.model_name || '—';
}

/* ====================================================================
   Update alerts table
==================================================================== */
function updateAlerts(rows) {
  const tbody = document.getElementById('alertBody');
  attackSources = {};
  const incidents = aggregateIncidents(rows);
  document.getElementById('alertCountBadge').textContent = incidents.length;

  if (!incidents.length) {
    tbody.innerHTML = '<tr><td colspan="5"><div class="empty-state">No incidents</div></td></tr>';
    updateSources();
    return incidents;
  }

  const maxScore = Math.max(...incidents.map(r => Number(r.score) || 0), 0.001);
  tbody.innerHTML = incidents.map(row => {
    const score = Number(row.score) || 0;
    const pct = Math.min((score / maxScore) * 100, 100).toFixed(0);
    return `<tr>
      <td style="color:var(--muted);white-space:nowrap">${fmtTime(row.ts)}</td>
      <td><span class="ip-tag">${row.src}</span></td>
      <td><span class="ip-tag dst">${row.dst}</span></td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
          <span style="font-size:11px;min-width:58px;text-align:right;color:var(--bad)">${score.toFixed(5)}</span>
        </div>
        <div style="margin-top:4px;font-size:10px;color:var(--muted)">${row.type}</div>
      </td>
      <td style="color:var(--warn);text-align:center">${row.hits || row.count || '—'}</td>
    </tr>`;
  }).join('');

  incidents.forEach(r => {
    if (!r.src || r.src === '—') return;
    if (!attackSources[r.src]) attackSources[r.src] = { count: 0, lastTs: '' };
    attackSources[r.src].count += 1;
    attackSources[r.src].lastTs = r.ts || '';
  });
  updateSources();
  return incidents;
}

/* ====================================================================
   Update attack sources panel
==================================================================== */
function updateSources() {
  const list = document.getElementById('sourceList');
  const entries = Object.entries(attackSources).sort((a, b) => b[1].count - a[1].count);
  document.getElementById('sourcesBadge').textContent = entries.length + ' IPs';

  if (!entries.length) {
    list.innerHTML = '<div class="empty-state">No active sources</div>';
    return;
  }
  list.innerHTML = entries.slice(0, 8).map(([ip, info]) => `
    <div class="source-item">
      <div>
        <div class="source-ip">${ip}</div>
        <div class="source-hits">last seen: ${fmtTime(info.lastTs)}</div>
      </div>
      <span class="source-badge">${info.count} alerts</span>
    </div>
  `).join('');
}

/* ====================================================================
   Update model info
==================================================================== */
function updateInfo(p) {
  document.getElementById('infoRunId').textContent = p.run_id || '—';
  document.getElementById('infoModel').textContent = p.model_name || '—';
  document.getElementById('infoTask').textContent = p.task_type || '—';
  document.getElementById('infoDir').textContent = p.score_direction || '—';
  document.getElementById('infoBuffers').textContent = p.sequence_buffer_count ?? '—';
  document.getElementById('infoDropped').textContent = p.dropped_rows ?? '—';
  document.getElementById('infoScheme').textContent =
    (p.feature_names || []).length ? `${p.feature_scheme || 'feature_scheme'} (${p.feature_names.length}f)` : '—';
  document.getElementById('uptimeBadge').textContent = fmtUptime(p.uptime_s);
}

/* ====================================================================
   API + SSE
==================================================================== */
function setConnection(connected, label = null) {
  isConnected = !!connected;
  document.getElementById('connDot').className = connected ? 'pulse' : 'pulse dead';
  document.getElementById('connLabel').textContent = label || (connected ? 'connected' : 'offline');
}

function resetView() {
  attackSources = {};
  latestAlerts = [];
  latestState = null;
  latestState = null;
  timelineStartMs = null;
  timelineTick = -1;
  lastTimelineAcceptedMs = null;
  lastTimelineState = '';
  scoreHistory.length = 0;
  timeLabels.length = 0;
  timelineChart.data.datasets[1].data.length = 0;
  timelineChart.update('none');
  updateAlerts([]);
}

function applyState(p) {
  if (!p) return;
  latestState = p;
  threshold = Number(p.threshold) || 0;
  const score = Number(p.display_score ?? p.recent_peak_score ?? p.max_score) || 0;
  const incidents = aggregateIncidents(latestAlerts);
  updateStatus(p, incidents);
  updateKPIs(p, incidents);
  updateGauge(score, threshold);
  pushTimeline(score, p.server_timestamp || p.poll_timestamp, threshold, p);
  updateInfo(p);
}

function applyAlerts(rows) {
  latestAlerts = Array.isArray(rows) ? rows : [];
  const incidents = updateAlerts(latestAlerts);
  if (latestState) {
    updateStatus(latestState, incidents);
    updateKPIs(latestState, incidents);
  }
}

function loadRuns() {
  return fetch('/api/runs').then(r => r.json()).then(j => {
    const runs = j.runs || [];
    const sel = document.getElementById('runSelect');
    const prev = currentRun;
    sel.innerHTML = runs.map(r => `<option value="${r}">${r}</option>`).join('');
    if (!currentRun && runs.length) currentRun = runs[0];
    if (currentRun && runs.includes(currentRun)) sel.value = currentRun;
    if (prev && currentRun !== prev) resetView();
  });
}

function disconnectStream() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function connectStream() {
  if (!currentRun) return;
  disconnectStream();
  setConnection(false, 'connecting…');
  eventSource = new EventSource(`/stream?run_id=${encodeURIComponent(currentRun)}`);
  eventSource.addEventListener('open', () => setConnection(true, 'streaming'));
  eventSource.addEventListener('snapshot', ev => {
    const payload = JSON.parse(ev.data || '{}');
    if (payload.state) applyState(payload.state);
    if (payload.alerts) applyAlerts(payload.alerts.rows || []);
  });
  eventSource.addEventListener('state', ev => applyState(JSON.parse(ev.data || '{}')));
  eventSource.addEventListener('alerts', ev => {
    const payload = JSON.parse(ev.data || '{}');
    applyAlerts(payload.rows || []);
  });
  eventSource.addEventListener('heartbeat', () => {});
  eventSource.onerror = () => {
    setConnection(false, 'reconnecting…');
    disconnectStream();
    reconnectTimer = setTimeout(connectStream, 2000);
  };
}

async function bootstrap() {
  try {
    await loadRuns();
    connectStream();
  } catch {
    setConnection(false, 'offline');
  }
}

document.getElementById('runSelect').addEventListener('change', e => {
  currentRun = e.target.value;
  resetView();
  connectStream();
});

bootstrap();
runsRefreshTimer = setInterval(loadRuns, 10000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server + SSE
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    runtime_root: Path = Path("runtime_logs")
    event_hub: DashboardEventHub | None = None
    ui_root: Path = Path(__file__).resolve().parent / "dashboard_ui"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            index_path = self.ui_root / "index.html"
            if index_path.exists():
                self._send_file(index_path)
            else:
                self._send_html(HTML)
            return
        if parsed.path.startswith("/static/"):
            rel_path = parsed.path[len("/static/"):]
            asset_path = (self.ui_root / rel_path).resolve()
            try:
                asset_path.relative_to(self.ui_root.resolve())
            except Exception:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if asset_path.exists() and asset_path.is_file():
                self._send_file(asset_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/runs":
            self._send_json({"runs": self._list_runs()})
            return
        if parsed.path == "/api/status":
            run_id = self._get_run_id(parsed.query)
            payload = self._get_state(run_id)
            self._send_json(payload, status=HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/health":
            run_id = self._get_run_id(parsed.query) or self._default_run_id()
            payload = self._get_state(run_id)
            controller_metrics = self._get_controller_metrics(run_id)
            self._send_json({
                "ok": bool(payload),
                "run_id": run_id,
                "status": payload.get("status", "") if payload else "",
                "queue_depth": payload.get("queue_depth", 0) if payload else 0,
                "raw_stats_queue_depth": controller_metrics.get("raw_stats_queue_depth", 0) if controller_metrics else 0,
                "uptime_s": payload.get("uptime_s", 0.0) if payload else 0.0,
            }, status=HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/alerts":
            run_id = self._get_run_id(parsed.query)
            limit = self._query_int(parsed.query, "limit", 50, minimum=1, maximum=5000)
            self._send_json({"rows": self._get_alerts(run_id, limit=limit)})
            return
        if parsed.path == "/api/export/alerts":
            run_id = self._get_run_id(parsed.query)
            rows = self._get_alerts(run_id, limit=100000)
            body = self._csv_body(rows)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{run_id or "alerts"}.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/flows":
            run_id = self._get_run_id(parsed.query)
            limit = self._query_int(parsed.query, "limit", 120, minimum=1, maximum=2000)
            metrics = self._get_controller_metrics(run_id)
            flows = list(metrics.get("active_flows", []) if metrics else [])
            total = int(metrics.get("active_flow_count", len(flows))) if metrics else 0
            self._send_json({"run_id": run_id, "rows": flows[:limit], "returned": min(len(flows), limit), "total": total}, status=HTTPStatus.OK if metrics else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/polling-stats":
            run_id = self._get_run_id(parsed.query)
            metrics = self._get_controller_metrics(run_id)
            self._send_json({
                "run_id": run_id,
                "polling_stats": metrics.get("polling_stats", {}) if metrics else {},
                "controller_metrics": metrics if metrics else {},
            }, status=HTTPStatus.OK if metrics else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/latency-breakdown":
            run_id = self._get_run_id(parsed.query)
            payload = self._get_state(run_id)
            self._send_json({
                "run_id": run_id,
                "last": payload.get("latency_breakdown", {}) if payload else {},
                "average": payload.get("avg_latency_breakdown", {}) if payload else {},
            }, status=HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/topology":
            run_id = self._get_run_id(parsed.query)
            host_limit = self._query_int(parsed.query, "host_limit", 60, minimum=1, maximum=500)
            link_limit = self._query_int(parsed.query, "link_limit", 60, minimum=1, maximum=500)
            metrics = self._get_controller_metrics(run_id)
            topology = dict(metrics.get("topology", {}) if metrics else {})
            if topology:
                topology["hosts"] = list(topology.get("hosts", []) or [])[:host_limit]
                topology["links"] = list(topology.get("links", []) or [])[:link_limit]
            self._send_json({"run_id": run_id, "topology": topology}, status=HTTPStatus.OK if metrics else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/model-info":
            run_id = self._get_run_id(parsed.query)
            payload = self._get_model_info(run_id)
            self._send_json(payload, status=HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/scalability-report":
            report = self._get_scalability_report()
            self._send_json(report, status=HTTPStatus.OK if report else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/stream":
            run_id = self._get_run_id(parsed.query)
            self._send_event_stream(run_id)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return  # suppress request logs

    @classmethod
    def _list_runs(cls) -> list[str]:
        if cls.event_hub is not None:
            return list(reversed(cls.event_hub.list_runs()))
        if not cls.runtime_root.exists():
            return []
        return sorted(
            [item.name for item in cls.runtime_root.iterdir() if item.is_dir()],
            reverse=True,
        )

    @staticmethod
    def _get_run_id(query: str) -> str:
        return parse_qs(query).get("run_id", [""])[0]

    @staticmethod
    def _query_int(query: str, name: str, default: int, minimum: int = 1, maximum: int = 1000) -> int:
        raw = parse_qs(query).get(name, [default])[0]
        try:
            value = int(raw)
        except Exception:
            value = int(default)
        return max(int(minimum), min(int(maximum), int(value)))

    @classmethod
    def _default_run_id(cls) -> str:
        runs = cls._list_runs()
        return runs[0] if runs else ""

    @classmethod
    def _read_json(cls, run_id: str, filename: str) -> dict:
        if not run_id:
            return {}
        path = cls.runtime_root / run_id / filename
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _get_state(cls, run_id: str) -> dict:
        if cls.event_hub is not None:
            payload = cls.event_hub.get_state(run_id)
            if payload:
                return payload
        return cls._read_json(run_id, "dashboard_state.json")

    @classmethod
    def _get_controller_metrics(cls, run_id: str) -> dict:
        return cls._read_json(run_id, "controller_metrics.json")


    @classmethod
    def _get_model_info(cls, run_id: str) -> dict:
        info = cls._read_json(run_id, "model_info.json")
        if info:
            return info
        payload = cls._get_state(run_id)
        if not payload:
            return {}
        return {
            "run_id": run_id,
            "bundle_info": payload.get("bundle_info", {}),
            "model_metrics": payload.get("model_metrics", {}),
            "feature_names": payload.get("feature_names", []),
        }

    @classmethod
    def _get_alerts(cls, run_id: str, limit: int = 20) -> list[dict]:
        if cls.event_hub is not None:
            rows = cls.event_hub.get_alerts(run_id, limit=limit)
            if rows:
                return rows
        return cls._read_alerts(run_id, limit=limit)

    @classmethod
    def _read_alerts(cls, run_id: str, limit: int = 20) -> list[dict]:
        if not run_id:
            return []
        path = cls.runtime_root / run_id / "alerts.csv"
        if not path.exists():
            return []
        try:
            import io
            with path.open("r", encoding="utf-8", newline="") as tf:
                header = tf.readline().rstrip("\n")
            if not header:
                return []
            with path.open("rb") as f:
                f.seek(0, 2)
                end_pos = f.tell()
                chunk = 4096
                data = b""
                while end_pos > 0 and data.count(b"\n") <= limit + 1:
                    take = min(chunk, end_pos)
                    end_pos -= take
                    f.seek(end_pos)
                    data = f.read(take) + data
            lines = [line for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip()]
            if lines and lines[0] == header:
                lines = lines[1:]
            payload = header + "\n" + "\n".join(lines[-limit:])
            return list(csv.DictReader(io.StringIO(payload)))[-limit:]
        except Exception:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            return rows[-limit:]

    @classmethod
    def _get_scalability_report(cls) -> dict:
        candidates = [
            cls.runtime_root / "scalability_report.json",
            cls.runtime_root.parent / "artifacts_v4" / "scalability_report.json",
            cls.runtime_root.parent / "artifacts_v4" / "benchmark" / "scalability_report.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return {}
        return {}

    def _sse_send_event(self, event: str, payload: dict) -> None:
        body = f"retry: 2000\nevent: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_event_stream(self, run_id: str) -> None:
        if not run_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "run_id is required")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        hub = self.event_hub
        q = hub.subscribe(run_id) if hub is not None else None
        try:
            snapshot = {"state": self._get_state(run_id), "alerts": {"rows": self._get_alerts(run_id, limit=50)}}
            self._sse_send_event("snapshot", snapshot)
            while True:
                try:
                    if q is None:
                        time.sleep(15.0)
                        self._sse_send_event("heartbeat", {"ts": time.time()})
                        continue
                    item = q.get(timeout=15.0)
                    self._sse_send_event(str(item.get("event", "message")), dict(item.get("payload", {})))
                except queue.Empty:
                    self._sse_send_event("heartbeat", {"ts": time.time()})
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            if hub is not None and q is not None:
                hub.unsubscribe(run_id, q)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + ("; charset=utf-8" if (content_type or "").startswith("text/") or content_type in {"application/javascript", "application/json"} else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _csv_body(rows: list[dict]) -> bytes:
        if not rows:
            return b""
        headers = list(rows[0].keys())
        output = [",".join(headers)]
        for row in rows:
            values = []
            for header in headers:
                text = str(row.get(header, ""))
                if any(ch in text for ch in [",", "\"", "\n"]):
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
            output.append(",".join(values))
        return ("\n".join(output)).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard for SDN realtime NIDS.")
    parser.add_argument("--runtime-root", default="runtime_logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stream-host", default="127.0.0.1")
    parser.add_argument("--stream-port", type=int, default=8765)
    args = parser.parse_args()

    DashboardHandler.runtime_root = Path(args.runtime_root)
    hub = DashboardEventHub(runtime_root=args.runtime_root)
    hub.start_udp_listener(host=args.stream_host, port=args.stream_port)
    DashboardHandler.event_hub = hub
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port} | SSE UDP ingest {args.stream_host}:{args.stream_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
        server.server_close()


if __name__ == "__main__":
    main()
