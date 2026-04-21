// charts.js — global chart helpers (no ES modules, plain script)

window.SDN = window.SDN || {};

(function () {
  var GRID  = 'rgba(99,140,200,.09)';
  var TICK  = '#556b90';
  var TIP_BG= '#172035';
  var TIP_BD= 'rgba(99,140,200,.28)';

  window.SDN.createScoreChart = function (canvas) {
    return new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Anomaly Score',
            data: [],
            borderColor: '#4f8ef7',
            backgroundColor: 'rgba(79,142,247,.07)',
            borderWidth: 1.5,
            pointRadius: 0,
            pointHoverRadius: 3,
            fill: true,
            tension: 0.3,
          },
          {
            label: 'Threshold',
            data: [],
            borderColor: '#f5a623',
            borderWidth: 1,
            borderDash: [5, 5],
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
            backgroundColor: TIP_BG,
            borderColor: TIP_BD,
            borderWidth: 1,
            titleColor: TICK,
            bodyColor: '#e4eaf8',
            padding: 10,
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': ' + Number(ctx.raw).toFixed(5);
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: TICK, font: { size: 10 }, maxTicksLimit: 8 }, grid: { color: GRID } },
          y: { min: 0, ticks: { color: TICK, font: { size: 10 }, maxTicksLimit: 5 }, grid: { color: GRID } },
        },
      },
    });
  };

  window.SDN.createGaugeChart = function (canvas) {
    return new Chart(canvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [0, 1],
          backgroundColor: ['#4f8ef7', 'rgba(99,140,200,.08)'],
          borderWidth: 0,
          circumference: 180,
          rotation: 270,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        cutout: '76%',
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });
  };

  window.SDN.updateScoreChart = function (chart, labels, scores, thresholds) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = scores;
    chart.data.datasets[1].data = thresholds;
    var hasAtk = thresholds.length > 0 && scores.some(function (s, i) { return s >= thresholds[i]; });
    chart.data.datasets[0].borderColor     = hasAtk ? '#f05252' : '#4f8ef7';
    chart.data.datasets[0].backgroundColor = hasAtk ? 'rgba(240,82,82,.07)' : 'rgba(79,142,247,.07)';
    chart.update('none');
  };

  window.SDN.updateGauge = function (chart, score, threshold) {
    var max   = Math.max(score * 1.6, threshold * 2, 0.001);
    var ratio = Math.min(score / max, 1);
    var col   = score >= threshold      ? '#f05252'
              : score >= threshold * 0.8 ? '#f5a623'
              : '#10c88a';
    chart.data.datasets[0].data[0] = ratio;
    chart.data.datasets[0].data[1] = 1 - ratio;
    chart.data.datasets[0].backgroundColor[0] = col;
    chart.update();
    return { col: col, max: max };
  };
})();
