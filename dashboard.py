"""Simple real-time dashboard using Flask + Server-Sent Events."""

import json
import logging
from threading import Thread

from flask import Flask, Response, render_template_string

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

# Suppress Flask/Werkzeug request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# Global state updated by bot
state = {
    'tick': 0,
    'period': 1,
    'pnl': 0.0,
    'positions': {},
    'signals': [],
    'active_strategies': [],
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #1a1a2e; color: #eee; padding: 20px;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #16213e; border-radius: 8px; padding: 20px; }
        .card h2 { color: #7f8c8d; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
        .pnl { font-size: 48px; font-weight: bold; }
        .pnl.positive { color: #2ecc71; }
        .pnl.negative { color: #e74c3c; }
        .tick { font-size: 24px; color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: right; border-bottom: 1px solid #2c3e50; }
        th { color: #7f8c8d; font-weight: normal; }
        td:first-child, th:first-child { text-align: left; }
        .pos-long { color: #2ecc71; }
        .pos-short { color: #e74c3c; }
        .signal { padding: 4px 8px; background: #2c3e50; border-radius: 4px; margin: 2px; display: inline-block; }
        #chart-container { height: 200px; }
        .status { display: flex; gap: 20px; align-items: baseline; }
    </style>
</head>
<body>
    <div class="grid">
        <div class="card">
            <h2>Performance</h2>
            <div class="status">
                <div class="pnl" id="pnl">$0</div>
                <div class="tick" id="tick">Tick 0</div>
            </div>
        </div>
        <div class="card">
            <h2>Active Strategies</h2>
            <div id="strategies"></div>
        </div>
        <div class="card">
            <h2>Positions</h2>
            <table>
                <thead><tr><th>Ticker</th><th>Position</th><th>Value</th></tr></thead>
                <tbody id="positions"></tbody>
            </table>
        </div>
        <div class="card">
            <h2>PnL Chart</h2>
            <div id="chart-container">
                <canvas id="chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const pnlHistory = [];
        const maxPoints = 200;

        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#3498db',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        ticks: {
                            color: '#7f8c8d',
                            callback: v => '$' + (v/1000000).toFixed(2) + 'M'
                        },
                        grid: { color: '#2c3e50' }
                    }
                }
            }
        });

        function formatMoney(n) {
            const sign = n >= 0 ? '' : '-';
            const abs = Math.abs(n);
            if (abs >= 1000000) return sign + '$' + (abs/1000000).toFixed(2) + 'M';
            if (abs >= 1000) return sign + '$' + (abs/1000).toFixed(1) + 'K';
            return sign + '$' + abs.toFixed(0);
        }

        function update(data) {
            // PnL
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = formatMoney(data.pnl);
            pnlEl.className = 'pnl ' + (data.pnl >= 0 ? 'positive' : 'negative');

            // Tick
            document.getElementById('tick').textContent =
                'Period ' + data.period + ' | Tick ' + data.tick;

            // Strategies
            const stratEl = document.getElementById('strategies');
            if (data.active_strategies && data.active_strategies.length > 0) {
                stratEl.innerHTML = data.active_strategies
                    .map(s => '<span class="signal">' + s + '</span>').join('');
            } else {
                stratEl.innerHTML = '<span style="color:#7f8c8d">None active</span>';
            }

            // Positions
            const posEl = document.getElementById('positions');
            let posHtml = '';
            for (const [ticker, info] of Object.entries(data.positions || {})) {
                const pos = info.position || 0;
                const price = info.price || 0;
                const value = pos * price;
                const cls = pos > 0 ? 'pos-long' : (pos < 0 ? 'pos-short' : '');
                posHtml += '<tr><td>' + ticker + '</td>' +
                    '<td class="' + cls + '">' + pos.toLocaleString() + '</td>' +
                    '<td>' + formatMoney(value) + '</td></tr>';
            }
            posEl.innerHTML = posHtml || '<tr><td colspan="3" style="color:#7f8c8d">No positions</td></tr>';

            // Chart
            pnlHistory.push(data.pnl);
            if (pnlHistory.length > maxPoints) pnlHistory.shift();
            chart.data.labels = pnlHistory.map((_, i) => i);
            chart.data.datasets[0].data = pnlHistory;
            chart.update('none');
        }

        // SSE connection
        const evtSource = new EventSource('/stream');
        evtSource.onmessage = (e) => update(JSON.parse(e.data));
        evtSource.onerror = () => console.log('SSE reconnecting...');
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/stream')
def stream():
    def generate():
        import time
        while True:
            yield f"data: {json.dumps(state)}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'})


def update_state(tick: int, period: int, pnl: float,
                 positions: dict, active: list) -> None:
    """Called by bot to update dashboard state.

    Args:
        tick: Current tick number
        period: Current period
        pnl: Current PnL in dollars
        positions: Dict of {ticker: {'position': int, 'price': float}}
        active: List of active strategy names
    """
    state['tick'] = tick
    state['period'] = period
    state['pnl'] = pnl
    state['positions'] = positions
    state['active_strategies'] = active


def run_dashboard(port: int = 5000) -> None:
    """Run dashboard server (call from background thread)."""
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)


def start_dashboard(port: int = 5000) -> Thread:
    """Start dashboard in background thread and return the thread."""
    t = Thread(target=run_dashboard, args=(port,), daemon=True)
    t.start()
    logging.info('Dashboard started at http://localhost:%d', port)
    return t
