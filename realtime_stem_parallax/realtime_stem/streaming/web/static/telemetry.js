/* Telemetry is independent of preview SSE, so idle/stalled scans remain observable. */
(() => {
  const el = id => document.getElementById(id);
  const fmt = (v, digits=1) => v == null ? 'N/A' : Number(v).toLocaleString(undefined,
    {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const gib = v => v == null ? 'N/A' : `${fmt(v / 2**30, 2)} GiB`;
  function chart(history, target) {
    const canvas = el('telemetry-chart'), ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;
    const end = history[history.length-1].time;
    const fpsMax = Math.max(target * 1.2, ...history.map(s => s.fps));
    const cpuMax = Math.max(100, ...history.map(s => s.cpu || 0));
    for (const [key, color, max] of [['fps', '#4fa3ff', fpsMax], ['cpu', '#3fbf6f', cpuMax],
                                    ['gpu', '#f0a34a', 100]]) {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
      let pen = false;
      for (const s of history) {
        if (s[key] == null) { pen = false; continue; }
        const x = w * (1 - (end-s.time)/60), y = h-6-(h-12)*s[key]/max;
        if (pen) ctx.lineTo(x,y); else ctx.moveTo(x,y);
        pen = true;
      }
      ctx.stroke();
    }
    ctx.fillStyle = '#8b949f'; ctx.font = '18px monospace';
    ctx.fillText(`FPS scale 0–${fmt(fpsMax/1000,0)}k · CPU 0–${fmt(cpuMax,0)}% · GPU 0–100%`, 8, 22);
  }
  function render(t) {
    el('telemetry-panel').hidden = !t.available;
    document.body.classList.toggle('has-telemetry', Boolean(t.available));
    if (!t.available) { el('telemetry').hidden = true; return; }
    el('telemetry').hidden = false;
    el('telemetry-status').textContent = `Scan ${t.scan} · ${t.status}`;
    el('telemetry-status').style.color = t.status === 'incomplete' ? 'var(--bad)' : 'var(--muted)';
    el('telemetry-status').title = t.endpoint;
    const r = t.resources || {}, b = t.host_batch_ms, integrity = t.integrity;
    const preview = el('rate'), previewAge = preview.dataset.updated;
    const rows = [
      ['Detector rate (1 s)', `${fmt(t.fps/1000)} kfps`],
      ['Scan average', `${fmt(t.average_fps == null ? null : t.average_fps/1000)} kfps`],
      ['120k reference', `${fmt(100*t.fps/t.target_fps)}%`],
      ['Frames / expected', `${fmt(t.frames,0)} / ${fmt(t.expected,0)}`],
      ['Batches / elapsed', `${fmt(t.batches,0)} / ${fmt(t.elapsed_s,2)} s`],
      ['Missing / duplicates', integrity ? `${fmt(integrity.missing,0)} / ${fmt(integrity.duplicates,0)}` : 'Checked at scan end'],
      ['Host batch p50/p95/p99', `${fmt(b.p50,2)} / ${fmt(b.p95,2)} / ${fmt(b.p99,2)} ms`],
      ['Batch interval jitter (σ)', `${fmt(t.batch_interval_jitter_ms,2)} ms`],
      ['Last batch age', `${fmt(t.last_batch_age_s,1)} s`],
      ['Preview updates', preview.textContent || 'Waiting'],
      ['Last preview age', previewAge ? `${fmt((performance.now()-Number(previewAge))/1000)} s` : 'N/A'],
      ['App CPU (100% = 1 core)', `${fmt(r.process_cpu_pct)}%`],
      ['Host CPU', `${fmt(r.host_cpu_pct)}% · ${fmt(r.cpu_cores,0)} cores`],
      ['App RAM', gib(r.process_rss_bytes)],
      ['Host RAM used / total', `${gib(r.host_ram_used_bytes)} / ${gib(r.host_ram_total_bytes)}`],
    ];
    for (const g of r.gpus || []) {
      rows.push([`GPU ${g.index} utilization`, `${fmt(g.util_pct,0)}%`],
        [`GPU ${g.index} memory`, g.memory_mib == null ? 'N/A (driver unsupported)' : `${fmt(g.memory_mib,0)} MiB`],
        [`GPU ${g.index} temperature / power`, `${fmt(g.temperature_c,0)} °C / ${fmt(g.power_w,1)} W`]);
    }
    if (!(r.gpus || []).length) rows.push(['GPU metrics', 'Unavailable']);
    const nets = (r.network || []).filter(n => n.route || n.rx_gbps > .001 || n.tx_gbps > .001);
    for (const n of nets) rows.push([
      `${n.interface}${n.route ? ' (stream route)' : ''} RX / TX`, `${fmt(n.rx_gbps,3)} / ${fmt(n.tx_gbps,3)} Gbit/s`],
      [`${n.interface} RX errors / drops`, `${fmt(n.rx_errors_delta,0)} / ${fmt(n.rx_drops_delta,0)} per sample`]);
    rows.push(['Resource sample age', `${fmt(t.resource_age_s)} s`]);
    const values = el('telemetry-values'); values.replaceChildren();
    for (const [name,value] of rows) {
      const row = document.createElement('div'); row.className = 'row';
      const label = document.createElement('label'); label.textContent = name;
      const output = document.createElement('span'); output.className = 'mono'; output.textContent = value;
      row.append(label,output); values.append(row);
    }
    const prev = t.previous_scan;
    el('telemetry-last').textContent = prev ? `Previous scan: ${prev.status}, ${fmt(prev.frames,0)} frames, ` +
      `${fmt(prev.average_fps == null ? null : prev.average_fps/1000)} kfps` : '';
    chart(t.history || [],t.target_fps);
  }
  async function poll() {
    try {
      const response = await fetch('telemetry', {cache: 'no-store', signal: AbortSignal.timeout(4000)});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      el('telemetry-status').textContent = 'Telemetry disconnected — displayed values are stale';
      el('telemetry-status').style.color = 'var(--bad)';
    } finally { setTimeout(poll,1000); }
  }
  poll();
})();
