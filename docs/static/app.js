const state = { board: null, selected: { left: null, right: null }, side: "left" };
const useApi = location.hostname === "127.0.0.1" || location.hostname === "localhost";
let cacheBust = "";

function pageBase() {
  let path = location.pathname;
  if (path.endsWith("index.html")) path = path.slice(0, -"index.html".length);
  if (!path.endsWith("/")) path += "/";
  return path;
}

function withBust(url) {
  return cacheBust ? `${url}${url.includes("?") ? "&" : "?"}t=${cacheBust}` : url;
}

function dashboardUrl() {
  return withBust(useApi ? "/api/dashboard" : `${pageBase()}dashboard.json`);
}

function chartUrl(code) {
  return withBust(useApi ? `/api/etf/${code}` : `${pageBase()}charts/${code}.json`);
}
const charts = {};

const $ = (id) => document.getElementById(id);

function codesOf(text) {
  return new Set((text || "").toUpperCase().split(/[^0-9A-Z]+/).filter((item) => item.length >= 4));
}

function loadHolds() {
  $("hold-left").value = localStorage.getItem("hold-left") || "";
  $("hold-right").value = localStorage.getItem("hold-right") || "";
}

function saveHolds() {
  localStorage.setItem("hold-left", $("hold-left").value);
  localStorage.setItem("hold-right", $("hold-right").value);
  renderTables();
}

function pctClass(value) {
  if (value == null || value === 0) return "";
  return value > 0 ? "up" : "down";
}

function fmt(value, digits = 2, suffix = "") {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function seal(el, mode) {
  el.textContent = mode || "—";
  el.className = "seal " + (mode === "开启" ? "on" : mode === "降速" ? "slow" : "off");
}

function listOf(side, kind) {
  const rows = state.board?.[side] || [];
  const query = $(side === "left" ? "q-left" : "q-right").value.trim();
  const includeWatch = $(side === "left" ? "hot-left" : "hot-right").checked;
  const actionKey = side === "left" ? "action_left" : "action_right";
  const toneKey = side === "left" ? "tone_left" : "tone_right";
  return rows.filter((row) => {
    const blob = `${row.code} ${row.name}`;
    if (query && !blob.includes(query)) return false;
    const tone = row[toneKey];
    const action = row[actionKey];
    const isSell = tone === "sell" || action === "周五再定";
    const isWatch = ["观察", "暂缓", "轨道关闭"].includes(action);
    if (kind === "sell") return isSell;
    if (tone === "buy") return true;
    return includeWatch && isWatch;
  });
}

function renderTables() {
  const empty = `<tr><td colspan="7" class="empty">还没有本地行情。点右上角「更新今日数据」。</td></tr>`;
  if (!state.board?.ready) {
    ["left-buy", "left-sell", "right-buy", "right-sell"].forEach((id) => { $(id).innerHTML = empty; });
    return;
  }
  const rightHolds = codesOf($("hold-right").value);
  drawBody("left", "buy", listOf("left", "buy"), rightHolds);
  drawBody("left", "sell", listOf("left", "sell"), rightHolds);
  drawBody("right", "buy", listOf("right", "buy"), rightHolds);
  drawBody("right", "sell", listOf("right", "sell"), rightHolds);
}

function drawBody(side, kind, rows, rightHolds) {
  const body = $(`${side}-${kind}`);
  if (!rows.length) {
    const label = kind === "buy" ? "买点" : "卖点";
    body.innerHTML = `<tr><td colspan="7" class="empty">今天没有${label}。</td></tr>`;
    return;
  }
  const actionKey = side === "left" ? "action_left" : "action_right";
  const toneKey = side === "left" ? "tone_left" : "tone_right";
  const posKey = side === "left" ? "position_left" : "position_right";
  const biasKey = side === "left" ? "weekly_bias6" : "bias20";
  body.innerHTML = rows.map((row) => {
    const isolated = side === "left" && rightHolds.has(row.code);
    const active = state.selected[side] === row.code ? "active" : "";
    return `<tr class="${active}" data-side="${side}" data-code="${row.code}">
      <td><span class="name">${row.name}</span><span class="code">${row.code}</span></td>
      <td>${row.category}</td>
      <td class="num ${pctClass(row.pct)}">${fmt(row.close, 3)}<br>${fmt(row.pct, 2, "%")}</td>
      <td class="num">${fmt(row[biasKey], 1, "%")}</td>
      <td class="num">${fmt(row.vol_ratio)}</td>
      <td><span class="tag ${row[toneKey]}">${row[actionKey]}</span>${isolated ? '<div class="flag">右侧已持有，左侧不抄</div>' : ""}</td>
      <td>${row[posKey]}</td>
    </tr>`;
  }).join("");
}

function renderCommander() {
  const market = state.board?.market;
  $("commander").hidden = !market;
  if (!market) return;
  $("headline").textContent = market.headline;
  $("tab-note").textContent = "开仓看 ETF 自己的周线 MA20";
  $("market-nums").innerHTML = [
    ["收盘", fmt(market.close, 2)],
    ["周线MA20", fmt(market.weekly_ma20, 2)],
    ["周线MA60", fmt(market.weekly_ma60, 2)],
    ["MACD", market.macd_state],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  $("steps").innerHTML = market.steps.map((step) => `<li><strong>${step.n} ${step.title}</strong><span>${step.text}</span></li>`).join("");
}

async function loadDashboard() {
  const res = await fetch(dashboardUrl());
  state.board = await res.json();
  const info = state.board.status || {};
  const err = info.error ? ` · ${info.error}` : "";
  $("status-line").textContent = info.updated_at
    ? `上次更新 ${info.updated_at} · 已缓存 ${info.symbols || 0} 个品种${err}`
    : `${info.message || "等待第一次更新"}${err}`;
  renderCommander();
  renderTables();
}

async function pollStatus() {
  const res = await fetch("/api/status");
  const info = await res.json();
  $("refresh").disabled = !!info.running;
  if (info.running) {
    $("status-line").textContent = `${info.message}（${info.done}/${info.total}）`;
    setTimeout(pollStatus, 1200);
    return;
  }
  await loadDashboard();
}

async function refreshData() {
  if (!useApi) {
    $("refresh").disabled = true;
    $("status-line").textContent = "正在读取已发布的看板…";
    cacheBust = Date.now().toString();
    try {
      await loadDashboard();
      const code = state.selected[state.side];
      if (code) await openRow(state.side, code);
    } catch (err) {
      $("status-line").textContent = `读取失败 · ${err.message || err}`;
    } finally {
      $("refresh").disabled = false;
    }
    return;
  }
  const limit = Number($("limit").value) || 120;
  const minAmount = (Number($("min-amount").value) || 0) * 10000;
  $("refresh").disabled = true;
  const res = await fetch(`/api/refresh?limit=${limit}&min_amount=${minAmount}`, { method: "POST" });
  const data = await res.json();
  if (!data.started && data.status?.running) {
    $("status-line").textContent = "已经在更新";
  }
  pollStatus();
}

function rowByCode(code) {
  return (state.board?.right || []).find((row) => row.code === code) || null;
}

function syncSheet() {
  const panel = $(`panel-${state.side}`);
  const dock = $(`${state.side}-dock`);
  const open = !!(panel && dock && !panel.hidden && !dock.hidden);
  document.body.classList.toggle("sheet-open", open);
}

function closeDock(side) {
  destroyChart(side);
  const dock = $(`${side}-dock`);
  dock.hidden = true;
  dock.innerHTML = "";
  state.selected[side] = null;
  renderTables();
  syncSheet();
}

function firstCode(side) {
  return (listOf(side, "buy")[0] || listOf(side, "sell")[0] || {}).code || null;
}

function destroyChart(side) {
  if (charts[side]) {
    charts[side].price.remove();
    charts[side].macd.remove();
    charts[side] = null;
  }
}

function dockMessage(side, text) {
  const dock = $(`${side}-dock`);
  dock.hidden = false;
  dock.innerHTML = `<p>${text}</p><div class="dock-bar"><button type="button" class="close-dock" data-close="${side}">关闭</button></div>`;
  syncSheet();
}

async function openRow(side, code) {
  state.selected[side] = code;
  renderTables();
  const dock = $(`${side}-dock`);
  dock.hidden = false;
  dock.innerHTML = `<div><p>正在读取 ${code} 的K线和条件对照…</p></div><div class="dock-bar"><button type="button" class="close-dock" data-close="${side}">关闭</button></div>`;
  syncSheet();
  const res = await fetch(chartUrl(code));
  if (!res.ok) {
    dockMessage(side, `没有读到 ${code} 的K线。`);
    return;
  }
  const data = await res.json();
  const signal = data.signal || rowByCode(code);
  if (!signal) {
    dockMessage(side, `没有找到 ${code} 的信号。`);
    return;
  }
  const checks = side === "left" ? signal.checks_left : signal.checks_right;
  const detail = side === "left" ? signal.detail_left : signal.detail_right;
  const action = side === "left" ? signal.action_left : signal.action_right;
  const stop = side === "left" && signal.stop ? `止损参考 ${fmt(signal.stop, 3)}，约在支撑下方 2.5%。` : "";
  dock.innerHTML = `
    <div>
      <h3>${signal.name}</h3>
      <p>${action}。${detail}</p>
      <p>${signal.rs_text || ""} ${stop}</p>
      <ul class="checks">
        ${(checks || []).map((item) => `<li><span class="mark ${item.ok ? "" : "no"}">${item.ok ? "是" : "否"}</span><span>${item.text}</span></li>`).join("")}
      </ul>
    </div>
    <div>
      <div class="dock-bar"><button type="button" class="close-dock" data-close="${side}">关闭</button></div>
      <div id="chart-${side}" class="chart"></div>
      <div id="macd-${side}" class="chart-sub"></div>
    </div>`;
  drawChart(side, data);
  syncSheet();
}

function drawChart(side, data) {
  destroyChart(side);
  const priceEl = document.getElementById(`chart-${side}`);
  const macdEl = document.getElementById(`macd-${side}`);
  const base = {
    layout: { background: { color: "transparent" }, textColor: "#1c2b33", fontFamily: "IBM Plex Mono" },
    grid: { vertLines: { color: "rgba(28,43,51,0.06)" }, horzLines: { color: "rgba(28,43,51,0.06)" } },
    rightPriceScale: { borderColor: "#d5e0e2" },
    timeScale: { borderColor: "#d5e0e2" },
    crosshair: { mode: 0 },
  };
  const price = LightweightCharts.createChart(priceEl, { ...base, autoSize: true });
  const candle = price.addCandlestickSeries({
    upColor: "#ef6a62", downColor: "#3cba8b", borderUpColor: "#ef6a62", borderDownColor: "#3cba8b", wickUpColor: "#ef6a62", wickDownColor: "#3cba8b",
  });
  candle.setData(data.candles || []);
  const ma20 = price.addLineSeries({ color: "#b85a32", lineWidth: 2, priceLineVisible: false });
  const ma60 = price.addLineSeries({ color: "#1d6b73", lineWidth: 1, priceLineVisible: false });
  ma20.setData(data.ma20 || []);
  ma60.setData(data.ma60 || []);
  const macd = LightweightCharts.createChart(macdEl, { ...base, autoSize: true });
  const hist = macd.addHistogramSeries({ priceLineVisible: false });
  hist.setData((data.hist || []).map((item) => ({
    time: item.time,
    value: item.value,
    color: item.value >= 0 ? "rgba(239,106,98,0.85)" : "rgba(60,186,139,0.85)",
  })));
  const fit = () => {
    price.timeScale().fitContent();
    macd.timeScale().fitContent();
  };
  fit();
  requestAnimationFrame(fit);
  charts[side] = { price, macd };
}

document.body.addEventListener("click", (event) => {
  const closer = event.target.closest("[data-close]");
  if (closer) {
    closeDock(closer.dataset.close);
    return;
  }
  const row = event.target.closest("tr[data-code]");
  if (!row) return;
  openRow(row.dataset.side, row.dataset.code);
});

function showSide(side) {
  state.side = side;
  $("panel-left").hidden = side !== "left";
  $("panel-right").hidden = side !== "right";
  $("tab-left").classList.toggle("is-on", side === "left");
  $("tab-right").classList.toggle("is-on", side === "right");
  $("tab-left").setAttribute("aria-selected", side === "left" ? "true" : "false");
  $("tab-right").setAttribute("aria-selected", side === "right" ? "true" : "false");
  const chart = charts[side];
  if (chart) {
    chart.price.timeScale().fitContent();
    chart.macd.timeScale().fitContent();
  }
  syncSheet();
  if (state.board?.ready && !state.selected[side]) {
    const code = firstCode(side);
    if (code) openRow(side, code);
  }
}

["q-left", "q-right", "hot-left", "hot-right"].forEach((id) => {
  $(id).addEventListener("input", renderTables);
  $(id).addEventListener("change", renderTables);
});
$("tab-left").addEventListener("click", () => showSide("left"));
$("tab-right").addEventListener("click", () => showSide("right"));
["hold-left", "hold-right"].forEach((id) => $(id).addEventListener("change", saveHolds));
$("refresh").addEventListener("click", refreshData);
if (!useApi) {
  document.querySelectorAll(".controls label").forEach((el) => { el.hidden = true; });
}
loadHolds();
loadDashboard().then(async () => {
  if (state.board?.ready && !state.selected[state.side]) {
    const code = firstCode(state.side);
    if (code) await openRow(state.side, code);
  }
  if (!useApi) return;
  const res = await fetch("/api/status");
  const info = await res.json();
  if (info.running) pollStatus();
});
