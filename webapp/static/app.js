const state = {
  tasks: [],
  currentRun: null,
  latestBacktest: null,
  liveCurve: [],
  chartHoverIndex: null,
  chartHoverX: null,
  chartHoverXPx: null,
  chartResizeObserver: null,
  logLines: [],
  socket: null,
  taskGroupOpen: new Map(),
};

const dom = {
  taskGroups: document.getElementById("task-groups"),
  taskLockChip: document.getElementById("task-lock-chip"),
  stopRunBtn: document.getElementById("stop-run-btn"),
  refreshLatestBtn: document.getElementById("refresh-latest-btn"),
  wsStatusChip: document.getElementById("ws-status-chip"),
  metricReturn: document.getElementById("metric-return"),
  metricDate: document.getElementById("metric-date"),
  runTaskName: document.getElementById("run-task-name"),
  runStatusText: document.getElementById("run-status-text"),
  runDuration: document.getElementById("run-duration"),
  logOutput: document.getElementById("log-output"),
  logCountChip: document.getElementById("log-count-chip"),
  chart: document.getElementById("equity-chart"),
  chartEmpty: document.getElementById("chart-empty"),
  chartOverlay: document.getElementById("chart-overlay"),
  portfolioTable: document.getElementById("portfolio-table"),
  portfolioAddBtn: document.getElementById("portfolio-add-btn"),
  portfolioSaveBtn: document.getElementById("portfolio-save-btn"),
  portfolioExtractBtn: document.getElementById("portfolio-extract-btn"),
  portfolioCashInput: document.getElementById("portfolio-cash-input"),
};

function setText(element, value) {
  if (!element) return;
  element.textContent = value;
}

function setClassName(element, value) {
  if (!element) return;
  element.className = value;
}

function setChartOverlayVisible(visible) {
  if (!dom.chartOverlay) return;
  dom.chartOverlay.style.display = visible ? "grid" : "none";
}

function positionChartOverlay(anchorX, anchorY, chartWidth, chartHeight) {
  if (!dom.chartOverlay) return;
  const overlay = dom.chartOverlay;
  const margin = 10;
  const offsetX = 12;
  const offsetY = 14;
  const tooltipWidth = overlay.offsetWidth || 140;
  const tooltipHeight = overlay.offsetHeight || 68;

  const minLeft = margin;
  const maxLeft = chartWidth - tooltipWidth - margin;
  const minTop = margin;
  const maxTop = chartHeight - tooltipHeight - margin;

  let left = anchorX + offsetX;
  if (left > maxLeft) left = anchorX - tooltipWidth - offsetX;
  left = Math.max(minLeft, Math.min(left, maxLeft));

  let top = anchorY + offsetY;
  top = Math.max(minTop, Math.min(top, maxTop));

  overlay.style.left = `${left}px`;
  overlay.style.top = `${top}px`;
}

function formatDuration(durationMs) {
  if (!Number.isFinite(durationMs) || durationMs < 0) return "--";
  const totalSeconds = Math.floor(durationMs / 1000);
  const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  return `${num >= 0 ? "+" : ""}${num.toFixed(2)}%`;
}

function getNiceStep(range, targetTicks = 6) {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const roughStep = range / Math.max(2, targetTicks);
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / power;

  let niceNormalized;
  if (normalized <= 1) niceNormalized = 1;
  else if (normalized <= 2) niceNormalized = 2;
  else if (normalized <= 2.5) niceNormalized = 2.5;
  else if (normalized <= 5) niceNormalized = 5;
  else niceNormalized = 10;

  return niceNormalized * power;
}

function formatAxisPct(value, step) {
  const absStep = Math.abs(step);
  const decimals = absStep >= 1 ? 0 : absStep >= 0.1 ? 1 : 2;
  const normalized = Number(value.toFixed(decimals));
  return `${normalized}%`;
}

function setPctStyle(element, value) {
  if (!element) return;
  element.classList.remove("is-positive", "is-negative");
  if (value > 0) element.classList.add("is-positive");
  if (value < 0) element.classList.add("is-negative");
}

function groupTasks(tasks) {
  return tasks.reduce((acc, task) => {
    if (!acc[task.category]) acc[task.category] = [];
    acc[task.category].push(task);
    return acc;
  }, {});
}

function isTaskRunning() {
  return state.currentRun && ["queued", "running"].includes(state.currentRun.status);
}

function isDefaultGroupOpen(category) {
  return category === "一键任务";
}

function renderTasks() {
  if (!dom.taskGroups) return;
  const grouped = groupTasks(state.tasks);
  dom.taskGroups.innerHTML = "";
  const categories = Object.entries(grouped).sort(([a], [b]) => {
    if (a === "一键任务") return -1;
    if (b === "一键任务") return 1;
    return 0;
  });
  categories.forEach(([category, tasks]) => {
    const groupEl = document.createElement("section");
    groupEl.className = `task-group${category === "一键任务" ? " task-group--pipeline" : ""}`;
    const details = document.createElement("details");
    const knownOpenState = state.taskGroupOpen.get(category);
    details.open = knownOpenState ?? isDefaultGroupOpen(category);
    details.innerHTML = `
      <summary>
        <span>${category}</span>
        <span class="task-count">${tasks.length}</span>
      </summary>
      <div class="task-list"></div>
    `;
    details.addEventListener("toggle", () => {
      state.taskGroupOpen.set(category, details.open);
    });
    const listEl = details.querySelector(".task-list");

    tasks.forEach((task) => {
      const card = document.createElement("article");
      card.className = `task-card accent-${task.accent}${task.category === "一键任务" ? " pipeline-card" : ""}`;

      const running = isTaskRunning();
      const disabled = running;
      const buttonLabel = running && state.currentRun?.task_id === task.id ? "运行中" : "执行";

      card.innerHTML = `
        <div>
          <div class="task-title-row">
            <h4>${task.name}</h4>
            <span class="task-meta">${task.supports_curve ? "实时曲线" : "实时日志"}</span>
          </div>
          <p>${task.description}</p>
        </div>
        <div class="task-card-footer">
          <span class="task-meta">${running && state.currentRun?.task_id === task.id ? "任务执行中" : "点击运行"}</span>
          <button class="task-run-btn" type="button" ${disabled ? "disabled" : ""}>${buttonLabel}</button>
        </div>
      `;

      card.querySelector("button").addEventListener("click", () => runTask(task.id));
      listEl.appendChild(card);
    });

    groupEl.appendChild(details);
    dom.taskGroups.appendChild(groupEl);
  });
}

function renderRunStatus() {
  const run = state.currentRun;
  const running = isTaskRunning();

  if (dom.stopRunBtn) dom.stopRunBtn.disabled = !running;
  setText(dom.taskLockChip, running ? "任务运行中" : "空闲中");
  setClassName(dom.taskLockChip, running ? "chip" : "chip chip-muted");

  if (!run) {
    setText(dom.runTaskName, "暂无运行任务");
    setText(dom.runStatusText, "Idle");
    setText(dom.runDuration, "--");
    return;
  }

  const statusMap = {
    queued: "排队中",
    running: "运行中",
    success: "已完成",
    failed: "失败",
    stopped: "已停止",
  };

  setText(dom.runTaskName, run.task_name);
  setText(dom.runStatusText, statusMap[run.status] || run.status);

  const startedAt = run.started_at ? new Date(run.started_at) : null;
  if (!startedAt || Number.isNaN(startedAt.getTime())) {
    setText(dom.runDuration, "--");
    return;
  }

  const finishedAt = run.finished_at ? new Date(run.finished_at) : new Date();
  const durationMs = finishedAt.getTime() - startedAt.getTime();
  setText(dom.runDuration, formatDuration(durationMs));
}

function appendLogs(items) {
  items.forEach((item) => {
    state.logLines.push(item.text);
  });
  if (state.logLines.length > 1200) {
    state.logLines = state.logLines.slice(-1200);
  }
  if (dom.logOutput) {
    dom.logOutput.textContent = state.logLines.join("\n");
    dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
  }
  setText(dom.logCountChip, `${state.logLines.length} 行`);
}

function resetLogs() {
  state.logLines = [];
  if (dom.logOutput) dom.logOutput.textContent = "";
  setText(dom.logCountChip, "0 行");
}

function getChartPoints() {
  if (state.liveCurve.length) return state.liveCurve;
  if (state.latestBacktest?.points?.length) return state.latestBacktest.points;
  return [];
}

function renderMetrics(points) {
  if (!points.length) {
    setText(dom.metricReturn, "--");
    setText(dom.metricDate, "--");
    return;
  }

  const safeHoverIndex = state.chartHoverIndex === null
    ? points.length - 1
    : Math.max(0, Math.min(points.length - 1, state.chartHoverIndex));
  const focusPoint = points[safeHoverIndex];

  setText(dom.metricReturn, formatPct(focusPoint.return_pct));
  setText(dom.metricDate, focusPoint.date || "--");
  setPctStyle(dom.metricReturn, Number(focusPoint.return_pct));
}

function renderChart() {
  if (!dom.chart || !dom.chartEmpty) return;
  const points = getChartPoints();
  renderMetrics(points);

  const chartHostRect = dom.chart.parentElement?.getBoundingClientRect();
  const chartRectNow = dom.chart.getBoundingClientRect();
  const viewportWidth = Math.max(
    Math.floor(chartHostRect?.width || chartRectNow.width || dom.chart.clientWidth || 880),
    320
  );
  const viewportHeight = Math.max(
    Math.floor(chartHostRect?.height || chartRectNow.height || dom.chart.clientHeight || 340),
    260
  );
  const width = viewportWidth;
  const height = viewportHeight;
  dom.chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  dom.chart.setAttribute("preserveAspectRatio", "xMidYMid meet");

  if (!points.length) {
    state.chartHoverIndex = null;
    state.chartHoverX = null;
    state.chartHoverXPx = null;
    dom.chart.innerHTML = "";
    dom.chartEmpty.style.display = "grid";
    setChartOverlayVisible(false);
    return;
  }

  dom.chartEmpty.style.display = "none";
  const pad = { top: 30, right: 8, bottom: 28, left: 34 };
  const gainColor = "#c63d36";
  const lossColor = "#2f8f4a";
  const values = points.map((point) => Number(point.return_pct || 0));
  const baselineValues = values.concat([0]);
  const minValue = Math.min(...baselineValues);
  const maxValue = Math.max(...baselineValues);
  const spread = Math.max(1, maxValue - minValue);

  // Use a zero-based "nice" axis so labels are integer-like (20%, 40%, ...).
  let paddedLower = Math.min(0, minValue - spread * 0.10);
  let paddedUpper = Math.max(0, maxValue + spread * 0.10);
  const step = getNiceStep(paddedUpper - paddedLower, 6);
  let lower = Math.floor(paddedLower / step) * step;
  let upper = Math.ceil(paddedUpper / step) * step;
  if (lower === upper) upper = lower + step;

  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const xForIndex = (index) => pad.left + (points.length === 1 ? plotWidth / 2 : (plotWidth * index) / (points.length - 1));
  const yForValue = (value) => pad.top + ((upper - value) / (upper - lower)) * plotHeight;
  const baselineY = yForValue(0);
  const safeHoverIndex = state.chartHoverIndex === null
    ? points.length - 1
    : Math.max(0, Math.min(points.length - 1, state.chartHoverIndex));

  const chartNodes = points.map((point, index) => ({
    x: xForIndex(index),
    y: yForValue(Number(point.return_pct || 0)),
    v: Number(point.return_pct || 0),
  }));

  const posSegments = [];
  const negSegments = [];
  const samePoint = (a, b) => Math.abs(a.x - b.x) < 0.01 && Math.abs(a.y - b.y) < 0.01;
  const addSegmentPiece = (segments, start, end) => {
    if (!segments.length) {
      segments.push([start, end]);
      return;
    }
    const lastSeg = segments[segments.length - 1];
    const lastPoint = lastSeg[lastSeg.length - 1];
    if (samePoint(lastPoint, start)) {
      lastSeg.push(end);
      return;
    }
    segments.push([start, end]);
  };

  for (let i = 0; i < chartNodes.length - 1; i += 1) {
    const current = chartNodes[i];
    const next = chartNodes[i + 1];
    const currentPos = current.v >= 0;
    const nextPos = next.v >= 0;

    if (currentPos && nextPos) {
      addSegmentPiece(posSegments, current, next);
      continue;
    }

    if (!currentPos && !nextPos) {
      addSegmentPiece(negSegments, current, next);
      continue;
    }

    const crossingRatio = (0 - current.v) / (next.v - current.v);
    const crossing = {
      x: current.x + (next.x - current.x) * crossingRatio,
      y: baselineY,
      v: 0,
    };

    if (currentPos) {
      addSegmentPiece(posSegments, current, crossing);
      addSegmentPiece(negSegments, crossing, next);
    } else {
      addSegmentPiece(negSegments, current, crossing);
      addSegmentPiece(posSegments, crossing, next);
    }
  }

  const toPathD = (segments) => segments
    .filter((segment) => segment.length > 1)
    .map((segment) => segment.map((node, idx) => `${idx === 0 ? "M" : "L"} ${node.x} ${node.y}`).join(" "))
    .join(" ");

  const toAreaD = (segments) => segments
    .filter((segment) => segment.length > 1)
    .map((segment) => `M ${segment[0].x} ${baselineY} ${segment.map((node) => `L ${node.x} ${node.y}`).join(" ")} L ${segment[segment.length - 1].x} ${baselineY} Z`)
    .join(" ");

  const posLinePath = toPathD(posSegments);
  const negLinePath = toPathD(negSegments);
  const posAreaPath = toAreaD(posSegments);
  const negAreaPath = toAreaD(negSegments);

  const gridLines = [];
  const ticks = [];
  for (let value = lower; value <= upper + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }

  ticks.forEach((value) => {
    const y = yForValue(value);
    const isZeroLine = Math.abs(value) < step / 1000;
    const lineStroke = isZeroLine ? "rgba(22,48,43,0.34)" : "rgba(22,48,43,0.10)";
    const lineDash = isZeroLine ? "" : ` stroke-dasharray="4 6"`;
    const labelFill = isZeroLine ? "rgba(22,48,43,0.82)" : "rgba(22,48,43,0.55)";
    gridLines.push(`
      <line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="${lineStroke}"${lineDash}></line>
      <text x="${pad.left - 8}" y="${y + 4}" text-anchor="end" fill="${labelFill}" font-size="11">${formatAxisPct(value, step)}</text>
    `);
  });

  const firstDate = points[0].date;
  const lastDate = points[points.length - 1].date;
  const focusPoint = points[safeHoverIndex];
  const focusPointX = xForIndex(safeHoverIndex);
  const focusPointY = yForValue(Number(focusPoint.return_pct || 0));
  const focusPointValue = Number(focusPoint.return_pct || 0);
  const focusPointColor = focusPointValue >= 0 ? gainColor : lossColor;
  const hoverLineX = state.chartHoverX === null ? focusPointX : state.chartHoverX;
  const chartWidthPx = dom.chart.clientWidth || width;
  const chartHeightPx = dom.chart.clientHeight || height;
  const xScale = chartWidthPx / width;
  const yScale = chartHeightPx / height;
  const hoverLineXPx = state.chartHoverXPx === null ? hoverLineX * xScale : state.chartHoverXPx;
  const focusPointYPx = focusPointY * yScale;
  const hoverLine = state.chartHoverIndex === null
    ? ""
    : `<line x1="${hoverLineX}" y1="${pad.top}" x2="${hoverLineX}" y2="${height - pad.bottom}" stroke="#d94e4e" stroke-width="1.8"></line>`;
  const showTooltip = points.length > 0;

  dom.chart.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${gridLines.join("")}
    <line x1="${pad.left}" y1="${baselineY}" x2="${width - pad.right}" y2="${baselineY}" stroke="rgba(22,48,43,0.40)" stroke-width="2"></line>
    ${negAreaPath ? `<path d="${negAreaPath}" fill="rgba(47,143,74,0.12)"></path>` : ""}
    ${posAreaPath ? `<path d="${posAreaPath}" fill="rgba(198,61,54,0.12)"></path>` : ""}
    ${negLinePath ? `<path d="${negLinePath}" fill="none" stroke="${lossColor}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
    ${posLinePath ? `<path d="${posLinePath}" fill="none" stroke="${gainColor}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
    ${hoverLine}
    <circle cx="${focusPointX}" cy="${focusPointY}" r="6" fill="${focusPointColor}" stroke="#ffffff" stroke-width="3"></circle>
    <text x="${pad.left}" y="${height - 10}" fill="rgba(22,48,43,0.58)" font-size="12">${firstDate}</text>
    <text x="${width - pad.right}" y="${height - 10}" text-anchor="end" fill="rgba(22,48,43,0.58)" font-size="12">${lastDate}</text>
    <rect id="chart-hitbox" x="${pad.left}" y="${pad.top}" width="${plotWidth}" height="${plotHeight}" fill="transparent" pointer-events="all"></rect>
  `;

  const chartHitbox = dom.chart.querySelector("#chart-hitbox");
  setChartOverlayVisible(showTooltip);
  if (showTooltip) positionChartOverlay(hoverLineXPx, focusPointYPx, chartWidthPx, chartHeightPx);
  if (!chartHitbox) return;

  const updateHoverByPointerEvent = (event) => {
    const chartRect = dom.chart.getBoundingClientRect();
    const hitboxRect = chartHitbox.getBoundingClientRect();
    if (!hitboxRect.width) return;
    const clampedClientX = Math.max(hitboxRect.left, Math.min(hitboxRect.right, event.clientX));
    const ratio = (clampedClientX - hitboxRect.left) / hitboxRect.width;
    const hoverIndex = Math.round(ratio * (points.length - 1));
    const hoverX = pad.left + ratio * plotWidth;
    const hoverXPx = clampedClientX - chartRect.left;
    if (
      hoverIndex === state.chartHoverIndex &&
      state.chartHoverXPx !== null &&
      Math.abs(state.chartHoverXPx - hoverXPx) < 0.2
    ) return;
    state.chartHoverIndex = hoverIndex;
    state.chartHoverX = hoverX;
    state.chartHoverXPx = hoverXPx;
    renderChart();
  };

  chartHitbox.addEventListener("pointermove", (event) => {
    updateHoverByPointerEvent(event);
  });

  chartHitbox.addEventListener("pointerleave", () => {
    if (state.chartHoverIndex === null) return;
    state.chartHoverIndex = null;
    state.chartHoverX = null;
    state.chartHoverXPx = null;
    renderChart();
  });
}

function renderAll() {
  renderTasks();
  renderRunStatus();
  renderChart();
}

async function fetchState() {
  const response = await fetch("/api/state");
  if (!response.ok) throw new Error("无法获取页面状态");
  const data = await response.json();
  state.tasks = data.tasks || [];
  state.currentRun = data.current_run;
  state.latestBacktest = data.latest_backtest;
  if (!isTaskRunning()) {
    state.liveCurve = [];
  }
  renderAll();
}

async function fetchLatestBacktest() {
  const response = await fetch("/api/backtest/latest");
  if (!response.ok) throw new Error("无法刷新历史回测");
  state.latestBacktest = await response.json();
  if (!state.liveCurve.length) renderAll();
}

async function runTask(taskId) {
  const response = await fetch(`/api/tasks/${taskId}/run`, { method: "POST" });
  const data = await response.json();
  if (!response.ok) {
    alert(data.detail || "启动任务失败");
    return;
  }

  state.currentRun = data.run;
  state.liveCurve = [];
  resetLogs();
  renderAll();
}

async function stopCurrentRun() {
  const response = await fetch("/api/runs/current/stop", { method: "POST" });
  const data = await response.json();
  if (!response.ok) {
    alert(data.detail || "停止任务失败");
    return;
  }
  state.currentRun = data.run;
  renderAll();
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  state.socket = socket;
  setText(dom.wsStatusChip, "连接中");
  setClassName(dom.wsStatusChip, "chip chip-ws-disconnected");

  socket.addEventListener("open", () => {
    setText(dom.wsStatusChip, "实时已连接");
    setClassName(dom.wsStatusChip, "chip chip-ws-connected");
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);

    if (message.type === "latest_backtest") {
      state.latestBacktest = message.data;
      renderAll();
      return;
    }

    if (message.type === "run_meta") {
      state.currentRun = message.run;
      if (!isTaskRunning() && state.currentRun?.status === "success") {
        fetchLatestBacktest().catch(() => {});
      }
      renderAll();
      return;
    }

    if (message.type === "logs") {
      appendLogs(message.items || []);
      return;
    }

    if (message.type === "curve") {
      state.liveCurve = mergeCurvePoints(state.liveCurve, message.items || []);
      renderChart();
      return;
    }

    if (message.type === "run_reset") {
      state.currentRun = null;
      state.liveCurve = [];
      renderAll();
    }
  });

  socket.addEventListener("close", () => {
    setText(dom.wsStatusChip, "连接断开，重连中");
    setClassName(dom.wsStatusChip, "chip chip-ws-disconnected");
    setTimeout(connectWebSocket, 1200);
  });
}

function mergeCurvePoints(existing, incoming) {
  const map = new Map(existing.map((point) => [point.date, point]));
  incoming.forEach((point) => {
    map.set(point.date, point);
  });
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/* ── 持仓管理 ── */

let portfolioData = [];
let portfolioAvailableCash = null;
let savedSnapshot = JSON.stringify({ positions: [], available_cash: null });

function normalizeAvailableCash(value) {
  const text = String(value ?? "").trim();
  if (!text) return null;
  const amount = Number(text);
  return Number.isFinite(amount) && amount > 0 ? amount : null;
}

function getAvailableCashFromDOM() {
  if (!dom.portfolioCashInput) return null;
  return normalizeAvailableCash(dom.portfolioCashInput.value);
}

function getSavedPortfolioState() {
  try {
    const saved = JSON.parse(savedSnapshot);
    if (Array.isArray(saved)) return { positions: saved, available_cash: null };
    return {
      positions: Array.isArray(saved.positions) ? saved.positions : [],
      available_cash: normalizeAvailableCash(saved.available_cash),
    };
  } catch {
    return { positions: [], available_cash: null };
  }
}

function buildPortfolioSnapshot(positions, availableCash) {
  return JSON.stringify({
    positions,
    available_cash: normalizeAvailableCash(availableCash),
  });
}

function isPortfolioDirty() {
  return buildPortfolioSnapshot(collectPortfolioFromDOM(), getAvailableCashFromDOM()) !== savedSnapshot;
}

function renderPortfolioTable() {
  if (!dom.portfolioTable) return;
  if (portfolioData.length === 0) {
    dom.portfolioTable.innerHTML = '<p class="portfolio-empty">暂无持仓，点击"+ 添加"录入</p>';
    updateSaveBtn();
    return;
  }
  const today = new Date().toISOString().slice(0, 10);
  const savedList = getSavedPortfolioState().positions;
  let html = "";
  portfolioData.forEach((pos, idx) => {
    const saved = savedList[idx];
    const isSaved = saved
      && saved.code === (pos.code || "")
      && saved.buy_price === (pos.buy_price || 0)
      && saved.buy_date === (pos.buy_date || "")
      && saved.shares === (pos.shares || 0);
    const cls = isSaved ? "portfolio-card is-saved" : "portfolio-card is-unsaved";
    html += `<div class="${cls}" data-idx="${idx}">
      <div class="portfolio-card-header">
        <span class="portfolio-card-label">#${idx + 1}</span>
        <button class="portfolio-del-btn" type="button" data-idx="${idx}">&times;</button>
      </div>
      <label>代码</label>
      <input type="text" class="pf-code" value="${pos.code || ""}" placeholder="600519">
      <label>买入价</label>
      <input type="number" class="pf-price" value="${pos.buy_price || ""}" step="0.01" placeholder="0.00">
      <label>日期</label>
      <input type="date" class="pf-date" value="${pos.buy_date || today}">
      <label>股数</label>
      <input type="number" class="pf-shares" value="${pos.shares || ""}" step="100" placeholder="100">
    </div>`;
  });
  dom.portfolioTable.innerHTML = html;
  dom.portfolioTable.querySelectorAll(".portfolio-del-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      portfolioData.splice(Number(btn.dataset.idx), 1);
      renderPortfolioTable();
    });
  });
  dom.portfolioTable.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => {
      updateCardDirtyState();
      updateSaveBtn();
    });
  });
  updateSaveBtn();
}

function updateCardDirtyState() {
  if (!dom.portfolioTable) return;
  const savedList = getSavedPortfolioState().positions;
  dom.portfolioTable.querySelectorAll(".portfolio-card").forEach((card) => {
    const idx = Number(card.dataset.idx);
    const saved = savedList[idx];
    const code = card.querySelector(".pf-code").value.trim();
    const price = parseFloat(card.querySelector(".pf-price").value) || 0;
    const date = card.querySelector(".pf-date").value;
    const shares = parseInt(card.querySelector(".pf-shares").value, 10) || 0;
    const isSaved = saved
      && saved.code === code
      && saved.buy_price === price
      && saved.buy_date === date
      && saved.shares === shares;
    card.classList.toggle("is-saved", isSaved);
    card.classList.toggle("is-unsaved", !isSaved);
  });
}

function updateSaveBtn() {
  if (!dom.portfolioSaveBtn) return;
  const dirty = isPortfolioDirty();
  dom.portfolioSaveBtn.classList.toggle("has-changes", dirty);
  dom.portfolioSaveBtn.textContent = dirty ? "保存 *" : "已保存";
}

function collectPortfolioFromDOM() {
  if (!dom.portfolioTable) return [];
  const cards = dom.portfolioTable.querySelectorAll(".portfolio-card");
  return Array.from(cards).map((card) => ({
    code: card.querySelector(".pf-code").value.trim(),
    buy_price: parseFloat(card.querySelector(".pf-price").value) || 0,
    buy_date: card.querySelector(".pf-date").value,
    shares: parseInt(card.querySelector(".pf-shares").value, 10) || 0,
  })).filter((p) => p.code);
}

async function loadPortfolio() {
  try {
    const res = await fetch("/api/portfolio");
    portfolioData = await res.json();
  } catch {
    portfolioData = [];
  }
  try {
    const metaRes = await fetch("/api/portfolio/meta");
    const meta = await metaRes.json();
    portfolioAvailableCash = normalizeAvailableCash(meta.available_cash);
  } catch {
    portfolioAvailableCash = null;
  }
  if (dom.portfolioCashInput) {
    dom.portfolioCashInput.value = portfolioAvailableCash === null ? "" : String(portfolioAvailableCash);
  }
  savedSnapshot = buildPortfolioSnapshot(portfolioData, portfolioAvailableCash);
  renderPortfolioTable();
  updateSaveBtn();
}

async function savePortfolio() {
  portfolioData = collectPortfolioFromDOM();
  const availableCash = getAvailableCashFromDOM();
  try {
    const res = await fetch("/api/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(portfolioData),
    });
    const data = await res.json();
    const metaRes = await fetch("/api/portfolio/meta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ available_cash: availableCash }),
    });
    const metaData = await metaRes.json();
    if (data.ok && metaData.ok) {
      portfolioAvailableCash = normalizeAvailableCash(metaData.available_cash);
      if (dom.portfolioCashInput) {
        dom.portfolioCashInput.value = portfolioAvailableCash === null ? "" : String(portfolioAvailableCash);
      }
      savedSnapshot = buildPortfolioSnapshot(portfolioData, portfolioAvailableCash);
      renderPortfolioTable();
    }
  } catch (err) {
    alert("保存失败: " + err.message);
  }
}

if (dom.portfolioAddBtn) {
  dom.portfolioAddBtn.addEventListener("click", () => {
    portfolioData = collectPortfolioFromDOM();
    portfolioData.push({ code: "", buy_price: 0, buy_date: new Date().toISOString().slice(0, 10), shares: 100 });
    renderPortfolioTable();
  });
}
if (dom.portfolioSaveBtn) {
  dom.portfolioSaveBtn.addEventListener("click", savePortfolio);
}
if (dom.portfolioCashInput) {
  dom.portfolioCashInput.addEventListener("input", updateSaveBtn);
}

async function handleExtractClick() {
  const btn = document.getElementById("portfolio-extract-btn");
  if (!btn) return;
  if (btn.disabled) return;
  btn.disabled = true;
  try {
    const res = await fetch("/api/portfolio/extract-backtest", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "提取持仓失败");
      return;
    }
    await loadPortfolio();
    await savePortfolio();
  } catch (err) {
    alert("提取失败: " + err.message);
  } finally {
    btn.disabled = false;
  }
}

if (dom.portfolioExtractBtn) {
  dom.portfolioExtractBtn.addEventListener("click", handleExtractClick);
}

window.addEventListener("resize", () => renderChart());
window.setInterval(() => {
  if (isTaskRunning()) renderRunStatus();
}, 1000);
if (dom.stopRunBtn) dom.stopRunBtn.addEventListener("click", stopCurrentRun);
if (dom.refreshLatestBtn) {
  dom.refreshLatestBtn.addEventListener("click", () => {
    fetchLatestBacktest().catch((error) => alert(error.message));
  });
}

if (dom.chart && "ResizeObserver" in window) {
  state.chartResizeObserver = new ResizeObserver(() => {
    renderChart();
  });
  state.chartResizeObserver.observe(dom.chart.parentElement || dom.chart);
}

fetchState()
  .then(connectWebSocket)
  .catch((error) => {
    alert(error.message);
  });
loadPortfolio();

window.addEventListener("load", () => {
  renderChart();
});
