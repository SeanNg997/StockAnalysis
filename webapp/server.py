from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "webapp" / "static"
RUN_OUTPUT_DIR = BASE_DIR / "output" / "web_console"
BACKTEST_DAILY_CSV = BASE_DIR / "output" / "backtest" / "backtest_daily.csv"
BACKTEST_STATE_JSON = BASE_DIR / "output" / "backtest" / "backtest_state.json"
PORTFOLIO_JSON = BASE_DIR / "output" / "portfolio.json"
PORTFOLIO_META_JSON = BASE_DIR / "output" / "portfolio_meta.json"


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    name: str
    description: str
    category: str
    accent: str
    command: list[str] | None = None
    steps: list[str] | None = None
    supports_curve: bool = False

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "accent": self.accent,
            "supports_curve": self.supports_curve,
        }


def _python_task(script_name: str) -> list[str]:
    return [sys.executable, "-u", str(BASE_DIR / "src" / script_name)]


TASKS: dict[str, TaskDefinition] = {
    "fetch_data": TaskDefinition(
        id="fetch_data",
        name="抓取数据",
        description="运行 py00，增量抓取或补齐 A 股日线数据。",
        category="数据准备",
        accent="amber",
        command=_python_task("py00_fetch_stock_data.py"),
    ),
    "clean_data": TaskDefinition(
        id="clean_data",
        name="数据清洗",
        description="运行 py01，生成清洗后的主板股票缓存。",
        category="数据准备",
        accent="teal",
        command=_python_task("py01_data_clean.py"),
    ),
    "build_features": TaskDefinition(
        id="build_features",
        name="特征工程",
        description="运行 py02，生成技术指标、动量和截面特征。",
        category="数据准备",
        accent="cyan",
        command=_python_task("py02_features.py"),
    ),
    "train_model": TaskDefinition(
        id="train_model",
        name="训练模型",
        description="运行 py03，按 config 中的固定回测起点执行 Walk-Forward 训练和预测。",
        category="模型与策略",
        accent="blue",
        command=_python_task("py03_model.py"),
    ),
    "today_strategy": TaskDefinition(
        id="today_strategy",
        name="今日策略",
        description="运行 py03 单日训练 + py04 生成交易决策（有持仓时自动生成持仓建议）。",
        category="模型与策略",
        accent="green",
        command=_python_task("py04_today.py") + ["--train-latest"],
    ),
    "run_backtest": TaskDefinition(
        id="run_backtest",
        name="运行回测",
        description="运行 py05，按 config 中的固定回测起点回测，并在页面中实时刷新收益曲线。",
        category="回测与报告",
        accent="rose",
        command=_python_task("py05_backtest.py"),
        supports_curve=True,
    ),
    "build_report": TaskDefinition(
        id="build_report",
        name="生成图表",
        description="运行 py06，输出收益曲线、回撤和热力图图片。",
        category="回测与报告",
        accent="violet",
        command=_python_task("py06_report.py"),
    ),
    "full_pipeline": TaskDefinition(
        id="full_pipeline",
        name="完整回测流水线",
        description="依次执行：抓数据 → 清洗 → 特征 → 训练 → 回测 → 图表，回测区间由 config 中的起始年份控制。",
        category="一键任务",
        accent="crimson",
        steps=["fetch_data", "clean_data", "build_features", "train_model", "run_backtest", "build_report"],
        supports_curve=True,
    ),
    "fast_strategy": TaskDefinition(
        id="fast_strategy",
        name="快速策略流水线",
        description="依次执行：抓数据 → 清洗 → 特征 → 今日策略。",
        category="一键任务",
        accent="jade",
        steps=["fetch_data", "clean_data", "build_features", "today_strategy"],
    ),
}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class TaskRun:
    task: TaskDefinition
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    returncode: int | None = None
    stop_requested: bool = False
    process: subprocess.Popen[str] | None = None
    progress_file: Path | None = None
    summary: dict[str, Any] | None = None
    error_message: str | None = None
    _log_lines: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=4000))
    _curve_points: list[dict[str, Any]] = field(default_factory=list)
    _status_events: list[dict[str, Any]] = field(default_factory=list)
    _log_seq: int = 0
    _curve_seq: int = 0
    _meta_seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def bump_meta(self) -> None:
        with self._lock:
            self._meta_seq += 1

    def append_log(self, line: str) -> None:
        cleaned = line.rstrip()
        if not cleaned:
            return
        with self._lock:
            self._log_seq += 1
            self._log_lines.append({"seq": self._log_seq, "text": cleaned})

    def add_curve_point(self, payload: dict[str, Any]) -> None:
        with self._lock:
            last_date = self._curve_points[-1]["date"] if self._curve_points else None
            if last_date == payload.get("date"):
                self._curve_points[-1] = payload
            else:
                self._curve_points.append(payload)
            self._curve_seq = len(self._curve_points)

    def add_status_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._status_events.append(payload)

    def set_summary(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.summary = payload
            self._meta_seq += 1

    def mark_started(self) -> None:
        self.status = "running"
        self.bump_meta()

    def mark_finished(self, status: str, returncode: int | None = None, error_message: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.returncode = returncode
            self.error_message = error_message
            self.finished_at = _utc_now()
            self._meta_seq += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "task_id": self.task.id,
                "task_name": self.task.name,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "supports_curve": self.task.supports_curve,
                "stop_requested": self.stop_requested,
                "summary": self.summary,
                "error_message": self.error_message,
                "meta_seq": self._meta_seq,
                "log_seq": self._log_seq,
                "curve_seq": self._curve_seq,
            }

    def logs_since(self, last_seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [item for item in self._log_lines if item["seq"] > last_seq]

    def curve_since(self, last_seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return self._curve_points[last_seq:]


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_run: TaskRun | None = None

    def current_run(self) -> TaskRun | None:
        with self._lock:
            return self._current_run

    def start_task(self, task_id: str) -> TaskRun:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="未知任务")

        with self._lock:
            if self._current_run and self._current_run.status in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="当前已有任务在运行，请先等待完成或手动停止")
            run = TaskRun(task=task)
            run.progress_file = RUN_OUTPUT_DIR / f"{run.id}_progress.jsonl"
            self._current_run = run

        worker = threading.Thread(target=self._execute_run, args=(run,), daemon=True)
        worker.start()
        return run

    def stop_current(self) -> TaskRun:
        run = self.current_run()
        if run is None:
            raise HTTPException(status_code=404, detail="当前没有正在运行的任务")
        if run.process is None or run.status not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="当前任务已经结束")

        run.stop_requested = True
        run.bump_meta()
        try:
            if os.name == "nt":
                run.process.terminate()
            else:
                os.killpg(os.getpgid(run.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return run

    def _execute_run(self, run: TaskRun) -> None:
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHON_BIN"] = sys.executable
        if run.progress_file is not None:
            env["STOCK_ANALYSIS_PROGRESS_FILE"] = str(run.progress_file)
            if run.progress_file.exists():
                run.progress_file.unlink()

        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            preexec_fn = os.setsid

        commands = self._resolve_commands(run.task)
        run.mark_started()

        progress_thread = threading.Thread(target=self._tail_progress_file, args=(run,), daemon=True)
        progress_thread.start()

        returncode = 0
        for i, (step_label, cmd) in enumerate(commands):
            if run.stop_requested:
                break
            if step_label:
                run.append_log(f"\n{'='*50}\n[{i+1}/{len(commands)}] {step_label}\n{'='*50}\n")

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    preexec_fn=preexec_fn,
                    creationflags=creationflags,
                )
            except Exception as exc:
                run.mark_finished("failed", error_message=str(exc))
                return

            run.process = process
            assert process.stdout is not None
            for line in process.stdout:
                run.append_log(line)
            returncode = process.wait()
            if returncode != 0:
                break

        progress_thread.join(timeout=1.0)
        self._drain_progress_file(run)

        if run.stop_requested and returncode != 0:
            run.mark_finished("stopped", returncode=returncode)
        elif returncode == 0:
            run.mark_finished("success", returncode=returncode)
        else:
            run.mark_finished("failed", returncode=returncode)

    def _resolve_commands(self, task: TaskDefinition) -> list[tuple[str, list[str]]]:
        if task.steps:
            result = []
            for step_id in task.steps:
                sub = TASKS[step_id]
                result.append((sub.name, sub.command))
            return result
        return [("", task.command)]

    def _tail_progress_file(self, run: TaskRun) -> None:
        idle_rounds = 0
        while True:
            changed = self._drain_progress_file(run)
            process = run.process
            process_alive = process is not None and process.poll() is None
            if changed:
                idle_rounds = 0
            else:
                idle_rounds += 1
            if not process_alive and idle_rounds >= 3:
                break
            time.sleep(0.25)

    def _drain_progress_file(self, run: TaskRun) -> bool:
        path = run.progress_file
        if path is None or not path.exists():
            return False

        changed = False
        state_path = path.with_suffix(".offset")
        offset = 0
        if state_path.exists():
            try:
                offset = int(state_path.read_text(encoding="utf-8"))
            except ValueError:
                offset = 0

        file_size = path.stat().st_size
        if offset > file_size:
            offset = 0

        with path.open("r", encoding="utf-8") as fp:
            fp.seek(offset)
            for raw_line in fp:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                self._ingest_progress_event(run, event)
                changed = True
            offset = fp.tell()

        state_path.write_text(str(offset), encoding="utf-8")
        return changed

    def _ingest_progress_event(self, run: TaskRun, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "equity":
            run.add_curve_point(payload)
        elif event_type == "summary":
            run.set_summary(payload)
        elif event_type == "status":
            run.add_status_event(payload)


manager = TaskManager()
app = FastAPI(title="Stock Analysis Web Console")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serialize_run(run: TaskRun | None) -> dict[str, Any] | None:
    return None if run is None else run.snapshot()


def load_latest_backtest_snapshot() -> dict[str, Any]:
    if not BACKTEST_DAILY_CSV.exists():
        return {
            "available": False,
            "points": [],
        }

    daily_df = pd.read_csv(BACKTEST_DAILY_CSV)
    if daily_df.empty:
        return {
            "available": False,
            "points": [],
        }

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    initial_value = float(daily_df.iloc[0]["portfolio_value"])
    curve_points = []
    for _, row in daily_df.iterrows():
        portfolio_value = float(row["portfolio_value"])
        curve_points.append(
            {
                "date": row["date"].date().isoformat(),
                "portfolio_value": portfolio_value,
                "cash": float(row.get("cash", portfolio_value)),
                "n_positions": int(row.get("n_positions", 0)),
                "n_trades": int(row.get("n_trades", 0)),
                "return_pct": round((portfolio_value / initial_value - 1) * 100, 4),
            }
        )

    cummax = daily_df["portfolio_value"].cummax()
    drawdown = (daily_df["portfolio_value"] - cummax) / cummax
    final_value = float(daily_df.iloc[-1]["portfolio_value"])

    return {
        "available": True,
        "points": curve_points,
        "summary": {
            "start_date": daily_df.iloc[0]["date"].date().isoformat(),
            "end_date": daily_df.iloc[-1]["date"].date().isoformat(),
            "observations": int(len(daily_df)),
            "initial_value": round(initial_value, 2),
            "final_value": round(final_value, 2),
            "total_return_pct": round((final_value / initial_value - 1) * 100, 4),
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 4),
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def state() -> dict[str, Any]:
    return {
        "tasks": [task.to_public() for task in TASKS.values()],
        "current_run": serialize_run(manager.current_run()),
        "latest_backtest": load_latest_backtest_snapshot(),
    }


@app.get("/api/backtest/latest")
def latest_backtest() -> dict[str, Any]:
    return load_latest_backtest_snapshot()


@app.get("/api/portfolio")
def get_portfolio() -> list:
    if PORTFOLIO_JSON.exists():
        return json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    return []


def _parse_available_cash(value: Any) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    return amount


def _load_portfolio_meta() -> dict[str, Any]:
    if not PORTFOLIO_META_JSON.exists():
        return {"available_cash": None}
    try:
        payload = json.loads(PORTFOLIO_META_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available_cash": None}
    if not isinstance(payload, dict):
        return {"available_cash": None}
    return {"available_cash": _parse_available_cash(payload.get("available_cash"))}


@app.get("/api/portfolio/meta")
def get_portfolio_meta() -> dict[str, Any]:
    return _load_portfolio_meta()


@app.post("/api/portfolio")
async def save_portfolio(request: Request) -> dict[str, Any]:
    positions = await request.json()
    PORTFOLIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_JSON.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(positions)}


@app.post("/api/portfolio/meta")
async def save_portfolio_meta(request: Request) -> dict[str, Any]:
    payload = await request.json()
    available_cash = _parse_available_cash(payload.get("available_cash") if isinstance(payload, dict) else None)
    PORTFOLIO_META_JSON.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_META_JSON.write_text(
        json.dumps({"available_cash": available_cash} if available_cash is not None else {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "available_cash": available_cash}


@app.post("/api/portfolio/extract-backtest")
def extract_backtest_positions() -> dict[str, Any]:
    """从回测状态文件提取最新持仓，覆盖写入 portfolio.json"""
    if not BACKTEST_STATE_JSON.exists():
        raise HTTPException(status_code=404, detail="回测状态文件不存在，请先运行回测")

    try:
        with open(BACKTEST_STATE_JSON, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"读取回测状态失败: {exc}")

    raw_positions = state.get("positions", {})
    if not raw_positions:
        return {"ok": True, "count": 0, "message": "回测状态中无持仓"}

    result = []
    for code, pos in raw_positions.items():
        short_code = code.split(".")[1] if "." in code else code
        buy_date = pos.get("buy_date", "")
        if buy_date:
            buy_date = pd.Timestamp(buy_date).strftime("%Y-%m-%d")
        result.append({
            "code": short_code,
            "buy_price": float(pos.get("buy_price", 0)),
            "buy_date": buy_date,
            "shares": int(float(pos.get("shares", 0))),
            "basis_amount": float(pos.get("basis_amount", 0)),
            "buy_cost": float(pos.get("buy_cost", 0)),
            "cash_dividends_received": float(pos.get("cash_dividends_received", 0)),
            "max_profit_pct": float(pos.get("max_profit_pct", 0)),
            "current_price": float(pos.get("current_price", 0)),
        })

    PORTFOLIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(result)}


@app.post("/api/tasks/{task_id}/run")
def run_task(task_id: str) -> dict[str, Any]:
    run = manager.start_task(task_id)
    return {"run": run.snapshot()}


@app.post("/api/runs/current/stop")
def stop_current_run() -> dict[str, Any]:
    run = manager.stop_current()
    return {"run": run.snapshot()}


@app.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "latest_backtest", "data": load_latest_backtest_snapshot()})

    active_run_id = None
    last_log_seq = 0
    last_curve_seq = 0
    last_meta_seq = -1

    try:
        while True:
            run = manager.current_run()
            if run is None:
                if active_run_id is not None:
                    await websocket.send_json({"type": "run_reset"})
                    active_run_id = None
                    last_log_seq = 0
                    last_curve_seq = 0
                    last_meta_seq = -1
                await asyncio.sleep(0.35)
                continue

            snapshot = run.snapshot()
            if snapshot["id"] != active_run_id:
                active_run_id = snapshot["id"]
                last_log_seq = 0
                last_curve_seq = 0
                last_meta_seq = -1
                await websocket.send_json({"type": "run_meta", "run": snapshot})

            if snapshot["meta_seq"] != last_meta_seq:
                last_meta_seq = snapshot["meta_seq"]
                await websocket.send_json({"type": "run_meta", "run": snapshot})

            log_items = run.logs_since(last_log_seq)
            if log_items:
                last_log_seq = log_items[-1]["seq"]
                await websocket.send_json({"type": "logs", "run_id": run.id, "items": log_items})

            if snapshot["status"] in {"queued", "running"}:
                curve_items = run.curve_since(last_curve_seq)
                if curve_items:
                    last_curve_seq += len(curve_items)
                    await websocket.send_json({"type": "curve", "run_id": run.id, "items": curve_items})

            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
