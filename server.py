# -*- coding: utf-8 -*-
r"""翻唱小網頁。跑起來：run_cover.bat（或 python server.py --open）

127.0.0.1:6970，只聽本機。
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote

HERE = os.path.dirname(os.path.abspath(__file__))
_extra = os.path.join(HERE, "extra_packages")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.insert(0, _extra)

import pipeline
import fetcher

WORK = os.path.join(HERE, "work")
INDEX_HTML = os.path.join(HERE, "index.html")

JOBS = {}
QUEUE = queue.Queue()
LOCK = threading.Lock()
CURRENT = {"id": None}


def new_job(name):
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = os.path.join(WORK, jid)
    os.makedirs(d, exist_ok=True)
    job = {
        "id": jid, "dir": d, "name": name,
        "status": "排隊中", "pct": 0, "stage": "排隊中", "msg": "",
        "created": time.time(), "result": None, "error": None,
        "model": "", "pitch": 0, "sep": "kim", "autotune": False,
    }
    with LOCK:
        JOBS[jid] = job
    return job


def public(job):
    return {k: v for k, v in job.items() if k != "dir"}


def runner():
    while True:
        job = QUEUE.get()
        if job is None:
            continue
        CURRENT["id"] = job["id"]
        job["status"] = "跑"
        try:
            def progress(stage, pct, msg=""):
                job["stage"], job["pct"], job["msg"] = stage, pct, msg

            res = pipeline.make_cover(
                job["dir"], job["src"], job["model_path"],
                pitch=job["pitch"], sep_model=job["sep"],
                autotune=job["autotune"], progress=progress,
            )
            job["result"] = {
                "cover": os.path.basename(res["cover"]),
                "vocals_fish": os.path.basename(res["vocals_fish"]),
                "instrumental": os.path.basename(res["instrumental"]),
                "seconds": res["seconds"],
                "model": res["model"],
                "pitch": res["pitch"],
            }
            job["status"] = "完成"
            job["stage"], job["pct"], job["msg"] = "完成", 100, ""
        except Exception as e:
            job["status"] = "失敗"
            job["error"] = f"{e}"
            job["msg"] = traceback.format_exc(limit=6)
            pipeline.log("JOB FAIL " + str(e))
        finally:
            CURRENT["id"] = None
            try:
                with open(os.path.join(job["dir"], "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({k: job[k] for k in
                               ("id", "name", "status", "result", "error", "model",
                                "pitch", "sep", "autotune", "created")},
                              f, ensure_ascii=False, indent=2)
            except Exception:
                pass


def recent_jobs(limit=12):
    out = []
    if not os.path.isdir(WORK):
        return out
    for name in sorted(os.listdir(WORK), reverse=True)[:40]:
        meta = os.path.join(WORK, name, "meta.json")
        if not os.path.isfile(meta):
            continue
        try:
            with open(meta, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def safe_name(s):
    s = os.path.basename(s or "song")
    s = re.sub(r'[\\/:*?"<>|]+', "_", s).strip() or "song"
    return s[:80]


LABELS = {
    "VOICE_HIGH60.pth": "高音 60%（唱歌預設）",
    "VOICE_TALK40.pth": "說話 40%",
    "VOICE_LOW30.pth": "低音 30%",
    "LAN2_YU1_70.pth": "蘭音 v2 70% ＋ 我 30%",
    "LAN2_YU1_50.pth": "蘭音 v2 50% ＋ 我 50%",
    "LAN1_YU1_70.pth": "蘭音 v1 70% ＋ 我 30%",
    "LAN1_YU1_50.pth": "蘭音 v1 50% ＋ 我 50%",
    "YU_V1.pth": "我自己（YU_V1）",
    "FISH_FINAL.pth": "FISH_FINAL（第一版）",
}


def vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
        )
        total, used = [int(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"total_mb": total, "used_mb": used, "free_mb": total - used}
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "FishCover/1.0"

    def log_message(self, fmt, *a):
        pass

    # ------------------------------------------------------------- util
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, download=None):
        if not os.path.isfile(path):
            self._json({"error": "no such file"}, 404)
            return
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and rng.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            from urllib.parse import quote
            self.send_header("Content-Disposition",
                             f"attachment; filename*=UTF-8''{quote(download)}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ------------------------------------------------------------- routes
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p in ("/", "/index.html"):
            self._file(INDEX_HTML)
            return

        if p == "/api/init":
            models = pipeline.list_voice_models()
            default = None
            for m in models:
                if m["file"] == "VOICE_HIGH60.pth":
                    default = m["path"]
            models_json = [{
                "file": m["file"], "path": m["path"], "size_mb": m["size_mb"],
                "label": LABELS.get(m["file"], m["file"]),
                "default": m["path"] == (default or (models[0]["path"] if models else None)),
            } for m in models]
            self._json({
                "models": models_json,
                "sep_models": [{"key": k, "file": v} for k, v in pipeline.SEP_MODELS.items()],
                "recent": recent_jobs(),
                "busy": CURRENT["id"],
                "vram": vram(),
                "sep_model_dir": pipeline.MODEL_DIR,
            })
            return

        if p == "/api/get_fetch_audio":
            dl_id = q.get("id", [""])[0]
            fn = q.get("fn", [""])[0]
            if not dl_id or not fn or ".." in dl_id or ".." in fn:
                self._json({"error": "bad request"}, 400)
                return
            target_path = os.path.join(WORK, "_downloads", safe_name(dl_id), safe_name(fn))
            self._file(target_path)
            return

        if p.startswith("/api/job/"):
            jid = p.rsplit("/", 1)[-1]
            job = JOBS.get(jid)
            if job:
                self._json(public(job))
                return
            meta = os.path.join(WORK, jid, "meta.json")
            if os.path.isfile(meta):
                with open(meta, encoding="utf-8") as f:
                    self._json(json.load(f))
                return
            self._json({"error": "no such job"}, 404)
            return

        if p.startswith("/files/"):
            parts = p.split("/", 3)
            if len(parts) < 4:
                self._json({"error": "bad path"}, 400)
                return
            jid, fname = parts[2], parts[3]
            if not re.fullmatch(r"[\w\-\.]+", jid) or ".." in fname:
                self._json({"error": "bad path"}, 400)
                return
            path = os.path.join(WORK, jid, safe_name(fname))
            self._file(path, download=q.get("dl", [None])[0])
            return

        self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p == "/api/fetch_url":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
            target_url = (data.get("url") or q.get("url", [""])[0]).strip()
            if not target_url:
                self._json({"error": "請提供網址或 BV 號"}, 400)
                return
            try:
                res = fetcher.fetch_url(target_url, cache_root=os.path.join(WORK, "_downloads"))
                dl_id = os.path.basename(os.path.dirname(res["path"]))
                self._json({
                    "ok": True,
                    "id": dl_id,
                    "title": res["title"],
                    "filename": res["filename"],
                    "source": res["source"],
                    "size": res["size"],
                    "download_url": f"/api/get_fetch_audio?id={dl_id}&fn={quote(safe_name(res['filename']))}"
                })
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if p == "/api/upload":
            name = safe_name(q.get("name", ["song.wav"])[0])
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json({"error": "空檔案"}, 400)
                return
            if length > 200 * 1024 * 1024:
                self._json({"error": "太大（上限 200MB）"}, 413)
                return

            models = {m["file"]: m for m in pipeline.list_voice_models()}
            want = q.get("model", [""])[0]
            model = models.get(want) or next(
                (m for m in pipeline.list_voice_models() if m["file"] == "VOICE_HIGH60.pth"),
                (pipeline.list_voice_models() or [None])[0])
            if not model:
                self._json({"error": "logs 裡找不到任何 .pth"}, 400)
                return

            job = new_job(name)
            ext = os.path.splitext(name)[1] or ".wav"
            src = os.path.join(job["dir"], "input" + ext)
            with open(src, "wb") as f:
                left = length
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            job.update({
                "src": src, "model_path": model["path"], "model": model["file"],
                "pitch": int(q.get("pitch", ["0"])[0] or 0),
                "sep": q.get("sep", ["kim"])[0],
                "autotune": q.get("autotune", ["0"])[0] in ("1", "true", "on"),
                "status": "排隊中",
            })
            QUEUE.put(job)
            self._json({"job": public(job)})
            return

        if p == "/api/analyze":
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json({"error": "空檔案"}, 400)
                return
            tmp = os.path.join(WORK, "_tmp_" + uuid.uuid4().hex[:6] + ".bin")
            os.makedirs(WORK, exist_ok=True)
            with open(tmp, "wb") as f:
                left = length
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            try:
                info = pipeline.measure_pitch(tmp)
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            self._json(info or {"error": "量不出來（可能沒人聲）"})
            return

        self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6970)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    threading.Thread(target=runner, daemon=True, name="job-runner").start()

    port = a.port
    srv = None
    for cand in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", cand), Handler)
            port = cand
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("6970–6989 都被佔了")

    url = f"http://127.0.0.1:{port}/"
    print("=" * 48, flush=True)
    print(f"  翻唱小網頁跑起來了：{url}", flush=True)
    print("  關掉這個視窗就結束。", flush=True)
    print("=" * 48, flush=True)
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
