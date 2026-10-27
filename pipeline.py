# -*- coding: utf-8 -*-
r"""翻唱管線：分軌 → 換嗓 → 混音。

重用 2026-10-26 那四首驗過的參數，不重新發明：
    分軌  audio_separator，模型 Kim_Vocal_2.onnx（快）或 MDX23C（好但慢）
    換嗓  Applio 的常駐 worker（rvc/infer/worker.py），模型只載一次
    混音  ffmpeg amix + loudnorm=I=-14（跟那四首 final 一樣）

差別只有一個：換嗓改走 worker IPC，而不是每次叫 core.py 重載 6 秒模型。
"""
import json
import os
import subprocess
import sys
import threading
import time

APPLIO = r"C:\Users\cirue\Applio"
PY = os.path.join(APPLIO, "env", "Scripts", "python.exe")
FFMPEG = os.path.join(APPLIO, "ffmpeg.exe")
WORKER = os.path.join(APPLIO, "rvc", "infer", "worker.py")
LOGS = os.path.join(APPLIO, "logs")

HERE = os.path.dirname(os.path.abspath(__file__))
_extra = os.path.join(HERE, "extra_packages")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.insert(0, _extra)

# 分軌模型沿用昨天抓好的那份，不重下 515MB
MODEL_DIR = os.path.normpath(os.path.join(HERE, "..", "01_實驗場", "翻唱4首_20261026", "models"))
DEFAULT_INDEX = os.path.join(LOGS, "lanyin_v6", "lanyin_v6.index")

SEP_MODELS = {
    "kim": "Kim_Vocal_2.onnx",
    "mdx23c": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
}

# 唱歌的預設參數（昨天四首的定案；protect 0.5 是那時用的，留著）
CONVERT = {
    "indexRate": 0.75,
    "protect": 0.5,
    "volumeEnvelope": 1.0,
    "f0Method": "rmvpe",
    "exportFormat": "WAV",
    "embedderModel": "contentvec",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 模型清單

def list_voice_models():
    """logs 底下所有可用的 .pth（排除訓練中途的 G_/D_）。"""
    out = []
    for name in os.listdir(LOGS):
        path = os.path.join(LOGS, name)
        if not name.lower().endswith(".pth"):
            continue
        if name.startswith(("G_", "D_")):
            continue
        if not os.path.isfile(path):
            continue
        out.append({
            "file": name,
            "path": path,
            "size_mb": round(os.path.getsize(path) / 1048576, 1),
            "mtime": os.path.getmtime(path),
        })
    return sorted(out, key=lambda m: -m["mtime"])


def guess_index(pth_path):
    """找模型對應的 index。lanyin 系列全部吃 v6 的 index（跟 rvc_client 一樣）。"""
    stem = os.path.splitext(os.path.basename(pth_path))[0].lower()
    if stem.startswith(("voice_", "lan1_", "lan2_", "tri_", "_", "yu_", "fish_final", "lanyin_v6")):
        return DEFAULT_INDEX if os.path.isfile(DEFAULT_INDEX) else ""
    for d in os.listdir(LOGS):
        idx = os.path.join(LOGS, d, f"{d}.index")
        if os.path.isfile(idx) and stem.startswith(d.lower()):
            return idx
    return DEFAULT_INDEX if os.path.isfile(DEFAULT_INDEX) else ""


# ---------------------------------------------------------------- 分軌

def to_wav(src, dst, sr=44100):
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-i", src, "-vn", "-ar", str(sr), "-ac", "2", "-c:a", "pcm_s16le", dst],
        check=True,
    )
    return dst


def duration_of(path):
    r = subprocess.run(
        [FFMPEG.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def separate(in_wav, outdir, sep_model="kim"):
    """回傳 (人聲路徑, 伴奏路徑)。"""
    os.environ["PATH"] = APPLIO + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, APPLIO)
    from audio_separator.separator import Separator

    model_file = SEP_MODELS.get(sep_model, SEP_MODELS["kim"])
    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"分軌模型不在：{model_path}")

    sep = Separator(
        model_file_dir=MODEL_DIR,
        output_dir=outdir,
        output_format="WAV",
        sample_rate=44100,
        use_autocast=True,
    )
    sep.load_model(model_filename=model_file)
    # audio_separator 回的是「檔名」，不是路徑（昨天踩過）
    files = [f if os.path.isabs(f) else os.path.join(outdir, f) for f in sep.separate(in_wav)]

    vocals = next((f for f in files if "(Vocals)" in f), None)
    inst = next((f for f in files if "(Instrumental)" in f), None)
    if not vocals or not inst:
        raise RuntimeError(f"分軌沒吐齊，拿到：{files}")

    v_dst = os.path.join(outdir, "vocals.wav")
    i_dst = os.path.join(outdir, "instrumental.wav")
    for src, dst in ((vocals, v_dst), (inst, i_dst)):
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
    return v_dst, i_dst


# ---------------------------------------------------------------- 換嗓

class RvcWorker:
    """Applio 常駐 worker 的殼。一次只留一個模型在顯存。"""

    def __init__(self):
        self.proc = None
        self.model = None
        self._job = 0
        self._done = {}
        self._cv = threading.Condition()
        self._ready = False
        self._dead = False
        self._logs = []

    def _spawn(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PATH"] = os.path.join(APPLIO, "env", "Scripts") + os.pathsep + APPLIO + os.pathsep + env.get("PATH", "")
        self._ready = False
        self._dead = False
        self._done = {}
        self.proc = subprocess.Popen(
            [PY, "-u", WORKER], cwd=APPLIO, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._read_loop, daemon=True, name="rvc-worker").start()
        t0 = time.time()
        while time.time() - t0 < 180:
            if self._ready:
                return
            if self._dead:
                raise RuntimeError("RVC worker 起不來（stdout 關了）：" + " | ".join(self._logs[-3:]))
            time.sleep(0.05)
        raise TimeoutError("RVC worker 180 秒沒 ready")

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._logs.append(line)
                continue
            if not msg.get("_applio_ipc"):
                self._logs.append(line)
                continue
            kind = msg.get("type")
            if kind == "ready":
                self._ready = True
            elif kind in ("done", "error", "unloaded"):
                with self._cv:
                    self._done[msg.get("id")] = msg
                    self._cv.notify_all()
            elif kind == "log":
                self._logs.append(str(msg.get("message")))
                if len(self._logs) > 200:
                    del self._logs[:100]
        with self._cv:
            self._dead = True
            self._cv.notify_all()

    def _send(self, payload):
        self._job += 1
        payload = {"id": self._job, **payload}
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._job

    def _wait(self, job_id, timeout=900.0):
        deadline = time.time() + timeout
        with self._cv:
            while job_id not in self._done:
                if self._dead:
                    raise RuntimeError("RVC worker 掛了：" + " | ".join(self._logs[-3:]))
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError(f"RVC job {job_id} 超時")
                self._cv.wait(min(left, 2.0))
            return self._done.pop(job_id)

    def use(self, pth_path):
        if self.model == pth_path and self.proc and self.proc.poll() is None and not self._dead:
            return
        self.close()
        self._spawn()
        job = self._send({"command": "preload_model", "pthPath": pth_path})
        res = self._wait(job)
        if not res.get("success"):
            raise RuntimeError(f"預載模型失敗：{res.get('error')}")
        self.model = pth_path
        log(f"模型載好：{os.path.basename(pth_path)}")

    def convert(self, src, dst, pth_path, pitch=0, autotune=False, index_rate=None, protect=None):
        self.use(pth_path)
        params = dict(CONVERT)
        if index_rate is not None:
            params["indexRate"] = float(index_rate)
        if protect is not None:
            params["protect"] = float(protect)
        params["pitch"] = int(pitch)
        params["pthPath"] = pth_path
        params["indexPath"] = guess_index(pth_path)
        params["f0Autotune"] = bool(autotune)
        params["f0AutotuneStrength"] = 1.0
        job = self._send({"command": "infer", "inputPath": src, "outputPath": dst, "params": params})
        res = self._wait(job)
        if not res.get("success"):
            raise RuntimeError(f"換嗓失敗：{res.get('error')}")
        out = res.get("outputPath") or dst
        if not os.path.isfile(out) or os.path.getsize(out) < 2048:
            raise RuntimeError(f"換嗓回報成功但檔案不對：{out}")
        return out

    def close(self):
        if not self.proc:
            return
        try:
            job = self._send({"command": "unload"})
            self._wait(job, timeout=30)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        self.model = None


# ---------------------------------------------------------------- 混音

def mix(inst, vocal, out_mp3, out_wav=None):
    fc = (
        "[0:a]aresample=48000,volume=1.0[i];"
        "[1:a]aresample=48000,volume=1.0[v];"
        "[i][v]amix=inputs=2:duration=longest:dropout_transition=0,volume=2,"
        "loudnorm=I=-14:TP=-1.5:LRA=11[mix]"
    )
    base = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", inst, "-i", vocal, "-filter_complex", fc, "-map", "[mix]",
            "-ar", "48000", "-ac", "2"]
    subprocess.run(base + ["-c:a", "libmp3lame", "-b:a", "320k", out_mp3], check=True)
    if out_wav:
        subprocess.run(base + ["-c:a", "pcm_s16le", out_wav], check=True)
    return out_mp3


# ---------------------------------------------------------------- 音高

def measure_pitch(path, seconds=40):
    """量人聲的中位音高（Hz），順便給個半音建議。"""
    import numpy as np
    import librosa

    try:
        dur = duration_of(path)
        offset = max(0.0, (dur - seconds) / 3.0)
        y, sr = librosa.load(path, sr=22050, mono=True, offset=offset, duration=seconds)
        f0 = librosa.yin(y, fmin=65, fmax=900, sr=sr)
        voiced = f0[(f0 > 65) & (f0 < 900)]
        if len(voiced) < 20:
            return None
        med = float(np.median(voiced))
    except Exception as e:
        log(f"量音高失敗：{e}")
        return None

    if med < 175:
        semi = 12
    elif med > 400:
        semi = -12
    else:
        semi = 0
    return {"median_hz": round(med, 1), "suggest_pitch": semi,
            "note": f"中位音高 {med:.0f} Hz"}


# ---------------------------------------------------------------- 整首

def make_cover(job_dir, src, pct_path, pitch=0, sep_model="kim",
               autotune=False, progress=None):
    """跑完整首。progress(stage, pct, msg) 會被一直叫。"""
    def emit(stage, pct, msg=""):
        log(f"{stage} {pct}% {msg}")
        if progress:
            try:
                progress(stage, pct, msg)
            except Exception:
                pass

    os.makedirs(job_dir, exist_ok=True)

    emit("準備", 3, "轉檔")
    in_wav = to_wav(src, os.path.join(job_dir, "song.wav"))

    emit("分軌", 8, "把人聲跟伴奏拆開（第一次跑要載模型）")
    t0 = time.time()
    vocals, inst = separate(in_wav, job_dir, sep_model)
    emit("分軌", 45, f"拆好了，{time.time() - t0:.0f} 秒")

    emit("換嗓", 48, "載模型")
    est = max(20.0, duration_of(vocals) * 0.5)
    t1 = time.time()
    worker = _get_worker()
    # 換嗓期間沒有真實進度，照時間往前爬（誠實說明：這是估計）
    th = _Ticker(lambda p: emit("換嗓", p, f"跑了 {time.time() - t1:.0f} 秒"), 48, 85, est)
    th.start()
    try:
        conv = worker.convert(vocals, os.path.join(job_dir, "vocals_fish.wav"),
                              pct_path, pitch=pitch, autotune=autotune)
    finally:
        th.stop()
    emit("換嗓", 86, f"換好了，{time.time() - t1:.0f} 秒")

    emit("混音", 90, "伴奏回來")
    out_mp3 = mix(inst, conv, os.path.join(job_dir, "cover.mp3"),
                  out_wav=os.path.join(job_dir, "cover.wav"))
    emit("完成", 100, os.path.basename(out_mp3))
    return {
        "cover": out_mp3,
        "vocals_fish": conv,
        "vocals": vocals,
        "instrumental": inst,
        "pitch": pitch,
        "model": os.path.basename(pct_path),
        "sep_model": sep_model,
        "seconds": round(time.time() - t0, 1),
    }


class _Ticker(threading.Thread):
    def __init__(self, emit, lo, hi, est):
        super().__init__(daemon=True)
        self.emit, self.lo, self.hi, self.est = emit, lo, hi, est
        self._stop = threading.Event()

    def run(self):
        t0 = time.time()
        while not self._stop.wait(1.0):
            frac = min(0.95, (time.time() - t0) / self.est)
            self.emit(int(self.lo + (self.hi - self.lo) * frac))

    def stop(self):
        self._stop.set()


_WORKER = None


def _get_worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = RvcWorker()
    return _WORKER
