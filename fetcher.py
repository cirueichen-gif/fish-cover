# -*- coding: utf-8 -*-
r"""B站與 YouTube 音訊抓取模組。
支援：
- B 站：BV 號、bilibili.com 連結、b23.tv 短網址
- YouTube：youtube.com、youtu.be、shorts
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
APPLIO = r"C:\Users\cirue\Applio"
FFMPEG = os.path.join(APPLIO, "ffmpeg.exe")
YTDLP = os.path.join(APPLIO, "env", "Scripts", "yt-dlp.exe")
CRED_PATH = Path.home() / ".dsh" / "bili-duty" / "bili_credential.json"
BILI_PY = r"C:\Users\cirue\.dsh\bili-duty\.venv\Scripts\python.exe"
BILI_WORKER = os.path.join(HERE, "bili_worker.py")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def sanitize_filename(name):
    clean = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return clean[:80] or "audio"


def resolve_bvid(s):
    s = s.strip()
    m = re.search(r"(BV[a-zA-Z0-9]{10})", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if "b23.tv" in s:
        url_match = re.search(r"https?://b23\.tv/\w+", s)
        target = url_match.group(0) if url_match else s
        req = urllib.request.Request(target, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                real_url = resp.geturl()
                m2 = re.search(r"(BV[a-zA-Z0-9]{10})", real_url, re.IGNORECASE)
                if m2:
                    return m2.group(1)
        except Exception:
            pass
    return None


def fetch_bilibili(bvid, out_dir):
    cmd = [BILI_PY, BILI_WORKER, bvid, out_dir, str(CRED_PATH), FFMPEG]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = p.stderr.strip()
        raise RuntimeError(f"B站解析失敗:\n{err}")
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("B站下載器未輸出結構化結果")


def fetch_youtube(url, out_dir):
    if not os.path.isfile(YTDLP):
        raise RuntimeError(f"找不到 yt-dlp: {YTDLP}")

    out_tmpl = os.path.join(out_dir, "%(title).80s.%(ext)s")
    cmd = [
        YTDLP,
        "--ffmpeg-location", FFMPEG,
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "--max-filesize", "200M",
        "--print", "after_move:filepath",
        "--print", "title",
        "-o", out_tmpl,
        url,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        last_err = err[-1] if err else "YouTube 下載失敗"
        raise RuntimeError(f"YouTube 下載失敗: {last_err}")

    lines = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    target_path = None
    title = None
    for line in lines:
        if os.path.isfile(line):
            target_path = line
        else:
            title = line

    if not target_path:
        mp3s = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".mp3")]
        if mp3s:
            target_path = mp3s[-1]
            title = os.path.splitext(os.path.basename(target_path))[0]
        else:
            raise RuntimeError("無法找到 YouTube 轉檔後的音訊檔案")

    if not title:
        title = os.path.splitext(os.path.basename(target_path))[0]

    return {
        "title": title,
        "filename": os.path.basename(target_path),
        "path": target_path,
        "size": os.path.getsize(target_path),
        "source": "youtube",
    }


def fetch_url(url_or_bv, cache_root=None):
    s = url_or_bv.strip()
    if not s:
        raise ValueError("請輸入網址或 BV 號")

    if not cache_root:
        cache_root = os.path.join(HERE, "work", "_downloads")
    os.makedirs(cache_root, exist_ok=True)
    import uuid
    target_dir = os.path.join(cache_root, f"dl_{uuid.uuid4().hex[:8]}")
    os.makedirs(target_dir, exist_ok=True)

    bvid = resolve_bvid(s)
    if bvid:
        return fetch_bilibili(bvid, target_dir)
    elif "youtube.com" in s or "youtu.be" in s:
        return fetch_youtube(s, target_dir)
    else:
        raise ValueError("不支援的連結格式。請輸入 B 站（BV號／連結／b23短網址）或 YouTube 網址。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "請傳入 URL"}))
        sys.exit(1)
    try:
        res = fetch_url(sys.argv[1])
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)
