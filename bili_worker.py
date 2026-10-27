# -*- coding: utf-8 -*-
r"""B站音訊下載 worker（在 bili-duty 虛擬環境中執行）。"""
import asyncio
import json
import os
import re
import subprocess
import sys
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from bilibili_api import video, Credential


def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "參數不足"}))
        sys.exit(1)

    bvid = sys.argv[1]
    out_dir = sys.argv[2]
    cred_path = sys.argv[3]
    ffmpeg_exe = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    cred_data = {}
    if os.path.isfile(cred_path):
        try:
            with open(cred_path, encoding="utf-8") as f:
                cred_data = json.load(f)
        except Exception:
            pass

    cred = Credential(
        sessdata=cred_data.get("sessdata"),
        bili_jct=cred_data.get("bili_jct"),
        buvid3=cred_data.get("buvid3"),
        dedeuserid=str(cred_data.get("dedeuserid", ""))
    ) if cred_data else None

    v = video.Video(bvid=bvid, credential=cred)

    async def run():
        info = await v.get_info()
        title = info.get("title", bvid)
        url_info = await v.get_download_url(page_index=0)
        audios = url_info.get("dash", {}).get("audio", [])
        if not audios:
            raise RuntimeError("影片沒有可用音訊軌")
        audios.sort(key=lambda a: a.get("bandwidth", 0), reverse=True)
        best = audios[0]
        candidate_urls = [best["baseUrl"]] + best.get("backupUrl", [])

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://www.bilibili.com/video/{bvid}",
        }

        import tempfile
        fd, tmp_m4s = tempfile.mkstemp(prefix=f"bili_{bvid}_", suffix=".m4s")
        os.close(fd)
        downloaded = False
        async with httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True) as client:
            for u in candidate_urls:
                try:
                    async with client.stream("GET", u) as r:
                        if r.status_code == 200:
                            with open(tmp_m4s, "wb") as f:
                                async for chunk in r.aiter_bytes(chunk_size=65536):
                                    f.write(chunk)
                            if os.path.isfile(tmp_m4s) and os.path.getsize(tmp_m4s) > 1000:
                                downloaded = True
                                break
                        else:
                            sys.stderr.write(f"Candidate status: {r.status_code}\n")
                except Exception as e:
                    sys.stderr.write(f"Candidate error: {e}\n")
                    continue

        if not downloaded:
            raise RuntimeError("所有音訊串流伺服器下載均未成功")

        clean_title = re.sub(r'[^\w\s\u4e00-\u9fa5\-\.]', '_', title).strip()[:60] or bvid
        target_mp3 = os.path.join(out_dir, f"{clean_title}.mp3")
        cmd = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", "-i", tmp_m4s, "-vn", "-b:a", "320k", target_mp3]
        subprocess.run(cmd, check=True)
        try:
            os.remove(tmp_m4s)
        except Exception:
            pass

        print(json.dumps({
            "title": title,
            "filename": os.path.basename(target_mp3),
            "path": target_mp3,
            "size": os.path.getsize(target_mp3),
            "source": "bilibili"
        }, ensure_ascii=False))

    asyncio.run(run())


if __name__ == "__main__":
    main()
