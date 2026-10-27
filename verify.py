# -*- coding: utf-8 -*-
"""驗一首翻唱：長度、能量、爆音、音高有沒有跑掉。

用法：python verify.py work\<jobid>
"""
import os
import subprocess
import sys
import warnings

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
FFPROBE = r"C:\Users\cirue\Applio\ffprobe.exe"


def dur(p):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", p], capture_output=True, text=True)
    return float(r.stdout.strip())


def f0(p, sr=24000):
    y, _ = librosa.load(p, sr=sr, mono=True)
    f, _, _ = librosa.pyin(y, fmin=70, fmax=1100, sr=sr, frame_length=2048, hop_length=256)
    return f


def main(job):
    j = os.path.abspath(job)
    song = os.path.join(j, "song.wav")
    voc = os.path.join(j, "vocals.wav")
    ins = os.path.join(j, "instrumental.wav")
    conv = os.path.join(j, "vocals_fish.wav")
    cov = os.path.join(j, "cover.mp3")

    for p in (song, voc, ins, conv, cov):
        if not os.path.isfile(p):
            print(f"缺檔：{p}")
            return 1

    print(f"原曲     {dur(song):7.2f}s")
    print(f"人聲     {dur(voc):7.2f}s")
    print(f"伴奏     {dur(ins):7.2f}s")
    print(f"換嗓人聲 {dur(conv):7.2f}s")
    print(f"成品     {dur(cov):7.2f}s")

    a = f0(voc)
    b = f0(conv)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() > 50:
        cents = 1200 * np.log2(b[m] / a[m])
        print(f"\n音高：中位差 {np.median(cents):+.0f} cent，"
              f"90% 在 {np.percentile(np.abs(cents), 90):.0f} cent 內，"
              f"超過 100 cent 的佔 {(np.abs(cents) > 100).mean() * 100:.1f}%")
        print(f"      人聲幀數 {m.sum()}／{n}（有聲比例 {m.mean() * 100:.0f}%）")
    else:
        print("\n音高：量不出來")

    for name, p in (("換嗓人聲", conv), ("伴奏", ins), ("成品", cov)):
        y, sr = sf.read(p) if not p.endswith(".mp3") else librosa.load(p, sr=None, mono=True)
        if y.ndim > 1:
            y = y.mean(axis=1)
        rms = float(np.sqrt((y ** 2).mean()))
        peak = float(np.abs(y).max())
        clip = float((np.abs(y) > 0.999).mean() * 100)
        print(f"{name:8s} rms {rms:.3f}  peak {peak:.3f}  削波 {clip:.3f}%  "
              f"靜音比例 {(np.abs(y) < 1e-3).mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
