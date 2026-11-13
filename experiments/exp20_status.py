"""
exp20_status.py
---------------
Exp20 の進み具合を表示する。BC などの結果は表示しない（止めるかどうかを結果で決めないため）。

使い方（numpy だけを使うので、fmri と venv のどちらの環境でもよい）：
    python experiments/exp20_status.py $EXP20/out
"""
import os, sys, subprocess
import numpy as np

# ── 設定 ──────────────────────────────────────────────────────────────────
MODELS  = ["final_subj01_pretrained_40sess_24bs", "final_subj01_pretrained_1sess_24bs"]
TARGETS = [200, 500, 1000]                      # 残り時間を出す刺激の数（200 = S3 を論文に使う最低ライン）


def complete_prefix(stim, dones):
    """処理の順の先頭から数えて、dones のすべてで全行が済んでいる刺激の数 K と、その行を返す。

    刺激は seed 42 の無作為な順で処理するので、先頭の K 枚は 1,000 枚からの無作為な標本になる。
    """
    starts = np.r_[0, np.flatnonzero(np.diff(stim)) + 1]   # 刺激ごとの行は連続している
    ends = np.r_[starts[1:], len(stim)]
    ok = np.array([all(np.asarray(d[s:e]).all() for d in dones) for s, e in zip(starts, ends)])
    K = len(ok) if ok.all() else int(np.argmin(ok))
    return K, np.arange(ends[K - 1] if K else 0)


def load_or_zeros(path, N, dtype=bool):
    return np.load(path, mmap_mode="r") if os.path.exists(path) else np.zeros(N, dtype)


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$EXP20/out")
    dirs = [f"{out_root}/{m}" for m in MODELS]
    have = [d for d in dirs if os.path.exists(f"{d}/meta.npz")]
    if not have:
        print("まだパス A を始めていない")
        return
    stim = np.load(f"{have[0]}/meta.npz")["stim"]
    N, C = len(stim), len(np.unique(stim))
    da = [load_or_zeros(f"{d}/done_prior.npy", N) for d in dirs]
    db = [load_or_zeros(f"{d}/done_unclip.npy", N) for d in dirs]
    K_A, _ = complete_prefix(stim, da)
    K_B, _ = complete_prefix(stim, da + db)

    for m, a, b in zip(MODELS, da, db):
        print(f"{m}: パス A {int(np.sum(a))}/{N} 行、パス B {int(np.sum(b))}/{N} 行")
    print(f"両モデルで済んだ刺激（先頭から）：パス A まで K_A = {K_A}、パス B まで K_B = {K_B}（全 {C}）")

    t = [np.asarray(load_or_zeros(f"{d}/times_unclip.npy", N, np.float32))[np.asarray(b)] for d, b in zip(dirs, db)]
    t = np.concatenate(t)
    if len(t):
        sec = float(np.median(t))
        print(f"パス B の 1 試行：中央値 {sec:.1f} 秒（{len(t)} 試行）")
        ends = np.r_[np.flatnonzero(np.diff(stim)) + 1, N]
        for K in [k for k in TARGETS if K_B < k <= C]:
            left = sum(int((~np.asarray(b[:ends[K - 1]])).sum()) for b in db)
            print(f"  K_B = {K} まで：あと {left} 試行、約 {left * sec / 3600:.1f} 時間")

    pids = subprocess.run(["pgrep", "-f", "exp20_mindeye2_pertrial|exp20_run"], capture_output=True, text=True).stdout.split()
    print("動いているプロセス：", " ".join(pids) if pids else "なし")
    print("STOP ファイル：", "あり" if os.path.exists(f"{out_root}/STOP") else "なし")
    if os.path.exists(f"{out_root}/runs.jsonl"):
        with open(f"{out_root}/runs.jsonl") as f:
            lines = f.read().splitlines()
        if lines:
            print("最後に正常に終わった実行：", lines[-1])


if __name__ == "__main__":
    main()
