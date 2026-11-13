#!/bin/bash
# Exp21（seed による分解）の推論。Exp20 の推論スクリプトを、--seed と出力先だけ変えて実行する。
#   r = 0 は Exp20 の出力（$EXP20/out、seed 42）。r = 1〜3 は seed = 42 + r × 10,000,000 で作り直す
#   パス A（S0〜S2）は全 1,000 刺激。パス B（S3）は 10 刺激ずつ、r = 1 → 2 → 3 の順に回す
#   （どこで止めても、全部の seed で同じ刺激がそろう。目標は先頭の K_TARGET 刺激）
# 何度実行しても、済んだところは飛ばして続きから行う。止めるときは $EXP20/out_seed/STOP を置く。
#
# 使い方: bash experiments/exp21_run.sh {pilot|main|status} [--stop_at=2026-11-10T08:00]
set -e
EXP20=${EXP20:-$HOME/exp20}
OUT=$EXP20/out_seed
M40=final_subj01_pretrained_40sess_24bs
M1=final_subj01_pretrained_1sess_24bs
SEEDS="1 2 3"
K_TARGET=${K_TARGET:-100}   # パス B の刺激の数（計画 §3.2）
CHUNK=10
seed_of() { echo $((42 + $1 * 10000000)); }

# 引数の r（0 は Exp20）について、先頭から数えてパス B まで両モデルで済んだ刺激の数（r を複数渡すと、全部でそろった数）
k_of() {
  python - "$EXP20" "$@" <<'EOF'
import os, sys
import numpy as np
E, rs = sys.argv[1], sys.argv[2:]
sys.path.insert(0, f"{E}/brain-control/experiments")
from exp20_status import complete_prefix, MODELS
d = lambda r, m: f"{E}/out/{m}" if r == "0" else f"{E}/out_seed/r{r}/{m}"
stim = np.load(f"{E}/out/{MODELS[0]}/meta.npz")["stim"]
dn = [np.load(p, mmap_mode="r") if os.path.exists(p) else np.zeros(len(stim), bool)
      for r in rs for m in MODELS for p in (f"{d(r, m)}/done_prior.npy", f"{d(r, m)}/done_unclip.npy")]
print(complete_prefix(stim, dn)[0])
EOF
}

MODE=$1
shift || true
export PYTHONUNBUFFERED=1   # print をすぐログに出す（出力は変わらない）
cd "$EXP20/MindEyeV2/src"
source fmri/bin/activate
if [ "$MODE" = status ]; then
  for r in 0 $SEEDS; do echo "r = $r: パス B まで済んだ刺激 $(k_of $r)"; done
  echo "全部の seed でそろった刺激：$(k_of 0 $SEEDS)（目標 ${K_TARGET}）"
  exit 0
fi
if pgrep -f exp20_mindeye2_pertrial.py > /dev/null; then
  echo "推論がもう動いている（Exp20 か Exp21）。同時には動かさない"; exit 1
fi
mkdir -p $OUT
for r in $SEEDS; do
  mkdir -p $OUT/r$r
  ln -sfn ../STOP $OUT/r$r/STOP   # 推論スクリプトは出力先の親の STOP を見る。$OUT/STOP 1 つで全部止まるようにする
done
cp "$EXP20/brain-control/experiments/exp20_mindeye2_pertrial.py" .
RUN="python exp20_mindeye2_pertrial.py --data_path=$EXP20/data --cache_dir=$EXP20/data --subj=1 --hidden_dim=4096 --n_blocks=4 --new_test --no-blurry_recon"

echo "== $(date '+%F %T') 開始: $MODE $*"
case $MODE in
  pilot)   # r = 1 で、パス A 各モデル 30 行（10 刺激）と、パス B 2 刺激。出力は本番でそのまま使う
    S=$(seed_of 1)
    $RUN --model_name=$M40 --stage=prior --seed=$S --limit=30 --out_dirs $OUT/r1/$M40
    $RUN --model_name=$M1  --stage=prior --seed=$S --limit=30 --out_dirs $OUT/r1/$M1
    $RUN --stage=unclip --seed=$S --limit=2 --out_dirs $OUT/r1/$M40 $OUT/r1/$M1 ;;
  main)
    for r in $SEEDS; do
      S=$(seed_of $r)
      $RUN --model_name=$M40 --stage=prior --seed=$S --out_dirs $OUT/r$r/$M40 "$@"
      $RUN --model_name=$M1  --stage=prior --seed=$S --out_dirs $OUT/r$r/$M1  "$@"
    done
    K=$(k_of 0 $SEEDS)
    while [ "$K" -lt "$K_TARGET" ]; do
      GOAL=$(( K + CHUNK < K_TARGET ? K + CHUNK : K_TARGET ))
      for r in $SEEDS; do
        n=$(( GOAL - $(k_of $r) ))
        if [ "$n" -gt 0 ]; then
          $RUN --stage=unclip --seed=$(seed_of $r) --limit=$n --out_dirs $OUT/r$r/$M40 $OUT/r$r/$M1 "$@"
        fi
      done
      K_NEW=$(k_of 0 $SEEDS)
      if [ "$K_NEW" -le "$K" ]; then echo "進まなかった（K = ${K}）。止める"; exit 1; fi
      K=$K_NEW
      echo "== $(date '+%F %T') 全部の seed でそろった刺激 $K / $K_TARGET"
    done ;;
  *)
    echo "使い方: bash exp21_run.sh {pilot|main|status} [--stop_at=...]"
    exit 1 ;;
esac
echo "== $(date '+%F %T') 終了: $MODE"
