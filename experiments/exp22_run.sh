#!/bin/bash
# Exp22（被験者 2・5・7）の推論。Exp20 の推論スクリプトを、--subj と出力先だけ変えて実行する。
# 出力は被験者ごとに $EXP20/out_subj0<s>/<model_name>/（STOP・ロック・runs.jsonl も被験者ごと）。
# 何度実行しても、済んだところは飛ばして続きから行う。止めるときは $EXP20/out_subj0<s>/STOP を置く。
#
# 使い方: bash experiments/exp22_run.sh <被験者 2|5|7> {repro|main} [--stop_at=2026-11-10T08:00]
#   repro : 40sess で 3 試行平均の 10 刺激（Exp20 の §6 と同じ再現確認）
#   main  : パス A（40sess → 1sess）のあと、パス B（両モデルを交互に）を K_B = KB_TARGET（既定 500）で自動で止める
set -e
EXP20=${EXP20:-$HOME/exp20}
S=$1
case "$S" in 2|5|7) ;; *) echo "被験者は 2、5、7 のどれか"; exit 1 ;; esac
OUT=$EXP20/out_subj0${S}
M40=final_subj0${S}_pretrained_40sess_24bs
M1=final_subj0${S}_pretrained_1sess_24bs
KB_TARGET=${KB_TARGET:-500}   # パス B の刺激の数（計画 §4、被験者 1 とそろえる）

# 先頭から数えて、パス B まで両モデルで済んだ刺激の数
k_b() {
  python - "$EXP20" "$S" <<'EOF'
import os, sys
import numpy as np
E, s = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, f"{E}/brain-control/experiments")
from exp20_status import complete_prefix
from exp22_status import models_of
d = [f"{E}/out_subj0{s}/{m}" for m in models_of(s)]
stim = np.load(f"{d[0]}/meta.npz")["stim"]
dn = [np.load(p, mmap_mode="r") if os.path.exists(p) else np.zeros(len(stim), bool)
      for x in d for p in (f"{x}/done_prior.npy", f"{x}/done_unclip.npy")]
print(complete_prefix(stim, dn)[0])
EOF
}
MODE=$2
shift 2 || true

export PYTHONUNBUFFERED=1   # print をすぐログに出す（出力は変わらない）
if pgrep -f exp20_mindeye2_pertrial.py > /dev/null; then
  echo "推論がもう動いている（Exp20〜22 のどれか）。同時には動かさない"; exit 1
fi
mkdir -p $OUT
cd "$EXP20/MindEyeV2/src"
source fmri/bin/activate
cp "$EXP20/brain-control/experiments/exp20_mindeye2_pertrial.py" .
RUN="python exp20_mindeye2_pertrial.py --data_path=$EXP20/data --cache_dir=$EXP20/data --subj=${S} --hidden_dim=4096 --n_blocks=4 --new_test --no-blurry_recon"

echo "== $(date '+%F %T') 開始: 被験者 ${S} $MODE $*"
case $MODE in
  repro)
    $RUN --model_name=$M40 --stage=prior --average_reps --limit=10 --out_dirs $OUT/repro_40sess
    $RUN --stage=unclip --out_dirs $OUT/repro_40sess ;;
  main)
    $RUN --model_name=$M40 --stage=prior --out_dirs $OUT/$M40 "$@"
    $RUN --model_name=$M1  --stage=prior --out_dirs $OUT/$M1  "$@"
    n=$(( KB_TARGET - $(k_b) ))
    if [ "$n" -gt 0 ]; then
      $RUN --stage=unclip --limit=$n --out_dirs $OUT/$M40 $OUT/$M1 "$@"
    fi
    echo "== $(date '+%F %T') K_B = $(k_b) / ${KB_TARGET}" ;;
  *)
    echo "使い方: bash exp22_run.sh <2|5|7> {repro|main} [--stop_at=...]"
    exit 1 ;;
esac
echo "== $(date '+%F %T') 終了: 被験者 ${S} $MODE"
