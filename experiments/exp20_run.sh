#!/bin/bash
# Exp20 の推論（§6 の再現確認、§7 のパイロット、§8 の本番）をまとめて実行する。
# 何度実行しても、済んだところは飛ばして続きから行う。止め方は手順書の §14.2。
#
# 使い方: bash experiments/exp20_run.sh {repro|pilot|main} [--stop_at=2026-11-10T08:00]
set -e
EXP20=${EXP20:-$HOME/exp20}
OUT=$EXP20/out
M40=final_subj01_pretrained_40sess_24bs
M1=final_subj01_pretrained_1sess_24bs

export PYTHONUNBUFFERED=1   # print をすぐログに出す（ログで進み具合を見るため。出力は変わらない）
cd "$EXP20/MindEyeV2/src"
source fmri/bin/activate
cp "$EXP20/brain-control/experiments/exp20_mindeye2_pertrial.py" .
RUN="python exp20_mindeye2_pertrial.py --data_path=$EXP20/data --cache_dir=$EXP20/data --subj=1 --hidden_dim=4096 --n_blocks=4 --new_test --no-blurry_recon"

MODE=$1
shift || true
echo "== $(date '+%F %T') 開始: $MODE $*"
case $MODE in
  repro)   # §6：3 試行平均で 10 刺激
    $RUN --model_name=$M40 --stage=prior --average_reps --limit=10 --out_dirs $OUT/repro_40sess
    $RUN --stage=unclip --out_dirs $OUT/repro_40sess ;;
  pilot)   # §7：各モデル 30 行（10 刺激）のパス A と、5 刺激 × 2 モデル × 3 試行のパス B。出力は本番でそのまま使う
    $RUN --model_name=$M40 --stage=prior --limit=30 --out_dirs $OUT/$M40
    $RUN --model_name=$M1  --stage=prior --limit=30 --out_dirs $OUT/$M1
    $RUN --stage=unclip --limit=5 --out_dirs $OUT/$M40 $OUT/$M1 ;;
  main)    # §8：パス A（40sess → 1sess）のあとパス B（両モデルを交互に）
    $RUN --model_name=$M40 --stage=prior --out_dirs $OUT/$M40 "$@"
    $RUN --model_name=$M1  --stage=prior --out_dirs $OUT/$M1  "$@"
    $RUN --stage=unclip --out_dirs $OUT/$M40 $OUT/$M1 "$@" ;;
  *)
    echo "使い方: bash exp20_run.sh {repro|pilot|main} [--stop_at=...]"
    exit 1 ;;
esac
echo "== $(date '+%F %T') 終了: $MODE"
