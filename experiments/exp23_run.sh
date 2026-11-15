#!/bin/bash
# Exp23（MindEye1 と Brain-Diffuser、被験者 1）の実行。
# 出力はリポジトリの外（$EXP20/exp23/）。何度実行しても、済んだところは飛ばす。
#
# 使い方:
#   bash experiments/exp23_run.sh bd_cpu
#       Brain-Diffuser の GPU を使わない段（データの準備 → 特徴量 → 回帰）。CPU だけで動かす
#       （Exp22 と並行してよい。nice をつけ、スレッドの数をしぼる）
#   bash experiments/exp23_run.sh me1 <出力の名前> {prior|vd} [--limit=N] [--average_reps] [--stop_at=...]
#       MindEye1 の推論（GPU）。出力は $EXP20/exp23/me1_runs/<出力の名前>/、STOP は $EXP20/exp23/me1_runs/STOP
#   bash experiments/exp23_run.sh bd <出力の名前> {vdvae|vd} [--limit=N] [--average_reps] [--stop_at=...]
#       Brain-Diffuser の再構成（GPU）。出力は $EXP20/exp23/bd_runs/<出力の名前>/、STOP は $EXP20/exp23/bd_runs/STOP
#   bash experiments/exp23_run.sh g2 {me1|bd} <出力の名前>
#       関門 G2：--average_reps の出力で CLIP の 2 択の識別精度を出す
#   本番の出力の名前は final_subj01（単一試行）、G2 は g2avg_subj01（3 試行の平均）
set -e
EXP20=${EXP20:-$HOME/exp20}
E=$EXP20/exp23
D=$E/bd_data
PY=$E/bd_env/bin/python
X=$(cd "$(dirname "$0")" && pwd)
M20=$EXP20/out/final_subj01_pretrained_40sess_24bs/meta.npz
VD=$E/versatile-diffusion/pretrained_pth/vd-four-flow-v1-0-fp16-deprecated.pth
export PYTHONUNBUFFERED=1

case "$1" in
bd_cpu)
  cd $E/brain-diffuser                                  # 元のコードと同じく、vdvae と versatile_diffusion を相対パスで読む
  if [ ! -f $D/meta.npz ]; then
    echo "== $(date '+%F %T') データの準備"
    OMP_NUM_THREADS=4 nice -n 10 $PY $X/exp23_bd_prepare.py --nsd_dir $EXP20/nsd_aws \
      --annots $E/me1_data/COCO_73k_annots_curated.npy --exp20_meta $M20 --out_dir $D
  fi
  echo "== $(date '+%F %T') 特徴量（CLIP-Vision と VDVAE を並行）"
  OMP_NUM_THREADS=4 nice -n 10 $PY $X/exp23_bd_features.py --feature clipvision --data_dir $D --vd_pth $VD \
    --device cpu --bs 16 > $D/log_features_clipvision.txt 2>&1 &
  P1=$!
  OMP_NUM_THREADS=8 nice -n 10 $PY $X/exp23_bd_features.py --feature vdvae --data_dir $D --vdvae_dir $E/vdvae_model \
    --device cpu --bs 30 > $D/log_features_vdvae.txt 2>&1 &
  P2=$!
  wait $P1; wait $P2
  OMP_NUM_THREADS=4 nice -n 10 $PY $X/exp23_bd_features.py --feature cliptext --data_dir $D --vd_pth $VD \
    --device cpu > $D/log_features_cliptext.txt 2>&1
  for f in cliptext clipvision vdvae; do
    echo "== $(date '+%F %T') 回帰 ${f}"
    OMP_NUM_THREADS=8 nice -n 10 $PY $X/exp23_bd_regression.py --feature $f --data_dir $D --check
  done
  echo "== $(date '+%F %T') bd_cpu 終了" ;;
me1)
  if pgrep -f "exp20_mindeye2_pertrial.py|exp23_bd_pertrial.py" > /dev/null; then
    echo "ほかの推論が GPU を使っている。同時には動かさない"; exit 1
  fi
  NAME=$2; STAGE=$3; shift 3
  cp $X/exp23_mindeye1_pertrial.py $E/mindeye1/src/     # MindEye1 の models.py と utils.py を import するため、src で動かす
  mkdir -p $E/me1_runs && cd $E/mindeye1/src
  $EXP20/MindEyeV2/src/fmri/bin/python exp23_mindeye1_pertrial.py --data_path $E/me1_data \
    --vd_dir $E/versatile-diffusion --exp20_meta $M20 --stage $STAGE --out_dir $E/me1_runs/$NAME "$@" ;;
bd)
  if pgrep -f "exp20_mindeye2_pertrial.py|exp23_mindeye1_pertrial.py" > /dev/null; then
    echo "ほかの推論が GPU を使っている。同時には動かさない"; exit 1
  fi
  NAME=$2; STAGE=$3; shift 3
  mkdir -p $E/bd_runs && cd $E/brain-diffuser
  $PY $X/exp23_bd_pertrial.py --data_dir $D --exp20_meta $M20 --vd_pth $VD --vdvae_dir $E/vdvae_model \
    --stage $STAGE --out_dir $E/bd_runs/$NAME "$@" ;;
g2)
  $PY $X/exp23_gate_g2.py --out_dir $E/$2_runs/$3 --data_dir $D ;;
*)
  echo "使い方は、このファイルの先頭を見る"; exit 1 ;;
esac
