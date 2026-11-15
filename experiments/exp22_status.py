"""
exp22_status.py
---------------
Exp22（被験者 2・5・7）の進み具合を表示する。exp20_status.py の表示を、被験者のモデルと出力先に置き換えて使う。
BC などの結果は表示しない（止めるかどうかを結果で決めないため）。

使い方（numpy だけを使うので、fmri と venv のどちらの環境でもよい）：
    python experiments/exp22_status.py <被験者 2|5|7> [$EXP20]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp20_status


def models_of(subj):
    return [f"final_subj0{subj}_pretrained_40sess_24bs", f"final_subj0{subj}_pretrained_1sess_24bs"]


if __name__ == "__main__":
    subj = int(sys.argv[1])
    assert subj in (2, 5, 7), "被験者は 2、5、7 のどれか"
    exp20 = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("EXP20", os.path.expanduser("~/exp20"))
    exp20_status.MODELS = models_of(subj)
    sys.argv = [sys.argv[0], f"{exp20}/out_subj0{subj}"]
    exp20_status.main()
