"""
exp16b_noise_sensitivity_curve.py
-----------------------------------
Exp16 の補足図: BC と識別精度のノイズ感度曲線

論文のメイン図として使う想定。
  - x 軸: ノイズ強度 σ
  - y 軸: 正規化した BC と識別精度（0=最低, 1=最高）

見せたいこと:
  BC は識別精度より敏感にノイズに反応する
  → BC は識別精度では捉えられない構造劣化を検出できる

Exp16 の保存済み数値を読み込んで再プロットするだけ。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Exp16 の保存済み数値を読み込む ──
    data = np.load(os.path.join(OUTPUT_DIR, "exp16_same_acc_diff_bc.npz"))
    noise_levels = data["noise_levels"]
    bc_vals      = data["bc_v1_noise"]
    bc_stds      = data["bc_v1_noise_std"]
    acc_vals     = data["acc_v1_noise"]
    bc_hvc_m     = float(data["bc_hvc"])
    bc_hvc_s     = float(data["bc_hvc_std"])
    acc_hvc      = float(data["acc_hvc"])
    cohens_d     = float(data["cohens_d"])
    p_val        = float(data["p_val"])
    ci_low       = float(data["ci_low"])
    ci_high      = float(data["ci_high"])

    # ── 正規化（0=最低値, 1=ノイズなし条件の値）──
    bc_max  = bc_vals[0]   # σ=0.0 のときの最大BC
    bc_min  = 1.0          # prior 支配のとき BC → 1
    acc_max = acc_vals[0]  # σ=0.0 のときの最大精度
    acc_min = 1 / 50       # チャンスレベル（50クラス）

    bc_norm  = (bc_vals  - bc_min)  / (bc_max  - bc_min)
    acc_norm = (acc_vals - acc_min) / (acc_max - acc_min)

    # HVC も同スケールで正規化
    bc_hvc_norm  = (bc_hvc_m  - bc_min)  / (bc_max  - bc_min)
    acc_hvc_norm = (acc_hvc   - acc_min) / (acc_max - acc_min)

    # ── プロット ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 正規化カーブ（メイン図）
    ax = axes[0]
    ax.plot(noise_levels, bc_norm, color="steelblue", marker="o",
            linewidth=2.5, markersize=7, label="BC (Degree of Brain Control)")
    ax.fill_between(
        noise_levels,
        bc_norm - bc_stds / (bc_max - bc_min),
        bc_norm + bc_stds / (bc_max - bc_min),
        alpha=0.2, color="steelblue"
    )
    ax.plot(noise_levels, acc_norm, color="tomato", marker="s",
            linewidth=2.5, markersize=7, label="Identification Accuracy", linestyle="--")

    ax.axhline(bc_hvc_norm, color="steelblue", linestyle=":", linewidth=1.5, alpha=0.6,
               label="HVC BC (no noise)")
    ax.axhline(acc_hvc_norm, color="tomato", linestyle=":", linewidth=1.5, alpha=0.6,
               label="HVC Accuracy (no noise)")

    idx_075 = list(noise_levels).index(0.75)
    ax.annotate(
        f"Same Acc (s=0.75)\nBC diff d={cohens_d:.2f}, p={p_val:.3f}",
        xy=(noise_levels[idx_075], bc_norm[idx_075]),
        xytext=(1.5, 0.35),
        fontsize=9, color="navy",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="navy"),
    )

    ax.set_xlabel("Noise level sigma", fontsize=11)
    ax.set_ylabel("Normalized score (0=floor, 1=no-noise baseline)", fontsize=11)
    ax.set_title("BC is more sensitive to noise degradation than accuracy\n(V1 noise injection)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    # (2) 降下率の比較（Δ per unit σ）
    ax = axes[1]
    bc_drop  = bc_norm[0]  - bc_norm
    acc_drop = acc_norm[0] - acc_norm

    ax.plot(noise_levels, bc_drop, color="steelblue", marker="o",
            linewidth=2.5, markersize=7, label="BC degradation")
    ax.plot(noise_levels, acc_drop, color="tomato", marker="s",
            linewidth=2.5, markersize=7, label="Accuracy degradation", linestyle="--")

    ax.fill_between(noise_levels, acc_drop, bc_drop,
                    where=(bc_drop >= acc_drop),
                    alpha=0.15, color="steelblue",
                    label="BC sensitivity advantage")

    ax.set_xlabel("Noise level sigma", fontsize=11)
    ax.set_ylabel("Degradation from no-noise baseline (normalized)", fontsize=11)
    ax.set_title("BC sensitivity advantage\n(BC degrades faster than accuracy)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    plt.suptitle(
        "BC detects structural degradation that accuracy cannot detect\n"
        f"V1 noise injection experiment (Subject1, relu7 features)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "exp16b_noise_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")

    # ── 感度の定量比較 ──
    # 精度が50%低下するときの σ と、BC が50%低下するときの σ を比較
    bc_half  = np.interp(0.5, bc_drop[::-1],  noise_levels[::-1])  # 逆順補間
    acc_half = np.interp(0.5, acc_drop[::-1], noise_levels[::-1])

    print(f"\n=== 感度比較（50%低下点）===")
    print(f"BC が 50% 低下する σ:    {bc_half:.3f}")
    print(f"識別精度が 50% 低下する σ: {acc_half:.3f}")
    print(f"BC の方が {acc_half/bc_half:.2f}x 敏感")
    print("完了!")


if __name__ == "__main__":
    main()
