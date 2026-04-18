"""
fig4_noise_sensitivity.py
--------------------------
論文 Figure 4: BC vs 識別精度のノイズ感度カーブ（論文品質）

左パネル: V1+noise における BC と accuracy の推移（正規化）
右パネル: σ=0.75 で精度マッチした V1+noise vs HVC の BC 比較

キーメッセージ:
  - BC は精度より敏感にノイズ劣化を反映する
  - 精度が同じでも BC は有意に異なる（Cohen's d=0.465, p=0.023）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
NPZ_PATH   = "outputs/exp16_same_acc_diff_bc.npz"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    d = np.load(NPZ_PATH, allow_pickle=True)
    noise_levels  = d["noise_levels"]
    bc_curve      = d["bc_v1_noise"]
    bc_std        = d["bc_v1_noise_std"]
    acc_curve     = d["acc_v1_noise"]
    bc_hvc        = float(d["bc_hvc"])
    bc_hvc_std    = float(d["bc_hvc_std"])
    acc_hvc       = float(d["acc_hvc"])
    cohens_d      = float(d["cohens_d"])
    ci_low        = float(d["ci_low"])
    ci_high       = float(d["ci_high"])
    p_val         = float(d["p_val"])

    # σ=0.75 のインデックス（精度マッチポイント）
    target_sigma  = 0.75
    match_idx     = np.argmin(np.abs(noise_levels - target_sigma))

    bc_match  = bc_curve[match_idx]
    acc_match = acc_curve[match_idx]

    # ── レイアウト ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    fig.subplots_adjust(wspace=0.38)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 左パネル: ノイズ感度カーブ（実スケール）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[0]

    # BC カーブ（左 y 軸）
    color_bc  = "#2c7bb6"
    color_acc = "#d7442a"

    ax.plot(noise_levels, bc_curve, "o-", color=color_bc,
            linewidth=2.0, markersize=5, label="BC (V1 + noise)", zorder=3)
    ax.fill_between(noise_levels,
                    bc_curve - bc_std,
                    bc_curve + bc_std,
                    color=color_bc, alpha=0.15, zorder=2)

    # HVC ベースライン（BC）
    ax.axhline(bc_hvc, color=color_bc, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"BC (HVC baseline) = {bc_hvc:.3f}")

    ax.set_xlabel("Noise level $\\sigma$ added to V1 signals", fontsize=10)
    ax.set_ylabel("BC", fontsize=10, color=color_bc)
    ax.tick_params(axis="y", labelcolor=color_bc, labelsize=9)

    # 右 y 軸: accuracy
    ax2 = ax.twinx()
    ax2.plot(noise_levels, acc_curve, "s--", color=color_acc,
             linewidth=2.0, markersize=5, label="Accuracy (V1 + noise)", zorder=3)
    ax2.axhline(acc_hvc, color=color_acc, linestyle=":", linewidth=1.2,
                alpha=0.7, label=f"Accuracy (HVC) = {acc_hvc:.3f}")
    ax2.set_ylabel("Identification accuracy", fontsize=10, color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc, labelsize=9)

    # マッチポイントのアノテーション
    ax.axvline(target_sigma, color="#555555", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.annotate(f"$\\sigma$ = {target_sigma}\n(acc matched)",
                xy=(target_sigma, bc_match),
                xytext=(target_sigma + 0.3, bc_match + 0.008),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    # 凡例をまとめる
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=8, loc="upper right", framealpha=0.9)

    ax.set_title("(A)  BC and accuracy under noise injection\n(V1 ROI, Subject 1)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 右パネル: 精度マッチ条件の BC 比較
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[1]

    labels    = [f"V1+noise\n($\\sigma$={target_sigma})", "HVC"]
    bc_vals   = [bc_match,  bc_hvc]
    bc_errs   = [float(bc_std[match_idx]), bc_hvc_std]
    acc_vals  = [acc_match, acc_hvc]
    bar_colors = [color_bc, "#555555"]

    bars = ax.bar(labels, bc_vals, color=bar_colors,
                  alpha=0.85, edgecolor="white", linewidth=1.5,
                  width=0.45, zorder=3,
                  yerr=bc_errs, capsize=5,
                  error_kw=dict(elinewidth=1.2, ecolor="#333333"))

    for bar, val, acc in zip(bars, bc_vals, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + bc_errs[0] + 0.002,
                f"BC={val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=bar.get_facecolor())
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.965,
                f"acc={acc:.3f}",
                ha="center", va="bottom", fontsize=8, color="#555555")

    # Cohen's d アノテーション
    y_bracket = max(bc_vals) + max(bc_errs) + 0.010
    ax.annotate("",
                xy=(0, y_bracket), xytext=(1, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(0.5, y_bracket + 0.003,
            f"Cohen's $d$ = {cohens_d:.3f}\n"
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}], $p$ = {p_val:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7",
                      edgecolor="#f9a825", linewidth=1.0))

    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_ylim(0.960, y_bracket + 0.045)
    ax.set_title("(B)  Same accuracy, different BC\n(matched at identification accuracy ≈ 0.17)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "BC captures structural degradation not detectable by identification accuracy",
        fontsize=11, y=1.02, color="#222222"
    )

    out_path = os.path.join(OUTPUT_DIR, "fig4_noise_sensitivity.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
