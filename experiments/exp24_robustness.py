"""
exp24_robustness.py
-------------------
Experiment 24: 今ある出力（Exp20・22）だけでできる 3 つの確認（GPU での生成はしない）

  A：カテゴリの分解（Exp20b）を被験者 1・2・5・7 で。被験者 1 は Exp20b の値が再現するかの確認
  B：session ごとの平均を引いた BC（同じ画像の試行が同じ session に入ることによる見かけの一貫性がないか）
  C：カテゴリだけのデコーダとの比較。刺激ごとの平均を「同じクラスタのほかの刺激の平均」に置き換え、
     試行ごとのずれ（同じ刺激の中のばらつき）はそのまま残した版を作り、BC・カテゴリを除いた BC・識別精度を比べる

  B・C の変換は列（次元）ごとに独立なので、段階の出力を列のかたまりごとに読み、元の出力・session の平均を引いた版・
  カテゴリだけの版の十分統計量を 1 回の読み込みでまとめて計算する（S1・S2 の全次元は大きいため）。
  BC と BC_beyond は Exp20・20b と同じ式（bc_from_stats、beyond_from_stats）。信頼区間も同じ
  （刺激をリサンプリングする bootstrap 1,000 回、同じ被験者の両モデルで同じ重み、バイアス補正パーセンタイル法）。

計画：~/brain-control/plan_exp24.md（予測 A1〜A6、B0〜B3、C1〜C4 は解析の前に固定した）

使い方（解析用の venv で、リポジトリの直下で実行する。GPU は使わない）：
    venv/bin/python experiments/exp24_robustness.py --selftest   # 合成データでの確認
    venv/bin/python experiments/exp24_robustness.py              # 本番
    venv/bin/python experiments/exp24_robustness.py --plot_only  # npz から図だけ描き直す
    venv/bin/python experiments/exp24_robustness.py --run_matched  # B5：同じ run の中だけで比べる（計画 §3.4）

  B5（計画 §3.4。B0 の結果を見たあと、計算の前に追加）：同じ画像の 3 試行はほとんど同じ run に入っていたため、
  試行の組の二乗距離で、同じ run の中だけの「違う刺激の組 ÷ 同じ刺激の組」を全体の比と比べる。
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp20_status import complete_prefix
from exp20_mindeye2_bc import SEED, N_BOOT, STAGES, EMBEDDERS, bc_from_stats, load_stage, summarize
from exp20b_beyond_category import KS_CLUSTER, gt_clip_all, make_clusters, beyond_from_stats
from exp22_multisubject_bc import SUBJECTS, VARIANTS, out_root, models_of

# ── 設定 ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
NPZ_PATH   = os.path.join(OUTPUT_DIR, "exp24_robustness.npz")
FIG_PATH   = os.path.join(OUTPUT_DIR, "exp24_robustness.png")
K_MAIN     = KS_CLUSTER[0]                               # 40（C はこのクラスタだけ）
STAGES_A   = ["S0", "S1", "S2", "S1_tok", "S2_tok"]      # 集合 A（K_A = 1000）で見る段階
COLORS     = {"40sess": "#2c7bb6", "1sess": "#7b3294"}


# ── 十分統計量（変換つき）────────────────────────────────────────────────────

def multi_stats(X, labels, shuf, transforms, chunk=8192):
    """X の先頭 len(labels) 行について、変換ごと・ラベルごとの十分統計量 (ssw_c, G, n) を 1 回の読み込みで計算する。

    exp20_mindeye2_bc.stimulus_stats と同じ計算に、列のかたまりごとの変換を足したもの。
    transforms: {名前: fn または None}。fn は (行, 列) の float64 を受け取り、同じ形を返す（行は X の並びのまま）。
    Returns: {名前: {"true": (ssw_c, G, n), "shuf": (ssw_c, G, n)}}
    """
    X2 = X.reshape(X.shape[0], -1)
    n_rows = len(labels)
    prep = {}
    for key, lab in {"true": labels, "shuf": shuf}.items():
        order = np.argsort(lab, kind="stable")
        cats, counts = np.unique(lab, return_counts=True)
        assert (counts == counts[0]).all(), "刺激ごとの試行数がそろっていない"
        prep[key] = (order, len(cats), int(counts[0]))
    acc = {(t, key): [np.zeros(C), np.zeros((C, C))] for t in transforms for key, (_, C, _) in prep.items()}
    for s in range(0, X2.shape[1], chunk):
        x_raw = np.asarray(X2[:n_rows, s:s + chunk], dtype=np.float64)
        for t, fn in transforms.items():
            x_t = x_raw if fn is None else fn(x_raw)
            for key, (order, C, n) in prep.items():
                x = x_t[order]
                x = x - x.mean(axis=0)                       # 桁落ちを防ぐため全体平均を引く（BC は変わらない）
                x = x.reshape(C, n, -1)
                m = x.mean(axis=1)
                acc[(t, key)][0] += ((x - m[:, None]) ** 2).sum(axis=(1, 2))
                with np.errstate(all="ignore"):              # matmul は値が正しくても警告を出すことがあるため
                    acc[(t, key)][1] += m @ m.T
    out = {}
    for t in transforms:
        out[t] = {}
        for key, (_, _, n) in prep.items():
            ssw_c, G = acc[(t, key)]
            assert np.isfinite(ssw_c).all() and np.isfinite(G).all(), "統計量に inf か nan"
            out[t][key] = (ssw_c, G, n)
    return out


def session_demean(sess):
    """同じ session の行の平均を、列ごとに引く変換。"""
    u, inv = np.unique(sess, return_inverse=True)
    onehot = np.zeros((len(sess), len(u)))
    onehot[np.arange(len(sess)), inv] = 1
    cnt = onehot.sum(axis=0)

    def fn(x):
        with np.errstate(all="ignore"):                      # matmul は値が正しくても警告を出すことがあるため
            means = (onehot.T @ x) / cnt[:, None]
        return x - means[inv]
    return fn


def category_only(labels, cl_of_stim, seed=SEED):
    """同じクラスタの中で試行をランダムに入れ替える変換（行の並べ替え）。出力はクラスタにしか依存しなくなる。

    各刺激の試行は、同じクラスタの刺激の試行の中からランダムに選ばれたものになる（ノイズはそのまま残る）。
    同じクラスタにほかの刺激がない刺激は入れ替わらない。
    Returns: (fn, ほかの刺激がないクラスタにいる刺激の数)
    """
    cats, inv = np.unique(labels, return_inverse=True)
    g_row = cl_of_stim[labels]
    perm = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    for g in np.unique(g_row):
        idx = np.flatnonzero(g_row == g)
        perm[idx] = idx[rng.permutation(len(idx))]
    n_stim_in_cluster = {g: len(np.unique(labels[g_row == g])) for g in np.unique(g_row)}
    n_lone = sum(1 for s in cats if n_stim_in_cluster[cl_of_stim[s]] == 1)

    def fn(x):
        return x[perm]
    return fn, n_lone


# ── 識別精度（試行ごと）──────────────────────────────────────────────────────

def ident_per_trial(pred, true_feat, img_index, cl_true):
    """試行ごとの、全体のペア識別精度と、クラスタの中のペア識別精度（同じクラスタにほかの画像がなければ nan）。

    exp19 の pairwise_identification と exp20b の within_cluster_identification を、試行ごとの値を返す形にしたもの。
    """
    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        r = zrows(pred.astype(np.float64)) @ zrows(true_feat.astype(np.float64)).T
    r_true = r[np.arange(len(pred)), img_index]
    overall = (r < r_true[:, None]).sum(axis=1) / (r.shape[1] - 1)
    within = np.full(len(pred), np.nan)
    for t in range(len(pred)):
        same = np.flatnonzero(cl_true == cl_true[img_index[t]])
        others = same[same != img_index[t]]
        if len(others):
            within[t] = np.mean(r[t, others] < r_true[t])
    return overall, within


def per_stim(values, img_index, K):
    """試行ごとの値を刺激ごとに平均する（nan の試行は除く。全部 nan の刺激は nan）。"""
    out = np.full(K, np.nan)
    for k in range(K):
        v = values[img_index == k]
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = v.mean()
    return out


def boot_mean(v, W):
    """刺激ごとの値 v を bootstrap の重み W で平均する（nan の刺激は除く）。"""
    ok = np.isfinite(v)
    with np.errstate(all="ignore"):
        return (W[:, ok] @ v[ok]) / W[:, ok].sum(axis=1)


# ── 1 人分の解析 ──────────────────────────────────────────────────────────

def analyze_subject(subj, clusters):
    root = out_root(subj)
    dirs = [f"{root}/{m}" for m in models_of(subj)]
    metas = [np.load(f"{d}/meta.npz") for d in dirs]
    stim = metas[0]["stim"]
    assert all(np.array_equal(m["stim"], stim) for m in metas), f"被験者 {subj}：モデルの間で試行の並びが違う"
    da = [np.load(f"{d}/done_prior.npy", mmap_mode="r") for d in dirs]
    db = [np.load(f"{d}/done_unclip.npy", mmap_mode="r") for d in dirs]
    K_A, rows_A = complete_prefix(stim, da)
    K_B, rows_B = complete_prefix(stim, da + db)
    print(f"被験者 {subj}：K_A = {K_A}、K_B = {K_B}")
    with open(f"{root}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), script="exp24", K_A=K_A, K_B=K_B)) + "\n")
    behav = metas[0]["behav"][:, 0, :]
    res = {"K_A": K_A, "K_B": K_B, "coco73k": metas[0]["coco73k"], "stim": stim}
    nv, ns = len(VARIANTS), len(STAGES)

    # ── B0：同じ画像の試行が何個の session・run に分かれているか（記述）──
    for name, rows in {"A": rows_A, "B": rows_B}.items():
        lab, ses, run = stim[rows], behav[rows, 2], behav[rows, 3]
        n_sess = np.array([len(np.unique(ses[lab == s])) for s in np.unique(lab)])
        res[f"n_sessions_{name}"] = np.array([(n_sess == c).mean() for c in (1, 2, 3)])
        pairs, same_run = 0, 0
        for s in np.unique(lab):
            idx = np.flatnonzero(lab == s)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    pairs += 1
                    same_run += (ses[idx[a]] == ses[idx[b]]) and (run[idx[a]] == run[idx[b]])
        res[f"same_run_pairs_{name}"] = same_run / pairs
        print(f"  集合 {name}：3 試行の session の数 1・2・3 の割合 = {np.round(res[f'n_sessions_{name}'], 4)}"
              f"、同じ run の試行の組 = {res[f'same_run_pairs_{name}']:.4f}")

    # ── 集合 A：カテゴリの分解（元の出力だけ）──
    labels_A = stim[rows_A]
    shuf_A = np.random.default_rng(SEED).permutation(labels_A)
    W_A = np.random.default_rng(SEED).multinomial(K_A, np.ones(K_A) / K_A, size=N_BOOT)
    cats_A = np.unique(labels_A)
    A = {"bc": np.full((nv, ns), np.nan), "boot_bc": np.full((nv, ns, N_BOOT), np.nan)}
    for k in KS_CLUSTER:
        for key in (f"beyond{k}", f"beyond{k}_shuf", f"share{k}"):
            A[key] = np.full((nv, ns), np.nan)
        for key in (f"boot_beyond{k}", f"boot_share{k}"):
            A[key] = np.full((nv, ns, N_BOOT), np.nan)
    for i, d in enumerate(dirs):
        for stage in STAGES_A:
            j = STAGES.index(stage)
            st = multi_stats(load_stage(d, stage, len(rows_A)), labels_A, shuf_A, {"orig": None})["orig"]
            fill_beyond(A, i, j, st, clusters, cats_A, W_A)
            print(f"  被験者 {subj} {VARIANTS[i]:>6} 集合 A {stage:<8} BC = {A['bc'][i, j]:.4f}  "
                  f"beyond(k={K_MAIN}) = {A[f'beyond{K_MAIN}'][i, j]:.4f}（割合 {A[f'share{K_MAIN}'][i, j]:.3f}）")
    finish(A)
    for key, v in A.items():
        res[f"A_{key}_A"] = v

    # ── 集合 B：A・B・C を 1 回の読み込みで ──
    labels_B = stim[rows_B]
    shuf_B = np.random.default_rng(SEED).permutation(labels_B)
    W_B = np.random.default_rng(SEED).multinomial(K_B, np.ones(K_B) / K_B, size=N_BOOT)
    cats_B = np.unique(labels_B)
    sess_B = behav[rows_B, 2]
    cat_fn, n_lone = category_only(labels_B, clusters[K_MAIN])
    res["C_n_lone"] = n_lone
    sess_rand = np.random.default_rng(SEED).permutation(sess_B)   # 対照：session の番号をランダムに入れ替える
    transforms = {"orig": None, "demean": session_demean(sess_B), "demean_rand": session_demean(sess_rand), "catonly": cat_fn}
    A_B = {"bc": np.full((nv, ns), np.nan), "boot_bc": np.full((nv, ns, N_BOOT), np.nan)}
    for k in KS_CLUSTER:
        for key in (f"beyond{k}", f"beyond{k}_shuf", f"share{k}"):
            A_B[key] = np.full((nv, ns), np.nan)
        for key in (f"boot_beyond{k}", f"boot_share{k}"):
            A_B[key] = np.full((nv, ns, N_BOOT), np.nan)
    B = {key: np.full((nv, ns), np.nan) for key in ("bc", "bc_dm", "bc_dm_shuf", "bc_dm_rand")}
    B.update({key: np.full((nv, ns, N_BOOT), np.nan) for key in ("boot_bc", "boot_bc_dm", "boot_bc_dm_rand")})
    Cc = {key: np.full((nv, ns), np.nan) for key in ("bc", "bc_shuf", "beyond", "beyond_real")}
    Cc.update({key: np.full((nv, ns, N_BOOT), np.nan) for key in ("boot_bc", "boot_beyond", "boot_beyond_real")})
    cl = clusters[K_MAIN][cats_B]
    for i, d in enumerate(dirs):
        for stage in STAGES:
            j = STAGES.index(stage)
            st = multi_stats(load_stage(d, stage, len(rows_B)), labels_B, shuf_B, transforms)
            fill_beyond(A_B, i, j, st["orig"], clusters, cats_B, W_B)
            B["bc"][i, j] = bc_from_stats(*st["orig"]["true"])
            B["bc_dm"][i, j] = bc_from_stats(*st["demean"]["true"])
            B["bc_dm_shuf"][i, j] = bc_from_stats(*st["demean"]["shuf"])
            B["boot_bc"][i, j] = [bc_from_stats(*st["orig"]["true"], w=w) for w in W_B]
            B["boot_bc_dm"][i, j] = [bc_from_stats(*st["demean"]["true"], w=w) for w in W_B]
            B["bc_dm_rand"][i, j] = bc_from_stats(*st["demean_rand"]["true"])
            B["boot_bc_dm_rand"][i, j] = [bc_from_stats(*st["demean_rand"]["true"], w=w) for w in W_B]
            Cc["bc"][i, j], Cc["beyond"][i, j] = beyond_from_stats(*st["catonly"]["true"], cl)
            Cc["bc_shuf"][i, j] = bc_from_stats(*st["catonly"]["shuf"])
            Cc["beyond_real"][i, j] = A_B[f"beyond{K_MAIN}"][i, j]
            boot_c = np.array([beyond_from_stats(*st["catonly"]["true"], cl, w) for w in W_B])
            Cc["boot_bc"][i, j], Cc["boot_beyond"][i, j] = boot_c[:, 0], boot_c[:, 1]
            Cc["boot_beyond_real"][i, j] = A_B[f"boot_beyond{K_MAIN}"][i, j]
            print(f"  被験者 {subj} {VARIANTS[i]:>6} 集合 B {stage:<12} BC = {B['bc'][i, j]:.4f}  "
                  f"session の平均を引く {B['bc_dm'][i, j]:.4f}（ランダムな session {B['bc_dm_rand'][i, j]:.4f}、Shuffled {B['bc_dm_shuf'][i, j]:.4f}）  "
                  f"カテゴリだけ BC {Cc['bc'][i, j]:.4f}・beyond {Cc['beyond'][i, j]:.4f}  "
                  f"本物 beyond {A_B[f'beyond{K_MAIN}'][i, j]:.4f}")
    finish(A_B)
    for key, v in A_B.items():
        res[f"A_{key}_B"] = v
    B["ci_bc_dm"], _ = summarize(B["boot_bc_dm"], B["bc_dm"])
    B["ci_bc"], _ = summarize(B["boot_bc"], B["bc"])
    B["ci_dm_minus_orig"], _ = summarize(B["boot_bc_dm"] - B["boot_bc"], B["bc_dm"] - B["bc"])
    B["ci_dm_minus_rand"], _ = summarize(B["boot_bc_dm"] - B["boot_bc_dm_rand"], B["bc_dm"] - B["bc_dm_rand"])
    B["diff_ci_bc_dm"], _ = summarize(B["boot_bc_dm"][0] - B["boot_bc_dm"][1], B["bc_dm"][0] - B["bc_dm"][1])
    for key, v in B.items():
        res[f"B_{key}"] = v
    Cc["ci_bc"], _ = summarize(Cc["boot_bc"], Cc["bc"])
    Cc["ci_beyond"], _ = summarize(Cc["boot_beyond"], Cc["beyond"])
    Cc["ci_real_minus_cat_beyond"], _ = summarize(Cc["boot_beyond_real"] - Cc["boot_beyond"], Cc["beyond_real"] - Cc["beyond"])

    # ── S3：識別精度（全体・クラスタの中）、本物とカテゴリだけの版 ──
    img_index = np.searchsorted(cats_B, labels_B)
    shape = (nv, len(EMBEDDERS))
    for key in ("ident", "ident_within", "ident_cat", "ident_within_cat"):
        Cc[key] = np.full(shape, np.nan)
        Cc[f"boot_{key}"] = np.full(shape + (N_BOOT,), np.nan)
    ident_within_k = {k: np.full(shape, np.nan) for k in KS_CLUSTER}
    for e, emb in enumerate(EMBEDDERS):
        gt_emb = np.load(f"{root}/gt_{emb}.npy")[cats_B]
        for i, d in enumerate(dirs):
            pred = np.load(f"{d}/s3_{emb}.npy")[rows_B].astype(np.float64)
            pred_cat = cat_fn(pred)
            for k in KS_CLUSTER:
                ident_within_k[k][i, e] = np.nanmean(ident_per_trial(pred, gt_emb, img_index, clusters[k][cats_B])[1])
            for tag, p in (("", pred), ("_cat", pred_cat)):
                ov, wi = ident_per_trial(p, gt_emb, img_index, cl)
                for key, vals in ((f"ident{tag}", ov), (f"ident_within{tag}", wi)):
                    Cc[key][i, e] = np.nanmean(vals)
                    Cc[f"boot_{key}"][i, e] = boot_mean(per_stim(vals, img_index, K_B), W_B)
            print(f"  被験者 {subj} {VARIANTS[i]:>6} {emb:<8} 識別精度 本物 {Cc['ident'][i, e]:.4f}"
                  f"（クラスタの中 {Cc['ident_within'][i, e]:.4f}）  カテゴリだけ {Cc['ident_cat'][i, e]:.4f}"
                  f"（クラスタの中 {Cc['ident_within_cat'][i, e]:.4f}）")
    for key in ("ident", "ident_within", "ident_cat", "ident_within_cat"):
        Cc[f"ci_{key}"], _ = summarize(Cc[f"boot_{key}"], Cc[key])
    # S3 で、入れ替えを 20 回変えたときのばらつき（BC、BC_beyond、識別精度）
    N_PERM = 20
    for key in ("perm_bc", "perm_beyond", "perm_ident", "perm_ident_within"):
        Cc[key] = np.full(shape + (N_PERM,), np.nan)
    for e, emb in enumerate(EMBEDDERS):
        gt_emb = np.load(f"{root}/gt_{emb}.npy")[cats_B]
        for i, d in enumerate(dirs):
            pred = np.load(f"{d}/s3_{emb}.npy")[rows_B].astype(np.float64)
            for p in range(N_PERM):
                fn_p, _ = category_only(labels_B, clusters[K_MAIN], seed=SEED + p)
                st_p = multi_stats(pred, labels_B, shuf_B, {"cat": fn_p})["cat"]
                Cc["perm_bc"][i, e, p], Cc["perm_beyond"][i, e, p] = beyond_from_stats(*st_p["true"], cl)
                ov, wi = ident_per_trial(fn_p(pred), gt_emb, img_index, cl)
                Cc["perm_ident"][i, e, p], Cc["perm_ident_within"][i, e, p] = ov.mean(), np.nanmean(wi)
            print(f"  被験者 {subj} {VARIANTS[i]:>6} {emb:<8} カテゴリだけ（入れ替え 20 回）BC {Cc['perm_bc'][i, e].mean():.4f}"
                  f" [{Cc['perm_bc'][i, e].min():.4f}, {Cc['perm_bc'][i, e].max():.4f}]  beyond {Cc['perm_beyond'][i, e].mean():.4f}"
                  f" [{Cc['perm_beyond'][i, e].min():.4f}, {Cc['perm_beyond'][i, e].max():.4f}]")
    Cc["ci_real_minus_cat_ident_within"], _ = summarize(Cc["boot_ident_within"] - Cc["boot_ident_within_cat"],
                                                        Cc["ident_within"] - Cc["ident_within_cat"])
    for key, v in Cc.items():
        res[f"C_{key}"] = v
    for k in KS_CLUSTER:
        res[f"A_ident_within{k}"] = ident_within_k[k]
    res["A_ident"] = Cc["ident"]
    return res


def fill_beyond(out, i, j, st, clusters, cats, W):
    """exp20b の analyze_set と同じ値を out に入れる（BC、BC_beyond、クラスタの割合、Shuffled、bootstrap）。"""
    for k in KS_CLUSTER:
        cl = clusters[k][cats]
        bc, bb = beyond_from_stats(*st["true"], cl)
        out["bc"][i, j], out[f"beyond{k}"][i, j] = bc, bb
        out[f"share{k}"][i, j] = (bc - bb) / (bc - 1)
        out[f"beyond{k}_shuf"][i, j] = beyond_from_stats(*st["shuf"], cl)[1]
        boot = np.array([beyond_from_stats(*st["true"], cl, w) for w in W])
        out["boot_bc"][i, j] = boot[:, 0]
        out[f"boot_beyond{k}"][i, j] = boot[:, 1]
        out[f"boot_share{k}"][i, j] = (boot[:, 0] - boot[:, 1]) / (boot[:, 0] - 1)


def finish(out):
    """信頼区間と、40sess − 1sess の差の信頼区間を足す（exp20b と同じ）。"""
    for key in ["bc"] + [f"{a}{k}" for k in KS_CLUSTER for a in ("beyond", "share")]:
        out[f"ci_{key}"], out[f"ci_raw_{key}"] = summarize(out[f"boot_{key}"], out[key])
        out[f"diff_ci_{key}"], out[f"diff_ci_raw_{key}"] = summarize(
            out[f"boot_{key}"][0] - out[f"boot_{key}"][1], out[key][0] - out[key][1])


# ── 予測の判定（計画 §2.2・3.3・4.3）──────────────────────────────────────

def predictions(R):
    j0, jd, jc = STAGES.index("S0"), STAGES.index("S3_dreamsim"), STAGES.index("S3_clip")
    lines = []
    def report(name, ok_list, subs):
        lines.append(f"  {name}：{sum(ok_list)} / {len(ok_list)} 人で成立（被験者 {', '.join(str(s) for s, ok in zip(subs, ok_list) if not ok) or 'なし'} で不成立）")
    new = [s for s in (2, 5, 7) if s in R]
    allS = [s for s in SUBJECTS if s in R]
    r = R
    # A1：全段階・両モデル・全 k で beyond の下限 > 1（集合 A の S0〜S2 と集合 B の全段階）
    ok = []
    for s in new:
        good = True
        for k in KS_CLUSTER:
            for st in ("A", "B"):
                ci = r[s][f"A_ci_beyond{k}_{st}"]
                good &= bool(np.all(ci[np.isfinite(ci[..., 0])][..., 0] > 1))
        ok.append(good)
    report("A1（BC_beyond の下限 > 1、全段階・両モデル・k = 10・40・100）", ok, new)
    report("A2（S3 DreamSim のクラスタの割合が両モデルで 0.30〜0.60）",
           [bool(np.all((r[s][f"A_share{K_MAIN}_B"][:, jd] >= 0.30) & (r[s][f"A_share{K_MAIN}_B"][:, jd] <= 0.60))) for s in new], new)
    report("A3（両モデルで S3 DreamSim の割合 > S0 の割合）",
           [bool(np.all(r[s][f"A_share{K_MAIN}_B"][:, jd] > r[s][f"A_share{K_MAIN}_B"][:, j0])) for s in new], new)
    report("A4（S3 DreamSim で 1sess − 40sess の割合の差の下限 ≤ 0）",
           [bool(-r[s][f"A_diff_ci_share{K_MAIN}_B"][jd, 1] <= 0) for s in new], new)
    report("A5（S3 DreamSim のクラスタの中の識別精度 > 0.5 かつ < 全体、両モデル）",
           [bool(np.all((r[s][f"A_ident_within{K_MAIN}"][:, 0] > 0.5) & (r[s][f"A_ident_within{K_MAIN}"][:, 0] < r[s]["A_ident"][:, 0]))) for s in new], new)
    report("A6（S3 DreamSim の BC_beyond の 40sess − 1sess の差の下限 > 0）",
           [bool(r[s][f"A_diff_ci_beyond{K_MAIN}_B"][jd, 0] > 0) for s in new], new)
    report("B0（90% 以上の画像で 3 試行が 3 つの session に分かれる、集合 B）",
           [bool(r[s]["n_sessions_B"][2] >= 0.90) for s in allS], allS)
    report("B1（S3 で session の平均を引いた BC と元の BC の差が ±0.05 以内、両モデル・両埋め込み）",
           [bool(np.all(np.abs(r[s]["B_bc_dm"][:, [jd, jc]] - r[s]["B_bc"][:, [jd, jc]]) <= 0.05)) for s in allS], allS)
    report("B4（S3 で本物の session とランダムな session で平均を引いた BC の差が ±0.05 以内、両モデル・両埋め込み）",
           [bool(np.all(np.abs(r[s]["B_bc_dm"][:, [jd, jc]] - r[s]["B_bc_dm_rand"][:, [jd, jc]]) <= 0.05)) for s in allS], allS)
    report("B2（session の平均を引いた BC の下限 > 1、全段階・両モデル）",
           [bool(np.all(r[s]["B_ci_bc_dm"][..., 0] > 1)) for s in allS], allS)
    report("B3（session の平均を引いても S3 DreamSim の 40sess − 1sess の差の下限 > 0）",
           [bool(r[s]["B_diff_ci_bc_dm"][jd, 0] > 0) for s in allS], allS)
    report("C1（カテゴリだけの版の S3 の BC_beyond ≤ 1.05、両モデル・両埋め込み）",
           [bool(np.all(r[s]["C_beyond"][:, [jd, jc]] <= 1.05)) for s in allS], allS)
    report("C2（カテゴリだけの版の S3 DreamSim の BC の下限 > 1、40sess。計画 §4.3 で「1.2〜1.6」から変えた）",
           [bool(r[s]["C_ci_bc"][0, jd, 0] > 1) for s in allS], allS)
    report("C3a（カテゴリだけの版の S3 DreamSim の全体の識別精度 ≥ 0.80、40sess）",
           [bool(r[s]["C_ident_cat"][0, 0] >= 0.80) for s in allS], allS)
    def near_half(s):
        ci = r[s]["C_ci_ident_within_cat"][0, 0]
        v = r[s]["C_ident_within_cat"][0, 0]
        return bool((ci[0] <= 0.5 <= ci[1]) or (0.45 <= v <= 0.55))
    report("C3b（カテゴリだけの版のクラスタの中の識別精度 ≈ 0.5、40sess、S3 DreamSim）", [near_half(s) for s in allS], allS)
    report("C4（本物 − カテゴリだけ：S3 DreamSim の BC_beyond とクラスタの中の識別精度の差の下限 > 0、両モデル）",
           [bool(np.all(r[s]["C_ci_real_minus_cat_beyond"][:, jd, 0] > 0) and np.all(r[s]["C_ci_real_minus_cat_ident_within"][:, 0, 0] > 0))
            for s in allS], allS)
    return lines


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(R, path):
    ink, muted = "#222222", "#666666"
    subs = [s for s in SUBJECTS if s in R]
    jd = STAGES.index("S3_dreamsim")
    x = np.arange(len(subs))
    off = {"40sess": -0.15, "1sess": 0.15}
    run = np.load(RUN_NPZ) if os.path.exists(RUN_NPZ) else None
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for i, v in enumerate(VARIANTS):
        y = np.array([R[s][f"A_share{K_MAIN}_B"][i, jd] for s in subs])
        ci = np.array([R[s][f"A_ci_share{K_MAIN}_B"][i, jd] for s in subs])
        ax.errorbar(x + off[v], y, yerr=np.clip([y - ci[:, 0], ci[:, 1] - y], 0, None), fmt="o", color=COLORS[v],
                    ms=7, capsize=0, elinewidth=2, label=v)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of BC − 1 explained by clusters", color=ink)
    ax.set_title(f"(A) Category share at S3, DreamSim (k = {K_MAIN})", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[0, 1]
    for i, v in enumerate(VARIANTS):
        y0 = np.array([R[s]["B_bc"][i, jd] for s in subs])
        y1 = np.array([R[s]["B_bc_dm"][i, jd] for s in subs])
        ci = np.array([R[s]["B_ci_bc_dm"][i, jd] for s in subs])
        ax.scatter(x + off[v] - 0.07, y0, facecolor="white", edgecolor=COLORS[v], s=45, lw=1.6, zorder=3,
                   label=f"{v}: all trials")
        ax.errorbar(x + off[v], y1, yerr=np.clip([y1 - ci[:, 0], ci[:, 1] - y1], 0, None), fmt="o",
                    color=COLORS[v], ms=7, capsize=0, elinewidth=2, label=f"{v}: session means removed")
        if run is not None:
            y2 = np.array([run[f"s{s}_ratio_run"][i, jd] for s in subs])
            ax.scatter(x + off[v] + 0.07, y2, marker="D", facecolor=COLORS[v], edgecolor="white", s=40, zorder=3,
                       label=f"{v}: same-run pairs only")
    ax.axhline(1, color="#999999", ls="--", lw=1)
    ax.set_ylabel("BC at S3 (DreamSim)", color=ink)
    ax.set_title("(B) Session and run checks", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0.95, 2.15)

    for ax, (xkey, xkey_c, ykey, ykey_c, xl, yl, title) in zip(
            (axes[1, 0], axes[1, 1]),
            (("C_ident", "C_ident_cat", None, "C_bc", "Pairwise identification, all distractors (DreamSim)",
              "BC at S3 (DreamSim)", "(C) Standard measures: decoders vs. category-only"),
             ("C_ident_within", "C_ident_within_cat", f"A_beyond{K_MAIN}_B", "C_beyond",
              "Pairwise identification within clusters (DreamSim)", f"BC beyond category (k = {K_MAIN})",
              "(D) Category-controlled measures"))):
        for i, v in enumerate(VARIANTS):
            xr = [R[s][xkey][i, 0] for s in subs]
            xc = [R[s][xkey_c][i, 0] for s in subs]
            yr = [R[s]["B_bc"][i, jd] if ykey is None else R[s][ykey][i, jd] for s in subs]
            yc = [R[s][ykey_c][i, jd] for s in subs]
            for a, b, c, d in zip(xr, yr, xc, yc):
                ax.plot([a, c], [b, d], color=COLORS[v], lw=0.8, alpha=0.35, zorder=1)
            ax.scatter(xr, yr, facecolor=COLORS[v], edgecolor="white", s=55, zorder=3, label=f"{v} decoder")
            ax.scatter(xc, yc, facecolor="white", edgecolor=COLORS[v], s=55, lw=1.6, zorder=3, label=f"{v} category-only")
        ax.axhline(1, color="#999999", ls="--", lw=1)
        ax.axvline(0.5, color="#999999", ls=":", lw=1)
        ax.set_xlabel(xl, color=ink)
        ax.set_ylabel(yl, color=ink)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlim(0.45, 1.0)

    for ax in axes[0]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"subj {s}" for s in subs])
    for ax in axes.ravel():
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbbbbb")
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"図を保存しました: {path}")


# ── 合成データでの確認 ────────────────────────────────────────────────────

def selftest():
    from exp20_mindeye2_bc import stimulus_stats
    from exp20b_beyond_category import within_cluster_identification
    from exp19_existing_metrics import pairwise_identification
    rng = np.random.default_rng(0)
    C, n, d, G = 120, 3, 50, 12
    cl_of_stim = np.repeat(np.arange(G), C // G)
    labels = np.repeat(np.arange(C), n)
    order = rng.permutation(C * n)
    labels = labels[order]
    clus = rng.normal(0, 1.0, (G, d))
    within = rng.normal(0, 0.7, (C, d))
    X = clus[cl_of_stim[labels]] + within[labels] + rng.normal(0, 1.0, (C * n, d))
    shuf = rng.permutation(labels)
    ok = True

    # 1. 変換なしなら exp20 の stimulus_stats と同じ
    st = multi_stats(X, labels, shuf, {"orig": None})["orig"]
    ref = stimulus_stats(X, [labels, shuf])
    same = all(np.allclose(a, b) for a, b in zip(st["true"][:2], ref[0][:2])) and np.isclose(bc_from_stats(*st["shuf"]), bc_from_stats(*ref[1]))
    print(f"  1. stimulus_stats と一致：{same}"); ok &= same

    # 2. session：繰り返しが別々の session なら、平均を引くと BC は上がる。同じ session なら下がる
    S = 30
    sess_diff = np.zeros(C * n, int)
    for s in range(C):
        idx = np.flatnonzero(labels == s)
        sess_diff[idx] = rng.choice(S, n, replace=False)
    sess_same = rng.integers(0, S, C)[labels]                # 刺激ごとに 1 つの session
    for name, sess, expect in (("別々の session", sess_diff, "上がる"), ("同じ session", sess_same, "下がる")):
        off = rng.normal(0, 1.0, (S, d))
        Xs = X + off[sess]
        r = multi_stats(Xs, labels, shuf, {"orig": None, "dm": session_demean(sess)})
        b0, b1 = bc_from_stats(*r["orig"]["true"]), bc_from_stats(*r["dm"]["true"])
        good = (b1 > b0) if expect == "上がる" else (b1 < b0)
        print(f"  2. {name}：元 {b0:.3f} → 平均を引く {b1:.3f}（{expect}はず）：{good}"); ok &= good
    # 実データと同じ規模（500 刺激 × 3 試行、40 session）で、session のずれがないとき：平均を引いても BC − 1 はわずかに減るだけ
    C2, S2, d2 = 500, 40, 50
    lab2 = rng.permutation(np.repeat(np.arange(C2), n))
    X2 = rng.normal(0, 0.8, (C2, d2))[lab2] + rng.normal(0, 1.0, (C2 * n, d2))
    sess2 = rng.integers(0, S2, C2 * n)
    r = multi_stats(X2, lab2, rng.permutation(lab2), {"orig": None, "dm": session_demean(sess2)})
    b0, b1 = bc_from_stats(*r["orig"]["true"]), bc_from_stats(*r["dm"]["true"])
    rel = (b1 - b0) / (b0 - 1)
    small = abs(rel) < 0.10
    print(f"  2. session のずれがないとき（実データの規模）：BC {b0:.3f} → {b1:.3f}（BC − 1 の変化 {rel:+.1%}、10% 未満のはず）：{small}"); ok &= small

    # 3. カテゴリだけの版：BC_beyond ≈ 1、BC > 1（入れ替えを 20 回変えた平均で見る）
    cats = np.unique(labels)
    r = multi_stats(X, labels, shuf, {"orig": None})
    bc_o, bb_o = beyond_from_stats(*r["orig"]["true"], cl_of_stim[cats])
    vals = []
    for sd in range(20):
        fn, n_lone = category_only(labels, cl_of_stim, seed=sd)
        rc = multi_stats(X, labels, shuf, {"cat": fn})["cat"]
        vals.append(beyond_from_stats(*rc["true"], cl_of_stim[cats]))
    bc_c, bb_c = np.mean(vals, axis=0)
    good = (abs(bb_c - 1) < 0.05) and (bc_c > 1.2) and (bb_o > 1.2)
    print(f"  3. カテゴリだけの版（20 回の平均）：BC {bc_c:.3f}、beyond {bb_c:.3f}（≈ 1 のはず）、本物：BC {bc_o:.3f}・beyond {bb_o:.3f}、"
          f"ほかの刺激がない刺激 {n_lone}：{good}"); ok &= good
    fn1, n1 = category_only(labels, np.where(np.arange(C) == 0, G, cl_of_stim))   # 刺激 0 だけのクラスタ
    print(f"  3. ほかの刺激がないクラスタを数える：{n1 == 1}"); ok &= (n1 == 1)

    # 4. 試行ごとの識別精度の平均が、exp19・exp20b の関数と一致
    gt = clus[cl_of_stim] + within + rng.normal(0, 0.3, (C, d))
    img_index = labels
    ov, wi = ident_per_trial(X, gt, img_index, cl_of_stim)
    with np.errstate(all="ignore"):
        g1 = np.isclose(ov.mean(), pairwise_identification(X, gt, img_index))
    g2 = np.isclose(np.nanmean(wi), within_cluster_identification(X, gt, img_index, cl_of_stim)[0])
    W = np.random.default_rng(SEED).multinomial(C, np.ones(C) / C, size=50)
    g3 = np.isclose(boot_mean(per_stim(ov, img_index, C), np.ones((1, C)))[0], ov.mean())
    print(f"  4. 識別精度が既存の関数と一致：全体 {g1}、クラスタの中 {g2}、重み 1 の bootstrap 平均 {g3}")
    ok &= g1 and g2 and g3
    _ = boot_mean(per_stim(wi, img_index, C), W)
    print("合成データでの確認：" + ("すべて OK" if ok else "失敗あり"))
    return ok


# ── B5：同じ run の中だけで比べる（計画 §3.4）────────────────────────────────

RUN_NPZ = os.path.join(OUTPUT_DIR, "exp24_run_matched.npz")


def gram(X, n_rows, chunk=8192):
    """先頭 n_rows 行の内積の行列（列のかたまりごとに読む）。"""
    X2 = X.reshape(X.shape[0], -1)
    Gm = np.zeros((n_rows, n_rows))
    for s in range(0, X2.shape[1], chunk):
        x = np.asarray(X2[:n_rows, s:s + chunk], dtype=np.float64)
        with np.errstate(all="ignore"):                      # matmul は値が正しくても警告を出すことがあるため
            Gm += x @ x.T
    return Gm


def pair_ratios(Gm, labels, run_key, n_boot=N_BOOT, seed=SEED):
    """試行の組の二乗距離から、組の BC（全体・同じ run）と、違う刺激の組の同じ run ÷ 違う run の比を出す。"""
    nrm = np.diag(Gm)
    D = nrm[:, None] + nrm[None, :] - 2 * Gm
    iu = np.triu_indices(len(labels), 1)
    d = D[iu]
    same_stim = labels[iu[0]] == labels[iu[1]]
    same_run = run_key[iu[0]] == run_key[iu[1]]
    ratio_all = d[~same_stim].mean() / d[same_stim].mean()
    ratio_run = d[~same_stim & same_run].mean() / d[same_stim & same_run].mean()
    between_run_vs_not = d[~same_stim & same_run].mean() / d[~same_stim & ~same_run].mean()
    # run をリサンプリングする bootstrap（同じ run の組だけを使う）
    runs = np.unique(run_key)
    r_of = np.searchsorted(runs, run_key[iu[0]])
    sw = np.bincount(r_of[same_stim & same_run], d[same_stim & same_run], len(runs))
    cw = np.bincount(r_of[same_stim & same_run], minlength=len(runs)).astype(float)
    sb = np.bincount(r_of[~same_stim & same_run], d[~same_stim & same_run], len(runs))
    cb = np.bincount(r_of[~same_stim & same_run], minlength=len(runs)).astype(float)
    Wr = np.random.default_rng(seed).multinomial(len(runs), np.ones(len(runs)) / len(runs), size=n_boot)
    with np.errstate(all="ignore"):
        boot = ((Wr @ sb) / (Wr @ cb)) / ((Wr @ sw) / (Wr @ cw))
    return dict(ratio_all=ratio_all, ratio_run=ratio_run, between_run_vs_not=between_run_vs_not,
                boot_ratio_run=boot, n_within_same_run=int((same_stim & same_run).sum()),
                n_within=int(same_stim.sum()), n_between_same_run=int((~same_stim & same_run).sum()))


def run_matched():
    jd, jc, j0 = STAGES.index("S3_dreamsim"), STAGES.index("S3_clip"), STAGES.index("S0")
    out = dict(subjects=np.array(SUBJECTS), variants=np.array(VARIANTS), stages=np.array(STAGES), n_boot=N_BOOT, seed=SEED)
    ok_all = []
    for subj in SUBJECTS:
        root = out_root(subj)
        dirs = [f"{root}/{m}" for m in models_of(subj)]
        meta = np.load(f"{dirs[0]}/meta.npz")
        stim = meta["stim"]
        da = [np.load(f"{d}/done_prior.npy", mmap_mode="r") for d in dirs]
        db = [np.load(f"{d}/done_unclip.npy", mmap_mode="r") for d in dirs]
        K_B, rows_B = complete_prefix(stim, da + db)
        behav = meta["behav"][:, 0, :]
        labels = stim[rows_B]
        run_key = behav[rows_B, 2] * 100 + behav[rows_B, 3]
        shape = (len(VARIANTS), len(STAGES))
        res = {k: np.full(shape, np.nan) for k in ("ratio_all", "ratio_run", "between_run_vs_not")}
        res["ci_ratio_run"] = np.full(shape + (2,), np.nan)
        for i, d in enumerate(dirs):
            for stage in STAGES:
                j = STAGES.index(stage)
                r = pair_ratios(gram(load_stage(d, stage, len(rows_B)), len(rows_B)), labels, run_key)
                for k in ("ratio_all", "ratio_run", "between_run_vs_not"):
                    res[k][i, j] = r[k]
                res["ci_ratio_run"][i, j], _ = summarize(r["boot_ratio_run"], r["ratio_run"])
                print(f"  被験者 {subj} {VARIANTS[i]:>6} {stage:<12} 組の BC 全体 {r['ratio_all']:.4f}  同じ run {r['ratio_run']:.4f}"
                      f" [{res['ci_ratio_run'][i, j, 0]:.4f}, {res['ci_ratio_run'][i, j, 1]:.4f}]  "
                      f"違う刺激：同じ run ÷ 違う run {r['between_run_vs_not']:.4f}"
                      f"（同じ刺激の組 {r['n_within']}、うち同じ run {r['n_within_same_run']}、違う刺激で同じ run {r['n_between_same_run']}）")
        frac = (res["ratio_run"] - 1) / (res["ratio_all"] - 1)
        res["frac"] = frac
        ok_all.append(bool(np.all(frac[:, [j0, jd, jc]] >= 0.80)))
        for k, v in res.items():
            out[f"s{subj}_{k}"] = v
    np.savez(RUN_NPZ, **out)
    print(f"結果を保存しました: {RUN_NPZ}")
    print(f"  B5（S0・S3 で、同じ run の組の BC − 1 が全体の 80% 以上、両モデル）：{sum(ok_all)} / {len(ok_all)} 人で成立"
          f"（被験者 {', '.join(str(s) for s, ok in zip(SUBJECTS, ok_all) if not ok) or 'なし'} で不成立）")


def selftest_run_matched():
    """合成データ：同じ刺激の試行が同じ run にあり、run ごとのずれがあると、全体の組の BC は大きく出るが、同じ run の組の BC は本来の値に近い。"""
    rng = np.random.default_rng(1)
    C, n, d, R = 400, 3, 40, 160
    labels = np.repeat(np.arange(C), n)
    run_of_stim = rng.integers(0, R, C)
    run_key = run_of_stim[labels]
    mu = rng.normal(0, 0.8, (C, d))
    X = mu[labels] + rng.normal(0, 1.0, (C * n, d))
    true = 1 + 0.64                                          # 1 + σ_s² / σ_w²
    r0 = pair_ratios(X @ X.T, labels, run_key, n_boot=50)
    Xr = X + rng.normal(0, 1.0, (R, d))[run_key]
    r1 = pair_ratios(Xr @ Xr.T, labels, run_key, n_boot=50)
    ok = abs(r0["ratio_all"] - true) < 0.15 and abs(r0["ratio_run"] - true) < 0.2 and r1["ratio_all"] > true + 0.3 and abs(r1["ratio_run"] - true) < 0.25
    print(f"  5. run のずれなし：全体 {r0['ratio_all']:.3f}・同じ run {r0['ratio_run']:.3f}（本来 {true:.2f}）。"
          f"run のずれあり：全体 {r1['ratio_all']:.3f}（大きく出る）・同じ run {r1['ratio_run']:.3f}（本来に近い）：{ok}")
    return ok


# ── 本体 ──────────────────────────────────────────────────────────────────

def load_results(path):
    z = np.load(path, allow_pickle=True)
    R = {}
    for key in z.files:
        if key.startswith("s") and "_" in key and key.split("_")[0][1:].isdigit():
            s = int(key.split("_")[0][1:])
            R.setdefault(s, {})[key[len(f"s{s}_"):]] = z[key]
    return R


def main():
    if "--selftest" in sys.argv:
        ok = selftest()
        ok &= selftest_run_matched()
        sys.exit(0 if ok else 1)
    if "--run_matched" in sys.argv:
        run_matched()
        return
    if "--plot_only" in sys.argv:
        R = load_results(NPZ_PATH)
        plot(R, FIG_PATH)
        print("\n".join(predictions(R)))
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("クラスタを作る（Exp20b と同じ）")
    emb = gt_clip_all()
    clusters = {k: make_clusters(emb, k) for k in KS_CLUSTER}
    coco_1 = np.load(f"{out_root(1)}/{models_of(1)[0]}/meta.npz")["coco73k"]
    R = {}
    for subj in SUBJECTS:
        coco = np.load(f"{out_root(subj)}/{models_of(subj)[0]}/meta.npz")["coco73k"]
        assert np.array_equal(coco, coco_1), f"被験者 {subj}：試行の並びと COCO の番号が被験者 1 と違う"
        R[subj] = analyze_subject(subj, clusters)

    # 被験者 1 は Exp20b と同じ値になるはず
    z = np.load(os.path.join(OUTPUT_DIR, "exp20b_beyond_category.npz"))
    for key in ["bc", f"beyond{K_MAIN}", f"share{K_MAIN}"]:
        for st in ("A", "B"):
            a, b = R[1][f"A_{key}_{st}"], z[f"{key}_{st}"]
            m = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[m], b[m], rtol=1e-9, atol=1e-9), f"被験者 1 の {key}（集合 {st}）が Exp20b と違う"
    assert np.allclose(R[1][f"A_ident_within{K_MAIN}"], z[f"identification_within{K_MAIN}"]), "被験者 1 のクラスタの中の識別精度が Exp20b と違う"
    print("被験者 1 の値は Exp20b と一致した")

    out = dict(subjects=np.array(SUBJECTS), variants=np.array(VARIANTS), stages=np.array(STAGES),
               embedders=np.array(EMBEDDERS), ks_cluster=np.array(KS_CLUSTER), n_boot=N_BOOT, seed=SEED)
    for k in KS_CLUSTER:
        out[f"clusters{k}"] = clusters[k]
    for s, res in R.items():
        for key, v in res.items():
            if key.startswith("A_boot") or key.startswith("B_boot") or key.startswith("C_boot"):
                continue                                     # bootstrap の標本は保存しない（大きいため）
            out[f"s{s}_{key}"] = v
    np.savez(NPZ_PATH, **out)
    print(f"結果を保存しました: {NPZ_PATH}")
    plot(R, FIG_PATH)
    print("予測の判定")
    print("\n".join(predictions(R)))
    print("完了!")


if __name__ == "__main__":
    main()
