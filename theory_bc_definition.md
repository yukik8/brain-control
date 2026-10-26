# BC（脳制御度）の理論的定義

## 1. 定義

### 記号

- $\mathbf{x}_i \in \mathbb{R}^d$: 試行 $i$ のデコードされた特徴量ベクトル
- $c_i \in \{1, \ldots, C\}$: 試行 $i$ が属するカテゴリラベル
- $S_c = \{i : c_i = c\}$: カテゴリ $c$ に属する試行のインデックス集合
- $\pi$: 試行インデックスのランダム置換

### Var(preserved)（保存分散）

$$\text{Var}_{\text{pres}} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{d} \sum_{k=1}^{d} \text{Var}_{i \in S_c}[x_{ik}]$$

同カテゴリの試行内での特徴量の分散。カテゴリ内の trial-to-trial variability を表す。

### Var(broken)（破壊分散）

$$\text{Var}_{\text{brok}} = \mathbb{E}_{\pi}\left[\frac{1}{C} \sum_{c=1}^{C} \frac{1}{d} \sum_{k=1}^{d} \text{Var}_{i \in S_c}[x_{\pi(i),k}]\right]$$

試行インデックスをランダムシャッフルした後の、同「カテゴリスロット」内の分散の期待値。
刺激とデコード特徴量の対応関係を破壊したときの分散。

### BC の定義

$$\text{BC} = \frac{\text{Var}_{\text{brok}}}{\text{Var}_{\text{pres}}}$$

---

## 2. 直感的解釈

### BC = 1 の意味

$$\text{BC} = 1 \iff \text{Var}_{\text{brok}} = \text{Var}_{\text{pres}}$$

→ デコードされた特徴量は、どの刺激が提示されたかに関わらず同じ分散構造を持つ。
つまり、**脳活動の情報は特徴量に伝わっていない**（generator prior のみが分散を決定）。

### BC > 1 の意味

$$\text{BC} > 1 \iff \text{Var}_{\text{brok}} > \text{Var}_{\text{pres}}$$

→ ランダムに対応付けたときより、正しい対応付けのときのほうが分散が **小さい**。
つまり、**同カテゴリの刺激を見たときの脳活動は、異なるカテゴリの刺激を見たときより一貫した特徴量を生成する**。

BC - 1 が脳の制御の強さを表す効果量（effect size）に相当する。

---

## 3. SNR との関係

線形デコーダー $\hat{\mathbf{x}} = W\mathbf{b} + \boldsymbol{\epsilon}$ を仮定する。ここで：
- $\mathbf{b}$: 脳活動（観測）
- $W$: Ridge 回帰の重み行列
- $\boldsymbol{\epsilon}$: ガウス残差（分散 $\sigma_\epsilon^2$、等方的と仮定）

カテゴリ $c$ の試行の脳活動を $\mathbf{b}^{(c)}_i = \boldsymbol{\mu}_c + \boldsymbol{\delta}_i$ と分解すると（$\boldsymbol{\mu}_c$: カテゴリ平均, $\boldsymbol{\delta}_i$: ノイズ）：

$$\text{Var}_{\text{pres}} \propto \|W\|_F^2 \cdot \sigma_\delta^2 + \sigma_\epsilon^2$$

$$\text{Var}_{\text{brok}} \propto \|W\|_F^2 \cdot (\sigma_\delta^2 + \sigma_\mu^2) + \sigma_\epsilon^2$$

ここで $\sigma_\mu^2$ はカテゴリ間の脳活動の分散。よって：

$$\text{BC} \approx 1 + \frac{\|W\|_F^2 \cdot \sigma_\mu^2}{\|W\|_F^2 \cdot \sigma_\delta^2 + \sigma_\epsilon^2}$$

→ これは $\text{SNR} = \sigma_\mu^2 / \sigma_\delta^2$ が増えると BC が増えることを示す。

**重要な含意**:
- BC ≈ 1 + SNR_effective（ただし SNR は特徴量空間での有効 SNR）
- 識別精度は SNR の単調増加関数だが、飽和する（天井効果）
- BC は SNR に比例して増え続けるため、高精度領域での差も検出できる

---

## 4. 識別精度との関係

識別精度（Top-1, LOO コサイン類似度）を $A$ とすると：

$$A \approx \Phi\left(\frac{\sqrt{n/2} \cdot d(\boldsymbol{\mu}_c, \boldsymbol{\mu}_{c'})}{\sigma_\delta}\right)$$

（$\Phi$: 正規分布の累積分布関数、$d$: カテゴリ間の距離、$n$: 1カテゴリあたりの試行数）

精度は信号対雑音比 $d/\sigma_\delta$ の **非線形関数**（シグモイド型）なのに対し、
BC は **線形的な近似**で増加する。

→ **高 SNR 領域では精度が飽和しても BC は増加し続ける**

実験的には：
- 全ROI で精度は 0.070〜0.645（約 9x の幅）
- 全ROI で BC は 1.034〜1.116（約 1.08x の幅）

BC の変動幅が小さいのは、全 ROI が依然として低 SNR 領域にあるためと解釈できる。

---

## 5. Prior 支配の定義（Exp13 との接続）

**generator prior** が再構成を決定する割合を $p_{\text{prior}}$ と定義すると：

$$\mathbf{x}_{\text{recon}} = p_{\text{prior}} \cdot \mathbf{g}(\mathbf{z}) + (1 - p_{\text{prior}}) \cdot f_{\text{brain}}(\mathbf{b})$$

ここで $\mathbf{g}(\mathbf{z})$ は generator の出力（prior に由来）、$f_{\text{brain}}(\mathbf{b})$ は脳依存の成分。

Exp13 の結果：
- Real: BC = 1.259 → 脳が再構成に寄与
- Shuffled: BC = 1.001 ≈ 1 → 純粋な prior 支配

つまり **BC = 1 は「prior 支配」の empirical な証拠**であり、
「BC = 1 ならば $p_{\text{prior}} \approx 1$」が成立する。

---

## 6. BC の定義の正当化（なぜ Var 比か）

情報理論的な観点から：

各カテゴリ内の分布が共通の分散を持つガウス分布だと仮定すると、特徴量の各次元 $k$ について：

$$I(c; \hat{x}_k) = h(\hat{x}_k) - h(\hat{x}_k \mid c) \le \frac{1}{2} \log \frac{\sigma_T^2}{\sigma_W^2} = \frac{1}{2} \log \text{BC}_k$$

（$\sigma_T^2$: 全体の分散、$\sigma_W^2$: カテゴリ内分散、$\text{BC}_k$: 次元 $k$ だけで計算した BC）

- 不等号になるのは、分散が同じならガウス分布のエントロピーが最大だから。カテゴリ間の差がカテゴリ内のばらつきより十分小さい（BC が 1 に近い）ときはほぼ等号になり、本研究（BC ≈ 1.1〜1.3）はこの領域にある。
- $\text{BC} = 1$ は $I = 0$（情報なし）、$\text{BC} > 1$ は $I > 0$（デコードされた特徴量がカテゴリ情報を保持している）に対応する。
- 特徴量ベクトル全体（$d$ 次元）の上界は共分散行列の行列式の比 $\det\Sigma_T / \det\Sigma_W$ で決まり、トレースで定義した BC の関数にはならない。したがって $\frac{1}{2} \log \text{BC}$ をベクトル全体の相互情報量とは読めない。

---

## 7. ANOVA の F 値・Calinski–Harabasz 指数との関係

$SS_W$, $SS_B$, $SS_T = SS_W + SS_B$ を、カテゴリ内・カテゴリ間・全体の平方和（全次元で合計）とする。

非復元抽出した部分集合の標本分散は母集団の分散の不偏推定量なので、各カテゴリの試行数 $n$ がそろっていれば、シャッフルの期待値は閉じた式で書ける：

$$\text{Var}_{\text{brok}} = \frac{SS_T}{d(N-1)}, \qquad \text{Var}_{\text{pres}} = \frac{SS_W}{d(N-C)}$$

$$\text{BC} = \frac{N-C}{N-1} \cdot \frac{SS_T}{SS_W} = 1 + \frac{C-1}{N-1}(F - 1), \qquad F = \frac{SS_B/(C-1)}{SS_W/(N-C)}$$

- $F$ はトレースで計算した一元配置分散分析の F 値で、カテゴリをクラスタとみなした **Calinski–Harabasz 指数**（Caliński & Harabasz, 1974）と同じもの。
- BC と F は同じ情報を持つ（一方が増えればもう一方も増える）。1,000 回のシャッフルが必要なのは p 値を出すときだけ。
- 違いはスケール。同じ効果でも F は 1 カテゴリあたりの試行数 $n$ にほぼ比例して大きくなるが、$\text{BC} \to 1 + \mathrm{tr}\,\Sigma_B / \mathrm{tr}\,\Sigma_W$ で $n$ によらない。F が検定統計量、BC が効果量にあたる。
- 実測（Exp19, Subject1 Real）: $n$ を 35 → 5 に減らすと CH は 10.25 → 2.31 に下がるが、BC は 1.257〜1.261 でほぼ一定。5被験者 × 3条件のすべてで、シャッフルによる BC と閉じた式の差は 0.0002 未満。

---

## 8. 限界と正直な制約

1. **線形デコーダーへの依存**: Ridge 回帰の性能に依存。非線形な情報は捉えられない。
2. **特徴量空間の選択**: 特定の特徴量空間での測定（Exp13/17/18 は GOD の cnn8, 1000次元、Exp15/16 は AlexNet relu7, 4096次元）。DreamSim空間（Phase 2 BC）との相関は弱く有意でない（r=0.257, p=0.072, Exp12）。
3. **カテゴリ単位の集計**: カテゴリ内試行数（35）が少ないと分散推定が不安定。
4. **BC と識別精度の高相関（r=0.98, Exp15）**: 同じデコーダーから両方を計算しているため当然。両者は独立ではないが、測定する性質は異なる（Exp16, d=0.436）。
