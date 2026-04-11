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
- 全ROI で精度は 0.058〜0.629（約 11x の幅）
- 全ROI で BC は 1.035〜1.107（約 1.07x の幅）

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

カテゴリ情報の伝達量 $I(c; \hat{\mathbf{x}})$ は、ガウス近似下で：

$$I(c; \hat{\mathbf{x}}) \approx \frac{1}{2} \log \frac{\text{Var}_{\text{brok}}}{\text{Var}_{\text{pres}}} = \frac{1}{2} \log \text{BC}$$

→ **BC は相互情報量の指数スケール版**であり、$\text{BC} = 1$ は $I = 0$（情報なし）に対応。

$\text{BC} > 1$ は $I > 0$ に対応し、デコードされた特徴量がカテゴリ情報を保持していることを示す。

---

## 7. 限界と正直な制約

1. **線形デコーダーへの依存**: Ridge 回帰の性能に依存。非線形な情報は捉えられない。
2. **特徴量空間の選択**: relu7（fc7, 4096次元）という特定の特徴量空間での測定。DreamSim空間（Phase 2 BC）との乖離が存在する（r=0.308, Exp12）。
3. **カテゴリ単位の集計**: カテゴリ内試行数（35）が少ないと分散推定が不安定。
4. **BC と識別精度の高相関（r=0.97, Exp15）**: 同じデコーダーから両方を計算しているため当然。両者は独立ではないが、測定する性質は異なる（Exp16, d=0.465）。
