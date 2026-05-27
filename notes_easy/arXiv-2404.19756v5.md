# KAN: Kolmogorov–Arnold Networks（研究上の位置づけ）

- arXiv: https://arxiv.org/abs/2404.19756
- 一次ソース: ../papers/arXiv-2404.19756v5/
- 正規ノート: ../notes/arXiv-2404.19756v5.md

---

## 一言で言うと

MLP の「固定活性化をノードに置き、線形重みをエッジに置く」設計を反転し、エッジ上の各重みを B-spline でパラメータ化された学習可能な 1 変数関数に置き換える Kolmogorov-Arnold Networks (KANs) を提案する論文である。著者は、小規模な AI + Science タスクでは KANs が MLPs より高い精度・良い scaling law・高い解釈性を示すと主張するが、その主張は smooth Kolmogorov-Arnold representation の存在や、科学タスクに多い compositional structure への適合という仮定に依存している。

## 何を議論する論文か

- **問題設定**: 非線形関数近似器として標準的に使われる Multi-Layer Perceptrons (MLPs) に対し、精度・パラメータ効率・解釈性の面で代替となるアーキテクチャを作れるかを問う。論文の冒頭では、MLPs は universal approximation theorem に支えられる一方で、Transformer では非 embedding パラメータの大半を消費し、attention layers に比べると post-analysis tools なしでは解釈しにくい、と位置づけられている。
- **対象範囲 / 仮定**: 主な経験的対象は regression、function fitting、PINN 型の Poisson 方程式、continual learning の 1D toy、knot theory、Anderson localization であり、論文自身も「small-scale AI + Science tasks」と限定している。理論面では、対象関数が KAN 形式の smooth representation を持ち、各活性関数 $\Phi_{l,i,j}$ が $(k+1)$ 回連続微分可能であることを仮定する。
- **既存研究との差分**: Kolmogorov-Arnold representation theorem をニューラルネットに使う試み自体は既存研究があるが、多くは元の depth-2, width-$(2n+1)$ 構成に留まる。本論文の差分は、KAN layer を 1D 関数の行列 $\mathbf{\Phi}$ として定義し、任意の width・depth に stack できるようにした点、さらに B-spline、grid extension、sparsification、pruning、symbolification を一体の実用手順として提示した点にある。
- **この論文で答えたい問い**: エッジ上の learnable activation functions によって、MLPs より良い neural scaling laws と解釈可能な中間表現を得られるか。さらに、その解釈性が数学・物理の法則や関係式の再発見に使えるかを、Table や Figure の実験で示そうとしている。

## 背景と前提

- **Universal Approximation Theorem (UAT)**: MLP の表現力を支える古典的な結果で、十分な幅を持つ 2-layer network が任意の連続関数を近似できることを保証する。ただし TeX の §2.3 では、UAT は必要なニューロン数 $N(\epsilon)$ が誤差許容 $\epsilon$ に対してどうスケールするかを与えず、場合によっては次元 $d$ に対して指数的に増えると述べる。
- **Kolmogorov-Arnold representation theorem**: 多変数連続関数を 1 変数関数と加算の合成で表せるという定理で、KAN の発想元である。ただし §2.1 は、元の theorem の 1D functions は non-smooth や fractal になり得るため、そのままでは機械学習で実用的に学習できない可能性がある、と caveat を置く。
- **Splines と MLPs の相補性**: TeX の Introduction は、splines は低次元関数では正確で局所調整しやすく resolution を変えられるが、compositional structures を利用できず curse of dimensionality (COD) に弱いと説明する。一方 MLPs は feature learning により COD に相対的に強いが、ReLU などで 1 変数関数を高精度に近似するのは効率が悪い。KAN は「外側は MLP 的に compositional structure を学び、内側は spline 的に univariate functions を学ぶ」設計として置かれる。
- **Neural scaling laws**: §2.3 では test RMSE $\ell$ がパラメータ数 $N$ に対して $\ell\propto N^{-\alpha}$ と減る現象として定義される。Sharma & Kaplan は $\alpha=(k+1)/d$、Michaud et al. は最大 arity $d^*=2$ の computational graph で $\alpha=(k+1)/2$、本論文は smooth KA representation を仮定して $\alpha=k+1$ を得る、と比較している。著者は cubic splines の $k=3$ を選び、$\alpha=4$ と述べる。
- **AI + Science における解釈性**: 本論文での interpretability は、feature attribution の後処理だけではなく、可視化された activation functions を読み、pruning や `fix_symbolic` によって人間が仮説を入れ、symbolic formulas を取り出す interactive workflow を指す。

## 提案手法

### コアアイデア

KAN は、MLP の scalar weights を learnable univariate functions に置き換える。TeX の abstract と Introduction では、MLPs は fixed activation functions を nodes に置くのに対し、KANs は learnable activation functions を edges に置く、と説明される。KAN の nodes は incoming signals を単に足し合わせるだけで、node 上には非線形性を置かない。

具体的には、各エッジの活性化関数 $\phi_{l,j,i}$ は 1 変数関数であり、実装では residual basis $b(x)=\mathrm{silu}(x)$ と B-spline の線形結合の和としてパラメータ化される。これにより、各エッジは「ただの重み」ではなく、入力値に応じて形を変える曲線になる。KAN layer はこれらの 1D functions の行列 $\mathbf{\Phi}_l$ として定義され、通常の deep network のように複数層を合成できる。

著者の設計上の狙いは 2 つある。第一に、network shape という external degrees of freedom で多変数関数の compositional structure を学ぶこと。第二に、各 spline の grid という internal degrees of freedom で 1 変数関数を高精度に近似することである。§2.4 の grid extension は、後者の internal degrees of freedom を学習途中で増やす手段として導入される。

### 重要な定義・数式

$$
f(\mathbf{x}) = f(x_1,\cdots,x_n)=\sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^n\phi_{q,p}(x_p)\right)
$$

**式の意味**: Kolmogorov-Arnold representation theorem の形であり、滑らかな $f:[0,1]^n\to\mathbb{R}$ を 1 変数関数 $\phi_{q,p}$ と $\Phi_q$、および加算で表す。TeX では Eq. (2.1), `eq:KART` として KAN の発想元に置かれる。

**記号の定義**:
- $f$ ... 入力 $\mathbf{x}=(x_1,\ldots,x_n)$ から実数を返す多変数関数
- $x_p$ ... $p$ 番目の入力変数
- $\phi_{q,p}:[0,1]\to\mathbb{R}$ ... 内側の 1 変数関数
- $\Phi_q:\mathbb{R}\to\mathbb{R}$ ... 外側の 1 変数関数
- $2n+1$ ... 元の Kolmogorov-Arnold theorem に現れる hidden width

**この論文での役割**: KAN の数学的動機である。ただし著者は、この式の 1D functions が non-smooth や fractal になり得るため、元の depth-2 構成に固執せず、より深く広い KAN に一般化する必要があると述べる。

$$
x_{l+1,j} = \sum_{i=1}^{n_l}\tilde{x}_{l,j,i}
= \sum_{i=1}^{n_l}\phi_{l,j,i}(x_{l,i}),\qquad j=1,\cdots,n_{l+1}
$$

**式の意味**: KAN layer の forward pass である。次層の $j$ 番目のノードは、前層の各 activation value $x_{l,i}$ を対応する edge activation $\phi_{l,j,i}$ に通した値を足し合わせるだけで得られる。TeX では Eq. (2.2), `eq:kanforward`。

**記号の定義**:
- $l$ ... layer index
- $n_l$ ... $l$ 層のノード数
- $x_{l,i}$ ... $l$ 層 $i$ 番目の neuron の activation value
- $\phi_{l,j,i}$ ... $(l,i)$ から $(l+1,j)$ へ向かう edge 上の learnable activation function
- $\tilde{x}_{l,j,i}$ ... post-activation、すなわち $\phi_{l,j,i}(x_{l,i})$

**この論文での役割**: MLP との構造差を最も直接に表す式である。MLP が affine transformations $\mathbf{W}$ と fixed nonlinearity $\sigma$ を交互に使うのに対し、KAN は線形変換と非線形性を $\mathbf{\Phi}_l$ にまとめる。

$$
\phi(x)=w_b b(x)+w_s{\rm spline}(x),\qquad
b(x)={\rm silu}(x)=x/(1+e^{-x}),\qquad
{\rm spline}(x)=\sum_i c_iB_i(x)
$$

**式の意味**: 実装で使う 1 本の edge activation function のパラメータ化である。固定の residual basis $b(x)$ と、B-spline basis $B_i(x)$ の線形結合を足す。

**記号の定義**:
- $\phi(x)$ ... KAN の 1 エッジに置かれる learnable activation function
- $w_b,w_s$ ... basis function と spline part の全体スケールを制御する trainable factors
- $b(x)$ ... 多くの場合 SiLU として設定される basis function
- $B_i(x)$ ... B-spline basis function
- $c_i$ ... B-spline coefficient。TeX の implementation details では $c_i\sim\mathcal{N}(0,\sigma^2)$、典型的に $\sigma=0.1$ で初期化するとある

**この論文での役割**: KAN が「learnable activation functions on edges」を実際にどう最適化可能な形にするかを定める。$w_b,w_s$ は表現力のためには冗長だが、activation magnitude の制御と最適化のために残されている。

$$
\left\|f-(\mathbf{\Phi}^G_{L-1}\circ\mathbf{\Phi}^G_{L-2}\circ\cdots\circ\mathbf{\Phi}^G_{1}\circ\mathbf{\Phi}^G_{0})\mathbf{x}\right\|_{C^m}
\leq CG^{-k-1+m}
$$

**式の意味**: Theorem 2.1, `Approximation theory, KAT` の近似誤差 bound である。$f$ が smooth KAN representation を持つなら、各 1D function を $k$-th order B-spline で grid size $G$ によって近似したとき、$C^m$ norm の誤差が $G^{-k-1+m}$ で減る。

**記号の定義**:
- $\mathbf{\Phi}_l$ ... 理想的な KAN layer
- $\mathbf{\Phi}_l^G$ ... grid size $G$ の B-spline functions で近似した KAN layer
- $k$ ... B-spline の order。実験では cubic splines として $k=3$ が使われる
- $m$ ... $C^m$ norm で見る derivative order、$0\leq m\leq k$
- $C$ ... $f$ とその representation に依存する定数
- $G$ ... spline grid size

**この論文での役割**: KAN が COD を回避し得るという理論的根拠である。ただし TeX は、収束率は dimension independent だが、定数 $C$ は representation に依存し dimension にも依存し得るため、その依存性は future work と明記している。

$$
\ell_{\rm total}
= \ell_{\rm pred}
+ \lambda \left(\mu_1 \sum_{l=0}^{L-1}\left|\mathbf{\Phi}_l\right|_1
+ \mu_2 \sum_{l=0}^{L-1}S(\mathbf{\Phi}_l)\right)
$$

**式の意味**: KAN を sparse で解釈しやすくするための training objective である。prediction loss に、activation functions の L1 norm と entropy regularization を足す。TeX では §2.5 `Simplification techniques` にある。

**記号の定義**:
- $\ell_{\rm pred}$ ... 予測損失
- $\ell_{\rm total}$ ... 正則化込みの全体目的関数
- $\lambda$ ... regularization magnitude の全体係数
- $\mu_1,\mu_2$ ... L1 penalty と entropy penalty の相対係数。通常 $\mu_1=\mu_2=1$
- $\left|\mathbf{\Phi}_l\right|_1$ ... $l$ 層の activation functions の L1 norm の総和
- $S(\mathbf{\Phi}_l)$ ... activation magnitudes の分布に対する entropy

**この論文での役割**: KAN の interpretability pipeline の入口である。この正則化により不要な functions を弱め、その後 node-level pruning と symbolification に進む。

### 実装 / アルゴリズム上の要点

- KAN の shape は $[n_0,n_1,\cdots,n_L]$ と表され、$n_i$ は $i$ 層目のノード数である。元の Kolmogorov-Arnold representation は shape $[n,2n+1,1]$ の 2-Layer KAN に対応する。
- KAN layer は $n_{\rm out}\times n_{\rm in}$ 個の 1D functions の行列 $\mathbf{\Phi}_l$ として扱われる。network 全体は ${\rm KAN}(\mathbf{x})=(\mathbf{\Phi}_{L-1}\circ\cdots\circ\mathbf{\Phi}_0)\mathbf{x}$ と書ける。
- 実装上、各 spline grid は training 中に input activations に応じて on the fly で更新される。これは splines が bounded regions 上で定義される一方、activation values は学習中にその範囲外へ動き得るためである。
- parameter count は、depth $L$、等幅 $N$、spline order $k$、grid intervals $G$ の簡単化された設定で $O(N^2L(G+k))\sim O(N^2LG)$ とされる。MLP の $O(N^2L)$ より大きく見えるが、著者は KANs は通常 MLPs より小さい $N$ で済むと主張する。
- **Grid extension** は、粗い grid $G_1$ の spline $f_{\rm coarse}(x)=\sum_i c_i B_i(x)$ を細かい grid $G_2$ の spline $f_{\rm fine}(x)=\sum_j c'_jB'_j(x)$ に least squares で写す手順である。KAN 内の全 splines に独立に適用される。
- **Sparsification / visualization / pruning / symbolification** が interpretability の主要 pipeline である。node pruning では incoming score $I_{l,i}$ と outgoing score $O_{l,i}$ の両方が threshold $\theta=10^{-2}$ を超える node を重要とみなす。visualization では activation の透明度を $\tanh(\beta A_{l,i,j})$、$\beta=3$ に比例させる。
- **Symbolification** では、`fix_symbolic(l,i,j,f)` によって activation $\phi_{l,i,j}$ を symbolic function $f$ に固定する。ただし affine shifts/scalings を考慮するため、sample の preactivation $x$ と postactivation $y$ から $y\approx cf(ax+b)+d$ の $(a,b,c,d)$ を fitting する。
- toy symbolic regression 例では、$[2,5,1]$ KAN を sparsification して 4/5 hidden neurons を落とし、$[2,1,1]$ に pruning した後、`sin`, `x^2`, `exp` を `fix_symbolic` し、Sympy で $1.0e^{1.0y^2+1.0{\rm sin}(3.14x)}$ を出す流れが示される。

## 実験・結果

- **データセット / ベンチマーク**: Toy datasets 5 種、15 special functions、Feynman dataset の 27 方程式、Poisson equation、1D continual learning の 5 Gaussian peaks、knot theory の signature prediction と unsupervised relation discovery、Anderson localization の Mosaic Model (MM), generalized Aubry-André model (GAAM), modified Aubry-André model (MAAM) が扱われる。
- **比較対象 / baseline**: Toy では MLPs with different depths and widths、special functions では fixed width 5 または 100 の MLPs で depths $\{2,3,4,5,6\}$、Feynman では human-constructed KAN、unpruned KAN、pruned KAN、fixed width 5 の MLPs with Tanh/ReLU/SiLU、PDE では PINN framework 内の MLPs、knot では Davies et al. の Deepmind's 4-layer width-300 MLP が比較される。
- **指標**: function fitting では test RMSE と Pareto frontier、PDE では $L^2$ squared loss と $H^1$ squared loss、および MSE、knot supervised では cross-entropy training 後の test accuracy、Anderson localization では fractal dimension $D_k=-\log(\mathrm{IPR}_k)/\log(N)$ から localized / extended を分類する accuracy が使われる。
- **主な結果**:
  - Toy datasets では、$J_0(20x)$、$\exp(\sin(\pi x)+y^2)$、$xy$、100D の $\exp(\frac{1}{100}\sum_{i=1}^{100}\sin^2(\frac{\pi x_i}{2}))$、4D の $\exp(\frac{1}{2}(\sin(\pi(x_1^2+x_2^2))+\sin(\pi(x_3^2+x_4^2))))$ を用いる。KANs は $G=\{3,5,10,20,50,100,200,500,1000\}$ で 200 steps ごとに grid points を増やし、MLPs と KANs はともに LBFGS で 1800 steps train される。Figure `fig:model_scaling` では KANs が理論線 $\alpha=k+1=4$ に近く、MLPs は Sharma & Kaplan の $\alpha=4/d$ よりも遅く plateau しやすいと述べる。
  - Grid extension の toy $f(x,y)=\exp(\sin(\pi x)+y^2)$ では、$[2,5,1]$ KAN の training samples は 1000、parameters は $15G$ なので interpolation threshold を $G=1000/15\approx 67$ と予想し、観測値 $G\sim 50$ とおおむね一致すると述べる。$[2,1,1]$ KAN は test RMSE が roughly $G^{-3}$、median-based view では $G^{-4}$ に近い。
  - Special functions は Table `tab:special_kan_shape` の 15 個で、`ellipj`, `ellipkinc`, `ellipeinc`, `jv`, `yv`, `kv`, `iv`, `lpmv`, `sph_harm` などが含まれる。Figure `fig:special_pf` では KANs の Pareto frontier が MLPs より consistently better とされる。例として ${\rm lpmv}(0,x,y)$ は best KAN shape [2,2,1] で test RMSE $5.25\times 10^{-5}$、MLP test RMSE $1.74\times 10^{-2}$ である。
  - Feynman dataset では、`Feynman_no_units` の少なくとも 2 variables を持つ問題を対象にする。Table `tab:feynman_kan_shape` は 27 方程式を含み、各 KAN は $G=3$ から始め、200 steps ごとに $G=\{3,5,10,20,50,100,200\}$ を cover する。著者は "MLPs and KANs behave comparably on average" と明記する。一方、auto-discovered KAN shapes は human constructions より小さいことが多く、相対論的速度合成 $\frac{u+v}{1+uv}$ では human-constructed shape [2,2,2,2,2,1] に対して pruned KAN shape [2,2,1] が得られ、rapidity trick $\tanh(\mathrm{arctanh}\,u+\mathrm{arctanh}\,v)$ と結びつけて説明される。
  - Poisson equation では $\Omega=[-1,1]^2$、zero Dirichlet boundary data、true solution $u=\sin(\pi x)\sin(\pi y^2)$ を用いる。PINN loss で $n_i=10000$, $n_b=800$, $\alpha=0.01$ とし、2-Layer width-10 KAN が 4-Layer width-100 MLP より 100 times more accurate ($10^{-7}$ vs $10^{-5}$ MSE) かつ 100 times more parameter efficient ($10^2$ vs $10^4$ parameters) と報告される。
  - Continual learning では 1D regression の 5 Gaussian peaks を順番に提示する。Figure `fig:continual-learning` で、KAN は現在の data がある region だけを remodel し、以前の region を保つ一方、MLP は新しい samples によって whole region を remodel して catastrophic forgetting を示す、と説明される。
  - Knot theory supervised では、17 knot invariants を inputs、signature を outputs とし、even signatures を one-hot vectors に encoding して cross-entropy loss で train する。$[17,1,14]$ KAN ($G=3$, $k=3$) は $81.6\%$ test accuracy、約 $2\times 10^2$ parameters を達成し、Deepmind's 4-layer width-300 MLP は $78.0\%$、約 $3\times 10^5$ parameters である。3 important variables の $[3,1,14]$ KAN は $78.2\%$。ablation では $\mu_r$ alone が $65.0\%$、$\mu_i$ alone が $43.8\%$ とされる。Table `tab:knot_sf` では formula F が $\mu_r,\lambda$ のみで $77.8\%$ test accuracy を示す。
  - Knot theory unsupervised では、signature を含む 18 variables をすべて inputs として扱い、real data を positive、features を random shuffle したものを negative とする。$[18,1,1]$ KAN の second layer activation を peak at zero の Gaussian に固定し、$\sum_{i=1}^{18}g_i(x_i)=0$ 型の関係を読む。$\lambda=\{10^{-2},10^{-3}\}$、seed $=\{0,1,\cdots,99\}$ の 200 networks は 3 clusters に分類され、signature relation、$-\log V+\log\mu_r+\log\lambda=0$ すなわち $V=\mu_r\lambda$、short geodesic の real part $g_r$ と injectivity radius の relation を再発見する。
  - Anderson localization では、MM、GAAM、MAAM の mobility edge を KAN で抽出する。GAAM の theory は $\alpha E+2\lambda-2=0$ で accuracy $99.2\%$、KAN auto は $21.06\alpha E+45.13\lambda-54.45=0$ に主要項を持つ式で $99.0\%$。MAAM の theory は $E+\exp(p)-\lambda\cosh p=0$ で $98.6\%$、KAN auto は $97.1\%$、manual step 2/3 は $97.7\%$、step 4A は $96.6\%$、step 4B は $95.4\%$ である。
- **著者が主張する貢献**: 任意 depth/width の KAN architecture、KAT による dimension-independent rate の近似理論、grid extension による fine-graining、sparsification/pruning/symbolification による interpretability pipeline、そして PDE・continual learning・数学・物理での small-scale AI + Science 実証である。コードは TeX の Introduction で `https://github.com/KindXiaoming/pykan` と `pip install pykan` が示される。

## 妥当性と限界

- **この主張を支える根拠**: KAN の精度主張は、Theorem KAT の $G^{-k-1+m}$ bound、toy functions での $\alpha=4$ 近傍の scaling、special functions の Pareto frontier、PDE での MSE と parameter count の比較に支えられる。解釈性主張は、activation functions を直接可視化できる構造、sparsification/pruning/symbolification の手順、knot と Anderson localization で実際に変数依存や symbolic formulas を取り出した例に支えられる。
- **著者が認めている limitations / future work**: KAT は smooth KAN representation の存在を仮定しており、任意 task に KA representations が存在するか、training がそれを見つけるかは未解決である。定数 $C$ は representation と dimension に依存し得る。数学的には元の Kolmogorov-Arnold theorem は shape $[n,2n+1,1]$ の restricted subclass に対応するだけで、deep KAN に対応する generalized theorem はまだない。training は同じ parameter count の MLPs より通常 10x slower とされる。PDE 実験では true solution が symbolic formula であるため MLP に不利かもしれない、と本文が述べる。continual learning は extremely simple 1D toy であり、高次元で locality をどう定義するかは不明である。
- **読者として注意すべき点**: 「KAN が COD を破る」という主張は、収束率の次元非依存性を指しており、全ての高次元関数に一様に有利という意味ではない。一般の Sobolev/Besov 空間では COD を避けられないという議論も §2.3 で触れられている。Feynman dataset では KAN と MLP は平均的には同程度であり、KAN の優位は special functions や symbolic/scientific structure のある task に偏っている可能性がある。interpretability も hyperparameters に敏感で、Appendix `Dependence on hyperparameters` は entropy penalty、random seeds、$\lambda$、$G$、$k$ の影響を挙げる。
- **追加で確認したい実験 / 疑問**: 同じ wall-clock time や同じ training compute で KAN と MLP の Pareto frontier がどう変わるか。symbolic formula ではない PDE solution や noisy data でも PDE の優位が保たれるか。language modeling や image classification など machine-learning-related tasks で KAN の inductive bias が有効か。grid extension、pruning、symbolification の hyperparameter sensitivity をどの程度自動化できるか。unsupervised relation discovery で、seed に頼らず complete set of relations を見つける方法があるか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **KAN (Kolmogorov-Arnold Network)** ... edge 上に learnable univariate activation functions を置き、node は incoming post-activations を加算する network。元の KA theorem に敬意を示すが、実際には任意 depth/width に一般化される。
- **MLP (Multi-Layer Perceptron)** ... fixed activation functions on nodes と affine transformations を交互に使う標準的な fully-connected feedforward network。KAN の主比較対象。
- **Learnable activation functions on edges** ... KAN の各 scalar weight を置き換える 1D function。B-spline と residual SiLU によってパラメータ化される。
- **B-spline** ... KAN の edge activation の spline part を構成する local basis。局所性により grid extension と continual learning の議論に関わる。
- **Grid extension** ... coarse grid の spline を fine grid の spline へ least squares で移す手順。モデルを大きく作り直すのではなく、activation functions の internal degrees of freedom を増やす。
- **External / internal degrees of freedom** ... external dofs は network graph の shape、internal dofs は activation function 内の grid points。KAN は両方を持つため、composition と univariate approximation を分担できると著者は説明する。
- **KAT (KAN Approximation Theorem)** ... smooth KAN representation が存在する場合に、B-spline grid size $G$ による近似誤差 bound を与える Theorem 2.1。UAT と違い scaling rate を議論するが、仮定は強い。
- **Scaling exponent $\alpha$** ... $\ell\propto N^{-\alpha}$ の指数。KAN では cubic splines の $k=3$ から $\alpha=k+1=4$ とされる。
- **Sparsification** ... activation functions の L1 norm と entropy に penalty をかけ、不要な functions を弱くする処理。pruning の前段階。
- **Pruning** ... node-level に incoming/outgoing scores を見て不要な neurons を消す処理。default threshold は $\theta=10^{-2}$。
- **Symbolification** ... learned activation を `sin`, `exp`, `log` などの symbolic function に置き換える処理。`fix_symbolic(l,i,j,f)` と affine fitting $y\approx cf(ax+b)+d$ が使われる。
- **Pareto frontier** ... 同じ model family 内で、他の fit に比べて「より単純かつより正確」に同時に支配されない点。special functions と Feynman で parameter count と RMSE の平面に描かれる。
- **Feynman_no_units** ... Feynman dataset のうち units を除いた式データ。KAN では univariate tasks は 1D spline になり trivial として、少なくとも 2 variables を持つ問題が対象。
- **PINN** ... PDE residual と boundary condition を loss に入れる physics-informed neural network framework。Poisson equation 実験で KAN と MLP の表現器比較に使われる。
- **IPR / fractal dimension** ... Anderson localization で states の localized / extended を測るための量。$\mathrm{IPR}_k$ から $D_k=-\log(\mathrm{IPR}_k)/\log(N)$ を計算し、$D_k=0$ は localized、$D_k=1$ は extended を示す。
- **Meridinal / longitudinal translation** ... knot theory の signature prediction で重要とされた variables。TeX では $\mu$ の real/imag parts $\mu_r,\mu_i$ と $\lambda$ が signature 依存の中心として扱われる。

## 読む順番の提案

- 最初に abstract と Figure `fig:kan_mlp` を読み、MLP と KAN の違いが「activation on nodes」対「activation on edges」であることを確認する。正規ノートの Summary 冒頭の「重み＝学習可能な 1D 関数」という説明につながる。
- 次に §2.1 の Eq. `eq:KART` と §2.2 の Eq. `eq:kanforward`, `eq:KAN_forward` を読む。ここで元の KA theorem と、著者が任意 depth/width の KAN layer に一般化する論理を押さえる。
- その後 §2.3 の Theorem `approx thm` とその直後の caveat を読む。正規ノートの KAT・$\alpha=k+1$・定数 $C$ の注意点につながる。
- §2.4 の grid extension と Figure `fig:grid-extension` で、$G$ を増やす実装上の意味、$G^{-3}$ と median で $G^{-4}$ に近いという差、interpolation threshold の議論を見る。
- §2.5 の `Simplification techniques` を読んで、L1/entropy regularization、visualization、pruning、symbolification の順番を確認する。正規ノートの interpretability pipeline と対応する。
- 実験は §3.1 toy、§3.2 special functions Table `tab:special_kan_shape`、§3.3 Feynman Table `tab:feynman_kan_shape`、§3.4 PDE、§3.5 continual learning の順で読む。KAN が強い例と、Feynman のように平均では MLP と同程度の例を分けて読む。
- 解釈性の応用は §4.3 knot theory の Table `tab:math-compare`, `tab:knot_sf`、§4.4 Anderson localization の Table `tab:al_sf` を優先する。正規ノートの Critical Thoughts にある「feature attribution 不要」「人間との対話的簡略化」「新発見ではなく再発見も多い」という論点と接続する。
- 最後に §6 Discussion と Appendix `Dependence on hyperparameters` を読む。速度、深い KAN の数学的基礎、一般 task への外挿、random seed や $\lambda,G$ 依存の限界を確認できる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2404.19756v5/`
- 正規ノート: `notes/arXiv-2404.19756v5.md`
