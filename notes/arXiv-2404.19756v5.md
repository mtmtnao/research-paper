# KAN: Kolmogorov–Arnold Networks

- arXiv: https://arxiv.org/abs/2404.19756
- source: ../papers/arXiv-2404.19756v5/
- authors: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark
- venue / year: TeX 中には明示なし（main source は neurips_2023.sty を preprint で使用）
- tags: [KAN, MLP-alternative, splines, interpretability, AI4Science, scaling-laws]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: 現代の深層学習の基幹である MLP は universal approximation theorem に裏付けられているが、(i) 解釈性が低く、(ii) 次元の呪い (COD) に弱い疎な構造を持つ関数（多くの科学・物理の関数がこれにあたる）に対して非効率。spline は逆に低次元では強いが COD に弱い。両者の良いところだけを取れないか。
- **手法**: Kolmogorov–Arnold 表現定理 $f(x)=\sum_{q=1}^{2n+1}\Phi_q(\sum_{p=1}^n\phi_{q,p}(x_p))$ をニューラルネット化した KAN を提案。
  - エッジに **学習可能な 1 変数関数（B-spline）**、ノードは単純な総和。線形重みは原理上ゼロ（実装では magnitude 制御の $w_b, w_s$ と residual の SiLU $b(x)$ は残す: $\phi(x)=w_b\,{\rm silu}(x)+w_s\sum_i c_i B_i(x)$）。
  - 元の定理は深さ 2 だが、KAN レイヤを行列 $\mathbf\Phi=\{\phi_{q,p}\}$ と再定義することで任意の width・depth に拡張: ${\rm KAN}(\mathbf x)=(\mathbf\Phi_{L-1}\circ\cdots\circ\mathbf\Phi_0)\mathbf x$。
  - **Grid extension**: 粗いグリッドで学習した spline を細かいグリッドに最小二乗で写しなおすことで、再学習なしにキャパシティを増やせる。
  - **解釈性のための簡略化**: L1 正則化に加え活性関数のエントロピー正則化（$\mu_1=\mu_2=1$, 全体係数 $\lambda$）→ ノード単位のプルーニング（incoming/outgoing $\max|\phi|_1>\theta=10^{-2}$）→ `fix_symbolic` で affine $y\approx cf(ax+b)+d$ に当てはめて記号化。
  - 理論: 各 $\phi$ が $C^{k+1}$ なら $\|f-f_G\|_{C^m}\le CG^{-k-1+m}$（Theorem「KAT」）。**$G$ に対する収束率は次元に依存しない**ため COD を回避できる、と著者は述べる。ただし定数 $C$ は表現に依存し、次元依存性の議論は future work。スケーリング指数 $\alpha=k+1$、$k=3$（cubic）で $\alpha=4$。
- **結果**:
  - **Toy 5 例**（Bessel $J_0(20x)$ / $\exp(\sin\pi x+y^2)$ / $xy$ / 100D $\exp(\frac{1}{100}\sum\sin^2)$ / 4D ネスト）で KAN は理論線 $\alpha=4$ にほぼ到達、MLP は早期に頭打ち（Fig. model_scaling）。
  - **Special functions 15 個**（scipy.special, e.g. `ellipj`, `jv`, `yv`, `kv`, `iv`, `lpmv`, `sph_harm`）で KAN の Pareto frontier が一貫して MLP より上。例: `lpmv(0,x,y)` 最良 KAN test RMSE $5.25\times10^{-5}$ vs MLP $1.74\times 10^{-2}$（Table tab:special_kan_shape）。
  - **Feynman dataset**: Table tab:feynman_kan_shape の 27 方程式で KAN と MLP は平均ほぼ互角（"MLPs and KANs behave comparably on average"）。ただし auto-pruning が見つける KAN は人手で構成した KAN より小さくなることが多い（例: 相対論的速度合成 $\frac{u+v}{1+uv}$ で人手 5 層 [2,2,2,2,2,1] → auto 2 層 [2,2,1]、これは rapidity trick に対応）。
  - **PDE (Poisson, $u_{xx}+u_{yy}=f$ on $[-1,1]^2$)**: 2-Layer width-10 KAN が 4-Layer width-100 MLP より MSE で 100 倍精度良 ($10^{-7}$ vs $10^{-5}$)、パラメータも 100 倍少 ($10^2$ vs $10^4$)。
  - **Continual learning**: 5 Gaussian peaks の 1D 逐次学習で MLP は catastrophic forgetting、KAN は spline の局所性のおかげで forgetting なし。
  - **Knot theory (DeepMind dataset)**: 17 knot invariants → signature 予測タスクで $[17,1,14]$ KAN（約 200 params, $G=3, k=3$）が **81.6%** 精度。DeepMind の 4-layer width-300 MLP（約 $3\times 10^5$ params）の 78.0% を上回る。$\mu_r,\mu_i,\lambda$ への依存を feature attribution 不要で可視化のみで特定。さらに $\mu_r,\lambda$ のみ用いる新公式 F（77.8%）を発見。
  - **Anderson localization (Mosaic / GAAM / MAAM)**: KAN が mobility edge を symbolic に抽出。GAAM では真公式 $\alpha E+2\lambda-2=0$（99.2% acc）に近い $21.06\alpha E+45.13\lambda-54.45=0$ を auto モードで取得（99.0%）。MAAM では人と KAN の対話的「step 2/3/4A/4B」で精度と簡潔さを段階的にトレードオフ。
- **貢献**: (1) 元の 2 層 KA 表現を任意 depth/width に一般化した KAN アーキ、(2) スケーリング指数 $\alpha=k+1$ を保証する近似定理 KAT と経験的サチュレーション、(3) sparsification + pruning + symbolic snap という解釈性パイプライン、(4) PDE・continual learning・科学発見（結び目理論・Anderson 局在）での実証、(5) pykan として実装公開 (`pip install pykan`)。

## Takeaway（自分にとっての要点）

- **「重み＝学習可能な 1D 関数」**という見方は単に表現を変えただけに見えるが、(i) 内部 dof (grid) と外部 dof (network shape) を分離できる、(ii) 個々の活性関数を symbolic にスナップできる、という二つの帰結が大きい。MLP の「線形 + 固定非線形」を「学習可能 1D 関数」に統合した、と読むのが正しい。
- スケーリング指数 $\alpha=k+1=4$ が次元 $d$ に依存しない、というのが KAT の核。**MLP の $\alpha=(k+1)/d$（Sharma–Kaplan）を本質的に超える**根拠なので、データに compositional sparsity がある時の理論的優位は明確。逆に言うと、その仮定が成り立たないタスクでは効かない可能性がある（著者も認めている）。
- Grid extension は実用上強い: 粗いグリッドで安く学習 → 細かくして refine、をネット全体を作り直さず on the fly でできる。MLP の neural scaling laws は「別サイズで再学習」前提なので、ここは設計勝ち。
- PDE 例の「$10^2$ params で $10^{-7}$ MSE」が一番衝撃的な数字。symbolic な真解を当てに行く設定なので過大評価される可能性は著者自身も指摘しているが、PINN 文脈で KAN を使う動機としては十分。
- 「**$\phi_{l,j,i}(x)$ の透明度を $\tanh(\beta A_{l,i,j})$ にして描く**」だけで Saliency / attribution が要らない、というのは interpretability の UX 設計として強い。Knot で feature attribution なしに $\mu_r$ 依存を読み取れたのはこの可視化のおかげ。
- KAN が見つけた相対論的速度合成の 2 層解は rapidity trick（$\tanh(\text{arctanh}\,u+\text{arctanh}\,v)$）に等しい、という解釈は、「KAN で `arctanh` が出現していなくても rapidity 構造を見つけ得る」ことを示している。**KAN は人間が知る構造とは違う最小表現を返しうる**。
- Continual learning の話は local basis の自然な帰結だが、高次元で「locality」をどう定義するかは未解決、と著者自身が note している。応用上はここを越えないと NLP/CV に来ない。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 理論（KAT, $\alpha=k+1$）と実験（toy / special function / PDE / 科学発見）が一貫しており、主張の射程が明確。
  - 「activation on edge × 学習可能 spline」という設計が、解釈性ツール群（pruning, fix_symbolic, suggest_symbolic, symbolic_formula）と素直に噛み合っている。可視化だけで attribution と等価以上の情報が出る、という UX 主張が強い。
  - Knot theory で DeepMind の $\sim 3\times 10^5$ params MLP を 200 params の KAN が上回ったのは見栄えする結果。Anderson 局在の MAAM で人間と「step 2 → 3 → 4A/4B」と対話的に式を単純化していくフローも、symbolic regression にはない強みとして提示されている。
- **弱み / 疑問**:
  - 著者自身が認める通り **KAN は MLP の約 10 倍遅い**（"we should be honest that we did not try hard to optimize KANs' efficiency"）。これは「異なる活性関数がバッチ計算を共有できない」構造的問題に由来し、エンジニアリングだけで埋まるかは未確定。
  - 評価は **small-scale AI + Science に限定**。language modeling・大規模分類など普通の ML タスクでの evidence は無く、"kansformers" は将来課題のまま。
  - KAT は「$\Phi_{l,i,j}$ が $(k+1)$ 回連続微分可能な KAN 表現が存在する」前提の上限。**現実のデータについてその表現が存在するか、また学習で見つかるかの保証はない**（著者も "Do KA representations exist for general tasks?" と問う）。
  - Toy のスケーリング実験で $\alpha=4$ に近づけるのに RMSE の **平均ではなく中央値**を取らないと $\alpha\approx 3$ にしかならない、というのはやや恣意的。境界効果と言い切れているかは怪しい。
  - Feynman dataset では「KAN と MLP は平均ほぼ互角」と自分で書いている。「方程式の依存が滑らか・単調すぎて KAN の優位が出ない」と説明されているが、これは **科学タスクの大半が KAN にとって有利とは限らない** ことの傍証でもある。
  - Knot theory で signature が DeepMind は $\mu_i$ 主、KAN は $\mu_r$ 主、と結果が分かれる点を "algorithmic choice" で済ませているのは弱い。本当に同等の発見なのか、別変数の最適解なのか踏み込んでいない。
  - 著者自身の limitation:
    - mathematical foundation は $[n,2n+1,1]$ にしか厳密対応していない（一般の深さの KA 定理は無い）。
    - locality を高次元でどう定義するかは未解決（continual learning の一般化が不明）。
    - 解釈性は人間側に KAN の「言語」に対する慣れを要求する、と書かれている（Feynman KAN は "cute but not always interpretable"）。
    - interpretability hyperparameters の appendix では、random seed や $\lambda$ や $G$ により pruned network の大きさ・解釈しやすさが変わる、とされている。
    - 解釈性とは別に "trade-off knob" の意味で精度と単純さの行き先がデータ依存（MAAM の Step 4A vs 4B）。
- **次に試したいこと**:
  - **同じ params / 同じ wall-clock 予算**での KAN vs MLP の Pareto curve（特に PDE と Feynman）。10x slow を flat に揃えた時に PDE の 100x 精度差がどう変わるか。
  - spline を **radial basis / Chebyshev / Fourier basis** に置き換えた版（KAN 自身が discussion で提案）の比較。oscillatory ターゲットには Fourier、滑らか低次元には Chebyshev が良さそうだが裏付けは無い。
  - "kansformer": Transformer の FFN を KAN に置換した小規模 LM。スケーリング指数が NLP では出るのか（compositional sparsity が無さそうなので $\alpha=4$ は期待しない）。
  - Knot dataset で KAN が $\mu_r$ を選ぶ理由を Hessian/勾配で分析し、DeepMind との差異がアーキの帰納バイアスかデータ依存かを切り分け。
  - "multi-head KAN"（discussion 中の案: 活性関数をグループ化して同一にしバッチ可能にする）の実装と速度・精度のトレードオフ測定。

## Notes / Quotes

- "KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline."（abstract）
- "$\alpha=k+1$ ... We choose $k=3$ cubic splines so $\alpha=4$ which is the largest and best scaling exponent compared to other works."（§2.3）
- "A 2-Layer width-10 KAN is 100 times more accurate than a 4-Layer width-100 MLP ($10^{-7}$ vs $10^{-5}$ MSE) and 100 times more parameter efficient ($10^2$ vs $10^4$ parameters)."（§3.4 PDE）
- "KANs are usually 10x slower than MLPs, given the same number of parameters. We should be honest that we did not try hard to optimize KANs' efficiency though"（§6 final takeaway）
- Theorem (KAT): "$\|f-(\mathbf\Phi^G_{L-1}\circ\cdots\circ\mathbf\Phi^G_0)\mathbf x\|_{C^m}\le CG^{-k-1+m}$"（§2.3）
- Implementation: residual $\phi(x)=w_b\,{\rm silu}(x)+w_s\sum_i c_i B_i(x)$、初期化は $w_s=1, c_i\sim\mathcal{N}(0,0.1^2)$、$w_b$ は Xavier。
- Sparsification: $\ell_{\rm total}=\ell_{\rm pred}+\lambda(\mu_1\sum_l|\mathbf\Phi_l|_1+\mu_2\sum_l S(\mathbf\Phi_l))$、デフォルト $\mu_1=\mu_2=1$。
- Pruning threshold $\theta=10^{-2}$（node-level、incoming/outgoing 双方が必要）。
- Knot: $[17,1,14]$ KAN ($G=3,k=3$) で約 $2\times 10^2$ params, 81.6% acc / DeepMind 4 層 width-300 MLP ($3\times 10^5$ params) 78.0% acc。発表後、Shi Lab が 60 params MLP で 80% 達成 ＝ "AI + Science tasks may not be that computationally demanding"（Table tab:math-compare のキャプション内に明記）。
- 既知の限界（discussion / appendix）: 数学的基礎が depth-2 対応の KA 定理にしか厳密対応していない / 異なる活性関数がバッチ計算できないため遅い / locality の高次元定義不明 / Feynman では MLPs and KANs behave comparably on average / interpretability が random seed や $\lambda$ や $G$ に依存しうる。
- (verified 2026-05-20) Feynman dataset 「26 方程式」→「27 方程式」に修正（Table tab:feynman_kan_shape を実際に数えると 27 行、kan.tex §3.3 末尾でも "all 54 pruned KANs"=27×2 と整合）。同箇所で人手/auto の shape [2,2,2,2,2,1]→[2,2,1] を明示。根拠: kan.tex Table tab:feynman_kan_shape 行 I.16.6、本文「resulting a total of 5 layers. However, the auto-discovered KANs are only 2 layers deep!」。
- (verified 2026-05-20) Related Papers の Lai & Shen 2021 の引用タイトルを bbl の正式タイトル「The Kolmogorov superposition theorem can break the curse of dimensionality when approximating high dimensional functions」に修正。根拠: kan.bbl の \bibitem{lai2021kolmogorov}。
- (verified 2026-05-20) Related Papers の "He & Xu 2018/2023" を bbl の著者名に合わせ「He, Li, Xu, Zheng 2018 / He & Xu 2023」に修正。根拠: kan.bbl の \bibitem{he2018relu} と \bibitem{he2023deep}。
- (verified 2026-05-27) venue/year を TeX で確認できる範囲（neurips_2023.sty preprint 使用、明示 year なし）に限定し、KAT の次元非依存は「収束率」に修正して定数 $C$ の caveat を追加 (kan.tex title block, Theorem KAT paragraph)
- (verified 2026-05-27) Knot supervised task を「18 不変量」から TeX 本文通り「17 knot invariants → signature」に修正し、feature attribution 不要という表現に合わせた (kan.tex §Application to Mathematics: Knot Theory, Table tab:math-compare)
- (verified 2026-05-27) pruning 最適性に関する TeX 本文で確認しにくい断定を削り、appendix の hyperparameter/random seed 依存に置換 (kan.tex Appendix Dependence on hyperparameters)

## Related Papers

- Kolmogorov 1957 / Arnold — 元定理。
- Poggio et al. 2020, "theoretical issues in deep networks" — KA 表現の "pathological" 問題と compositional sparsity の話。
- Sharma & Kaplan 2020 — Neural scaling law $\alpha=(k+1)/d$（MLP の bound）。
- Michaud et al. 2023 "Precision Machine Learning" — $\alpha=(k+1)/d^*$, $d^*=2$（最大 arity）、MLP は $\alpha=1$ あたりでサチる、と報告。本論文の比較対象。
- Lai & Shen 2021 — "The Kolmogorov superposition theorem can break the curse of dimensionality when approximating high dimensional functions"。depth-2 で COD を回避する近似理論を経験的に検証。
- Davies et al. 2021 *Nature* "Advancing mathematics by guiding human intuition with AI" — Knot signature の DeepMind 研究。本論文 §3.5 のベースライン。
- Ganeshan et al. 2015 / Wang et al. 2020 / Biddle et al. 2010 — GAAM / Mosaic / MAAM の閉形式 mobility edge。
- Raissi et al. 2019 PINN, Karniadakis 2021 — PDE 比較対象。
- Fakhoury et al. 2022 ExSpliNet — B-spline / learnable activation 関連の先行研究。
- He, Li, Xu, Zheng 2018 (ReLU と finite elements) / He & Xu 2023 (deep neural networks and finite elements) — ReLU-k と spline、MLP と spline の橋渡し。
