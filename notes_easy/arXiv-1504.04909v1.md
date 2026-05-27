# Illuminating search spaces by mapping elites（illumination algorithm と MAP-Elites の提案）

- arXiv: https://arxiv.org/abs/1504.04909
- 一次ソース: ../papers/arXiv-1504.04909v1/
- 正規ノート: ../notes/arXiv-1504.04909v1.md

---

## 一言で言うと

この論文は、探索アルゴリズムの目的を「単一の最高性能解を返す」ことから、「ユーザが選んだ feature space の各領域で最高性能の解を返す」ことへ拡張する **illumination algorithm** という枠組みを導入し、その具体例として **Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)** を提案する。著者は、neural networks、simulated soft robot morphologies、real soft robot arm の 3 領域で、MAP-Elites が特徴空間全体の fitness potential を可視化し、多様な高性能解を返し、場合によっては単一最良解の探索でも baseline を上回ると主張している。

## 何を議論する論文か

- **問題設定**: 多くの search / optimization algorithms は、search space の中から単一または少数の高性能解を探す。一方で研究者や設計者は、たとえば「ロボットの高さと重さの各組合せで、最速の設計は何か」のように、ユーザが選んだ特徴軸ごとの性能分布を知りたいことがある。本論文はこの「feature space の各点での最高性能解を探す」問題を扱う。
- **対象範囲 / 仮定**: 対象は主に black-box optimization 問題で、解 $x$ の performance $f(x)$ と feature descriptor $b(x)$ は評価できるが、性能関数の勾配や解析式には依存しない。feature space はユーザが選び、計算資源に応じて離散化する。search space は高次元または無限次元でもよいが、feature space は低次元で可視化・保存できることが前提である。
- **既存研究との差分**: Novelty Search、Novelty Search + Local Competition (NS+LC)、Multi-Objective Landscape Exploration (MOLE) は多様性や repertoire を扱う先行手法だが、著者はそれらをより複雑で、feature space の各 cell の最良解を直接保持する設計ではないと位置づける。MAP-Elites は各 cell に elite を 1 個だけ保存し、archive と population を分けない。
- **この論文で答えたい問い**: MAP-Elites は、feature space 全体の性能地図を作り、各 cell の高性能解を保ちながら、多様な解と単一の高性能解の両方を見つけられるか。さらに、NS+LC、EA、EA+Diversity、random sampling、grid search などの比較対象より、global reliability、precision、coverage、global performance の面で有利かを検証する。

## 背景と前提

- **search space と feature space の区別**: search space は genome / genotype $x$ が属する空間で、実際の探索はこの空間で行われる。feature space は、ユーザが関心を持つ $N$ 個の variation dimensions で定義される低次元空間で、MAP-Elites はそこを cell に離散化する。TeX の Details 節は、robot morphology の例で「直接 feature space を探索することはできない」と明記している。
- **black-box optimization**: 物理系や複雑なシミュレーションでは、性能を評価することはできても勾配を計算できない。論文は hill climbing、simulated annealing、evolutionary algorithms、gradient ascent/descent、Bayesian optimization、multi-objective optimization を通常の search / optimization algorithms として挙げ、MAP-Elites はそのうち特に evolutionary computation の語彙で説明される。
- **evolutionary computation の用語**: solution は organism / phenotype / individual、記述は genome / genotype、性能は fitness、評価関数は fitness function と呼ばれる。mutation は genome をランダムに変える操作、crossover は 2 つの parent descriptor から子を作る操作である。
- **deceptive problem と diversity**: TeX は、local optima がある black-box optimization では「良い解の近くを変えればさらに良い解に近づく」という heuristic が deceptive problem で破綻すると説明する。Novelty Search は performance ではなく behavior / feature space 上の novelty を選択圧にする手法として紹介され、NS+LC や MOLE は MAP-Elites の前身にあたる illumination algorithms として扱われる。
- **この論文での illumination algorithm**: Optimization vs. Illumination Algorithms 節では、illumination algorithms を「feature space の各点で最高性能の解を返す」アルゴリズムと定義する。生物学の言葉では phenotype-fitness map を照らす手法であり、任意の illumination algorithm は optimization algorithm としても使えるため、optimization algorithms の上位集合だと著者は述べる。

## 提案手法

### コアアイデア

MAP-Elites は、ユーザが選んだ $N$ 次元の feature space を cell に離散化し、各 cell に「その cell に入る候補のうち、これまでに見つかった最高性能の解」を 1 つだけ保存する。TeX の Fig. `mapElitesPseudocodeSimpleVersion` では、性能値を保存する map を $\mathcal{P}$、対応する解を保存する map を $\mathcal{X}$ として表す。

手順は単純である。最初に $G$ 個の random solutions を生成し、各候補の feature descriptor と performance を測る。同じ cell に複数の候補が入る場合は最高性能のものだけを残す。その後は、既に埋まっている archive / map から elite を 1 つランダムに選び、mutation や crossover によって offspring を作り、offspring の feature descriptor $\mathbf{b'}$ と performance $p'$ を評価する。対応 cell が空なら保存し、既存 occupant より $p'$ が高ければ置換する。

この設計の重要点は、archive と population を分けないことである。NS+LC では current population と archive を持ち、feature distance や nearest neighbors に基づく動的な選択圧がある。MAP-Elites は「map に保存された elite 群そのもの」を探索母集団として使い、既存 archive から一様に sampling して offspring を作るため、著者は selection pressure がより simple, intuitive, predictable だと主張する。

### 重要な定義・数式

$$
\mathbf{b'} \leftarrow \mathrm{feature\_descriptor}(\mathbf{x'}), \qquad
p' \leftarrow \mathrm{performance}(\mathbf{x'})
$$

**式の意味**: Fig. `mapElitesPseudocodeSimpleVersion` の評価ステップをまとめた式である。候補解 $\mathbf{x'}$ をシミュレーションまたは実機で評価し、feature space 上の位置 $\mathbf{b'}$ と性能 $p'$ を得る。

**記号の定義**:
- $\mathbf{x'}$ ... 新しく生成された candidate solution / offspring
- $\mathbf{b'}$ ... $\mathbf{x'}$ の feature descriptor。$N$ 次元 feature space 内の cell を決める
- $p'$ ... $\mathbf{x'}$ の performance / fitness
- $\mathrm{feature\_descriptor}$ ... TeX の pseudocode に出てくる、候補解の特徴を記録する関数
- $\mathrm{performance}$ ... TeX の pseudocode に出てくる、候補解の性能を記録する関数

**この論文での役割**: MAP-Elites は search space 内で候補を作るが、保存と比較は feature descriptor によって決まる cell ごとに行う。この式は、search space から feature-performance map へ候補を投影する入口である。

$$
\text{if } \mathcal{P}(\mathbf{b'}) = \emptyset \text{ or } \mathcal{P}(\mathbf{b'}) < p'
\text{ then } \mathcal{P}(\mathbf{b'}) \leftarrow p', \quad
\mathcal{X}(\mathbf{b'}) \leftarrow \mathbf{x'}
$$

**式の意味**: 候補解を archive に採用するかどうかを決める MAP-Elites の中心ルールである。対応 cell が空なら候補を保存し、既に elite がいる場合は候補の性能が高いときだけ置換する。

**記号の定義**:
- $\mathcal{P}$ ... feature space の各 cell に保存された performance の map
- $\mathcal{X}$ ... feature space の各 cell に保存された solution の map
- $\mathcal{P}(\mathbf{b'})$ ... $\mathbf{b'}$ に対応する cell の現 occupant の performance
- $\mathcal{X}(\mathbf{b'})$ ... $\mathbf{b'}$ に対応する cell の現 occupant の solution
- $\emptyset$ ... その cell がまだ空であること

**この論文での役割**: 各 cell に elite を 1 つだけ保存するための更新式であり、MAP-Elites が global な performance competition ではなく local cell competition を行うことを表す。MOLE との違いとして、低・中性能領域の cell での小さな改善も保存される点が重要になる。

$$
M_{x,y} = \max_{i \in [1, \cdots, k]} m_i(x, y)
$$

**式の意味**: Methods の Global reliability 節で定義される、cell $(x,y)$ ごとの best known performance である。全 treatment・全 run の final map $\mathcal{M}=m_1,\cdots,m_k$ の中で、その cell における最大性能を $M_{x,y}$ とする。

**記号の定義**:
- $M_{x,y}$ ... coordinates $x,y$ において、全 run・全 algorithm が見つけた最高性能
- $\mathcal{M}=m_1,\cdots,m_k$ ... 全 treatment の全 run から得られた final map の集合
- $m_i(x,y)$ ... $i$ 番目の final map が cell $(x,y)$ に持つ performance
- $k$ ... final map の総数

**この論文での役割**: 真の cell-wise optimum は未知なので、評価指標では「全実験で観測された最高性能」を近似的な分母として使う。Global reliability と precision はどちらも $M_{x,y}$ によって正規化される。

$$
G(m) = \frac{1}{n(M)} \sum_{x,y} \frac{m(x, y)}{M(x,y)}
$$

**式の意味**: map $m$ の **Global reliability** を定義する式である。各 cell で、その run が見つけた性能を best known performance で割り、全ての fillable cells にわたって平均する。ある run がその cell を埋めていない場合、TeX はその cell の performance を 0 として扱う。

**記号の定義**:
- $G(m)$ ... map $m$ の global reliability
- $m(x,y)$ ... 評価対象 map $m$ の cell $(x,y)$ の performance
- $M(x,y)$ ... cell $(x,y)$ の best known performance
- $n(M)$ ... $M$ の non-zero entries 数、すなわちどれかの run が埋めた unique cells の数
- $x,y$ ... TeX では 2D feature map の座標。任意次元へ一般化可能と Methods に書かれている

**この論文での役割**: Illumination algorithm の中心指標である。空 cell を 0 とするため、性能が高い少数 cell だけを返す optimization algorithm はこの指標で不利になり、feature space 全体を高性能に埋める能力が測られる。

$$
P(m) = \frac{1}{n(m)} \sum_{x,y} \frac{m(x, y)}{M(x,y)}
$$

**式の意味**: map $m$ の **Precision (opt-in reliability)** を定義する式である。Global reliability と違い、平均はその run が実際に埋めた cell のみで取る。TeX では和の範囲を $\textrm{filled}_{m}(x,y)=1$ の cell に制限している。

**記号の定義**:
- $P(m)$ ... map $m$ の precision / opt-in reliability
- $n(m)$ ... map $m$ で埋まった cell の数。TeX の式の分母は $n(m)$
- $\textrm{filled}_{m}(x,y)$ ... algorithm が cell $(x,y)$ に解を生成したとき 1、そうでなければ 0 の binary matrix
- $m(x,y)$, $M(x,y)$ ... Global reliability と同じく、評価対象 map の性能と best known performance

**この論文での役割**: 「ある algorithm が解を返した cell に限れば、その解はどれだけ良いか」を測る。soft robot 実験で MAP-Elites が EA / EA+D より precision で有意に悪いという結果は、多数 cell に評価予算を分散することのトレードオフを示す。

### 実装 / アルゴリズム上の要点

1. ユーザが performance measure $f(x)$ と、$N$ 個の feature dimensions を選ぶ。
2. feature space を離散化する。粒度は user preference または available computational resources によって決まる。
3. 空の $N$ 次元 map $(\mathcal{P}, \mathcal{X})$ を作る。
4. 初期化として $G$ 個の random genomes を生成し、feature と performance を評価し、対応 cell に置く。同じ cell に複数入る場合は最高性能のものだけを残す。
5. 以後、map 内の elite をランダムに選び、mutation and/or crossover で offspring を作る。
6. offspring の $\mathbf{b'}$ と $p'$ を評価し、対応 cell が空または既存 occupant より高性能なら $\mathcal{P}$ と $\mathcal{X}$ を更新する。
7. termination criterion は、時間、計算資源、map cells の一定割合、average fitness、問題解の個数などにできる。
8. 本論文では hierarchical version と batched parallelized version を使う。Methods は「larger cells から始め、探索中に subdivide する」「slave nodes に評価 batch を送る」と説明し、全実験がこの hierarchical, paralleled version で行われたと明記している。

## 実験・結果

- **データセット / ベンチマーク**: 実験は 3 つの search spaces で行われる。(1) 8-pixel retina の左右両側に object があるかを答える neural network domain、(2) Voxcad simulator 上の 10 x 10 x 10 voxel multi-material soft robots、(3) dynamixel AX-18 servos 3 個と flexible tubes からなる real soft robot arm である。
- **比較対象 / baseline**: Retina では traditional single-objective EA、Novelty Search + Local Competition (NS+LC)、random sampling と比較する。MOLE は同じ domain / feature space で過去に使われたが、本ドラフトでは解像度が異なるため fair comparison は未実施で、30 runs を merge した MOLE 図との anecdotal comparison のみである。Simulated soft robots では NSGA-II 実装の traditional EA と EA+Diversity (EA+D) と比較する。Real soft robot arm では random sampling と traditional grid search と比較する。
- **指標**: 論文は Global Performance、Global reliability、Precision (opt-in reliability)、Coverage の 4 指標を使う。全 $p$ 値の検定は Methods の Statistics 節にある通り two-tailed Mann-Whitney U test である。Coverage は、理論的に fillable な cell 数が未知なため、全 run・全 treatment のいずれかで埋まった unique cells 数で近似する。
- **主な結果**: Retina では map resolution 512 x 512、feature 1 は connection cost、feature 2 は network modularity、performance は 256 input patterns に対する percent correct answers である。20 independent runs を行い、MAP-Elites は Global Performance、Global reliability、Precision、Coverage の全てで 3 control algorithms より有意に高いと報告される ($p < 1 \times 10^{-7}$, Fig. `fig:retina_results`)。本文は MAP-Elites を "10,000 evaluations"、Methods は "number of iterations: 10,000" と記しており、hierarchical schedule は 16 x 16 から始まり、iteration 0 で 64 x 64、1250 で 128 x 128、2500 で 256 x 256、5000 で 512 x 512 へ変わる。
- **主な結果**: Simulated soft robots では、各 voxel は empty か、4 種の material、すなわち bone (dark blue, stiff)、soft support tissue (light blue, deformable)、in-phase muscle (green, cyclical volumetric actuation of 20%)、opposite-phase muscle (red, counter-cyclical volumetric actuation of 20%) のいずれかとして扱われる。feature 1 は percentage of bones、feature 2 は percentage of voxels filled、performance は 10 simulated seconds の covered distance、map は 128 x 128 である。10 runs を開始したが完了分のみを使い、EA=7、EA+D=5、MAP-Elites=8 runs で解析する。MAP-Elites は global reliability と coverage で有意に良い ($p<0.002$)。Global performance は median が高いが有意差なし ($p>0.05$)。Precision は MAP-Elites が有意に悪い ($p<0.01$)。
- **主な結果**: Real soft robot arm では、solution は 3 関節角度を指定する 3 numbers で、各 servo は -150 から +150 steps の範囲で動く。feature space は camera image 上の arm endpoint の $x$-value を 64 cells に離散化した 1D 空間で、performance は endpoint の $y$-value の最大化である。本文は MAP-Elites と random sampling に 640 evaluations、grid search に 729 evaluations ($9\times9\times9$) と書く一方、Methods の Experimental parameters は MAP-Elites / random sampling を 420 evaluations、grid search を 512 evaluations / 8 steps とする。結果として、高い $x$ values では全手法がほぼ境界を見つけ、中間 $x$ values (approximately 400-600) では MAP-Elites が grid search と random sampling を上回り、低 $x$ region (200-500) では grid search の coverage が低いと報告される。ただし TeX は "too preliminary to provide reliable statistical results" と明記する。
- **著者が主張する貢献**: (1) "illumination algorithms" という用語と問題設定を導入したこと、(2) Cully et al. 2015 で brief に使われた MAP-Elites を詳細に記述し性質を調べたこと、(3) feature space 全体の性能地図を作る評価指標を整理したこと、(4) neural networks、simulated soft robots、real soft robot arm で preliminary evidence を示したことである。

## 妥当性と限界

- **この主張を支える根拠**: MAP-Elites の主張は、Fig. `fig:retina_results`、Fig. `fig:retina_paths`、Fig. `fig:softbotsPlots`、Fig. `fig:softbotThumbnails`、Fig. `fig:softArm` と、Methods の 4 指標によって支えられている。特に retina では 20 independent runs で 4 指標すべてに有意差があり、soft robots では reliability / coverage の有意差と precision の悪化が報告される。lineage analysis では、多くの elites は feature space 上の近い parent から生じるが、最終 elite の lineage は長距離の paths を通ることがあり、複数領域を同時探索する利点の定性的根拠として使われている。
- **著者が認めている limitations / future work**: 冒頭の Author's Note は、本稿が preliminary draft であり、全実験は final version までに redo され data are subject to change と明記する。Discussion でも、preliminary nature のため empirical performance について確定的に結論しないでほしいと述べる。Future work として、固定された feature space を超えて新しい cell types を追加できないため open-ended evolution にはならないこと、ever expanding feature space を扱う illumination algorithms が必要なことが書かれている。Alternate variants 節では、cell ごとに複数 genome を保存する、offspring を作る cell selection に bias を入れる、近傍内 crossover を使うなどは future research が必要とされる。
- **読者として注意すべき点**: MOLE との fair comparison は本ドラフトでは未実施で、30 runs を merge した過去図との anecdotal な比較に留まる。Simulated soft robot は run 数が EA=7、EA+D=5、MAP-Elites=8 と不揃いで、未完了 run がある。Real soft arm は本文と Methods で evaluation counts が一致せず、著者自身も reliable statistical results ではないとする。Coverage の分母は真の fillable cell 数ではなく、観測された union cells による近似である。
- **追加で確認したい実験 / 疑問**: TeX 中の TODO にも、MOLE や plain Novelty Search との比較、hierarchical version の効果を criteria ごとに報告すること、NS+LC や MOLE 以外の performance + behavioral diversity MOO との比較が残っている。読者としては、同一評価予算での MAP-Elites / NS+LC / MOLE / EA+D の learning curves、hierarchical schedule の ablation、feature dimensions の選び方に対する感度、precision が追加評価で本当に追いつくかを確認したい。ただし、これらは TeX から導かれる確認課題であり、論文内で実証済みではない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **MAP-Elites**: Multi-dimensional Archive of Phenotypic Elites。feature space の各 cell に、その cell で見つかった最高性能の solution と performance を保存する illumination algorithm。
- **illumination algorithm**: feature space の各点で highest-performing solution を返し、各 region の fitness potential を照らすアルゴリズム。著者は optimization algorithms の superset とする。
- **optimization algorithm**: search space で最高性能の solution、または Pareto front 上の少数 solution を返すことを主目的とするアルゴリズム。本論文では illumination と対比される。
- **feature space / behavior space**: ユーザが選ぶ $N$ 個の variation dimensions からなる低次元空間。MAP-Elites の map はこの空間を離散化した cell 群である。
- **search space**: genome / genotype $x$ の全候補が属する空間。MAP-Elites は feature space ではなく search space で候補を生成する。
- **cell**: 離散化された feature space の 1 区画。MAP-Elites では各 cell に elite を最大 1 つ保存する。
- **elite**: ある cell において、これまでに見つかった最高 performance の solution。
- **genome / genotype**: candidate solution の記述 $x$。soft robot 実験では CPPN network が genome として扱われる。
- **phenotype**: genome から生成・実体化された形や振る舞い。robot morphology や neural network の実際の構造・挙動が該当する。
- **fitness / performance**: solution の品質を表す値。retina では percent answers correct、soft robots では covered distance、real arm では endpoint の $y$ coordinate。
- **NS+LC**: Novelty Search + Local Competition。novelty objective と local competition を使う先行 illumination algorithm。本論文では nearest neighbor 計算や archive / population の複雑さが MAP-Elites との差分として説明される。
- **MOLE**: Multi-Objective Landscape Exploration。performance と feature space 上の diversity を目的にする先行 illumination algorithm。著者は global performance competition のため、中・低性能領域の cell improvement を保持しにくいと論じる。
- **Global Performance**: run 内で見つかった single highest-performing solution を、domain 内の最高可能性能または観測された最高性能で正規化した指標。
- **Global reliability**: 全 fillable cells に対し、cell ごとの best known performance にどれだけ近い解を返せたかを平均する指標。空 cell は 0 として扱われる。
- **Precision (opt-in reliability)**: algorithm が実際に埋めた cell だけを対象に、cell-wise best known performance に対する比を平均する指標。
- **Coverage**: map 内で埋まった cells 数を、理論的に fillable な cell 数の近似で割る指標。本論文では全 treatment / run の union filled cells を分母の近似にする。
- **CPPN**: Compositional Pattern-Producing Network。soft robot 実験で voxel の有無と material type を決める indirect / generative encoding。入力は voxel の Cartesian coordinates $(x,y,z)$ と center からの距離 $d$。
- **NEAT**: soft robot 実験では CPPN network 自体を進化させる原理として使われる algorithm。Sferes$_{v2}$ 実装では crossover や speciation による genetic diversity を含まないと TeX に書かれている。

## 読む順番の提案

- まず Abstract と Author's Note を読む。ここで、論文の主張が大きい一方で、preliminary draft で data are subject to change だという前提を押さえる。正規ノートの Summary 冒頭と Critical Thoughts の注意点につながる。
- 次に Background and Motivation の後半、Optimization vs. Illumination Algorithms、Fig. `conceptFig` を読む。optimization と illumination の違い、feature space をユーザが選ぶという立場、MAP-Elites が NS+LC / MOLE に続く手法であることが分かる。正規ノートの Takeaway「feature space は探索空間ではなく写像先」に対応する。
- その後、Fig. `mapElitesPseudocodeSimpleVersion` と Details of the MAP-Elites algorithm を読む。$\mathcal{P}$、$\mathcal{X}$、$\mathbf{b'}$、$p'$、cell replacement rule が実装の核である。
- Criteria for Measuring the Algorithms と Methods の Quantifiable measurements を続けて読む。Global Performance、Global reliability、Precision、Coverage の違いを理解してから実験図を見ると、結果の意味が読みやすい。TeX には table 環境はなく、結果は主に figures と captions に埋め込まれている。
- 実験は retina、simulated soft robots、real soft robot arm の順に読む。Fig. `fig:retina_results` と `fig:retina_paths` は「単一最良解でも勝つことがある」主張と lineage / stepping stones の根拠、Fig. `fig:softbotsPlots` と `fig:softbotThumbnails` は「feature space を照らす」主張の見本、Fig. `fig:softArm` は実機での preliminary evidence である。
- 最後に Discussion and Conclusion、Alternate variants、`mapElitesNoComments.bbl` を確認する。Discussion は著者自身の限界認識と future work、`mapElitesNoComments.bbl` は Cully et al. 2015、Lehman and Stanley 2011、Clune et al. 2013、Cheney et al. 2013、Mouret and Doncieux 2010/2012 などの正式タイトル確認に役立つ。正規ノートの Related Papers と Critical Thoughts はこの段階で照合するとよい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1504.04909v1/`
- 正規ノート: `notes/arXiv-1504.04909v1.md`
