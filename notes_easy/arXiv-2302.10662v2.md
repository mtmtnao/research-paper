# Snakes and Ladders: a Treewidth Story（長い ladder を短くしても treewidth を保つ tight な簡約定数を与えるグラフ理論論文）

- arXiv: https://arxiv.org/abs/2302.10662
- 一次ソース: ../papers/arXiv-2302.10662v2/
- 正規ノート: ../notes/arXiv-2302.10662v2.md

---

## 一言で言うと

グラフ $G$ に含まれる ladder、すなわち $2 \times (k+1)$ grid graph が 4 つの cornerpoints だけで外部と接続する誘導部分グラフ、を長さ 4 まで短くしても treewidth は変わらないことを示す論文である。さらにこの定数 4 は一般グラフでは tight であり、display graph では common chain を 4 leaf labels まで縮めても treewidth を保つことを示して、phylogenetics の未解決問題に答える。

## 何を議論する論文か

- **問題設定**: 長い ladder を短くする、または短い ladder を長くしても、グラフの treewidth $tw(G)$ が保存される条件を調べる。TeX の abstract は ladder を "$2 \times (k+1)$ grid graph" で、かつ "only connected to the rest of $G$ via its four cornerpoints" と定義している。
- **対象範囲 / 仮定**: 主結果は任意の連結な無向グラフ $G$ を対象にする。display graph への応用では、同じ taxa 集合 $X$ 上の 2 本の unrooted binary phylogenetic trees $T_1,T_2$ を考え、$|X|\geq 4$ かつ $T_1 \neq T_2$ を仮定する。
- **既存研究との差分**: Kelk et al. の Theorem `thm:unbounded` は、$tw(G)=k$ に依存する値 $f(k)$ 以上の ladder なら任意に長くしても treewidth が変わらないことを与えていた。本論文は、$k$ に依存しない普遍定数 4 を与え、さらに tightness を示す。
- **この論文で答えたい問い**: 一般グラフでは ladder をどこまで短くしても treewidth を保てるか。phylogenetics では、common chain reduction rule が display graph の treewidth を保存する定数長を持つか。

## 背景と前提

- **tree decomposition / treewidth**: グラフの頂点を bag に入れ、その bag を木 $\mathbb{T}$ 上に配置する表現である。各頂点を含む bag は $\mathbb{T}$ 上で連結に現れなければならず、これを running intersection property と呼ぶ。最大 bag サイズから 1 を引いた幅を最小化した値が $tw(G)$ である。
- **minor と treewidth**: TeX では、ladder of length $k$ は ladder of length $(k+1)$ の minor であり、treewidth は minor を取る操作で増えない、と述べている。したがって ladder を短くする操作だけなら treewidth は増えない。難しいのは「短くしても treewidth が下がらない」ことを、逆向きに ladder を長くする tree decomposition の構成で示す部分である。
- **disconnecting ladder**: ladder 内の square $\{u,v,w,x\}$ の水平辺 $\{u,w\}$ と $\{v,x\}$ がグラフ全体の edge cut になる場合を disconnecting と呼ぶ。Lemma `lem:disconnecting` により、この場合は ladder を任意に長くしても treewidth は増えない。
- **display graph と common chain**: 2 本の phylogenetic trees $T_1,T_2$ で同じ leaf label を同一視したグラフが $D(T_1,T_2)$ である。common chain は、両方の木で同じ順序に現れる leaf labels の列で、display graph では ladder-like structure を作る。chain の $k$ leaves は display graph 内で $k-1$ squares の ladder を誘導する。
- **先行研究や baseline との関係**: subtree reduction は Kelk et al. により display graph の treewidth-preserving と知られていた。一方、common chain reduction が treewidth-preserving かは未解決だった。protrusion や finite integer index に基づく一般理論は「ある定数」を示す可能性はあるが、本論文の tight bounds には届きにくい、という位置づけである。

## 提案手法

### コアアイデア

著者は treewidth の forbidden minor 的な特徴づけに頼らず、minimum-width tree decomposition $(\mathcal{B},\mathbb{T})$ を直接変形する。基本方針は、長さ 3 または 4 の局所的な ladder を見て、そこに新しい rung $\{u',v'\}$ を挿入しても bag の最大サイズを増やさない tree decomposition を作ることである。これを繰り返せば、ladder を任意の長さまで伸ばせる。短縮後のグラフは長いグラフの minor なので、伸長で treewidth が増えないことが短縮で treewidth が下がらないことを保証する。

証明の中心は Theorem `thm:main` の case analysis である。Case 1 は 1 つの square の 4 頂点を含む bag がある場合、Case 2 は上側列 $\{a,u,w,c\}$ と下側列 $\{b,v,x,d\}$ から 2 頂点ずつ含む bag がある場合を処理する。残りでは、$\{u,w,v\}$ を含む bag $B_1$ と $\{v,w,x\}$ を含む bag $B_2$ を、minimum-width tree decomposition の中で距離が最小になるように選ぶ。ある ladder vertex を含む bag 集合が tree decomposition 上を複雑に伸びる現象を著者は "snakes" と呼び、`reeling in $a$ and $b$` という relabel と bag 挿入の操作で制御する。

### 重要な定義・数式

TeX 中に機械学習論文のような目的関数や更新式はない。中核になるのは、tree decomposition の定義、width/treewidth の定義、ladder の定義、tightness 用の構成である。

$$
\begin{aligned}
&\cup_{i=1}^q B_i = V(G),\\
&\forall e = \{ u,v \} \in E(G), \exists B_i \in \mathcal{B} \mbox{ s.t. } \{u,v\} \subseteq B_i,\\
&\forall v \in V(G), \text{ all the bags } B_i \text{ that contain } v \text{ form a connected subtree of } \mathbb{T}.
\end{aligned}
$$

**式の意味**: tree decomposition の 3 条件 (tw1), (tw2), (tw3) である。全頂点が bag に現れ、全辺がどこかの bag で覆われ、同じ頂点を含む bag は木 $\mathbb{T}$ 上で連結にまとまる必要がある。

**記号の定義**:
- $G=(V,E)$ ... 対象の無向グラフ
- $\mathcal{B}=\{B_1,\dots,B_q\}$ ... bag の multiset
- $\mathbb{T}$ ... bag と一対一対応する node を持つ木
- $B_i$ ... $V(G)$ の部分集合である bag

**この論文での役割**: ladder を長くした後も、これら 3 条件を満たす tree decomposition を構成できるかが証明の本体である。特に (tw3) が、$B_1$ と $B_2$ の間の path $P$ や snakes の議論を支える。

$$
\operatorname{width}(\mathcal{B}, \mathbb{T}) = \max_{i=1}^q |B_i|-1,\qquad
tw(G)=\min_{(\mathcal{B},\mathbb{T})}\operatorname{width}(\mathcal{B}, \mathbb{T}).
$$

**式の意味**: ある tree decomposition の幅は最大 bag サイズから 1 を引いた値であり、treewidth は全 tree decompositions の中でその幅を最小化した値である。TeX では width を $\max_{i=1}^q |B_i|-1$ と定義し、treewidth を "the smallest width among all tree decompositions" と述べている。

**記号の定義**:
- $\operatorname{width}(\mathcal{B}, \mathbb{T})$ ... decomposition の幅
- $|B_i|$ ... bag $B_i$ の頂点数
- $tw(G)$ ... グラフ $G$ の treewidth

**この論文での役割**: Case 1, Case 2 では size-5 bag を導入するため、$tw(G)\geq 4$ なら幅を増やさずに済む。一方、$tw(G)=3$ では size-5 bag が使えない可能性があり、Lemma `lem:main2` と Theorem `thm:main3` の追加議論が必要になる。

$$
L \text{ is a } 2\times(k+1) \text{ grid graph},\qquad
G[V(L)] = L,\qquad
\text{only cornerpoints of } L \text{ can be incident to an edge with an endpoint outside } L.
$$

**式の意味**: TeX の ladder 定義を記法で整理したものである。ladder $L$ は $2\times(k+1)$ grid graph で、$G$ の中で誘導部分グラフとして現れ、外部と接続できるのは degree-2 vertices である 4 つの cornerpoints だけである。

**記号の定義**:
- $L$ ... length $k$ の ladder
- $G[V(L)]$ ... $L$ の頂点集合が $G$ で誘導する部分グラフ
- cornerpoints ... ladder の endpoints、すなわち degree-2 vertices

**この論文での役割**: この境界条件により、ladder 内部の頂点の incident edges を局所的に把握できる。Case analysis では、例えば $u,w,v,x$ などの ladder vertices がどの bag に現れるべきかをこの定義に基づいて制約する。

$$
\begin{aligned}
E_{\text{ladder}}=
\{&\{1,2\}, \{2,3\}, \{4,5\}, \{5,6\},\\
&\{1,4\}, \{2,5\}, \{3,6\}\}.
\end{aligned}
$$

**式の意味**: Lemma `lem:alwaystight` で構成する $G_2(t)$ の length 2 ladder 部分の辺集合である。TeX では vertices $1,2,3,4,5,6$ を導入し、この 7 本の辺で ladder of length 2 を作る。

**記号の定義**:
- $G_2(t)$ ... treewidth exactly $t$ を持つ tightness 用グラフ
- $t$ ... $t\geq 3$ の整数
- $E_{\text{ladder}}$ ... $G_2(t)$ 内の ladder 部分の辺集合

**この論文での役割**: この ladder に $(t-1)$ 頂点の clique $7,8,\ldots,7+t-2$ を接続し、長さ 2 から 3 に伸ばすと treewidth が $t$ から $t+1$ に上がる例を作る。したがって、treewidth が大きければ length 2 まで縮められる、という期待を否定する。

$$
\begin{array}{ll}
\text{bag 1:} & \{1,3,7\} \cup C\\
\text{bag 2:} & \{1,3,5\} \cup C\\
\text{bag 3:} & \{1,4,5\} \cup C\\
\text{bag 4:} & \{3,5,6\} \cup C\\
\text{bag 5:} & \{1,2,3,5\}
\end{array}
\qquad
C=\{8,\ldots,7+t-2\}.
$$

**式の意味**: Lemma `lem:alwaystight` で $G_2(t)$ の treewidth が高々 $t$ であることを示す 5 bag の tree decomposition である。TeX では bag 2 を bag 1 に接続し、bag 3, 4, 5 を bag 2 に接続する。各 bag は高々 $t+1$ 頂点を持つ。

**記号の定義**:
- $C$ ... clique vertices のうち vertex $7$ を除いた集合
- bag 1--5 ... $G_2(t)$ の tree decomposition に使う bag
- $t+1$ ... bag サイズの上限で、width は高々 $t$

**この論文での役割**: 上界側の構成である。下界側は、$G_3(t)$ に対して $|C|+5$ 個の connected subgraphs からなる bramble を作り、minimum hitting set が $t+2$ elements であることから Seymour and Thomas の結果により $tw(G_3(t))\geq t+1$ を得る。

### 実装 / アルゴリズム上の要点

- step1: ladder が disconnecting なら Lemma `lem:disconnecting` を使い、その ladder は任意に長くしても treewidth が増えないとする。
- step2: disconnecting でない length 2 以上の ladder では Observation `obs:k4` により $K_4$ minor を得て、$tw(G)\geq 3$ を確認する。
- step3: Theorem `thm:main` では $tw(G)\geq 4$ を仮定し、長さ 3 の ladder に rung を 1 本挿入する tree decomposition を構成する。Case 1 と Case 2 は size-5 bag を許容できるため処理しやすい。
- step4: Case 3 と Case 4 では、$\{u,w,v\}\subseteq B_1$ と $\{v,w,x\}\subseteq B_2$ を満たす bag の距離を最小化し、path $P$ と running intersection property を使って不可能な配置を排除する。必要に応じて `reeling in $a$ and $b$` で relabel し、新しい path of bags を挿入する。
- step5: $tw(G)=3$ で size-5 bag が使えない問題は、Lemma `lem:main2` では両側に buffer square がある長さ 5 ladder、Theorem `thm:main3` では biconnected component と degree-2 cornerpoint の Lemma `lem:pointed` を使って処理する。
- step6: display graph では、common chain が誘導する ladder の周辺構造を利用する。Theorem `thm:preserve2` では、Fig. `fig:buffercycle` の yellow-highlighted cycle が一般グラフの buffer square と同じ separator-blocking effect を持つため、common chain を 4 leaf labels まで縮められる。

## 実験・結果

- **データセット / ベンチマーク**: TeX 中に empirical dataset や benchmark はない。論文はグラフ理論の定理証明と構成例に基づく。
- **比較対象 / baseline**: 経験的 baseline ではないが、主な比較対象は Kelk et al. の Lemma `lem:disconnecting`, Lemma `lem:protoplusone`, Theorem `thm:unbounded`、および Author's note 2 で言及される Marchand et al. の `almob2023` / `wabi2022` の Theorem 2 である。
- **指標**: 評価される量は treewidth $tw(G)$、tree decomposition の width、ladder length、display graph の treewidth である。tightness では bramble の minimum hitting set size も使う。
- **主な結果**: Theorem `thm:main` は、$tw(G)\geq 4$ かつ ladder length が 3 以上なら ladder を任意に長くしても treewidth が変わらないことを示す。Lemma `lem:main2` は任意の treewidth で ladder length 5 以上なら同じ結論を示す。Theorem `thm:main3` は任意の treewidth で ladder length 4 以上なら同じ結論を示す。
- **主な結果**: Fig. `fig:prism` は、treewidth 3 のグラフで 3 squares の ladder を 1 square 伸ばすと treewidth が 4 に上がる例を与える。したがって Theorem `thm:main3` の定数 4 は一般グラフでは tight である。
- **主な結果**: Lemma `lem:alwaystight` は、任意の $t\geq 3$ に対して、length 2 ladder を含み treewidth exactly $t$ の $G_2(t)$ が存在し、その ladder を 1 つ伸ばして $G_3(t)$ にすると treewidth が exactly $t+1$ になることを示す。
- **主な結果**: Lemma `lem:preserve1` は、subtree reduction と common chain reduction を common chains が 5 leaf labels になるまで適用しても display graph の treewidth が変わらないことを示す。Theorem `thm:preserve2` はこれを 4 leaf labels まで強め、Fig. `fig:chains` により 3 leaf labels への truncation は treewidth-preserving とは限らない、つまり 4 leaf labels が best possible であることを示す。
- **著者が主張する貢献**: Abstract と Introduction によれば、長い ladder は treewidth 計算時に短縮でき、bounded treewidth graphs の minimal forbidden minors は long ladders を含めない。phylogenetics では common chain reduction rule が display graph の treewidth-preserving であることを示す。Author's note 2 では、Theorem `thm:main3` は `almob2023` で別技法により既に証明されていたとし、Theorem `thm:main` は $tw(G)\geq 4$ の場合の "very mild strengthening"、Theorem `thm:preserve2` は display graphs への strengthening、Lemma `lem:alwaystight` は本論文独自の tightness result と位置づけている。

## 妥当性と限界

- **この主張を支える根拠**: 一般グラフの上界は、minimum-width tree decomposition を直接変形し、ladder を 1 square 伸ばしても width が増えないことを case analysis で示す。短縮方向の treewidth 不変性は、短い ladder が長い ladder の minor であることと、逆向きの伸長で treewidth が増えないことを合わせて得る。
- **この主張を支える根拠**: tightness は、Fig. `fig:prism` の具体例と、Lemma `lem:alwaystight` の $G_2(t),G_3(t)$ 構成で示す。後者では、$G_2(t)$ の 5 bag decomposition による上界と、$G_3(t)$ の bramble による下界を組み合わせる。
- **この主張を支える根拠**: display graph 版は、common chain が ladder を誘導すること、$T_1\neq T_2$ なら display graph が $K_4$ minor を含み treewidth at least 3 になること、chain 周辺の restricted structure が buffer cycle を与えることを使う。
- **著者が認めている limitations / future work**: Discussion では、証明は constructive にできるが、distance-minimizing tree decomposition の仮定は proof restart により処理する必要があると述べる。また、各 square に高々 1 chord を持つ ladder には手法が適用できると示唆するが、より複雑な recursive low-pathwidth structures への一般化は future work として残す。
- **著者が認めている limitations / future work**: ladder の検出について、endpoints を guess する trivial $O(n^4)$-time algorithm より改善できるかを open にしている。さらに、3 本以上の phylogenetic trees から作る display graph で common chains がどのような ladder-like structures を誘導し、本結果が拡張できるかも open としている。
- **読者として注意すべき点**: 本論文の「実験・結果」は empirical な精度比較ではなく、定理、反例、bramble lower bound による数学的結果である。したがって baseline や metric を ML 論文のように読むと誤解する。
- **読者として注意すべき点**: Theorem `thm:main3` 自体は Author's note 2 により `almob2023` で先行して証明されていたと明記される。著者は、Theorem `thm:main` を $tw(G)\geq 4$ の場合のごく弱い強化、Theorem `thm:preserve2` を display graph 版の強化、Lemma `lem:alwaystight` を本論文独自の結果として切り分けている。
- **追加で確認したい実験 / 疑問**: TeX 中の future work に沿えば、長い ladder を実際に検出して reduction rule として使う場合、$O(n^4)$ より速い認識アルゴリズムが作れるかが自然な確認点である。また TeX は各 square に高々 1 chord を持つ ladder に proof technique が適用できると述べるが、formal な theorem としては展開していない。3 本以上の tree の display graph で common chains が誘導する ladder-like structures への拡張も、TeX 中では open として扱われている。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **tree decomposition**: bag の multiset $\mathcal{B}$ と、それらを node とする木 $\mathbb{T}$ の組。証明では bag を削除、コピー、relabel、pendant attachment する対象として扱う。
- **running intersection property**: 同じ頂点を含む bag が $\mathbb{T}$ 上で連結な subtree をなすという (tw3) の別名。Case 3, Case 4 で「この頂点がそこにあるなら途中の bag にも現れる」という推論に使う。
- **ladder**: $2\times(k+1)$ grid graph。$G$ 内で induced であり、外部への接続は 4 つの cornerpoints だけに限られる。
- **square**: ladder 内で 4-cycle を誘導する頂点集合。Theorem `thm:main` の図では $\{u,v,w,x\}$ などが square として使われる。
- **cornerpoints**: ladder の endpoints、すなわち degree-2 vertices。外部と接続できる ladder vertices は cornerpoints だけである。
- **disconnecting ladder**: ある square の水平辺 $\{u,w\}$ と $\{v,x\}$ が edge cut になり、グラフを分ける ladder。Lemma `lem:disconnecting` により扱いやすい場合である。
- **distance-minimizing tree decomposition**: $\{u,w,v\}$ を含む bag $B_1$ と $\{v,w,x\}$ を含む bag $B_2$ の距離を、minimum-width decompositions の中で最小にしたもの。Case 4 の矛盾導出に使う。
- **snakes**: ladder vertex を含む bag 集合が tree decomposition 上で病的に伸びる現象を著者が指す語。Introduction では、これを制御する部分が title の由来だと説明される。
- **reeling in**: 例えば $a,b$ について、$p_{ua}$ 上の $u$ を $a$ に、$p_{vb}$ 上の $v$ を $b$ に relabel し、$B_1$ のコピー列を挿入する操作。新しい rung を持つ graph の tree decomposition を作る。
- **display graph**: 2 本の unrooted binary phylogenetic trees $T_1,T_2$ で同じ leaf label を同一視して得るグラフ $D(T_1,T_2)$。
- **common chain**: 両方の phylogenetic trees に同じ順序で現れる leaf labels の列。ただし TeX の脚注では、最初の 2 leaves や最後の 2 leaves の unordered な場合、pendant の場合も含む技術的定義を与えている。
- **subtree reduction / cherry reduction**: 2 つの labels $x,y$ が両方の trees で共通の parent を持つとき leaves $x,y$ を消し、その parent に label $xy$ を割り当てる操作を cherry reduction と呼ぶ。これを exhaustively 適用したものが subtree reduction である。
- **bramble**: $G_3(t)$ の treewidth lower bound を示すための connected subgraphs の族。各 subgraph は互いに touch し、minimum hitting set のサイズが $t+2$ であることから treewidth at least $t+1$ を得る。
- **buffer square / buffer cycle**: $tw(G)=3$ の場合に size-5 bag を避けるため、separator にならないことを保証する周辺構造。一般グラフの Theorem `thm:main3` では extra square、display graph の Theorem `thm:preserve2` では Fig. `fig:buffercycle` の cycle がこの役割を持つ。

## 読む順番の提案

- まず abstract と Introduction を読み、問題が「common chain reduction が display graph treewidth を保つか」から始まり、一般グラフの ladder shortening に拡張されることを確認する。正規ノートでは `Summary（著者の主張）` の問題設定と対応する。
- 次に Preliminaries の tree decomposition 定義、ladder 定義、Lemma `lem:disconnecting`, Observation `obs:k4`, Lemma `lem:plusone`, Theorem `thm:unbounded` を読む。ここが正規ノートの `Related Papers` と `Notes / Quotes` の前提になる。
- Theorem `thm:main` は証明が長いので、最初は Case 1/2 が size-5 bag を作る処理、Case 3/4 が distance-minimizing decomposition と `reeling in` を使う処理、という粒度で読む。正規ノートの `Takeaway` にある "distance-minimizing" と "reeling in the snakes" に対応する。
- その後、Lemma `lem:main2`, Lemma `lem:pointed`, Theorem `thm:main3` を読む。ここで定数が 5 から 4 に改善され、$tw(G)=3$ が bottleneck になる理由が見える。正規ノートの `Critical Thoughts` の「$tw=3$ がボトルネック」というコメントにつながる。
- Tightness 節では Fig. `fig:prism` と Lemma `lem:alwaystight` を見る。特に $G_2(t)$ の clique 接続、5 bag decomposition、$G_3(t)$ の bramble が、正規ノートの `Notes / Quotes` にある tightness メモの根拠である。
- 最後に phylogenetics 節の Lemma `lem:preserve1` と Theorem `thm:preserve2`、Fig. `fig:buffercycle`, Fig. `fig:chains` を読む。正規ノートの `Summary` の display graph 特化版と、`Critical Thoughts` の future work に対応する。
- Discussion and future work は、constructive proof、chord 入り ladder、より複雑な low-pathwidth structures、$O(n^4)$ より速い ladder detection、3 本以上の trees への拡張という未解決点を確認するために読む。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2302.10662v2/`
- 正規ノート: `notes/arXiv-2302.10662v2.md`
