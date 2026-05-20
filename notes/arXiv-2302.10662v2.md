# Snakes and Ladders: a Treewidth Story

- arXiv: https://arxiv.org/abs/2302.10662
- source: ../papers/arXiv-2302.10662v2/
- authors: Steven Chaplick, Steven Kelk, Ruben Meuwese, Matúš Mihalák, Georgios Stamoulis (Maastricht University)
- venue / year: WG 2023 (proceedings 版が先行) / 当該 arXiv v2 は 2024 年版（Author's note 2 で "January 2024" と言及）
- tags: [treewidth, graph-theory, parameterized-complexity, phylogenetics, reduction-rules]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: グラフ $G$ から「ladder（$2\times(k+1)$ グリッドが 4 つのコーナーポイント経由でしか外と繋がっていない誘導部分グラフ）」を短くしたとき、treewidth $tw(G)$ は保たれるか。特に phylogenetics で **common chain reduction rule**（2 本の系統樹に共通する label 列を定数長に縮める操作）を display graph 上で treewidth-preserving に行える定数は存在するか（Kelk 2017 の未解決問題）。Theorem `thm:unbounded`（[kelk2017treewidth]）からは treewidth $k$ に依存した関数 $f(k)$ までは知られていたが、$k$ に依存しない普遍定数があるかは未解決だった。
- **手法**: tree decomposition を直接書き換える first-principles の証明。各 ladder 頂点を含む bag 集合が tree decomposition 内で病的にうねる "snakes" を、distance-minimizing な分解の選び方と "reeling in the snakes" という操作で飼いならす。Case 1（ある bag が ladder の 1 square 全てを含む）/ Case 2（bag が上 2 点・下 2 点を含む）/ Case 3（$B_1, B_2$ 隣接）/ Case 4（間に内部 bag あり）の case 分析で「extra rung を 1 本挿入しても width が上がらない」ことを示し、iteratively に任意長まで伸ばせるとする。display graph 特化版は biconnectivity と $c, d$ 周りの構造を使った付加議論。下界（tightness）は bramble 構成（Seymour-Thomas）で示す。
- **結果**:
  - **Theorem `thm:main`**: $tw(G) \geq 4$ かつ ladder 長 $\geq 3$ なら任意長に伸ばしても treewidth 不変。
  - **Lemma `lem:main2`**: 任意の $tw(G)$ で ladder 長 $\geq 5$ なら不変。
  - **Theorem `thm:main3`（メイン結果）**: 任意の $tw(G)$ で ladder 長 $\geq 4$ なら不変。定数 4 は tight（Fig. `fig:prism` のグラフ：treewidth 3 で長さ 3 の ladder を 1 増やすと treewidth が 4 に上がる）。
  - **Lemma `lem:alwaystight`**: 任意の $t \geq 3$ に対し、treewidth ちょうど $t$ で長さ 2 の ladder を持ち、それを長さ 3 にすると treewidth が $t+1$ に上がるグラフ $G_2(t)$ が存在（高い treewidth でも「長さ 2 始め」では足りないことを analytical に示す）。
  - **Theorem `thm:preserve2`**: display graph 上では common chain を 4 leaf labels（= ladder 長 3）に縮めても treewidth 不変。これは tight（3 labels まで縮めると下がるケースあり、Fig. `fig:chains`）。
  - 帰結: subtree reduction と common chain reduction（4 labels まで縮める版）の双方が display graph の treewidth を保存することが確定。
- **貢献**: (1) ladder 短縮の普遍定数の存在と最良値（4）の確定、(2) 全 treewidth に対し tightness を解析的に示した（先行の経験的 tightness を強化）、(3) display graph 上の chain reduction が treewidth-preserving であるという Kelk 2017 以来の open problem の解決、(4) treewidth 計算に使える新しい safe reduction rule（[Abu-Khzam2022] の survey に追加可能）、(5) bounded treewidth の forbidden minor が長い ladder を含み得ないという系。

## Takeaway（自分にとっての要点）

- 「path（degree-2 chain）の suppress は treewidth 不変」は folklore だが、ladder のような **やや構造のある再帰部分** に同種の reduction が成り立つかは（驚くべきことに）あまり整理されていなかった、という指摘自体が面白い。protrusion 系の一般機械（[BodlaenderFLPST16] や finite integer index [BodlaenderF01]）では「ある定数」は出るがタイトな 4 は得られない、という positioning。
- 証明の中心アイデア: minimum-width tree decomposition の中から **distance-minimizing**（$\{u,w,v\}$ を含む bag と $\{v,w,x\}$ を含む bag の距離が最小）なものを取ることで矛盾誘導の足場を作る。これは reduction rule を直接書き換える系の証明テクとして応用可能（Discussion でも constructive / polynomial-time に直せると主張）。
- $tw \geq 4$ と $tw = 3$ で挙動が変わり、$tw=3$ がボトルネックだから定数 4 になる、という現象は綺麗。さらに $tw$ をいくら大きくしても「長さ 2 始め」は無理（`lem:alwaystight` の bramble 構成）。
- display graph で 1 だけ短くできる（5 → 4）のは「両端の構造的制約」を活かしているから（buffer cycle、Fig. `fig:buffercycle`）。一般グラフでは右端側に buffer がないのが効く。
- Discussion で「各 square に高々 1 本の chord がある ladder」にも proof technique がそのまま乗ると示唆。Case 1/2 は 4-clique を内包できるので 0 or 1 chord OK、reeling 部分は構造的に 1 chord/square を induce する。一般化の方向性が示されている。
- Author's note 2 に「Theorem main3 は実は [almob2023]（WABI 2022 [wabi2022]）で別技法で先に証明されていた」と正直に記載。Theorem main の $tw \geq 4$ 強化と Theorem preserve2 の display graph 強化、`lem:alwaystight` だけが本論文 unique という素直な棲み分け表明は読み手として助かる。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 結果が **tight**：上界（定数 4 で十分）と下界（3 では足りない、しかも全 treewidth でそうなる例を構成）が揃っている。
  - 古典的（Kelk 2017）に提示された具体的な open problem を解いている点で位置付けが明確。
  - 証明が tree decomposition の構成的書き換えに基づくため、Discussion 冒頭で「polynomial-time に新しい decomposition を作れる」と主張できる（既存の forbidden-minor 経由の存在証明より algorithmic に強い）。
  - Discussion で chord 入り ladder への一般化方針まで述べていて、拡張余地を読者に渡している。
  - 先行研究 [almob2023] と独立に到達したことを正直に明記し、自分たちの貢献を `lem:alwaystight` と display graph 版（Theorem preserve2）と $tw \geq 4$ 版に絞っている。
- **弱み / 疑問**:
  - Theorem `thm:main` の case 分析（特に Case 4 / Subcase 4.2 / Fig. `fig:trickycase2`）が長く、ad-hoc な書き換え操作のオンパレードで、再現・検証コストが高い。著者自身 "much of the hard work and creativity lies" in controlling the snakes と書いているとおりで、もう少し抽象的な lemma に括れないのか、という疑問が残る。
  - Author's note 2 で述べているとおり、Theorem `thm:main3` 本体は [almob2023] と独立だが先んじられている。「$tw \geq 4$ への強化」は本論文も "very mild strengthening" と認めており、純粋に新しい部分（`lem:alwaystight` と display graph 版）の独立価値は読み手次第。
  - reduction rule としての algorithmic 側の議論が薄い。Discussion で「trivial $O(n^4)$ で ladder を探せる」と書いているが、改良アルゴリズムや実装上の cost-benefit は今後の課題として丸投げ。
  - chord 入り ladder への一般化が proof technique レベルでスケッチされるだけで、formal な statement / proof は無い。「more general low-pathwidth recursive structures」も open のまま。
  - 3 本以上の系統樹から作る display graph での chain 構造はノータッチ（最後に open として明記）。
  - Lemma `lem:alwaystight` の $G_2(t)$ 構成は clique $C$ をくっつけて treewidth を意図的に押し上げる人工的な例。実応用（phylogenetics）でこういう構造が現れるかの議論は無く、結果は「全 treewidth で tight」と言うための meta-statement に近い。
- **次に試したいこと**:
  - Discussion で示唆された **「各 square に高々 1 chord ある ladder」** バージョンを formal に書き下し、tight constant がいくつになるか確認する。
  - 「distance-minimizing tree decomposition を取る」テクが、ほかの reduction rule（subtree reduction の別証、protrusion 系など）にも汎用的に使えないか試す。
  - common chain reduction を実装に組み込んだうえで、Kelk 2017 系の display graph treewidth ベースの dissimilarity 計算パイプライン（[van2022embedding] など）で実速度がどれだけ落ちるか測る。
  - 3 本以上の tree から作る display graph 上の chain/ladder 類似構造の formalization。
  - bramble lower bound（Seymour-Thomas）に依らない、よりシンプルな tightness 証明（例えば forbidden-minor 直接構成）が `lem:alwaystight` に対しても可能か。

## Notes / Quotes

- "Getting these *snakes* under control is where much of the hard work and creativity lies, and is the inspiration for the title of this paper."（Introduction）
- "minimal forbidden minors for bounded treewidth graphs cannot contain long ladders."（Abstract / Introduction 末尾の系）
- Theorem `thm:main3` の constant 4 は tight：Fig. `fig:prism`（treewidth 3、長さ 3 の ladder、伸ばすと treewidth 4。pentagonal prism が minor として現れることが理由として figure caption の hidden コメントに書かれている）。
- Lemma `lem:alwaystight`：$G_2(t)$ は ladder の頂点 1–6 と $(t-1)$ 頂点の clique $\{7,8,\ldots,7+t-2\}$ からなる。頂点 1, 3 は clique 全体に、頂点 4, 6 は頂点 7 以外の clique 全体に隣接。tree decomposition は 5 bag、width $t$。$G_3(t)$ の lower bound は $|C|+5$ subgraph からなる bramble と minimum hitting set $H$（$|H|=t+2$）で示す（Seymour-Thomas）。
- "Author's note 2"（Introduction）: Theorem `thm:main3` は独立に [almob2023]（WABI 2022 [wabi2022] が源流）で証明済み。本論文の独自貢献は (a) Theorem `thm:main` の $tw \geq 4$ への mild strengthening、(b) display graph 上の Theorem `thm:preserve2`、(c) 全 treewidth tightness の `lem:alwaystight`。
- 証明の鍵: distance-minimizing tree decomposition と "reeling in (the snakes) $a$ and $b$" 操作（Subcase 3.2、Subcase 4.1）。
- display graph 版で 1 短くできる根拠は、ladder の右端側にも biconnectivity 由来の cycle が "buffer" の役を果たすこと（Fig. `fig:buffercycle`）。
- Discussion: proof は constructive にできる、各 square に高々 1 chord ある ladder にも proof technique は通る、低 pathwidth 再帰構造への一般化と $O(n^4)$ より速い ladder 検出は open、3 本以上の tree への一般化も open。
- (verified 2026-05-20) Related Papers の [kelk2017treewidth] のタイトル・著者を訂正 ("A note on convex characters..." → "Treewidth distance on phylogenetic trees" by Kelk, Stamoulis & Wu, TCS 2018) — 根拠: afterALMOB.bbl の \bibitem{kelk2017treewidth}

## Related Papers

- Kelk, Stamoulis & Wu, "Treewidth distance on phylogenetic trees", Theoretical Computer Science 731, 99–117 (2018) / [kelk2017treewidth]（display graph と treewidth の関係、subtree reduction が treewidth-preserving、common chain の open problem の原典）
- Bryant & Lagergren [bryant2006compatibility] — display graph の定義と多 tree への拡張。
- Allen & Steel [AllenSteel2001] — common chain reduction rule の原典。
- Janssen et al. [janssen2018treewidth], van Iersel et al. [van2022embedding] — display graph の treewidth を使った系統樹間距離計算と Courcelle 定理応用。
- Seymour & Thomas, *Graph searching and a min-max theorem for tree-width* [seymour1993graph] — bramble による treewidth 下界、`lem:alwaystight` で使用。
- Bodlaender et al. [BodlaenderFLPST16], Bodlaender & van Antwerpen-de Fluiter [BodlaenderF01] — protrusion / finite integer index に基づく一般 reduction の枠組み（本論文のタイトな結果は届かない、と positioning）。
- Abu-Khzam et al. [Abu-Khzam2022] — treewidth 計算向け reduction rule の survey。本論文の結果は新しい safe reduction rule として追加可能。
- Sanders [sanders1996linear] — ladder を扱った先行研究だが treewidth $\leq 4$ 認識用で ladder topology を破壊する形だった、と本論文が differentiator として明示。
- Chaplick et al. [chaplick2023snakesWG, chaplick2023snakes] — 本論文の WG 2023 proceedings / 初回 arXiv 版。
- [almob2023]（[wabi2022] 元）— Theorem `thm:main3` を独立に別技法で証明していた先行研究（Author's note 2 で明示）。
