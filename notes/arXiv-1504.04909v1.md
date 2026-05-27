# Illuminating search spaces by mapping elites

- arXiv: https://arxiv.org/abs/1504.04909
- source: ../papers/arXiv-1504.04909v1/
- authors: Jean-Baptiste Mouret, Jeff Clune
- venue / year: arXiv preprint, 2015（著者自身が "preliminary draft" と明記）
- tags: [evolutionary-computation, MAP-Elites, illumination-algorithm, soft-robotics]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 通常の最適化アルゴリズム（hill climbing, EA, Bayesian optimization, multi-objective EA 等）は「単一あるいは Pareto 前線上の少数解」を返すだけで、特徴空間のどこにどんな性能の解が存在するか（phenotype-fitness map）を教えてくれない。Novelty Search や Novelty Search + Local Competition (NS+LC)、Multi-Objective Landscape Exploration (MOLE) など「レパートリ取得型」の先行手法もあるが、archive と population の二重管理や global な性能競合のため挙動が複雑で、特徴空間を一様に埋めるのは不得手。
- **手法**: **MAP-Elites**（Multi-dimensional Archive of Phenotypic Elites）を提案。ユーザが選んだ $N$ 次元の feature space を離散化したセル群を用意し、各セルにそのセル内最良解（elite）を1つだけ保持する。各 iteration で (1) 既存 elite から1つランダム選択、(2) mutation/crossover で子を生成、(3) 子の feature descriptor $b(x)$ と performance $f(x)$ を評価、(4) 該当セルが空または現 elite より高性能なら置換、を繰り返す。archive と population の区別を捨て「archive のみ」にすることで cycling を回避し、計算量はセル参照 $O(1)$。粗→細の階層版（hierarchical MAP-Elites）と並列化版（slave node にバッチを farm out）も実装。本論文の全実験はこの hierarchical/parallel 版で実施。
- **結果**: 3 ドメインで検証。
  - **(1) Retina 神経網（8 画素入力、左右両側に object があるかを判定、performance = 256 入力に対する正答率）**: feature1=connection cost、feature2=Newman modularity、最終解像度 512×512、10,000 iteration。比較は traditional EA / NS+LC / random sampling、各 20 runs。MAP-Elites が global performance, global reliability, precision, coverage の4指標すべてで有意に勝つ（$p < 1\times10^{-7}$, two-tailed Mann-Whitney U）。MOLE との直接定量比較は本ドラフトでは未実施だが、過去研究では 30 runs 分の MOLE をマージしてようやく得た map に 1 run の MAP-Elites が近いと anecdotal に報告。
  - **(2) Voxcad シミュレーションの soft robot（10×10×10 voxel、4 種材料: bone/soft tissue/位相同期 muscle/逆位相 muscle、形態は CPPN で間接符号化し、CPPN network 自体は Sferes$_{v2}$ の NEAT 原理で直接符号化、performance = 10 秒走行距離）**: feature1=bone 割合、feature2=voxel 充填率、最終 128×128。比較は EA と EA+Diversity（NSGA-II ベース、diversity = feature space 内平均距離）。runs 数は EA=7, EA+D=5, MAP-Elites=8（一部 run が締切前に未完）。global reliability と coverage で MAP-Elites が有意に良い（$p<0.002$）。global performance は中央値で MAP-Elites が高いが有意差なし（$p>0.05$）。precision は MAP-Elites が有意に悪い（$p<0.01$、評価予算を多数セルに薄く配分するため）。map 内では bone を増やすと遅くなる傾向、また voxel 充填率 ~7% の縦に長い「1 voxel 厚シート」の高性能アイランドが偶然見つかった。
  - **(3) 実物の soft robot arm（dynamixel AX-18 サーボ3個 + フレキシブルチューブ、解は3関節角度、performance = 末端の y 座標）**: feature space は末端 x 座標を 64 セルに離散化。比較は random sampling と 8 段グリッド探索。MAP-Elites と random sampling は 420 評価（実験パラメータ節）／本文では 640 評価と記載、grid search は決定的で $9\times9\times9=729$ 評価（本文）／512 評価（パラメータ節）と本文中に不一致あり。中間 x 領域（≈400–600）で MAP-Elites が両者を上回り、低 x 領域では grid search はほぼ点が無く、MAP-Elites の方が高性能解を多く返す。「これらは予備的なので reliable な statistics ではない」と著者自身が明記。
- **貢献**: (1) "illumination algorithm" という概念を導入し optimization algorithm の上位集合と位置付けた、(2) シンプルでパラメータの少ない MAP-Elites を初めて詳細に記述・検討した（Cully et al. 2015 [cully2015robots; bbl では arXiv:1407.3501] では簡略にしか説明されていなかった、と TeX が明記）、(3) Global Performance / Global Reliability / Precision (opt-in reliability) / Coverage の4指標を illumination algorithm 評価基準として定式化、(4) 3 ドメインで NS+LC, EA, EA+D, random sampling, grid search との比較を提示。

## Takeaway（自分にとっての要点）

- **核心は「セル＝1個の elite」という割り切り**。Novelty Search のように k 近傍を毎回計算する必要も、archive と population を別々に持つ必要もない。これだけで cycling が消え、選択圧の動きが直感的に追える（uniform sample from filled archive）。
- **feature space は探索空間ではなく「ユーザの興味の写像先」**。直接 feature space を探索することはできず（多対一マッピングのため）、genotype 空間を探索しつつ評価で $b(x)$ を測って投影する、という方向性が重要。逆に、feature space で直接ステップ可能な問題なら MAP-Elites は不要（exhaustive search で済む）。
- **「optimization 性能で見ても勝つことがある」が面白い**。retina で traditional EA より single best が高い理由として、deceptive な探索景観で多様な stepping stones を同時保有することがローカルオプティマ脱出に効く、と説明。lineage 図（Fig. retina_paths 右）が遠隔セルからの長距離経由を示唆。
- **4 指標の切り分け**は他の QD/illumination 手法を評価するときにそのまま使える: (a) Global Performance（最良1点）, (b) Global Reliability（埋まり得る全セル平均、空セルは0）, (c) Precision（自分が埋めたセルだけの平均）, (d) Coverage（埋めたセル数比）。optimization アルゴリズムは (a)(c) は取れるが (b)(d) は構造的に取れない、という事実が指標設計に組み込まれている。
- **precision で MAP-Elites が負ける場面がある**（soft robot）という正直な報告は重要。多セル探索で評価予算が分散するという原理的トレードオフを示しており、「精度を犠牲にして網羅を取る」アルゴリズムだと理解しておくべき。
- **hierarchical 版**（粗いセルから始めて時間とともに細かく分割）は本論文の全実験で使われており、retina では 16×16 → 64×64 → 128×128 → 256×256 → 512×512 と段階的に細分化している（iteration 0, 1250, 2500, 5000）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - アルゴリズムが本当に小さい（疑似コード十数行）。TeX 中でも「simple, intuitive, and predictable」として、NS+LC や MOLE より実装・理解が容易な点を強調している。
  - 「optimization vs illumination」という概念整理が明快で、evaluation metric も4指標に切り分けたうえで「optimization は構造的に reliability/coverage は取れない」と認めている。後続研究の比較プロトコルとして使いやすい。
  - 3 ドメイン（NN, sim soft robot, real soft robot）の異質性が高く、特に real soft robot で実機評価をしている点は強い。
  - lineage 分析（Fig. retina_paths 右、stepping stones が feature space を横断する）は「単なる多点探索ではなく相互利用が起きている」ことの定性的証拠を与える。
- **弱み / 疑問**:
  - 著者自身が冒頭の "Author's Note" で「preliminary draft、データは差し替わる予定」と明言している。実際、soft robot は runs 数が EA=7/EA+D=5/MAP-Elites=8 と不揃いで、real arm の評価回数は本文（640/729）と実験パラメータ節（420/512）で食い違う。
  - **MOLE との定量比較が無い**（本文中で「MOLE 30 runs マージで初めて 1 run MAP-Elites 相当」と anecdotal にだけ書かれている）。最近接の先行手法と直接比較しないのは弱い。
  - **同じ評価予算の解釈に注意が必要**。TeX では controls と MAP-Elites に同じ評価回数を割り当てたと書かれているが、MAP-Elites は archive 全体に評価を分散する一方、EA は少数セルに集中する。したがって、特に precision の比較では評価配分の違いが結果に直接影響している。
  - feature dimensions の選び方が結果を支配するはずだが、その選定論は「ユーザに任せる、可用計算量で粒度を決める」止まり。soft robot の「voxel 7% 列の1 voxel 厚シート」高性能領域は simulator の quirk と認めており、feature の選び方次第で誤誘導的な map が出る危険がある。
  - **precision の劣化を「もっと回せば追いつく」と仮説で済ませている**（実証なし）。多セル化に対する評価予算の必要量に関する理論的議論が無い。
  - retina で MOLE/Novelty Search を「やる予定」と書きながら未実施、また Hierarchical 版の効果に関する ablation も "Report whether preliminary experience show this is a good idea" という TODO で抜けたまま。
  - 著者自身の認める limitations: (i) 既存セルを超えて feature space を拡張できない＝open-ended evolution にならない、(ii) precision が optimization 系より劣る、(iii) feature space の cell が物理的に達成不可能な領域があり coverage の理想値=1 にならない。
- **次に試したいこと**:
  - 同じ評価予算で MAP-Elites / NS+LC / MOLE / EA+D を比較する Pareto curve（横軸=評価数、縦軸=4 指標）を見る（MOLE 比較未実施という TeX 記述に基づく評者補足）。
  - hierarchical 版の split timing と final resolution の感度解析。retina の階段スケジュール（0/1250/2500/5000 で倍化）が他ドメインでも妥当なのか確認したい（階層版の効果検証 TODO に基づく評者補足）。
  - "anomalous island" のような simulator quirk を発見器として MAP-Elites を使えるか確認する（soft robot 結果に基づく評者補足）。

## Notes / Quotes

- "Optimization algorithms try to find the highest-performing solution in a search space. ... A different kind of algorithm, which we call \emph{illumination algorithms}, are designed to return the highest-performing solution at each point in the feature space." (§Optimization vs. Illumination Algorithms)
- "Any illumination algorithm can also be used as an optimization algorithm, making illumination algorithms a superset of optimization algorithms." (§Optimization vs. Illumination Algorithms)
- "MAP-Elites only needs to look up the current occupant of the cell, which is $O(1)$." — Novelty Search の $O(n\log n)$ との対比（§Differences）。
- "MAP-Elites does away with the archive vs. population distinction by having only an archive. It thus avoids cycling..." (§Differences)
- "It is not guaranteed that all cells in the feature space will be filled" — 物理的不可能セル + 探索失敗の2要因（§Details）。
- "While MAP-Elites significantly outperforms all controls on both reliability and precision (opt-in reliability), the gap is much narrower for precision, as is to be expected." (§Search space 1)
- soft robot precision: "MAP-Elites is significantly worse at precision than the two control algorithms ($p<0.01$). This result is likely explained by the fact that the control algorithms allocate all of their evaluations to very few cells..." (§Simulated soft robots)
- 著者自身の限界: "One drawback to MAP-Elites, however, is that it does not allow the addition of new types of cells over time that did not exist in the original feature space. It thus, by definition, cannot exhibit open-ended evolution." (§Discussion)
- retina 階層スケジュール: 16×16 開始 → 64×64 (iter 0) → 128×128 (1250) → 256×256 (2500) → 512×512 (5000), batch 2000, 10,000 iterations, initial batch 20,000（§Experimental parameters）。
- 統計検定: 全 $p$ 値は two-tailed Mann-Whitney U test（§Methods/Statistics）。
- "Author's Note: This paper is a preliminary draft ... All of the experiments in this paper will be redone before the final version of the paper is published, and the data are thus subject to change." — 冒頭。
- (verified 2026-05-20) Related Papers の Cully et al. 2015 を「Nature」→「bbl 表記の arXiv:1407.3501」に修正、タイトルを "Robots that can adapt like natural animals" に訂正（main.bbl の cully2015robots と一致）。
- (verified 2026-05-20) 貢献欄の "[Cully et al. 2015 Nature]" を bbl 表記（cully2015robots / arXiv:1407.3501）に揃えた。
- (verified 2026-05-20) Related Papers の "Mouret & Doncieux 2012, Sferes_v2" は bbl 上の2件（mouret2010sferesv2=Sferes_v2 本体 2010年、Mouret2012=behavioral diversity 論文 2012年）を混同していたため2行に分割した（mapElitesNoComments.tex L489 で両者が別 cite として使われていることを確認）。
- (verified 2026-05-26) TeX 外の後発研究名・普及状況・Final 版 arXiv 有無に関する外部推定を削除し、評者独自の実験案には「評者補足」を明記 (mapElitesNoComments.tex, mapElitesNoComments.bbl)。
- (verified 2026-05-26) hierarchical 版について「実装上ほぼ必須レベル」を、TeX で確認できる「本論文の全実験で使用」に修正 (mapElitesNoComments.tex, Methods)。
- (verified 2026-05-26) tags から TeX に無い quality-diversity を削除し、soft robot の CPPN/NEAT 符号化説明を本文記述に合わせて修正 (mapElitesNoComments.tex, Simulated soft robots)。

## Related Papers

- Lehman & Stanley 2011, "Evolving a diversity of virtual creatures through novelty search and local competition" — NS+LC、MAP-Elites の直接の起源。
- Lehman & Stanley 2011, "Abandoning Objectives: Evolution Through the Search for Novelty Alone" — Novelty Search 本体。
- Clune, Mouret, Lipson 2013, "The Evolutionary Origins of Modularity" — MOLE と retina ドメインの出典。
- Cully, Clune, Tarapore, Mouret 2015, "Robots that can adapt like natural animals" (bbl: arXiv:1407.3501) — MAP-Elites を初めて使った応用論文。本論文はその algorithm 部分を独立化したもの。
- Cheney, MacCurdy, Clune, Lipson 2013, "Unshackling Evolution: Evolving Soft Robots with Multiple Materials and a Powerful Generative Encoding" — Voxcad 上の soft robot 実験基盤と CPPN+NEAT 設定の元。
- Nguyen, Yosinski, Clune 2015, "Innovation Engines: Automated Creativity and Improved Stochastic Optimization via Deep Learning" — "goal switching" と「セル別に別々に探索するより MAP-Elites 統合の方が良い」根拠の参照先。
- Stanley & Miikkulainen 2002 (NEAT), Stanley 2007 (CPPN), Hiller & Lipson 2014 (Voxcad) — 直接符号・間接符号・シミュレータの基盤。
- Mouret & Doncieux 2010, "Sferesv2: Evolvin'in the multi-core world" (bbl: mouret2010sferesv2) — Sferes_v2 実装プラットフォーム本体。
- Mouret & Doncieux 2012, "Encouraging behavioral diversity in evolutionary robotics: an empirical study" (bbl: Mouret2012) — Sferes 版 NEAT の詳細（CPPN の直接符号で使用）。
