# Tree of Thoughts: Deliberate Problem Solving with Large Language Models（LLM 推論を思考単位の木探索として定式化する研究）

- arXiv: https://arxiv.org/abs/2305.10601
- 一次ソース: ../papers/arXiv-2305.10601v2/
- 正規ノート: ../notes/arXiv-2305.10601v2.md

---

## 一言で言うと

この論文は、LLM の推論を左から右への単一路生成だけに閉じず、`thought` と呼ぶまとまった中間ステップをノードにした探索木として扱う Tree of Thoughts (ToT) を提案する。Game of 24、Creative Writing、5x5 Mini Crosswords で、GPT-4 + ToT が IO prompting や Chain-of-Thought (CoT) prompting より高い性能を示す、というのが著者の主張である。

## 何を議論する論文か

- **問題設定**: LLM は推論時に token-level, left-to-right な自己回帰生成を行うため、探索、戦略的 lookahead、初期決定が重要なタスクで失敗しやすい。Introduction では、これを「System 1」的な高速・連想的処理に近いものとして位置づけ、より deliberate な「System 2」的計画過程で補うことを考える。
- **対象範囲 / 仮定**: 入力 $x$ から出力 $y$ を生成する問題を、自然言語列の部分解を状態として探索する問題に写す。中心的な仮定は、問題ごとに適切な粒度の `thought` を設計でき、LM 自身が候補 thought や状態をおおまかに評価できる、という点である。
- **既存研究との差分**: IO prompting は直接 $y$ を生成する。CoT は $z_1,\ldots,z_n$ という中間 thought を逐次生成するが、各 thought 列は一本道で、局所的な分岐や backtracking はない。CoT-SC は複数の完全な CoT 軌跡を独立にサンプルして多数決するが、軌跡内の途中状態を探索しない。ToT は途中の thought 単位で複数候補を生成・評価し、BFS や DFS によって木を探索する。
- **この論文で答えたい問い**: token-level の単一路生成では難しい planning/search タスクに対して、LM 自身による thought 生成、状態評価、探索アルゴリズムを組み合わせることで、より強い問題解決能力を得られるか。

## 背景と前提

- この論文では、$p_\theta$ をパラメータ $\theta$ を持つ事前学習済み LM と書く。小文字 $x,y,z,s,\cdots$ は language sequence、大文字 $S,\cdots$ は language sequences の集合を表す。
- IO prompting は、問題入力 $x$ を task instruction や few-shot 例で包んだ prompt から出力 $y$ を生成する方法で、本文では $y \sim p_\theta^{IO}(y|x)$ と簡略化される。
- CoT prompting は、入力 $x$ と出力 $y$ の間に thought 列 $z_1,\cdots,z_n$ を入れる。各 $z_i$ は「coherent language sequence that serves as a meaningful intermediate step」で、数学 QA なら中間式のようなものを指す。ただし CoT では thought の粒度は明示的に分解されないことが多い。
- CoT-SC は $k$ 個の CoT 軌跡 $[z^{(i)}_{1\cdots n}, y^{(i)}]$ を i.i.d. にサンプルし、最頻の出力を返す。出力空間が小さい multi-choice QA などでは有効だが、途中ステップ単位の局所探索はしない。
- ToT の思想的背景として、Newell, Shaw, and Simon 以降の「問題解決を combinatorial problem space の木探索として見る」古典 AI / cognitive science の見方が使われている。Related Work では A*, MCTS, NeuroLogic A*esque decoding、RAP、self-reflection / self-refine 系などと比較される。

## 提案手法

### コアアイデア

ToT は、問題解決を「状態 $s=[x,z_{1\cdots i}]$ をノード、次の thought を枝とする木の探索」として扱う。1 つの thought は token ではなく、問題に応じて意味のあるまとまりにする。Table 1 (`Task overview`) では、Game of 24 の thought は 3 個の中間式、Creative Writing の thought は短い writing plan、5x5 Crosswords の thought は clue に入れる単語として整理されている。

ToT の具体化は、本文で 4 つの問いに分けられる。

- Thought decomposition: thought の粒度を問題ごとに決める。小さすぎると評価できず、大きすぎると多様で有望な候補を生成しにくい。
- Thought generator $G(p_\theta,s,k)$: 現在状態 $s$ から次の thought 候補を $k$ 個作る。Creative Writing では CoT prompt から i.i.d. sample、Game of 24 と Crosswords では propose prompt で複数候補を列挙する。
- State evaluator $V(p_\theta,S)$: 候補状態の進捗を heuristic として評価する。候補ごとに `sure/likely/impossible` や scalar value を付ける Value と、複数候補を比較して有望なものに Vote する方式がある。
- Search algorithm: 木構造に合わせて BFS または DFS を使う。Game of 24 と Creative Writing は深さが浅いので BFS、Crosswords は最大 10 thought steps の深い探索なので DFS と backtracking を使う。

### 重要な定義・数式

$$
s = [x, z_{1 \cdots i}]
$$

**式の意味**: ToT における木のノード、すなわち状態 $s$ を定義している。状態は元の入力 $x$ と、ここまでに選ばれた thought 列 $z_{1\cdots i}$ からなる部分解である。

**記号の定義**:
- $s$ ... ToT の探索木における state / node
- $x$ ... 問題入力
- $z_{1\cdots i}$ ... 現在までの thought 列
- $i$ ... thought step の深さ

**この論文での役割**: IO や CoT では生成列全体を一方向に進めるのに対し、ToT はこの $s$ を単位として分岐、評価、枝刈り、backtracking を行う。

$$
z^{(j)} \sim p_\theta^{CoT}(z_{i+1} \mid s)
= p_\theta^{CoT}(z_{i+1} \mid x, z_{1\cdots i})
\quad (j=1 \cdots k)
$$

**式の意味**: Sample 型の thought generation を表す。現在状態 $s$ を条件として、次の thought $z_{i+1}$ の候補を $k$ 回 i.i.d. に生成する。

**記号の定義**:
- $z^{(j)}$ ... $j$ 番目に生成された次 thought 候補
- $p_\theta^{CoT}$ ... CoT prompt による LM の条件付き生成分布
- $k$ ... 生成する候補数
- $s=[x,z_{1\cdots i}]$ ... 現在の部分解

**この論文での役割**: Creative Writing のように thought space が豊かなタスクでは、独立サンプルによって多様な plan や passage 候補を得るために使われる。制約が強い Game of 24 / Crosswords では、別式の propose prompt $[z^{(1)},\cdots,z^{(k)}]\sim p_\theta^{propose}(z_{i+1}^{(1\cdots k)}\mid s)$ が使われる。

$$
V(p_\theta, S)(s) \sim p_\theta^{value}(v \mid s)
\quad \forall s \in S
$$

**式の意味**: Value 型の state evaluation を表す。候補集合 $S$ の各状態 $s$ について、LM に状態の見込みを評価させ、値 $v$ を返す。

**記号の定義**:
- $V(p_\theta,S)$ ... LM を用いた state evaluator
- $S$ ... frontier など、評価対象の状態集合
- $s$ ... 評価対象の状態
- $v$ ... scalar value、または `sure/likely/impossible` のような分類
- $p_\theta^{value}$ ... value prompt による LM の評価分布

**この論文での役割**: Game of 24 では「24 に到達できそうか」を `sure/maybe/impossible` で評価し、Crosswords では残り clue が埋められるかを評価して DFS の pruning に使う。Creative Writing のように直接値付けしにくい場合は、本文の別式 $V(p_\theta,S)(s)=\mathds{1}[s=s^*]$ と vote prompt が使われる。

$$
S_t = \arg \max_{S \subset S'_t, |S| = b} \sum_{s \in S} V_t(s)
$$

**式の意味**: ToT-BFS の枝刈りを表す。時刻 $t$ で生成された候補集合 $S'_t$ から、評価値の合計が最大になる $b$ 個の状態を残す。

**記号の定義**:
- $S_t$ ... 深さ $t$ で保持する状態集合
- $S'_t$ ... 深さ $t$ で生成された全候補状態
- $b$ ... breadth limit、各段で残す候補数
- $V_t(s)$ ... 深さ $t$ における状態 $s$ の評価値
- $\arg\max$ ... 目的を最大化する選択を返す演算

**この論文での役割**: Game of 24 では $b=5$ の BFS が主要設定で、成功率 74% を得る。DFS の場合は、本文で $V(p_\theta,\{s\})(s) \le v_{th}$ のとき subtree を prune し、親状態へ backtrack すると説明される。

### 実装 / アルゴリズム上の要点

- step1: 問題ごとに thought の粒度を決める。Game of 24 は 3 個の中間式、Creative Writing は 1 個の plan を挟む depth 2、Crosswords は clue に単語を入れる 5-10 steps。
- step2: 現在状態から候補 thought を作る。自由度が高い場合は Sample、制約が強く重複が問題になる場合は Propose を使う。
- step3: 候補状態を LM 自身に評価させる。Game of 24 は 3 回 value sampling、Creative Writing は 5 回 vote、Crosswords は残り clue の可能性評価と confidence の集約を行う。
- step4: 探索する。BFS は各段で上位 $b$ 個を残す。DFS は有望な候補から深く進み、評価が閾値以下なら prune して backtrack する。
- step5: 最終出力を生成する。BFS では最後に $S_T$ の最良状態から出力を生成し、Crosswords では 100 DFS steps を上限として、最も深く探索された state を board に render する。

## 実験・結果

- **データセット / ベンチマーク**: 主実験は Game of 24、Creative Writing、5x5 Mini Crosswords の 3 タスク。本文では、GPT-4 Chat Completion mode、temperature 0.7、実験期間 May 5-16, 2023 と明記されている。
- **比較対象 / baseline**: IO prompting、CoT prompting、CoT-SC、iterative-refine、oracle best-of-100、ToT の ablation (`+best state`, `-prune`, `-backtrack`) が使われる。タスクごとに prompt 例数や sampling 回数は異なる。
- **指標**: Game of 24 は 100 games の success rate。Creative Writing は GPT-4 zero-shot prompt による 1-10 coherency score と、著者サブセットによる blind human comparison。Mini Crosswords は Letter / Word / Game の success rate (%)。
- **主な結果**: Game of 24 では Table 2 で IO 7.3%、CoT 4.0%、CoT-SC (k=100) 9.0%、ToT (b=1) 45%、ToT (b=5) 74%。Creative Writing では Figure 5 (`fig:write_results`) で IO 6.19、CoT 6.93、ToT 7.56、human comparison は ToT 勝ち 41、CoT 勝ち 21、同等 38。Mini Crosswords では Table 3 で IO 38.7/14/0、CoT 40.6/15.6/1、ToT 78/60/20。
- **著者が主張する貢献**: ToT は LM 推論を thought 単位の木探索に一般化し、thought decomposition、generation、evaluation、search を独立に変えられる modular な枠組みを与える。3 つの planning/search タスクで、既存 prompting より強い結果を示す。

Game of 24 の設定は、4nums.com の 1,362 games から人間の solving time で難しめの index 901-1,000 を使う。成功条件は、有効な式で 24 になり、入力数字を各 1 回だけ使うこと。ToT は 3 thought steps、propose prompt、BFS、$b=5$、`sure/maybe/impossible` 評価を使う。Figure 3 の error analysis では、CoT samples の約 60% が最初の step、すなわち最初の 3 words で既に失敗していると説明される。

Creative Writing は randomwordgenerator.com から 4 random sentences を取り、4 paragraphs の各段落末尾がその文になる passage を生成する。ToT は $k=5$ plans を生成して 5 votes で best plan を選び、その plan から $k=5$ passages を生成して同様に vote する。iterative-refine は IO を 6.19 から 7.67、ToT を 7.56 から 7.91 に改善し、著者はこれを ToT framework 内の第三の thought generation approach と見なせると述べる。

Mini Crosswords は GooBix の 156 games から test indices $1,6,\cdots,91,96$ の 20 games、prompting 用に $136,141,146,151,156$ を使う。ToT は DFS で最も有望な word clue を埋め、既に埋めた word / letter は変更しないため最大 10 intermediate steps となる。`+best state` は 82.4/67.5/35 で 7/20 games を解く。`-prune` は 65.4/41.5/5、`-backtrack` は 54.6/20/5 で、特に backtracking の重要性を示す ablation になっている。

付録では、GSM8K と StrategyQA に zero-shot ToT-BFS を適用し、GSM8K は IO 51、CoT 86、ToT 90、StrategyQA は IO 73、CoT 82、ToT 83 と報告する。GPT-3.5 では Game of 24 が IO 6%、CoT 3%、ToT 19%、Creative Writing が IO 4.47、CoT 5.16、ToT 6.62。Game of 24 で GPT-4 generation + GPT-3.5 evaluation は 64%、GPT-3.5 generation + GPT-4 evaluation は 31% で、著者は thought generation が bottleneck だと解釈している。

コストは Appendix `Cost and efficiency` にまとまっている。Game of 24 では IO best-of-100 が 1.8k / 1.0k tokens、\$0.13、33%、CoT best-of-100 が 6.7k / 2.2k、\$0.47、49%、ToT が 5.5k / 1.4k、\$0.74、74%。Creative Writing では IO が 0.9k / 0.4k、\$0.06、CoT が 0.9k / 0.4k、\$0.07、ToT が 4k / 2.9k、\$0.32。

## 妥当性と限界

- **この主張を支える根拠**: Game of 24 では CoT-SC や best-of-100 CoT と比較しても ToT が高く、Figure 3 の error analysis が「初期 step の誤りが致命的」という問題設定を直接支えている。Mini Crosswords では `-backtrack` の Word success が 60 から 20 に落ち、backtracking が性能に効いていることを示す。Creative Writing では自動 GPT-4 score だけでなく blind human comparison も併用している。
- **著者が認めている limitations / future work**: Discussion では、GPT-4 が既に得意な多くの既存タスクでは deliberate search は不要かもしれないこと、この研究は GPT-4 を難しくする 3 つの比較的単純なタスクに限ること、ToT は IO / CoT より多くの計算資源や API cost を要すること、off-the-shelf LM のみを使っており ToT-style fine-tuning は未検討であることが述べられる。
- **読者として注意すべき点**: ToT の性能は thought 粒度、prompt、評価 heuristic、探索予算に依存する。特に Crosswords では、GPT-4 が rare or obsolete words を知らず、`agend` を typo と見なして `impossible` と prune する例が footnote で挙げられている。LM 自身による評価は有用な heuristic だが、完全な verifier ではない。
- **追加で確認したい実験 / 疑問**: 同一 token / cost budget で CoT-SC、iterative-refine、ToT の Pareto curve を比較すると、性能改善が探索構造由来か計算量由来かをより切り分けやすい。Game of 24 では式検算は外部プログラムで決定的にできるため、LM value prompt と symbolic verifier の比較も重要である。thought decomposition を人手設計せずに選ぶ方法は TeX 中には示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Tree of Thoughts (ToT)**: thought をノード展開単位として、生成、評価、探索を組み合わせる LM inference framework。本文では IO、CoT、CoT-SC、self-refinement を limited depth and breadth の特殊例として見られると述べる。
- **thought**: 問題解決へ向かう「coherent language sequence」。Game of 24 では 1 行の中間式、Creative Writing では writing plan、Crosswords では clue を埋める word。
- **state**: $s=[x,z_{1\cdots i}]$。入力とここまでの thought 列からなる部分解。
- **thought decomposition**: thought の粒度設計。LM が多様な候補を作れる程度に small で、かつ進捗を評価できる程度に big である必要がある。
- **thought generator $G$**: 現在状態から次 thought 候補を作る機構。Sample と Propose がある。
- **Sample**: CoT prompt から $k$ 個の thought を i.i.d. に生成する方式。本文では Creative Writing に使われる。
- **Propose**: propose prompt で複数 thought を同じ文脈から列挙する方式。本文では Game of 24 と Crosswords に使われる。
- **state evaluator $V$**: 状態の探索価値を推定する heuristic。LM 自身を value prompt または vote prompt で使う。
- **Value**: 各状態を独立に score / class で評価する方式。Game of 24 の `sure/maybe/impossible` や Crosswords の clue feasibility が例。
- **Vote**: 複数状態を比較し、最も有望な $s^*$ を選ぶ方式。Creative Writing の plan / passage 選択で使う。
- **BFS**: breadth-first search。各 step で $b$ 個の promising states を保持する。Game of 24 と Creative Writing で使用。
- **DFS**: depth-first search。有望な状態を深く探索し、評価が閾値以下なら prune して backtrack する。Mini Crosswords で使用。
- **pruning**: evaluator が見込みなしと判断した subtree を探索しないこと。
- **backtracking**: ある枝が失敗または終了したとき、親状態に戻って別候補を探索すること。
- **oracle best state**: Crosswords の ablation で、heuristic が選んだ出力ではなく、探索中の最良状態を oracle 的に出力した場合。

## 読む順番の提案

- まず Abstract と Introduction を読み、token-level left-to-right decoding への問題意識と、Newell らの tree search に基づく位置づけを押さえる。正規ノートの Summary 冒頭に対応する。
- 次に Background の IO / CoT / CoT-SC の定式化を読む。ここで $p_\theta$、$x,y,z,s$、language sequence、thought の使われ方を確認すると、正規ノートの「IO/CoT/CoT-SC/self-refine は ToT の特殊例」という整理が読みやすくなる。
- その後、Section 3 `Tree of Thoughts` の 4 要素、Algorithm 1 ToT-BFS、Algorithm 2 ToT-DFS を読む。特に $s=[x,z_{1\cdots i}]$、$G(p_\theta,s,k)$、$V(p_\theta,S)$、BFS の $S_t$ 更新式を見る。
- 実験は Table 1 で 3 タスクの thought 粒度を比べてから、Game of 24、Creative Writing、Mini Crosswords の順に読む。Table 2、Figure 3、Figure 5 (`fig:write_results`)、Table 3 が主要な数値の根拠で、正規ノートの Results / Critical Thoughts と対応する。
- 最後に Discussion と Appendix `Additional Experiment Results` / `Cost and efficiency` を読む。限界、GPT-3.5 結果、GSM8K / StrategyQA、コスト表が、正規ノートの limitations と cost の記述につながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2305.10601v2/`
- 正規ノート: `notes/arXiv-2305.10601v2.md`
