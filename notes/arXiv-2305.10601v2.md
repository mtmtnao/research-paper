# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

- arXiv: https://arxiv.org/abs/2305.10601
- source: ../papers/arXiv-2305.10601v2/
- authors: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- venue / year: NeurIPS 2023 (preprint v2, 2023)
- tags: [LLM, reasoning, search, planning, prompting]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: LLM は推論時に token-level・left-to-right の自己回帰デコードに閉じており、初手の数 token が致命的になる探索・先読み・後戻りを要するタスク（Game of 24 等）で破綻する。CoT も「1 本道で sequentially にしか thoughts を sample しない」ため局所探索や backtrack を持たない。
- **手法**: Tree of Thoughts (ToT)。問題を「state $s=[x, z_{1..i}]$ をノードとする木の探索」として定式化し、以下の 4 要素で具体化する。(1) **Thought decomposition**: タスク依存に thought の粒度を決める（Crosswords は単語、Game of 24 は 1 行の式、Creative Writing は段落プラン）。(2) **Thought generator $G$**: CoT prompt から i.i.d.\ Sample するか、propose prompt で k 個列挙する。(3) **State evaluator $V$**: LM 自身に各 state を sure/likely/impossible 的に Value させるか、複数 state を比較して Vote させる。few lookahead + commonsense で構成。(4) **Search**: BFS（各ステップ上位 b 個保持）か DFS（閾値 $v_{th}$ 以下で枝刈り＆親へ backtrack）。IO/CoT/CoT-SC/self-refine は ToT の特殊例として位置付けられる。
- **結果（GPT-4, temperature 0.7, 2023/5/5–16）**:
  - **Game of 24**（4nums.com の index 901–1000, 100 問）: IO 7.3% / CoT 4.0% / CoT-SC k=100 9.0% / ToT b=1 45% / **ToT b=5 74%**。oracle 比較でも IO best-of-100 33%, CoT best-of-100 49% を ToT が上回る。IO+Refine (k=10) 27%。CoT 失敗例の約 60% は最初の 1 step（最初の 3 単語）で既に詰んでいる。
  - **Creative Writing**（4 文を末尾に持つ 4 段落 passage 生成, 100 問）: GPT-4 zero-shot coherency スコア IO 6.19 / CoT 6.93 / **ToT 7.56**。著者陣の盲検比較で ToT 勝 41 / CoT 勝 21 / 同等 38。iterative-refine は IO→7.67, ToT→7.91 に上昇。
  - **5×5 Mini Crosswords**（GooBix 156 ゲーム中 index 1,6,…,96 の 20 ゲーム）: Letter/Word/Game 成功率は IO 38.7/14/0、CoT 40.6/15.6/1、**ToT 78/60/20**。oracle best state では 82.4/67.5/**35** (= 7/20 解ける)。ablation: −prune 65.4/41.5/5、−backtrack 54.6/20/5。
  - 付録: GSM8K IO 51 / CoT 86 / **ToT 90**、StrategyQA 73/82/**83**（zero-shot ToT-BFS、5 plan vote → 5 solution vote）。GPT-3.5 では Game of 24 ToT は 19% に留まるが Creative Writing は GPT-3.5+ToT > GPT-4+IO。GPT-4 gen + GPT-3.5 eval = 64%, 逆 = 31% → ボトルネックは思考生成側。
  - **コスト**: Game of 24 ToT は 5.5k generated / 1.4k prompt tokens, \$0.74/問（CoT best-of-100 \$0.47 と同オーダー）。Creative Writing で CoT の約 5 倍。
- **貢献**: (1) LM 推論を木探索として一般化する ToT フレームワーク、(2) 思考生成 (Sample/Propose) × 状態評価 (Value/Vote) × 探索 (BFS/DFS) のモジュラーな組合せ提示、(3) GPT-4 でも難しい 3 つの新タスク（Game of 24、Creative Writing、5×5 Crosswords）と systematic ablation、(4) LM の "System 1" を古典 AI 由来の "System 2" で増強する概念的位置付け（Newell & Simon の探索木へのオマージュ）。

## Takeaway（自分にとっての要点）

- ToT の本質は「**思考の粒度を problem-aware に決め、その粒度で LM 自身に valuation させる**」こと。Sample vs Propose、Value vs Vote の 2×2 を問題に合わせて選ぶのが設計指針。リッチな空間（Creative Writing 段落）は Sample + Vote、制約の強い空間（Game of 24 の式、Crosswords の単語）は Propose + Value。
- Game of 24 で CoT が IO より悪い（4.0% < 7.3%）のは意外。中間式を逐次サンプルしても、最初の 1 step を誤ると詰むので逆効果になる場面がある。これは「CoT は常に効く」というナイーブな信念への反例として覚えておく。
- BFS/DFS の選択は思考木の深さに依存：浅い木（T≤3）は BFS+b=5、深く局所評価困難な木（Crosswords は最大 10 step）は DFS + value 閾値で枝刈り + backtrack。Crosswords の −backtrack ablation が word 60→20% に崩れるのは backtrack 必要性の定量証拠。
- GPT-3.5 ablation（gen=GPT-4 / eval=GPT-3.5 で 64%）は **生成は大モデル・評価は小モデル**という非対称配置でコスト最適化できる示唆。実装上はこちらが現実的。
- IO/CoT/CoT-SC を ToT の特殊例として表現する図式（depth/breadth に縮退）は、self-consistency と debate と ToT を同じ座標系で比較する叩き台になる。
- iterative-refine（Self-Refine 系）は ToT と排他ではなく「思考生成方式の 3 つ目」として ToT に組み込める、と著者は明言（Creative Writing で ToT→7.91 を確認）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - "deliberate" を 4 つの質問（decompose / generate / evaluate / search）に分解した整理が clean で再現しやすい。コードと prompt が公開（GitHub: princeton-nlp/tree-of-thought-llm）。
  - Game of 24 / Crosswords で「best-of-N CoT との fair comparison」「ablation (−prune, −backtrack, +best state)」を出している点が同時期の prompting 論文に比べて誠実。特に CoT 失敗の 60% が first step で起きるという error analysis は ToT の必要性を直接裏付ける。
  - thought の粒度・search algorithm を独立に動かせる modularity は、後続研究（RAP、Self-eval guided decoding 等）と組み合わせやすい。
- **弱み / 疑問**:
  - 著者自身が Discussion で認めているとおり、ToT は **GPT-4 が CoT で既に解ける多くのタスクには不要**で、API コストが大きい。GSM8K で 86→90 は誤差レベル。"deliberate" が要らない領域では割に合わない。
  - Game of 24 / Crosswords 共に**評価ベンチが小さい**（各 100問・20問）。Crosswords 20 問で Game 成功率を 5% 単位で比較するのは noise が大きそう。
  - Crosswords は GPT-4 が知らない obsolete 語（例: ``agend''）で正解 state が impossible 判定されて pruning される、と著者が認めている。LM-as-heuristic の限界の典型例。
  - state evaluator も LM 自身なので、**生成と評価が同じ failure mode を共有する**懸念。GPT-3.5 ablation でも gen 側がボトルネックと言いつつ、eval 側の校正は議論薄い。
  - "thought" の粒度は手動設計（Game of 24=式、Crosswords=単語）。自動で粒度を決める手段が無く、結局タスクごとに prompt engineering が必要。一般性の主張と実運用コストにギャップがある。
  - cost 比較表（Table tab:cost_game）は ToT(b=5) と CoT(best-of-100) のみで、CoT-SC(k=100) や IO+Refine との token 量を揃えた pareto curve は描かれていない。
- **次に試したいこと**:
  - 同じ token 予算で ToT-BFS / CoT-SC / Self-Refine / Multi-agent Debate (Du+ 2023) を並べた pareto curve。debate と ToT は orthogonal（debate は roll-out 全体、ToT は step-wise）なので組み合わせ実験が見たい。
  - state evaluator を**別モデル**（小モデル, RM, 外部 verifier）に差し替えた時の精度／コスト変化。Game of 24 なら式の数値検算は決定的に判定可能で、LM eval は要らないはず。
  - Crosswords で外部辞書 retrieval を pruning ヒューリスティックに併用（``agend'' 問題への直接対処）。
  - 自動 thought-granularity 探索（粒度を 1 hyperparameter として ToT 自身に探させる）。
  - ToT のロールアウトログを SFT/PRM 学習データに distill して、1 pass で ToT 相当の精度を出すモデルが作れるか（著者の future work と整合）。

## Notes / Quotes

- "It is perhaps surprising that underlying all this progress is still the original autoregressive mechanism for generating text, which makes token-level decisions one by one and in a left-to-right fashion." (introduction)
- "while existing methods (detailed below) sample continuous language sequences for problem solving, ToT actively maintains a tree of thoughts, where each thought is a coherent language sequence that serves as an intermediate step toward problem solving." (introduction)
- 思考の粒度は ``small enough'' で diverse に生成でき、``big enough'' で評価可能であるべき、というトレードオフ（§3, Thought decomposition）。
- BFS と DFS の擬似コードは Algorithm 1, 2。DFS は $V(p_\theta,\{s\})(s) \le v_{th}$ で枝刈り → 親へ backtrack。
- "around 60\% of CoT samples already failed the task after generating the first step" (Game of 24 error analysis)。これが ToT 採用の最強の動機。
- "we limit DFS search steps to 100, and simply render the deepest explored state ... into the final output." (Crosswords)。深さ優先で 100 step が予算。
- 著者明記の **limitations**: (1) GPT-4 が既に解けるタスクでは ToT は不要、(2) API コストが大きい、(3) 3 タスクしか試していない、(4) off-the-shelf LM のみで、ToT-style fine-tuning は未検討、(5) Crosswords で稀な語を impossible 判定する pruning ミス（``agend'' の例）。
- Broader Impact では「ToT は判断が高レベル自然言語で読める分、interpretability と人間整合の機会を増やす」と主張。

## Related Papers

- Wei+ 2022, *Chain-of-Thought Prompting* — ToT が一般化する直接の base。
- Wang+ 2022, *Self-Consistency (CoT-SC)* — ToT 中で「solutions の bandit」として位置付けられる baseline。
- Madaan+ 2023 *Self-Refine* / Shinn+ 2023 *Reflexion* — iterative refine を ToT の thought generation の 3 つ目として包含可能、と著者が明言。
- Hao+ 2023 *RAP* (Reasoning via Planning) — 並行研究、MCTS で類似の枠組み。著者は「タスクが simpler で modularity に欠ける」と評価。
- Xie+ 2023 *Self-eval guided decoding* — tree search + LM self-eval の隣接研究、ただし PAL（code）表現に依存。
- Lu+ 2021 *NeuroLogic A\*esque decoding* — A\* 風 lookahead を beam-search に持ち込む先行例、文生成限定。
- Newell, Shaw & Simon (1959, 1972) — 探索木としての問題解決という思想的源流。
- Kahneman *Thinking, Fast and Slow* (System 1/2) — Discussion で参照される認知科学フレーム。
- Du+ 2023 *Multiagent Debate* (arXiv:2305.14325) — 同時期の "LM 自身に推論を改善させる" 直交アプローチ。組合せ実験の余地大。
