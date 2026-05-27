# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

- arXiv: https://arxiv.org/abs/2305.10601
- source: ../papers/arXiv-2305.10601v2/
- authors: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- venue / year: TeX 中には明示なし（main.tex は neurips_2023.sty を [final] で使用）
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
- **貢献**: (1) LM 推論を木探索として一般化する ToT フレームワーク、(2) 思考生成 (Sample/Propose) × 状態評価 (Value/Vote) × 探索 (BFS/DFS) のモジュラーな組合せ提示、(3) GPT-4 でも難しい 3 つの新タスク（Game of 24、Creative Writing、5×5 Crosswords）と systematic ablation、(4) LM の "System 1" を古典 AI 由来の "System 2" で増強する概念的位置付け（Newell, Shaw, and Simon の探索木に基づく問題解決への参照）。

## Takeaway（自分にとっての要点）

- ToT の本質は「**思考の粒度を problem-aware に決め、その粒度で LM 自身に valuation させる**」こと。Sample vs Propose、Value vs Vote の 2×2 を問題に合わせて選ぶのが設計指針。リッチな空間（Creative Writing 段落）は Sample + Vote、制約の強い空間（Game of 24 の式、Crosswords の単語）は Propose + Value。
- Game of 24 で CoT が IO より悪い（4.0% < 7.3%）のは意外。中間式を逐次サンプルしても、最初の 1 step を誤ると詰むので逆効果になる場面がある。これは「CoT は常に効く」というナイーブな信念への反例として覚えておく。
- BFS/DFS の選択は思考木の深さに依存：浅い木（T≤3）は BFS+b=5、深く局所評価困難な木（Crosswords は最大 10 step）は DFS + value 閾値で枝刈り + backtrack。Crosswords の −backtrack ablation が word 60→20% に崩れるのは backtrack 必要性の定量証拠。
- GPT-3.5 ablation（gen=GPT-4 / eval=GPT-3.5 で 64%）は、生成モデルと評価モデルを分けるとコストを下げつつ decent results を得られる可能性がある、という著者の示唆につながる。
- IO/CoT/CoT-SC/self-refinement を ToT の特殊例として表現する図式（limited depth and breadth）は、既存 prompting 手法を同じ枠組みで比較するための整理になる。
- iterative-refine（Self-Refine 系）は ToT と排他ではなく「思考生成方式の 3 つ目」として ToT に組み込める、と著者は明言（Creative Writing で ToT→7.91 を確認）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - "deliberate" を 4 つの質問（decompose / generate / evaluate / search）に分解した整理が clean で再現しやすい。コードと prompt が公開（GitHub: princeton-nlp/tree-of-thought-llm）。
  - Game of 24 / Crosswords で「best-of-N CoT との fair comparison」「ablation (−prune, −backtrack, +best state)」を出している点が同時期の prompting 論文に比べて誠実。特に CoT 失敗の 60% が first step で起きるという error analysis は ToT の必要性を直接裏付ける。
  - thought の粒度・search algorithm を独立に動かせる modularity は、著者が ToT の conceptual benefit として明示している。
- **弱み / 疑問**:
  - 著者自身が Discussion で認めているとおり、ToT は **GPT-4 が既に得意な多くの既存タスクには不要かもしれず**、API コストが大きい。付録でも GSM8K / StrategyQA の ToT 改善は「only slightly」と述べている。
  - Game of 24 / Crosswords 共に**評価ベンチが小さい**（各 100問・20問）。Crosswords 20 問で Game 成功率を比較する点は、サンプルサイズ上の不安が残る（評者補足）。
  - Crosswords は GPT-4 が知らない obsolete 語（例: ``agend''）で正解 state が impossible 判定されて pruning される、と著者が認めている。LM-as-heuristic の限界の典型例。
  - state evaluator も LM 自身なので、**生成と評価が同じ failure mode を共有する**懸念がある（評者補足）。GPT-3.5 ablation では Game of 24 の bottleneck が thought generation と述べられているが、eval 側の校正までは詳述されていない。
  - "thought" の粒度はタスクごとに設計されている（Game of 24=式、Crosswords=単語）。自動で粒度を決める手段は TeX 中には示されていない。
  - cost 比較表（Table \ref{tab:cost_game}）は IO(best-of-100) / CoT(best-of-100) / ToT の比較で、CoT-SC(k=100) や IO+Refine との token 量を揃えた pareto curve は示されていない。
- **次に試したいこと**:
  - 同じ token 予算で ToT-BFS / CoT-SC / self-refinement を並べた pareto curve（評者補足）。
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
- (verified 2026-05-27) venue/year を TeX で確認できる範囲（neurips_2023.sty を [final] で使用）に限定し、NeurIPS 2023 掲載・preprint v2 断定を削除 (main.tex)
- (verified 2026-05-27) Critical Thoughts / Takeaway の推測的評価を「評者補足」と明示、または TeX 根拠のある表現に弱めた (main.tex)
- (verified 2026-05-27) main.bbl に無い Du+ 2023 / Multiagent Debate の関連論文記述を削除し、関連論文タイトルを bbl で確認できる範囲に修正 (main.bbl)

## Related Papers

- Wei+ 2022, *Chain of thought prompting elicits reasoning in large language models* — CoT の baseline / ToT が一般化する対象。
- Wang+ 2022, *Self-consistency improves chain of thought reasoning in language models* — CoT-SC baseline。
- Madaan+ 2023, *Self-refine: Iterative refinement with self-feedback* / Shinn+ 2023, *Reflexion: an autonomous agent with dynamic memory and self-reflection* — self-reflection 関連。
- Hao+ 2023, *Reasoning with language model is planning with world model* — 並行研究、MCTS で類似の枠組み。著者は「タスクが simpler で modularity に欠ける」と評価。
- Xie+ 2023, *Decomposition enhances reasoning via self-evaluation guided decoding* — tree search + LM self-eval の隣接研究、ただし PAL（code）表現に依存。
- Lu+ 2021, *Neurologic a*esque decoding: Constrained text generation with lookahead heuristics* — A* 風 lookahead を beam-search / top-k sampling decoding に持ち込む先行例、文生成限定。
- Newell, Shaw, and Simon (1959) / Newell, Simon, et al. (1972) — 探索木としての問題解決という思想的源流。
- Kahneman 2011, *Thinking, fast and slow* — System 1/2 の認知科学フレーム。
