# Reflexion: Language Agents with Verbal Reinforcement Learning

- arXiv: https://arxiv.org/abs/2303.11366
- source: ../papers/arXiv-2303.11366v4/
- authors: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- venue / year: TeX 中には明示なし（main.tex は neurips_2023.sty を preprint で使用）
- tags: [LLM-agent, verbal-RL, self-reflection, code-generation, decision-making, reasoning]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: LLM ベースの言語エージェント（ReAct, SayCan, Toolformer 等）は試行錯誤から素早く学ぶ手段がない。従来 RL は大量のサンプルと fine-tuning が必要で、巨大 LLM では現実的でない。一方 in-context example だけでは credit assignment ができず、長軌跡での失敗から学べない。
- **手法**: 重みを更新せず「言語的フィードバック」で強化する **Reflexion** を提案。3 つのモデル — Actor $M_a$（ReAct/CoT で行動・テキスト生成）、Evaluator $M_e$（EM / LLM 判定 / ヒューリスティック / 自動生成ユニットテストで報酬付与）、Self-Reflection $M_{sr}$（{軌跡, スカラー報酬} を一人称の自然言語振り返りに変換）— が協調する。生成された振り返り $sr_t$ はエピソード記憶 *mem* に追記され（容量 $\Omega$ は通常 1–3）、次試行の Actor のコンテキストに渡される。これが「semantic gradient」として作用する。失敗→反省→記憶→再試行のループを Evaluator が pass を出すか max trials まで繰り返す（Algorithm 1）。
- **結果**:
    - **AlfWorld**（134 タスク, 12 trials）: ReAct + Reflexion が 130/134 解決、ReAct ベースラインを絶対値 22% 上回る。ベースラインは trial 6–7 で頭打ち。self-evaluation は LLM による binary classification と、同一行動・同一応答が 3 サイクル超または 30 action 超で reflect する hand-written heuristic の 2 種を実装。
    - **HotPotQA**（100 問）: ベースライン CoT/ReAct/CoT(GT) は temperature 0.7 で再試行しても確率的に改善できないのに対し、Reflexion は約 20% 改善。CoT(GT) で 14% 改善。ablation で self-reflection はエピソード記憶 (EPM) だけよりさらに +8% absolute（HotPotQA ablation 図, fig:reasoning:hotpotqa c）。
    - **コード生成（pass@1）**（Table `tbl:programming:success`）: HumanEval Python 91.0（GPT-4 80.1 SOTA を上回る）, HumanEval Rust 68.0（vs 60.0）, MBPP Rust 75.4（vs 70.9）, LeetcodeHard Python 15.0（vs 7.5）. ただし **MBPP Python は 77.1 で GPT-4 ベースライン 80.1 に劣る**。著者は、MBPP Python の自己生成テスト false positive 率が 16.3% と高いこと（HumanEval Python は 1.4%, Table `tbl:programming:failures`）を分析している。
    - **他モデルでの検証**（Appendix A）: CoT(GT) + text-davinci-003 で 0.60→0.77、gpt-3.5-turbo 0.57→0.71、gpt-4 0.68→0.80。ReAct + gpt-4 は 0.39→0.51。一方 **starchat-beta では HumanEval pass@1 が 0.26→0.26 と改善しない** → self-correction は強い大規模モデルの emergent 能力と主張。
    - **コード ablation**（HumanEval Rust 50 hardest, Table `tbl:programming:ablation`）: base 0.60 / test gen 抜き 0.52 / self-reflection 抜き 0.60 / フル Reflexion 0.68。著者は、test generation と self-reflection の協調を検証する ablation として提示している。
- **貢献**: (1) 重み更新なしで言語エージェントを強化する verbal RL の枠組みを定式化（policy = LLM weights + memory）、(2) decision-making / reasoning / coding という異質な 3 領域で同一枠組みが効くことを実証、(3) **LeetcodeHardGym**（GPT-4 pretraining cutoff 2022-10-08 以降の Leetcode hard 40 問 × 19 言語）を新規公開、(4) feedback source（環境/heuristic/LLM/unit test）・記憶構造・モデルサイズに対する網羅的 ablation。

## Takeaway（自分にとっての要点）

- 重要構図: **policy を「LLM 重み + テキスト記憶」と再定義**することで、勾配なしで policy iteration を回している。記憶はただのバッファでなく「次回プロンプトに刺さる semantic gradient」として位置付けられている点が地味に効いている。
- Evaluator の中身がタスクごとに違う（reasoning は EM、decision-making は LLM 分類 + ヒューリスティック、code は self-generated unit tests）。**Reflexion ≒ 自前評価器 + 自己反省ループ**で、評価器の質に性能が直結する。コード生成で MBPP Python が伸びないのも評価器（テスト）の質劣化が原因と著者自身が分析（FP 16.3%）。
- AlfWorld のヒューリスティック「同じ行動・同じ応答が 3 サイクル超 or 30 action 超 → reflect」のような **失敗検出器の手作り感**は実装上の重要ポイント。
- **エピソード記憶単体では効果限定的**で、自然言語の一人称反省が +8% absolute を持ってくる（HotPotQA ablation, fig:reasoning:hotpotqa c）。「直近の trajectory を見せる」だけでは credit assignment が足りない、というのは強い知見。
- starchat-beta で改善ゼロ → 著者は、self-correction を指定する能力を stronger, larger models の emergent quality と述べている。
- 記憶は sliding window で $\Omega = 1–3$ と小さい。長期化したくなったら vector DB / SQL に置き換えろ、と著者自身が future work に書いている。
- coding では **false negative (テスト失敗だが本当は正しい) のほうが false positive より望ましい** — FN なら反省で立て直せるが、FP は誤ったまま提出する。テスト生成の calibration がボトルネック。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 3 領域横断（sequential decision / single-step reasoning / code）で同じ枠組みが効くことを示した。
  - HumanEval Python で当時 SOTA の GPT-4 (80.1) を 91.0 に押し上げた数字は強い。
  - LeetcodeHardGym を GPT-4 pre-training cutoff date（2022-10-08）後に release された hard-rated Leetcode 40 問で作っている。
  - ablation では「test generation omission は baseline より下がる」「starchat-beta では 0.26→0.26」など、主要主張に対する失敗例も示している。
- **弱み / 疑問**:
  - **trial を増やす構造なので計算コストが大きい**（評者補足）。Table `tbl:programming:success` は trial 数や token 量を揃えた比較ではないため、同じ token 予算で別の sampling / selection 手法を回した場合との優劣は TeX 中には示されていない。
  - **Evaluator が間違うと反省も間違う**問題に対する根本対策はなく、MBPP Python での FP 16.3% という具体的な失敗例が出ている。コード以外でも、HotPotQA の EM grading が意味的に同じ別表現を拒否する可能性は残る（評者補足）。
  - 「self-reflection は emergent quality of stronger, larger models」と述べているが、プログラミングでの追加検証は starchat-beta の HumanEval Python だけ。サイズ・学習データ・指示追従能力のどれが効いているのかは TeX 中には切り分けられていない。
  - 著者自身が limitations で認めている: (a) 言語版でも policy optimization は **non-optimal local minima** に陥る、(b) 記憶が sliding window で長期化しない、(c) test-driven 評価は **非決定的関数・API 依存関数・ハード依存・並行処理**を扱えない。
  - 「memory size $\Omega$ を 1–3 にした」理由は max context LLM limitations と説明されているが、最適値スイープは TeX 中には示されていない。
  - 評価数は HotPotQA 100 問、AlfWorld 134 environments、LeetcodeHard 40 問。より大きいサンプルで同じ傾向が出るかは TeX 中には示されていない（評者補足）。
  - self-reflection の「質」を直接測る定量指標は TeX 中には示されていない。
- **次に試したいこと**:
  - 同じ予算で **Reflexion vs CodeT / stochastic beam search** の cost-accuracy を比較する（評者補足; CodeT と beam search は Related work で言及）。
  - Evaluator のキャリブレーションを改善し、自己生成 unit test の FP 率を下げたとき MBPP Python が GPT-4 baseline を超えるか検証する（評者補足）。
  - Reflexion の経験メモリを、著者が future work として挙げる vector embedding databases / SQL databases に置き換える（評者補足）。
  - heuristic Evaluator（行動ループ検出など）を learned にしたとき AlfWorld の 130/134 がさらに伸びるか検証する（評者補足）。

## Notes / Quotes

- 「This self-reflective feedback acts as a `semantic' gradient signal by providing the agent with a concrete direction to improve upon」(Introduction) — semantic gradient という比喩が論の核。
- Reflexion の 3 モデル: Actor $M_a$ / Evaluator $M_e$ / Self-Reflection $M_{sr}$（§3）。policy $\pi_\theta$, $\theta = \{M_a, mem\}$ と明示されている。
- AlfWorld の失敗検出ヒューリスティック: 同じ行動 → 同じ応答が 3 サイクル続く、または 30 step 超で self-reflect（§4.1）。
- HotPotQA ablation（fig:reasoning:hotpotqa c）: CoT(GT) → +EPM → +Self-reflection で +8% absolute。
- Programming pass@1（Table `tbl:programming:success`）: HumanEval PY 91.0, HumanEval RS 68.0, MBPP PY 77.1（GPT-4 80.1 に負け）, MBPP RS 75.4, LeetcodeHard PY 15.0。
- 失敗分析（Table `tbl:programming:failures`）: MBPP PY の FP=0.16, FN=0.59 / HumanEval PY の FP=0.01, FN=0.40。
- Code ablation（Table `tbl:programming:ablation`, HumanEval Rust hardest 50）: 0.60 / 0.52 / 0.60 / 0.68。
- 他モデル評価（Appendix A）: CoT(GT) + GPT-4 0.68→0.80、ReAct + GPT-4 0.39→0.51、starchat-beta 0.26→0.26。
- 著者の限界認識: 「local minima に陥る」「sliding window memory は将来 vector DB / SQL に拡張すべき」「非決定的・API 依存・並行処理関数では test-driven が効かない」（§5 Limitations）。
- Broader impact: verbal RL は black-box policy の解釈性問題に光を当てうる（§7）。
- (verified 2026-05-20) Sutton & Barto を 1998 → 2018 に修正 (main.bbl の bibitem [Sutton and Barto, 2018]{Sutton1998}, 該当書籍は第2版 2018)。
- (verified 2026-05-20) HotPotQA ablation の図参照を「Fig. 3c」から具体的な図番号なしのラベル参照表現に変更 (TeX 中に明示的な図番号は無く、\ref{fig:reasoning:hotpotqa} のラベル参照のみ存在)。
- (verified 2026-05-27) venue/year を TeX で確認できる neurips_2023.sty preprint 使用に限定し、NeurIPS 2023 断定を削除 (main.tex)。
- (verified 2026-05-27) 後続研究の増加、Tree-of-Thoughts、Llama-3-8B、反省ログ蒸留、starchat-beta の小型 OSS 断定など TeX / main.bbl で裏取りできない固有主張を削除または評者補足に変更 (main.tex, main.bbl)。
- (verified 2026-05-27) programming 表の参照を TeX label ベースに修正し、数値を Table `tbl:programming:success` / `tbl:programming:failures` / `tbl:programming:ablation` と照合 (main.tex)。

## Related Papers

- Yao+ 2023, **ReAct** — Reflexion の Actor として直接利用。長軌跡の thought-action 生成 baseline。
- Wei+ 2022, **Chain-of-Thought** — reasoning タスクで Actor として利用。
- Madaan+ 2023, **Self-Refine** — 単一生成タスク向けの自己改善（記憶・隠れ制約・decision making なし）。
- 関連する reasoning / decision-making 手法: Pryzant+ 2023（prompt 自動最適化）, Paul+ 2023 Refiner（critic fine-tune）, Xie+ 2023（stochastic beam search）, Kim+ 2023（retry pattern）, Goodman+ 2023（meta qualitative eval）。
- Chen+ 2022, **CodeT** — self-generated unit tests による code 評価。
- Le+ 2022, **CodeRL** / Chen+ 2023, **Self-Debugging** — RL / debugging 系コード生成。ground truth テストに依存する点で pass@1 適格性なし、と著者は差別化。
- Li+ 2022, **AlphaCode** — hidden test での評価。
- Shridhar+ **ALFWorld**, Yang+ **HotPotQA**, Chen+ **HumanEval**, Austin+ **MBPP**, Cassano+ **MultiPL-E** — 評価環境。
- Brooks+ 2022 — in-context policy iteration の先行研究、memory コンポーネントの着想元。
- Sutton & Barto 2018（『Reinforcement Learning: An Introduction』）— credit assignment problem の参照。
