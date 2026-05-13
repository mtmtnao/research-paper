# Reflexion: Language Agents with Verbal Reinforcement Learning

- arXiv: https://arxiv.org/abs/2303.11366
- source: ../papers/arXiv-2303.11366v4/
- authors: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- venue / year: NeurIPS 2023
- tags: [LLM-agent, verbal-RL, self-reflection, code-generation, decision-making, reasoning]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: LLM ベースの言語エージェント（ReAct, SayCan, Toolformer 等）は試行錯誤から素早く学ぶ手段がない。従来 RL は大量のサンプルと fine-tuning が必要で、巨大 LLM では現実的でない。一方 in-context example だけでは credit assignment ができず、長軌跡での失敗から学べない。
- **手法**: 重みを更新せず「言語的フィードバック」で強化する **Reflexion** を提案。3 つのモデル — Actor $M_a$（ReAct/CoT で行動・テキスト生成）、Evaluator $M_e$（EM / LLM 判定 / ヒューリスティック / 自動生成ユニットテストで報酬付与）、Self-Reflection $M_{sr}$（{軌跡, スカラー報酬} を一人称の自然言語振り返りに変換）— が協調する。生成された振り返り $sr_t$ はエピソード記憶 *mem* に追記され（容量 $\Omega$ は通常 1–3）、次試行の Actor のコンテキストに渡される。これが「semantic gradient」として作用する。失敗→反省→記憶→再試行のループを Evaluator が pass を出すか max trials まで繰り返す（Algorithm 1）。
- **結果**:
    - **AlfWorld**（134 タスク, 12 trials）: ReAct + Reflexion が 130/134 解決、ReAct ベースラインを絶対値 22% 上回る。ベースラインは trial 6–7 で頭打ち。Heuristic（同一行動 3 回ループ or 30 step 超過で reflect）と LLM 判定の両方で機能。
    - **HotPotQA**（100 問）: ベースライン CoT/ReAct/CoT(GT) は temperature 0.7 で再試行しても確率的に改善できないのに対し、Reflexion は約 20% 改善。CoT(GT) で 14% 改善。ablation で self-reflection はエピソード記憶 (EPM) だけよりさらに +8% absolute（Fig. 3c）。
    - **コード生成（pass@1）**（Table 1）: HumanEval Python 91.0（GPT-4 80.1 SOTA を上回り新 SOTA）, HumanEval Rust 68.0（vs 60.0）, MBPP Rust 75.4（vs 70.9）, LeetcodeHard Python 15.0（vs 7.5）. ただし **MBPP Python は 77.1 で GPT-4 ベースライン 80.1 に劣る**。原因は MBPP Python の自己生成テスト false positive 率が 16.3% と高いこと（HumanEval Python は 1.4%, Table 2）。
    - **他モデルでの検証**（Appendix A）: CoT(GT) + text-davinci-003 で 0.60→0.77、gpt-3.5-turbo 0.57→0.71、gpt-4 0.68→0.80。ReAct + gpt-4 は 0.39→0.51。一方 **starchat-beta（小型 OSS）では HumanEval pass@1 が 0.26→0.26 と全く改善しない** → self-correction は強い大規模モデルの emergent 能力と主張。
    - **コード ablation**（HumanEval Rust 50 hardest, Table 3）: base 0.60 / test gen 抜き 0.52 / self-reflection 抜き 0.60 / フル Reflexion 0.68。テスト生成だけでも、振り返りだけでも不十分でセットで効く。
- **貢献**: (1) 重み更新なしで言語エージェントを強化する verbal RL の枠組みを定式化（policy = LLM weights + memory）、(2) decision-making / reasoning / coding という異質な 3 領域で同一枠組みが効くことを実証、(3) **LeetcodeHardGym**（GPT-4 pretraining cutoff 2022-10-08 以降の Leetcode hard 40 問 × 19 言語）を新規公開、(4) feedback source（環境/heuristic/LLM/unit test）・記憶構造・モデルサイズに対する網羅的 ablation。

## Takeaway（自分にとっての要点）

- 重要構図: **policy を「LLM 重み + テキスト記憶」と再定義**することで、勾配なしで policy iteration を回している。記憶はただのバッファでなく「次回プロンプトに刺さる semantic gradient」として位置付けられている点が地味に効いている。
- Evaluator の中身がタスクごとに違う（reasoning は EM、decision-making は LLM 分類 + ヒューリスティック、code は self-generated unit tests）。**Reflexion ≒ 自前評価器 + 自己反省ループ**で、評価器の質に性能が直結する。コード生成で MBPP Python が伸びないのも評価器（テスト）の質劣化が原因と著者自身が分析（FP 16.3%）。
- AlfWorld のヒューリスティック「同じ行動を 3 回繰り返す or 30 step 超 → reflect」のような **失敗検出器の手作り感**は実装上の重要ポイント。LLM 判定とほぼ同等に動いた。
- **エピソード記憶単体では効果限定的**で、自然言語の一人称反省が +8% absolute を持ってくる（Fig. 3c）。「直近の trajectory を見せる」だけでは credit assignment が足りない、というのは強い知見。
- starchat-beta で改善ゼロ → **自己反省は emergent property**。小型 OSS でエージェント基盤を組むときの注意点。
- 記憶は sliding window で $\Omega = 1–3$ と小さい。長期化したくなったら vector DB / SQL に置き換えろ、と著者自身が future work に書いている。
- coding では **false negative (テスト失敗だが本当は正しい) のほうが false positive より望ましい** — FN なら反省で立て直せるが、FP は誤ったまま提出する。テスト生成の calibration がボトルネック。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 3 領域横断（sequential decision / single-step reasoning / code）で同じ枠組みが効くことを示し、後続研究のテンプレになった（実際このあと「verbal RL / self-refine 系」が爆発的に増えた）。
  - HumanEval Python で当時 SOTA の GPT-4 (80.1) を 91.0 に押し上げた数字は強い。
  - LeetcodeHardGym を GPT-4 cutoff 後の問題で作っているのは benchmark contamination への自覚が見えて良い。
  - ablation が誠実：「テスト生成だけ抜くと baseline より下がる」「starchat-beta では効かない」など、自分に不利な結果も載せている。
- **弱み / 疑問**:
  - **trial を増やせば自動的に良くなる構造で計算コストが大きい**。Table 1 は trial 数や token 量を揃えた fair comparison になっていない。同じ token 予算で best-of-N sampling や CodeT を回したとき本当に勝てるかが見えない。
  - **Evaluator が間違うと反省も間違う**問題に対する根本対策はなく、MBPP Python での FP 16.3% という具体的な失敗例が出ている。コード以外（HotPotQA EM）でも、EM が semantically 同じ答えを拒否すると無駄な反省が走るはず。
  - 「self-reflection は emergent」と結論しているが、検証は starchat-beta 一つだけ。サイズ・学習データ・指示追従能力のどれが効いているのかは切り分けられていない。
  - 著者自身が limitations で認めている: (a) 言語版でも policy optimization は **non-optimal local minima** に陥る、(b) 記憶が sliding window で長期化しない、(c) test-driven 評価は **非決定的関数・API 依存関数・ハード依存・並行処理**を扱えない。
  - 「memory size $\Omega$ を 1–3 にした」根拠が context 長制約だけで、最適値スイープがない。3 が最適かどうかは未確認。
  - HotPotQA は 100 問しか評価していない。AlfWorld 134、LeetcodeHard 40 と、いずれもサンプルサイズは小さめ。
  - 反省テキストの「質」が結果を支配しているはずだが、定性例以外で反省そのものを評価する手段がない。
- **次に試したいこと**:
  - 同じ予算で **Reflexion vs Self-Consistency vs Tree-of-Thoughts vs CodeT** の pareto curve（cost-accuracy）を引く。
  - Evaluator のキャリブレーションを改善（unit test の mutation testing で FP を下げる、HotPotQA で EM の代わりに LLM judge）。FP 率を 16.3% → 5% 以下に下げたとき MBPP Python が GPT-4 baseline を超えるか検証。
  - 反省ログを SFT データに蒸留して「1 試行で Reflexion 並み」を狙う self-distillation。
  - sliding window を retrieval-based memory（embedding 検索）に置き換え、過去 100 trial 規模で性能が単調改善するか。
  - heuristic Evaluator（行動ループ検出など）を learned にしたとき AlfWorld の 130/134 がさらに伸びるか。
  - 小型 OSS（Llama-3-8B 等）で「反省を外部の強モデルに任せる」非対称 Reflexion が emergent gap を埋めるかどうか。

## Notes / Quotes

- 「This self-reflective feedback acts as a `semantic' gradient signal by providing the agent with a concrete direction to improve upon」(Introduction) — semantic gradient という比喩が論の核。
- Reflexion の 3 モデル: Actor $M_a$ / Evaluator $M_e$ / Self-Reflection $M_{sr}$（§3）。policy $\pi_\theta$, $\theta = \{M_a, mem\}$ と明示されている。
- AlfWorld の失敗検出ヒューリスティック: 同じ行動 → 同じ応答が 3 サイクル続く、または 30 step 超で self-reflect（§4.1）。
- HotPotQA ablation（Fig. 3c）: CoT(GT) → +EPM → +Self-reflection で +8% absolute。
- Programming pass@1（Table 1）: HumanEval PY 91.0, HumanEval RS 68.0, MBPP PY 77.1（GPT-4 80.1 に負け）, MBPP RS 75.4, LeetcodeHard PY 15.0。
- 失敗分析（Table 2）: MBPP PY の FP=0.16, FN=0.59 / HumanEval PY の FP=0.01, FN=0.40。
- Code ablation（Table 3, HumanEval Rust hardest 50）: 0.60 / 0.52 / 0.60 / 0.68。
- 他モデル評価（Appendix A）: CoT(GT) + GPT-4 0.68→0.80、ReAct + GPT-4 0.39→0.51、starchat-beta 0.26→0.26。
- 著者の限界認識: 「local minima に陥る」「sliding window memory は将来 vector DB / SQL に拡張すべき」「非決定的・API 依存・並行処理関数では test-driven が効かない」（§5 Limitations）。
- Broader impact: verbal RL は black-box policy の解釈性問題に光を当てうる（§7）。

## Related Papers

- Yao+ 2023, **ReAct** — Reflexion の Actor として直接利用。長軌跡の thought-action 生成 baseline。
- Wei+ 2022, **Chain-of-Thought** — reasoning タスクで Actor として利用。
- Madaan+ 2023, **Self-Refine** — 単一生成タスク向けの自己改善（記憶・隠れ制約・decision making なし）。最も近い競合。
- Shinn 系の周辺: Pryzant+ 2023（prompt 自動最適化）, Paul+ 2023 Refiner（critic fine-tune）, Xie+ 2023（stochastic beam search）, Kim+ 2023（retry pattern）, Goodman+ 2023（meta qualitative eval）。
- Chen+ 2022, **CodeT** — self-generated unit tests による code 評価。Reflexion のテスト生成部分の源流。
- Le+ 2022, **CodeRL** / Chen+ 2023, **Self-Debugging** — RL / debugging 系コード生成。ground truth テストに依存する点で pass@1 適格性なし、と著者は差別化。
- Li+ 2022, **AlphaCode** — hidden test での評価。
- Shridhar+ **ALFWorld**, Yang+ **HotPotQA**, Chen+ **HumanEval**, Austin+ **MBPP**, Cassano+ **MultiPL-E** — 評価環境。
- Brooks+ 2022 — in-context policy iteration の先行研究、memory コンポーネントの着想元。
- Sutton & Barto 1998 — credit assignment problem の参照。
