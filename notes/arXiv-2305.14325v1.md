# Improving Factuality and Reasoning in Language Models through Multiagent Debate

- arXiv: https://arxiv.org/abs/2305.14325
- source: ../papers/arXiv-2305.14325v1/
- authors: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch
- venue / year: arXiv 2023-05（TeX は neurips_2023.sty 同梱／実掲載先は TeX 中には明示なし）
- tags: [multi-agent, LLM, reasoning, factuality]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: LLM は単体で推論・事実応答をすると自信ありげに hallucination を起こす。CoT・self-consistency・reflection 等の既存手法はいずれも「単一モデル内部」の改善に閉じている。
- **手法**: 複数の LLM インスタンスを agent として並行に走らせ、各 agent が他 agent の回答を context に受け取って自分の回答を更新する「debate」を複数 round 繰り返す（"society of minds" 風）。黒箱 API のみで動作し、prompt は全タスク共通。CoT 等とも直交して併用可能。consensus prompt の「頑固さ」で収束速度を制御できる。
- **結果**: 3 agent × 2 round 設定で、Arithmetic 67.0→81.8、GSM8K 77.0→85.0、Chess Δpawn 91.4→122.9（reasoning, Table 1）、Biographies 66.0→73.8、MMLU 63.9→71.1、Chess move validity 29.3→45.2（factuality, Table 2）。reasoning では Single agent / reflection / majority voting を上回り、factuality では Single agent / reflection を上回る（factuality では「個別応答が比較不能」として majority voting は baseline から除外）。agent 数・round 数を増やすと単調に改善（Arithmetic では round 4 で頭打ち）。chatGPT × Bard の異種混合 debate でも両者単独より良い（GSM8K 20問で 11/14→17）。
- **貢献**: (1) 黒箱 LLM だけで使える multi-agent debate 法、(2) 524 人の計算機科学者経歴を使った新規 factuality ベンチマーク、(3) agent 数・round 数・prompt の "stubbornness"・summarization・persona 初期化・異種モデル混合の網羅的アブレーション。

## Takeaway（自分にとっての要点）

- debate は「正解を増幅する投票」ではなく、全 agent が最初は間違えていても議論の過程で正解に辿り着くケースがある → majority voting と本質的に違う動作。
- 不確実な事実では agent 間で答えがばらけるが、各 agent に直接「自信ある？」と聞くと全員「ある」と答える。一方で debate に晒すと意見を曲げやすく、逆に確信のある事実は説得しても曲げない → **「説得されやすさ」が calibration の代理指標**になる可能性。これは応用余地が大きい。
- prompt を「他者の意見を尊重しろ」寄りにすると早く合意するが精度は落ち、「自分の根拠を保持しろ」寄り（stubborn）にすると収束は遅いが最終精度は上がる。RLHF された LLM は元々 agreeable すぎる点に注意。
- agent 数が増えると context が膨らむので、他 agent の応答を一度 LLM で summarize して渡すと精度も改善する（concat より良い）。
- MMLU では agent ごとに professor/doctor/mathematician 等の persona を当てると 71.1→74.2。同一 prompt より「異なる初期視点」のほうが有効。
- debate 出力を distillation の教師信号として使えば self-improvement loop になる、という discussion は実用上の本筋（推論コストの言い訳と表裏一体だが）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 黒箱前提・prompt 共通という制約が強く、誰でも明日から再現できる。
  - reasoning と factuality を同じ機構で同時に改善した点、特に factuality で reflection が逆効果（MMLU 63.9→57.7）なのに debate は効くという対比は説得力がある。
  - 異種モデル混合（chatGPT × Bard）が効くなら、将来的に小モデル群を集めて大モデル相当の精度を得る方向の根拠になる。
- **弱み / 疑問**:
  - 計算コストが agent 数 × round 数で乗算される。Table 1/2 の比較は token 量を揃えていないので、同じ予算を CoT + self-consistency に振った場合との fair comparison が抜けている。
  - 「debate が単なる多数決ではない」の主張は qualitative example 中心で、定量的に「全員不正解→正解」がどれくらいの割合で起きるかは示されていない。
  - 合意してしまった誤答は自信満々で出てくる、と limitations 自身が認めている。安全性が要る領域では使いにくい。
  - GSM8K 20問だけの異種モデル実験はサンプル小さすぎ。
  - Biographies ベンチマークが「計算機科学者 524 名」に偏っており、LLM の学習データに含まれやすい層。一般人物だとさらに崩れる可能性。
- **次に試したいこと**:
  - 「説得されやすさ」を信頼度のスコアとして数値化し、calibration（ECE）が confidence prompt より改善するか検証。
  - 同じ token 予算で self-consistency / CoT-SC / Tree-of-Thoughts と並べた pareto curve を引く。
  - debate ログを SFT データに distill して、1 agent モデルがどこまで debate 性能に近づけるか。
  - 「stubborn 側」と「agreeable 側」を意図的にミックスした非対称 debate の効果。

## Notes / Quotes

- "the purpose of our debate isn't just to amplify a correct answer -- all models can initially be wrong but arrive at the correct answer through the debate process." (experiments.tex)
- "ease of persuasion may be a method to assess factual confidence." (experiments.tex)
- consensus prompt は agent の "stubbornness" で debate 長を制御できる（method §2.2）。
- 大規模 agent 数では他者応答を summarize してから渡す（experiments）。
- 既知の限界: long debate になると LLM が直近 turn しか見なくなる、誤答に収束しても自信満々（discussion）。

## Related Papers

- Minsky, *Society of Mind* (1988) — 思想的源流。
- Kojima+ 2022, Zero-shot CoT — orthogonal に併用される baseline。
- Reflexion / Self-Refine (Madaan+ 2023) — 単一 agent 内省 baseline。
- Wang+ Self-Consistency, AlphaCode — majority voting baseline。
- Kadavath+ 2022 "Language Models (Mostly) Know What They Know" — confidence 推定の比較対象。
- Hendrycks+ MMLU, GSM8K (Cobbe+), BIG-Bench Chess State Tracking — 評価データセット。
