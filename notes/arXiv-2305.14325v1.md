# Improving Factuality and Reasoning in Language Models through Multiagent Debate

- arXiv: https://arxiv.org/abs/2305.14325
- source: ../papers/arXiv-2305.14325v1/
- authors: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch
- venue / year: TeX 中には明示なし（main.tex は neurips_2023.sty を preprint で使用）
- tags: [multi-agent, LLM, reasoning, factuality]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: LLM は単体で推論・事実応答をすると自信ありげに hallucination を起こす。CoT・self-consistency・reflection 等の既存手法はいずれも「単一モデル内部」の改善に閉じている。
- **手法**: 複数の LLM インスタンスを agent として並行に走らせ、各 agent が他 agent の回答を context に受け取って自分の回答を更新する「debate」を複数 round 繰り返す（"society of minds" 風）。黒箱 API のみで動作し、同じ procedure と prompt templates を調査対象タスクに使う。CoT 等とも直交して併用可能。consensus prompt の「頑固さ」で収束速度を制御できる。
- **結果**: 3 agent × 2 round 設定で、Arithmetic 67.0→81.8、GSM8K 77.0→85.0、Chess Δpawn 91.4→122.9（reasoning, Table 1）、Biographies 66.0→73.8、MMLU 63.9→71.1、Chess move validity 29.3→45.2（factuality, Table 2）。reasoning では Single agent / reflection / majority voting を上回り、factuality では Single agent / reflection を上回る（factuality では「個別応答が比較不能」として majority voting は baseline から除外）。agent 数・round 数を増やすと単調に改善（Arithmetic では round 4 で頭打ち）。chatGPT × Bard の異種混合 debate でも両者単独より良い（GSM8K 20問で 11/14→17）。
- **貢献**: (1) 黒箱 LLM だけで使える multi-agent debate 法、(2) 524 人の計算機科学者経歴を使った新規 factuality ベンチマーク、(3) agent 数・round 数・prompt の "stubbornness"・summarization・persona 初期化・異種モデル混合の分析。

## Takeaway（自分にとっての要点）

- debate は「正解を増幅する投票」ではなく、全 agent が最初は間違えていても議論の過程で正解に辿り着くケースがある、と著者は qualitative example として示している。
- 不確実な事実では agent 間で答えがばらけるが、各 agent に直接 confidence を聞くと高 confidence を返す。一方で debate では意見を変えやすく、確信のある事実では説得されにくいことから、著者は "ease of persuasion" が factual confidence 評価に使えるかもしれないと述べている。
- prompt を「他者の意見を尊重しろ」寄りにすると早く合意し、「自分の根拠を保持しろ」寄り（stubborn）にすると収束は遅いが最終精度は上がる。著者は、LLM agents が relatively "agreeable" だったことについて、instruction tuning や RLHF が一因かもしれないと述べている。
- agent 数が増えると context が膨らむので、他 agent の応答を一度 LLM で summarize して渡すと精度も改善する（concat より良い）。
- MMLU では agent ごとに professor/doctor/mathematician 等の persona を当てると 71.1→74.2。この設定では different initialization prompts による追加改善が示されている。
- debate 出力を追加学習データとして distill し、元の base model の self-improvement に戻すという discussion は、著者が計算コストへの応答として挙げている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 黒箱前提で、likelihood や gradient などの model-internal information を必要としない点は実装上の制約が小さい。
  - reasoning と factuality を同じ機構で同時に改善した点、特に factuality で reflection が逆効果（MMLU 63.9→57.7）なのに debate は効くという対比は説得力がある。
  - 異種モデル混合（chatGPT × Bard）でも、GSM8K 20問では Bard 単独 11問・chatGPT 単独 14問から joint debate 17問に改善している。
- **弱み / 疑問**:
  - 計算コストが agent 数 × round 数で増えることは著者自身も limitation として認めている。Table 1/2 では token 量を揃えた比較は示されていない（評者補足）。
  - 「debate が単なる多数決ではない」の主張は qualitative example 中心で、定量的に「全員不正解→正解」がどれくらいの割合で起きるかは示されていない。
  - 合意してしまった誤答は自信満々で出てくる、と limitations 自身が認めている。安全性が要る領域では使いにくい。
  - GSM8K 20問だけの異種モデル実験はサンプル小さすぎ。
  - Biographies ベンチマークは「well-known computer scientists」524 名に限定されている。別種の人物集合で同じ傾向が出るかは TeX 中には示されていない。
- **次に試したいこと**:
  - 「説得されやすさ」を信頼度のスコアとして数値化し、calibration が直接 confidence を聞く場合より改善するか検証（評者補足）。
  - 同じ token 予算で self-consistency / CoT-SC と並べた比較を行う（評者補足）。
  - debate ログを学習データに distill して、1 agent モデルがどこまで debate 性能に近づけるか検証（著者の self-improvement loop 議論に基づく評者補足）。
  - 「stubborn 側」と「agreeable 側」を意図的にミックスした非対称 debate の効果を調べる（評者補足）。

## Notes / Quotes

- "the purpose of our debate isn't just to amplify a correct answer -- all models can initially be wrong but arrive at the correct answer through the debate process." (experiments.tex)
- "ease of persuasion may be a method to assess factual confidence." (experiments.tex)
- consensus prompt は agent の "stubbornness" で debate 長を制御できる（method §2.2）。
- 大規模 agent 数では他者応答を summarize してから渡す（experiments）。
- 既知の限界: long debate になると LLM が直近 turn しか見なくなる、誤答に収束しても自信満々（discussion）。
- (verified 2026-05-26) 「prompt は全タスク共通」を、TeX の "same methodology and prompt templates" と Appendix の task-specific prompt table に合わせて修正 (text/abstract.tex, text/introduction.tex, tables/prompt_settings.tex)
- (verified 2026-05-26) 「網羅的」「誰でも明日から再現」「小モデル群で大モデル相当」など TeX 根拠より強い表現を削除または TeX で確認できる表現に修正 (text/introduction.tex, text/experiments.tex, text/discussion.tex)
- (verified 2026-05-26) Critical Thoughts の評者独自提案は TeX 事実と区別するため「評者補足」を明記 (text/experiments.tex, text/discussion.tex)
- (verified 2026-05-26) venue/year を TeX で確認できる範囲（neurips_2023.sty preprint 使用、実掲載先明示なし）に限定 (main.tex)

## Related Papers

- Minsky, *Society of Mind* (1988) — 思想的源流。
- Kojima+ 2022, Zero-shot CoT — orthogonal に併用される baseline。
- Reflexion / Self-Refine (Madaan+ 2023) — 単一 agent 内省 baseline。
- Wang+ Self-Consistency, AlphaCode — majority voting baseline。
- Kadavath+ 2022 "Language Models (Mostly) Know What They Know" — confidence 推定の比較対象。
- Hendrycks+ MMLU, GSM8K (Cobbe+), BIG-Bench Chess State Tracking — 評価データセット。
