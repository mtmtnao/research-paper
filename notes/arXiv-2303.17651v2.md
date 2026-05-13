# Self-Refine: Iterative Refinement with Self-Feedback

- arXiv: https://arxiv.org/abs/2303.17651
- source: ../papers/arXiv-2303.17651v2/
- authors: Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark
- venue / year: NeurIPS 2023 (arXiv v2, 2023)
- tags: [LLM, self-refinement, prompting, iterative-feedback, in-context-learning]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: LLM は単発生成では複雑な要件（対話応答の多目的性、コード可読性など定義の難しい目標）を一発で満たせない。既存の refinement 手法は (a) ドメイン特化の supervised refiner を訓練する、(b) 外部 reward / 人手アノテーションに依存する、のいずれかで、追加学習・追加データなしで汎用に使える方法が無い。
- **手法**: **Self-Refine** ― 同じ単一 LLM $\mathcal{M}$ が「生成 → 自己フィードバック → 自己リファイン」を反復する。3 つの few-shot prompt $(p_{gen}, p_{fb}, p_{refine})$ だけで動き、追加学習・RL・supervised data は不要。$y_0 = \mathcal{M}(p_{gen}\|x)$ を出した後、$fb_t = \mathcal{M}(p_{fb}\|x\|y_t)$ で**actionable かつ specific**（具体的に直すべき箇所と方法を含む）なフィードバックを得て、$y_{t+1} = \mathcal{M}(p_{refine}\|x\|y_0\|fb_0\|\dots\|y_t\|fb_t)$ で履歴付きリファインする（Algorithm 1, Eq. (1)-(4)）。停止条件は最大反復回数（実験では up to 4 iterations）か feedback 内のスカラー stop score。温度 0.7。
- **結果**: 7 タスク（Dialogue Response Generation, Code Optimization, Code Readability Improvement, GSM8K (Math Reasoning), Sentiment Reversal, Acronym Generation, CommonGen-Hard）× 3 ベース LLM（GPT-3.5 text-davinci-003, ChatGPT gpt-3.5-turbo, GPT-4）で評価。Table 1 では全 21 セル中 20 セルでベースを上回り、平均約 20% 絶対改善。代表例（GPT-4 base → +Self-Refine）: Dialogue Response 25.4→74.6 (+49.2)、Sentiment Reversal 3.8→36.2 (+32.4)、CommonGen-Hard 15.0→45.0 (+30.0)、Code Readability 27.4→56.2 (+28.8)、Acronym 30.4→56.0 (+25.6)、Code Optimization 27.3→36.0 (+8.7)、GSM8K 92.9→93.1 (+0.2)。GSM8K だけほぼ無改善（後述）。Code Optimization では Codex でも同様の傾向（§sota）。
- **貢献**: (1) 単一 LLM の test-time iterative self-refine フレームワークを提示、(2) supervision-free / refiner-free / multi-aspect / iterative の 4 条件を同時に満たす唯一の手法であると整理（Table 2: PEER, CodeRL, Self-correction, Reflexion 等と差別化）、(3) CommonGen-Hard と Acronym Generation の 2 タスクを新規導入、(4) feedback 品質・反復回数・サンプル数比較・モデルサイズ依存性などのアブレーション。

## Takeaway（自分にとっての要点）

- **同じモデル**だけで生成・批評・修正を全部回せる、追加学習も外部 reward も不要というのが本質。プロンプト 3 本 + ループだけで動くので、API しか叩けない環境にそのまま乗る。
- **「actionable かつ specific」が feedback の核**。Table 3 のアブレーション（Sentiment Reversal: 43.2 vs generic 31.2 vs no-feedback 0、Code Opt: 27.5 vs 26.0 vs 24.8、Acronym: 56.4 vs 54.0 vs 48.0）が示すのは、「Improve the efficiency」のような generic な自己反省はほぼ効かず、「この for ループを (n(n+1))/2 に置き換えろ」レベルの具体性が必要、ということ。Reflection 系手法を組むときの実装指針として強い。
- 反復による gain は **early iterations に集中**（Fig. 4: Constrained Gen は $y_0\to y_1$ で +11.3、$y_1\to y_2$ で +6.4、$y_2\to y_3$ で +3.0 と diminishing returns）。コスト最適化なら 1-2 round で打ち切るのが現実解。
- **GSM8K の停滞 (92.9→93.1, +0.2)** は重要。ChatGPT は 94% の事例で feedback を "everything looks good" と返してしまう。つまり「自分のミスを自分で見つける」能力がそもそも無いタスクでは self-refine は機能しない。外部検証器（compiler, calculator, unit test）と組むのが本筋。
- **1 vs $k$ sampling 比較**: $k=4$ 個独立にサンプルさせて best-of-k と比べても、Self-Refine がそれら全部より好まれる（§Analysis）。つまり「多数生成して選ぶ」より「1 本を直す」方が効率がよい。
- **弱いモデル (Vicuna-13B) では破綻**: feedback フォーマットを守れず、oracle feedback を渡しても refine 指示に従えない。**few-shot 追従能力＋指示追従能力の二つが揃った強モデル前提**であることが明示されている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 黒箱 API のみ・prompt 3 本だけで再現可能。誰でも明日からそのまま使える簡潔さは強い。
  - 7 タスク × 3 モデルの広い grid を一貫した設定（最大 4 iter, T=0.7）で回し、20/21 セルで改善を出した網羅性。
  - feedback quality と iteration の効果を分離したアブレーション（Table 3, Fig. 4）と、定性分析（失敗 70 事例中 33% が feedback の位置誤り、61% が修正案の誤り、refiner 自体の失敗はわずか 6%）まで踏み込んでいて「どこがボトルネックか」を診断している。
  - Table 2 の整理が良い: 関連手法（PEER, Self-critique, CodeRL, Self-correction, Reflexion, Augmenter, Re³）と (supervision-free refiner / supervision-free feedback / multi-aspect / iterative) の 4 軸でマトリクス比較し、本研究のニッチを明確化している。
- **弱み / 疑問**:
  - **計算コストが iter 数で線形に増える**が、Table 1 は base 1 回 vs Self-Refine $\leq$4 回の比較で、同じ token 予算を majority-vote / self-consistency / best-of-k に投じた場合との fair comparison が無い（§Analysis に 1 vs $k=4$ の sampling 比較はあるが、「単一の prompt をそのまま $k$ 回引いた best」を選ぶ簡易設定で、self-consistency 等の正式比較ではない）。
  - **GSM8K でほぼ改善しない**ことを著者自身が認め、「外部信号があれば +5% 以上」と注釈しているが、これはそもそも純 self-refine の限界を示している。事実検証や算術のように「正答可否を自己判断できないタスク」では本手法は無力。
  - **多くの評価が GPT-4-as-judge**。Dialogue / Sentiment / Acronym / Code Readability は GPT-4-pref（人間との一致 68-82%）が主指標で、生成器と判定器を同じ系列モデルで使っている circularity がある。著者は人間 A/B も併用しているがサブセットのみ。
  - 著者明示の limitations: (a) **強い few-shot/instruction-following 能力が前提**で弱いモデル（Vicuna-13B）では動かない、(b) GPT-3.5/ChatGPT/GPT-4/Codex はクローズドで pretraining 詳細不明、(c) **英語のみ**、(d) 有害テキスト生成への悪用に対するガードは無い。
  - 「fb_t に stop score を埋め込んで打ち切る」と書かれているが、その自己停止判定の精度（早すぎ/遅すぎ）の定量評価が見当たらない。
  - Acronym と CommonGen-Hard は本論文発の自作タスクで、難易度設計が手法に有利でないかという独立検証はまだ。
- **次に試したいこと**:
  - 同 token 予算で self-consistency / Tree-of-Thoughts / Reflexion / multi-agent debate との pareto curve（精度 vs total tokens）を引く。
  - GSM8K で feedback 役だけ外部検証器（Python 実行 / calculator）に置き換えるハイブリッド構成を組み、純 LLM-feedback と分離評価。
  - feedback prompt の「actionable・specific」要件を満たしているかを LLM で自動判定させ、不合格時に feedback だけリトライする 2 段構成で改善するか。
  - Self-Refine の対話ログを SFT データに distill して、小モデル単発生成で大モデル + Self-Refine の出力品質にどこまで迫れるか（実用化の本命）。
  - 「stop score」の校正実験：iteration 早期に "no further refinement needed" と判断したケースの真の品質を測り、停止判定の precision/recall を出す。

## Notes / Quotes

- "the same LLM as the generator, refiner and the feedback provider" (abstract) — 本手法の核となる主張。
- Algorithm 1: refinement 時に **過去の全 $(y_i, fb_i)$ 履歴**を prompt に concat する設計（Eq. 4）。「同じミスを繰り返さない」ためと著者は説明。長い iter で context が膨らむ点は議論されていない。
- "we use greedy decoding with a temperature of 0.7" (§Instantiating) — 形式上は矛盾した記述だが原文ママ。実質はサンプリング温度 0.7。
- ChatGPT の GSM8K feedback の 94% が "everything looks good"（§Results）→ self-feedback の決定的失敗モード。
- 失敗の内訳（70 事例, §Qualitative Analysis）: feedback の位置誤り 33% / 修正案誤り 61% / refiner の実装ミス 6%。**問題のほぼ 94% が feedback 側**にある。
- 成功事例の 33% は「partially incorrect feedback」でも refiner が補正している → refiner には一定の robustness。
- Table 2: Reflexion は (supervision-free feedback ✓/✗, multi-aspect ✗, iterative ✗) として整理されており、Self-Refine の差別化点は **multi-aspect feedback** と **iterative** の同時達成にあると著者は主張。
- Vicuna-13B は few-shot prompt のフォーマット追従に失敗し、oracle feedback を与えても refine が機能しない（§Analysis "Does Self-Refine work with weaker models?"）。

## Related Papers

- Welleck+ 2022 "Self-Correction" — supervised refiner、対比対象。
- Schick+ 2022 PEER — Wikipedia 編集ベースの supervised refiner、Table 2 で比較。
- Saunders+ 2022 Self-critique — モデルが自身の出力を批評する系譜。
- Shinn+ 2023 Reflexion — prompted refiner、Table 2 で最も近い同時代研究。
- Peng+ 2023 LLM-Augmenter, Yang+ 2022 Re³ — prompted refinement の先行例。
- Le+ 2022 CodeRL, Liu+ 2022 Rainier, Lu+ 2022 Quark — scalar reward / RL ベースの代替路線。
- Cobbe+ 2021 GSM8K, Lin+ 2019 CommonGen, Mehri & Eskenazi 2020 FED (Dialogue), Zhang+ 2015 Yelp (Sentiment) — 評価データセット。
- Pie (Madaan+) — Code Optimization タスクの出典で baseline。
- Chiang+ 2023 Vicuna — 弱モデル比較対象。
- Brown+ 2020 GPT-3 — few-shot in-context learning の基礎。
