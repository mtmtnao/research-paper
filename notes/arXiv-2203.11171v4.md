# Self-Consistency Improves Chain of Thought Reasoning in Language Models

- arXiv: https://arxiv.org/abs/2203.11171
- source: ../papers/arXiv-2203.11171v4/
- authors: Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (Google Research, Brain Team)
- venue / year: ICLR 2023
- tags: [LLM, reasoning, decoding, chain-of-thought, prompting]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: Chain-of-Thought (CoT) prompting は LLM の多段推論を改善するが、デフォルトの **greedy decoding** は単一の推論経路に固定され、途中のミスや局所最適に弱い。Verifier (Cobbe+ 2021) や re-ranker (Thoppilan+ 2022) は追加の教師信号や訓練を要する。
- **手法**: **Self-consistency** という新しい decoding 戦略。CoT prompt をそのまま使い、(1) decoder からサンプリングで多様な reasoning path $(\mathbf{r}_i, \mathbf{a}_i)$ を $m$ 個生成、(2) reasoning path を周辺化し、最終回答 $\mathbf{a}_i$ について **majority vote** を取る。追加学習・追加 verifier・人手アノテーション全て不要。同一の事前学習モデルの上で動くため "self-ensemble"。Table 1 で 5 種類の aggregation 戦略を比較し、PaLM-540B では **majority vote と normalized weighted sum がほぼ同等**（GSM8K でそれぞれ 74.4 / 74.1）、**normalized weighted avg だけが崩壊**（22.1）と示し、本文では実装が単純な majority vote を採用。
- **結果**: 4 モデル (UL2-20B, LaMDA-137B, PaLM-540B, GPT-3 code-davinci-001/002) × 多数の reasoning タスクで CoT に対して一貫して改善。PaLM-540B での代表ゲイン: GSM8K 56.5→74.4 (+17.9), SVAMP 79.0→86.6 (+7.6), AQuA 35.8→48.3 (+12.5), MultiArith 94.7→99.3 (+4.6), StrategyQA 75.3→81.6 (+6.3), ARC-c 85.2→88.7 (+3.5)。GPT-3 code-davinci-002 では GSM8K +17.9, SVAMP +11.0, AQuA +12.2, ARC-c +3.9。CoT が standard prompting より悪化する NLP タスク（ANLI R1/R2/R3, e-SNLI, RTE）でも self-consistency は両方を上回る（例: e-SNLI 81.0→88.4, RTE 79.1→86.3）。Sample-and-rank, beam search, prompt permutation/multi-prompt ensemble すべてに勝つ（LaMDA-137B GSM8K で 40 path SC=27.7 vs prompt-permutation ensemble=19.2）。imperfect prompt (中間数値をランダムに壊した CoT) でも 14.9→23.4 と回復、zero-shot CoT との組み合わせは 43.0→69.2 (PaLM-540B GSM8K)。10 runs × 40 samples 平均、temperature は UL2/LaMDA で T=0.5, k=40、PaLM で T=0.7, k=40、GPT-3 で T=0.7 (no top-k)。
- **貢献**: (1) 教師信号も追加モジュールも不要な "sample-and-marginalize" decoding を提案、(2) reasoning 系統の幅広いベンチマーク (AddSub, MultiArith, ASDiv, AQuA, SVAMP, GSM8K, CSQA, StrategyQA, ARC-e/c, Letter, Coinflip) で当時の SoTA を多数更新、(3) decoding 多様性が beam search よりも reasoning 精度に効くこと、aggregation 重み付けはほぼ majority vote と等価であることを実証、(4) 「最終回答の一致率 = consistency」を **モデルの不確実性指標** として使える（Figure 7: GSM8K で consistency と accuracy が高相関）と示唆。

## Takeaway（自分にとっての要点）

- **「サンプリング多様性 × 周辺化」が再ランキングより効く**: ベイズ的に解釈すれば $p(\mathbf{a}\mid x)=\sum_\mathbf{r} p(\mathbf{r},\mathbf{a}\mid x)$ を $\mathbf{r}$ で周辺化しているだけ。sample-and-rank が「最尤の 1 経路」しか拾わないのに対して、SC は経路を捨てて回答だけを多数決にするので「異なる正しい推論 → 同じ答え」が増幅される。
- **重み付き集約と単純多数決がほぼ等価**: Table 1 で normalized weighted sum 74.1 vs majority vote 74.4。著者は「LLM は各経路の尤度を区別できるほど calibration されていない」と注釈で書いており、これは self-consistency が高 calibration を要求しないという実務上の強み。
- **beam search は逆効果**: UL2-20B AQuA で beam size を増やすと top-beam の精度がむしろ単調減少 (23.6→19.3→16.1→15.0→10.2)。理由は多様性が落ちるから。reasoning では「正解にたどり着く幾つもの異なる経路を見たい」のであって、最尤の 1 経路の精緻化ではない。
- **CoT が逆効果な NLP タスクでも CoT+SC は standard prompt を超える**（Table 4）。CoT を「常に rationale を要求する一貫した枠組み」として再評価する根拠になる。
- **consistency 自体が confidence 推定**: Figure 7 で agreement 率と accuracy が高相関。verifier を別途学習する代わりに「多数決の票差」をそのまま不確実性スコアにできる。
- **imperfect prompt と zero-shot CoT で大きく回復**: prompt engineering の手間に対するロバスト性が示されており、prompt の質に過敏な多くの手法と差別化される。
- **後続への直接の影響**: ToT, RAP, debate (arXiv-2305.14325) などの「複数推論枝を集約する」系統の元ネタとして広く参照される。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 完全に prompt 層・decoding 層に閉じており、verifier・fine-tuning・追加注釈なしで動く。実装数行で再現できる単純さは大きい。
  - 4 モデル × 約 12 タスクという網羅性。スケール効果も検証されていて、大モデルほどゲインが大きい（UL2-20B で +3〜6% に対し LaMDA/GPT-3/PaLM では +9〜23%）という観察は scaling 仮説と整合的。
  - beam search・sample-and-rank・prompt ensemble・model ensemble の全てとの比較を載せており、「単に多数決を取っただけ」批判への先回りが効いている。
  - aggregation 戦略を Table 1 で 5 種類並べて比較し、計算量最小の majority vote を選んだ判断の透明性。
- **弱み / 疑問**:
  - 推論コストが path 数 × runs 倍になる（著者も Conclusion で limitation として明記）。Table 1 以降ほぼ全て 40 paths × 10 runs であり、現実的な予算下での精度/コスト pareto は十分には示されていない。実用では path 数を 5〜10 に落とすことが推奨されているが、その点の定量比較は Figure 2 に留まる。
  - **回答が固定集合に属するタスクに限られる**（著者自身が §2 末尾で認めている）。自由生成タスクへの拡張は "consistency metric を定義できれば" と仮説的にとどまり、本論では未検証。
  - 著者が limitation として認めている「nonsensical / non-factual な reasoning path」: StrategyQA の例で人口数値が架空、にもかかわらず最終回答は当たる、というケースを挙げており、**rationale の faithfulness は保証されない**。consistency が高くても rationale を信用できるとは限らない。
  - confidence 指標としての consistency は GSM8K のみで Figure 7 にプロットされており、calibration 指標 (ECE, Brier) などとの定量比較が欠けている。Reflexion 等の他の不確実性推定との比較もない。
  - LaMDA-137B と PaLM-540B が非公開モデルである点。Appendix に prompt は載っているが、モデルが API から消えた現状では完全な再現は GPT-3 / UL2 系列に限られる。
  - 多数決が**バイアスを増幅する可能性**: モデルが系統的に誤る種類の問題では、サンプル数を増やすほど誤答が安定して選ばれる。本論ではこの failure mode の体系的分析がない（誤答に高 consistency が出る例の頻度は未報告）。
- **次に試したいこと**:
  - 同じ token 予算で **CoT-SC vs Tree-of-Thoughts vs Multiagent Debate (arXiv-2305.14325)** を並べた pareto curve を引く。SC は最も "stupid simple" な baseline であるべきで、より高コストな手法がどこから優位になるかを切り分ける。
  - consistency をスカラ confidence として使い、**ECE / selective accuracy** を temperature-based confidence や p(true) prompting と比較。
  - 誤答に高 consistency が出る問題のクラスタリング（systematic error の特定）。SC は「典型的なミスを増幅する」と思われるので、その失敗モードのカタログ化。
  - sampling 多様性のための **温度以外の介入** (persona prompt, role variation, prompt permutation を 1 batch 内で混ぜる) と SC を組み合わせた場合の追加ゲイン。
  - SC で得られた多数派回答+rationale を **distillation の教師信号** として 1-shot inference の小モデルに転写（著者も future work に挙げている）。

## Notes / Quotes

- "Self-consistency leverages the intuition that complex reasoning tasks typically admit multiple reasoning paths that reach a correct answer." (Introduction)
- "we hypothesize that correct reasoning processes, even if they are diverse, tend to have greater agreement in their final answer than incorrect processes." (§2)
- "the language model regards those generations as 'similarly likely'" → calibration が悪いため重み付けが majority vote と等価になる、と脚注で説明 (§2)
- aggregation 比較 (PaLM-540B, Table 1): greedy 56.5 / majority vote 74.4 / normalized weighted sum 74.1 / unnormalized weighted avg 56.3 / normalized weighted avg 22.1（正規化平均だけ崩壊する）
- "beam search yields a lower diversity in the outputs, while in self-consistency the diversity of the reasoning paths is the key" (§3.3)
- "low consistency as an indicator that the model has low confidence; i.e., self-consistency confers some ability for the model to 'know when it doesn't know'." (§3.4)
- 制限事項: "One limitation of self-consistency is that it incurs more computation cost." / "language models can sometimes generate incorrect or nonsensical reasoning paths ... further work is needed to better ground models' rationale generations." (Conclusion)
- Ethics: rationale が non-factual なことがあるため出力を慎重に扱うべき、と明記 (Ethics Statement)
- 設定詳細: 40 samples × 10 runs 平均, 標準偏差 ≤0.5 のため Table 2/3 では省略。temperature: UL2/LaMDA T=0.5 k=40, PaLM T=0.7 k=40, GPT-3 T=0.7 no top-k。
- imperfect prompt: 中間数値だけ random に置換、最終回答だけ正解を残した CoT で 17.1→14.9→(+SC)23.4 (LaMDA-137B GSM8K)。
- zero-shot CoT (PaLM-540B GSM8K): 43.0→69.2 (+26.2)。
- (verified 2026-05-20) Summary の aggregation 比較で「unnormalized weighted avg が最悪 (22.1)」と誤っていたのを「normalized weighted avg が最悪 (22.1)」に訂正 (main.tex Table 1, L257-258: Weighted avg (unnormalized)=56.3, Weighted avg (normalized)=22.1)。
- (verified 2026-05-20) Related Papers の評価ベンチマーク列挙から「Hendrycks+ MMLU」を削除し、実際に使用されている MAWPS / ANLI / e-SNLI を追加 (main.tex §3.1 のタスク一覧、Table 5 の NLP タスク。MMLU はこの論文では使用されていない)。

## Related Papers

- Wei+ 2022, "Chain-of-Thought Prompting Elicits Reasoning..." — そのまま土台にしている。SC は CoT の decoding を差し替える形。
- Cobbe+ 2021, "Training Verifiers to Solve Math Word Problems" — verifier ベースの GSM8K SoTA。SC は教師なしでこれを上回る比較対象。
- Thoppilan+ 2022, LaMDA — re-ranker ベース。SC は re-ranker 不要。
- Kojima+ 2022, "Large Language Models are Zero-Shot Reasoners" (Zero-shot CoT) — Table 5 で組合せ実験。
- Du+ 2023, Multi-Agent Debate (arXiv-2305.14325) — SC を majority voting baseline として明示的に超えると主張する後続。
- Yao+ 2023, Tree of Thoughts — SC を「単純なフラット sampling」と位置付け、探索構造を入れた一般化。
- Holtzman+ Nucleus sampling / Fan+ top-k sampling — SC が前提とする多様サンプリング手法。
- Cobbe+ GSM8K, Patel+ SVAMP, Ling+ AQuA, Talmor+ CommonsenseQA, Geva+ StrategyQA, Clark+ ARC, Koncel-Kedziorski+ MAWPS (AddSub/MultiArith/ASDiv), Nie+ ANLI, Camburu+ e-SNLI — 評価ベンチマーク群。
