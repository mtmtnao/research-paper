# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- arXiv: https://arxiv.org/abs/2201.11903
- source: ../papers/arXiv-2201.11903v6/
- authors: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (Google Research, Brain Team)
- venue / year: NeurIPS 2022
- tags: [LLM, prompting, reasoning, chain-of-thought, emergent-ability]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 大規模言語モデルはスケールしても、算数文章題・常識推論・記号操作など「多段推論」を要するタスクで性能が頭打ちになる（GPT-3 175B でも GSM8K は 15.6%）。一方、rationale を生成させる手法は従来 finetuning か神経記号的手法に依存し、コスト高だった。標準 few-shot prompting は推論タスクではスケールに対する伸びがフラットで、限界がある。
- **手法**: few-shot prompt の各 exemplar を `〈input, chain of thought, output〉` の3つ組に拡張するだけ。手で書いた **8 個** の CoT exemplar（AQuA だけ 4 個）を全 arithmetic ベンチで使い回す。finetune は一切しない。decoding は基本 greedy。LaMDA だけ exemplar 順序 5 seed 平均。Equation only / Variable compute (dots) / CoT-after-answer の3種類の ablation で「CoT が効くのは中間ステップを自然言語で順次出すからだ」と切り分けする（§3.3）。
- **結果**:
  - **算数**: PaLM 540B で GSM8K 17.9→**56.9** (+39.0)、SVAMP 69.4→79.0、MAWPS 79.2→93.3、ASDiv 72.1→73.9、AQuA 25.2→35.8。外部電卓を足すと GSM8K は 58.6 まで。GPT-3 175B (text-davinci-002) でも GSM8K 15.6→46.9、Codex (code-davinci-002) は 19.7→**63.1** (+43.4)。GSM8K・SVAMP・MAWPS で当時 SOTA を更新（GSM8K 旧 SOTA 55; Cobbe+ 2021）。
  - **emergent**: CoT の利得は ~100B 未満ではほぼゼロまたは負（LaMDA 8B GSM8K は 3.2→1.6 で下がる）、100B 級から急に効き始める。これは Wei+ 2022 の emergent abilities の典型例。
  - **常識**: PaLM 540B で StrategyQA は標準 prompting 68.6 から CoT で旧 SOTA 69.4 を上回る（本文は 75.6%、Table `tab:all-lm-commonsense` は 77.8%）、Sports Understanding 84%（unaided sports enthusiast）→95.4%、Date Understanding と SayCan も改善。CSQA は伸びが小さい（78.1→79.9）。
  - **記号**: Last letter concatenation と Coin flip で PaLM 540B はほぼ 100% 達成。さらに「2語で訓練 → 3・4語で評価」の **OOD（長さ汎化）** でも CoT は標準 prompting が壊れるところを救う。
  - **ablation**: 等式のみ・ドットだけ・回答後 CoT のいずれも GSM8K では効果なし → 「中間自然言語ステップを答え前に逐次出すこと」自体が本質。
  - **頑健性**: 3 人のアノテーター A/B/C どれでも、また GSM8K train から無作為に取った exemplar でも、すべて standard prompting を大幅に上回る。exemplar 順序・個数にも比較的鈍感（coin flip だけ順序ぶれ大）。
  - **誤答分析**: LaMDA 137B GSM8K で、正解 50 例のうち付録では 49 例が correct logic and math、1 例が correct by chance と分類されている（本文は「2 例を除き正しい」と記述）。誤答 50 例のうち 46% は minor mistake（電卓ミス・記号対応ミス・1 ステップ抜け）、54% は意味理解や coherence の重大エラー。PaLM 62B→540B で 45 件のエラーの相当部分が解消される（§A.1）。
- **貢献**: (1) finetune ゼロ・8 exemplar だけで多段推論能力を引き出す **chain-of-thought prompting** を提案、(2) 算数・常識・記号という3領域 12 ベンチマークで、3 系統のモデル（GPT-3 / LaMDA / PaLM、加えて UL2・Codex）にわたり一貫した改善を実証、(3) これが ~100B 級でしか出ない **emergent ability** であることを示した、(4) ablation/robustness で「効くのは『答え前の自然言語多段ステップ』であり、計算量増・知識アクセス・形式表現ではない」と切り分けた、(5) coin-flip と last-letter-concatenation の合成ベンチを公開。

## Takeaway（自分にとっての要点）

- CoT は「prompt の **output** 側を拡張する」アプローチであり、instruction tuning など「input 側」拡張と直交する（Related Work）。
- **scale との関係が決定的**: 論文では、10B 未満の多くのモデルで CoT が性能を落とし、small models は fluent but illogical chains of thought を出すと記述している。
- ablation の Equation only と Dots-only がともに失敗した事実は重要。「単に計算量を増やすトークンを挟む」「単に式に翻訳させる」では足りず、**自然言語による段階的言語化** が効く、という著者主張の根拠になっている。
- annotator A/B/C で差はあるが全員 baseline を大きく上回る（特に GSM8K では3者ともに大差で勝つ）→ **prompt 工学に強くは依存しない**。ただし coin flip の Annotator A 99.6% vs C 71.4% のように、タスクによっては桁違いに振れる点には注意。
- 「答え前」に CoT を出すことが必要（CoT-after-answer ablation では baseline と同程度）という切り分けは、著者の「sequential reasoning embodied in the chain of thought is useful」という主張の根拠になっている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **シンプルさ**: finetune なしで、few-shot exemplars に CoT を足すだけで prior best を上回る設定がある。TeX は full prompts と GPT-3 API 実験を reproducibility の根拠として挙げている。
  - **網羅性**: 3 ドメイン × 多数のモデルサイズ × annotator/exemplar の頑健性チェックを行っている。著者の `\input{fables/all-lm-tables}` では全モデル × 全ベンチの表を出している。
  - **切り分け**: Equation only / Dots / CoT-after-answer の3者を全部潰してから「自然言語の逐次中間ステップが本質」と結論しており、説得力が高い。
  - **OOD（length generalization）**: 記号推論で、few-shot exemplars より長い入力に対しても CoT の scaling curve が上向きになることを示している。
- **弱み / 疑問**:
  - 著者自身が認める limitations（§6）：(1) これが本当に "reasoning" かは未解決、(2) finetune 用に大量 CoT を作るのは依然高コスト、(3) **正しい reasoning path の保証がない**（合っているように見える CoT が誤答を出す／誤った CoT が偶然正解を出す両ケースあり）、(4) ~100B 必要なので実運用コストが高い。
  - 評価指標が exact match accuracy 中心で、**中間ステップの正しさ**はサンプル 50 件の手動チェックしかない。CoT 全体での "rationale faithfulness" の定量評価が抜けている。
  - prompt 工学非依存と言いつつ、coin flip の A 99.6%/C 71.4%、5 要素の逆順タスクは「3 人目の著者だけが解ける CoT を書けた」（§A.2）など、**特定タスクでは依然著しく prompt 依存**。論文中も "Prompt engineering still matters" と保留している。
  - **prior best との比較条件が揃っていない**: 表 \ref{tab:flagship-table} の prior best は finetuning ベース、CoT は 8-shot prompting で、比較条件は apples-to-apples ではない（評者補足）。
  - **token コスト**: CoT は出力長が伸びるので、出力 token 数を揃えた比較が欲しい（TeX 中には明示されていない / 評者補足）。Self-consistency と組み合わせる場合の追加コストも、本論文では主比較の対象ではない。
  - **モデル間転移の不完全さ**: 同じ prompt でも GPT-3 では CSQA/StrategyQA で CoT が効かないと自己申告しており（§A.2 最後）、汎用性は完全ではない。
- **次に試したいこと**:
  - **CoT faithfulness 検証**: 著者が limitation として挙げる「正しい reasoning path の保証がない」点を、より大規模に評価する（評者補足）。
  - **小モデルへの誘導**: 著者が future work として挙げる「smaller models に reasoning を誘導する方法」を検証する（評者補足）。
  - **token 予算固定比較**: 同じ出力 token 数で「直接答え（多サンプル）」「CoT 1 回」「CoT + self-consistency」を比較する（self-consistency は本文で言及あり / 評者補足）。
  - **synthetic CoT 自動生成**: 著者も §A.2 で示唆している通り、CoT を LLM 自身に書かせて validation で選別するパイプラインを試す（評者補足）。
  - **失敗パターン分類の自動化**: §A.1 の 45 件手動分類を大規模化する（評者補足）。

## Notes / Quotes

- 「CoT が効くのは scale が大きい時だけで、10B 未満では fluent but illogical な chain が出てむしろ性能を落とす」（§3.2）。
- 「Equation only / Variable compute (dots) / CoT-after-answer は3つともベースライン同等 → CoT の本質は『答え前の中間自然言語ステップ』」（§3.3）。
- 「standard prompting は LLM 能力の **lower bound** にすぎず、CoT のような prompt 拡張で実は解ける課題が眠っている」（§6 Discussion）。
- 著者自認の limitation: "there is no guarantee of correct reasoning paths, which can lead to both correct and incorrect answers" (§6)。
- "the emergence of chain-of-thought reasoning only at large model scales makes it costly to serve in real-world applications" (§6)。
- 「3 人目の著者だけが reverse-5-list の CoT を書けた、2 人目までは何度試しても解けなかった」（§A.2）— prompt 工学が本質的に効く事例として正直に書いている。
- LaMDA 137B GSM8K 誤答 50 件のうち 46% は minor mistake（電卓・記号・1 ステップ抜け）、54% は重大な意味理解・coherence エラー（§3.2 / Appendix）。
- (verified 2026-05-20) §5/§4 と書かれていた節番号誤りを §7 Related Work / §6 Discussion に訂正（neurips_2022.tex 全 \section{} 列挙で確認）。数値（GSM8K 17.9→56.9 / SVAMP 69.4→79.0 / MAWPS 79.2→93.3 / AQuA 25.2→35.8 / Codex 19.7→63.1 等）、annotator A=99.6 vs C=71.4、PaLM 62B 45 件エラー分類、StrategyQA 本文 75.6% / Table 77.8%、Sports 95.4% vs unaided 84% などは fables/llm-math.tex, fables/all-lm-tables.tex および本文（neurips_2022.tex）で確認。
- (verified 2026-05-27) TeX/bbl に無い後続研究名・外部批判・データ汚染推測を削除し、StrategyQA は本文 75.6% と Table 77.8% の差を明記。LaMDA 137B GSM8K 正解 50 例分析は付録の 49 correct logic and math / 1 correct by chance と本文の「2 例を除く」記述差が見える形に修正 (neurips_2022.tex, fables/all-lm-tables.tex, neurips_2022.bbl)

## Related Papers

- Ling+ 2017 "Program induction by rationale generation: Learning to solve and explain algebraic word problems" — natural-language rationale で算数を解く先行研究。
- Cobbe+ 2021 "Training verifiers to solve math word problems" — GSM8K と finetuned GPT-3 verifier の比較対象。
- Brown+ 2020 "Language models are few-shot learners" — standard few-shot prompting baseline。
- Wei+ 2022 "Emergent abilities of large language models" — CoT 効果が scale で emergent という本文中の参照先。
- Nye+ 2021 "Show your work: Scratchpads for intermediate computation with language models" — program execution の逐次中間計算に関する関連研究。
- Wang+ 2022 "Self-consistency improves chain of thought reasoning in language models" — CoT の sampled generations 多数決として本文で言及。
- Zelikman+ 2022 "STaR: Bootstrapping reasoning with reasoning" — synthetic datasets / reasoning bootstrapping の関連研究。
