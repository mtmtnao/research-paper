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
  - **算数**: PaLM 540B で GSM8K 17.9→**56.9** (+39.0)、SVAMP 69.4→79.0、MAWPS 79.2→93.3、ASDiv 72.1→73.9、AQuA 25.2→35.8。外部電卓を足すと GSM8K は 58.6 まで。GPT-3 175B (text-davinci-002) でも GSM8K 15.6→46.9、Codex (code-davinci-002) は 19.7→**63.1** (+43.4)。GSM8K・SVAMP・MAWPS で当時 SOTA を更新（GSM8K 旧 SOTA 55、cobbe2021）。
  - **emergent**: CoT の利得は ~100B 未満ではほぼゼロまたは負（LaMDA 8B GSM8K は 3.2→1.6 で下がる）、100B 級から急に効き始める。これは Wei+ 2022 の emergent abilities の典型例。
  - **常識**: PaLM 540B で StrategyQA 68.6→**75.6**（旧 SOTA 69.4 超え）、Sports Understanding 84%（unaided sports enthusiast）→95.4%、Date Understanding と SayCan も改善。CSQA は伸びが小さい（78.1→79.9）。
  - **記号**: Last letter concatenation と Coin flip で PaLM 540B はほぼ 100% 達成。さらに「2語で訓練 → 3・4語で評価」の **OOD（長さ汎化）** でも CoT は標準 prompting が壊れるところを救う。
  - **ablation**: 等式のみ・ドットだけ・回答後 CoT のいずれも GSM8K では効果なし → 「中間自然言語ステップを答え前に逐次出すこと」自体が本質。
  - **頑健性**: 3 人のアノテーター A/B/C どれでも、また GSM8K train から無作為に取った exemplar でも、すべて standard prompting を大幅に上回る。exemplar 順序・個数にも比較的鈍感（coin flip だけ順序ぶれ大）。
  - **誤答分析**: LaMDA 137B GSM8K で正解 50 例中 48 例は CoT も論理的に正しい。誤答 50 例のうち 46% は minor mistake（電卓ミス・1 ステップ抜け）、54% は意味理解の重大エラー。PaLM 62B→540B でこれらの大半が解消される（§A.1）。
- **貢献**: (1) finetune ゼロ・8 exemplar だけで多段推論能力を引き出す **chain-of-thought prompting** を提案、(2) 算数・常識・記号という3領域 12 ベンチマークで、3 系統のモデル（GPT-3 / LaMDA / PaLM、加えて UL2・Codex）にわたり一貫した改善を実証、(3) これが ~100B 級でしか出ない **emergent ability** であることを示した、(4) ablation/robustness で「効くのは『答え前の自然言語多段ステップ』であり、計算量増・知識アクセス・形式表現ではない」と切り分けた、(5) coin-flip と last-letter-concatenation の合成ベンチを公開。

## Takeaway（自分にとっての要点）

- CoT は「prompt の **output** 側を拡張する」アプローチであり、instruction tuning など「input 側」拡張と直交する（§5）。後発の self-consistency や ToT もすべてこの線上にある。
- **scale との関係が決定的**: 10B 未満では CoT はむしろ性能を落とす（流暢だが論理的に破綻した chain が出るため）。手元の小モデルで効かなくても落胆せず、まず大モデルで挙動を確認するのが正しい順序。
- ablation の Equation only と Dots-only がともに失敗した事実は重要。「単に計算量を増やすトークンを挟む」「単に式に翻訳させる」では足りず、**自然言語による段階的言語化** が効く、という主張は thinking tokens / pause token 系研究と直接比較できる。
- annotator A/B/C で差はあるが全員 baseline を大きく上回る（特に GSM8K では3者ともに大差で勝つ）→ **prompt 工学に強くは依存しない**。ただし coin flip の Annotator A 99.6% vs C 71.4% のように、タスクによっては桁違いに振れる点には注意。
- 「答え前」に CoT を出すことが必要（CoT-after-answer ablation で否定済み）→ logit lens 的に見ても、**推論を前方計算に押し込む** ことが性能の源泉だと示唆される。これは現代の "reasoning model" の基礎。
- emergent ability の議論はその後 Schaeffer らから「metric の選び方アーティファクト」と反論されたが、CoT は accuracy 自体が跳ねるので、その批判は当たりにくい。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **シンプルさ**: 「prompt に8例書くだけ」で finetune した SOTA を抜く、という再現性の高さは衝撃的で、明日からどんなチームでも追試できる。
  - **網羅性**: 3 モデルファミリ × 3 ドメイン × 多数のサイズ × annotator/exemplar の頑健性チェック、と必要なアブレーションがほぼ全部入っている。著者の `\input{fables/all-lm-tables}` を見ると全モデル × 全ベンチの完全表まで公開している。
  - **切り分け**: Equation only / Dots / CoT-after-answer の3者を全部潰してから「自然言語の逐次中間ステップが本質」と結論しており、説得力が高い。
  - **OOD（length generalization）**: 記号推論で短い例から長い例へ伸ばせる、という結果は単純 memorization 説の反証として強い。
- **弱み / 疑問**:
  - 著者自身が認める limitations（§6）：(1) これが本当に "reasoning" かは未解決、(2) finetune 用に大量 CoT を作るのは依然高コスト、(3) **正しい reasoning path の保証がない**（合っているように見える CoT が誤答を出す／誤った CoT が偶然正解を出す両ケースあり）、(4) ~100B 必要なので実運用コストが高い。
  - 評価指標が exact match accuracy 中心で、**中間ステップの正しさ**はサンプル 50 件の手動チェックしかない。CoT 全体での "rationale faithfulness" の定量評価が抜けている。
  - prompt 工学非依存と言いつつ、coin flip の A 99.6%/C 71.4%、5 要素の逆順タスクは「3 人目の著者だけが解ける CoT を書けた」（§A.2）など、**特定タスクでは依然著しく prompt 依存**。論文中も "Prompt engineering still matters" と保留している。
  - **prior best との比較条件が揃っていない**: 表 \ref{tab:flagship-table} の prior best は finetuning ベース、CoT は 8-shot prompting で、訓練データ量は桁違い。「prompt だけで勝った」は政治的に強いが、apples-to-apples ではない。
  - **token コスト**: CoT は出力長が伸びるので推論コストも単純な few-shot より重い。Self-consistency と組み合わせれば更に重くなる。token 予算を揃えた比較は本論文には無い。
  - **モデル間転移の不完全さ**: 同じ prompt でも GPT-3 では CSQA/StrategyQA で CoT が効かないと自己申告しており（§A.2 最後）、汎用性は完全ではない。
  - **データ汚染リスク**: GSM8K・SVAMP は当時 web 上にあり PaLM の事前学習に混入している可能性。CoT の効果と暗記の切り分けは行われていない。
- **次に試したいこと**:
  - **CoT faithfulness 検証**: 中間ステップを改変したらどう答えが変わるか、causal intervention（Lanham+ 2023 系）で「CoT は本当に答えに使われているか」を測る。
  - **小モデルへの蒸留**: PaLM 540B の CoT 出力を distillation データにし、~10B モデルが CoT 性能をどこまで取り戻せるか（後の Magister+ 2022, Hsieh+ 2023 で実証済みだが手元で再現したい）。
  - **token 予算固定比較**: 同じ出力 token 数で「直接答え（多サンプル）」「CoT 1 回」「CoT + self-consistency」をパレートで比較。
  - **synthetic CoT 自動生成**: 著者も §A.2 で示唆している通り、CoT を LLM 自身に書かせて validation で選別するパイプライン（後の Auto-CoT, STaR と同じ方向）。
  - **失敗パターン分類の自動化**: §A.1 の 45 件手動分類を、LLM judge で大規模化して "where does CoT break" の体系的マップを作る。

## Notes / Quotes

- 「CoT が効くのは scale が大きい時だけで、10B 未満では fluent but illogical な chain が出てむしろ性能を落とす」（§3.2）。
- 「Equation only / Variable compute (dots) / CoT-after-answer は3つともベースライン同等 → CoT の本質は『答え前の中間自然言語ステップ』」（§3.3）。
- 「standard prompting は LLM 能力の **lower bound** にすぎず、CoT のような prompt 拡張で実は解ける課題が眠っている」（§4 Discussion）。
- 著者自認の limitation: "there is no guarantee of correct reasoning paths, which can lead to both correct and incorrect answers" (§6)。
- "the emergence of chain-of-thought reasoning only at large model scales makes it costly to serve in real-world applications" (§6)。
- 「3 人目の著者だけが reverse-5-list の CoT を書けた、2 人目までは何度試しても解けなかった」（§A.2）— prompt 工学が本質的に効く事例として正直に書いている。
- LaMDA 137B GSM8K 誤答 50 件のうち 46% は minor mistake（電卓・記号・1 ステップ抜け）、54% は重大な意味理解エラー（§3.2）。

## Related Papers

- Ling+ 2017 / Cobbe+ 2021 (GSM8K) — natural-language rationale で算数を解くという発想の起源、finetune 系の比較対象。
- Brown+ 2020 (GPT-3) — few-shot prompting の元祖、本論文の standard prompting baseline。
- Wei+ 2022 "Emergent Abilities of Large Language Models" — CoT 効果が scale で emergent という主張の理論的支柱、同じ著者群。
- Nye+ 2021 (Scratchpad) — Python プログラムの逐次実行予測。CoT の前身に近い finetune 版。
- Kojima+ 2022 "Let's think step by step" — Zero-shot CoT、本論文の直接の派生。
- Wang+ 2022 Self-Consistency — CoT のサンプル多数決、本論文の脚注で言及。
- Lewkowycz+ 2022 Minerva、Suzgun+ 2022 BIG-Bench Hard — CoT を前提に scientific reasoning / hard tasks を解く後続。
- Madaan+ 2023 Self-Refine、Yao+ 2023 Tree-of-Thoughts — CoT を反復・分岐に拡張する系統。
- Du+ 2023 Multi-agent Debate（このリポジトリの隣のノート arXiv-2305.14325v1）— CoT と直交して併用される後続。
