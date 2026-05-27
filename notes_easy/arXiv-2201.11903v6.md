# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（プロンプト内の中間推論が大規模言語モデルの多段推論を引き出す実証）

- arXiv: https://arxiv.org/abs/2201.11903
- 一次ソース: ../papers/arXiv-2201.11903v6/
- 正規ノート: ../notes/arXiv-2201.11903v6.md

---

## 一言で言うと

few-shot prompting の exemplar を従来の `input--output` ペアから `〈input, chain of thought, output〉` の 3 つ組にするだけで、十分大きい言語モデルに算数・常識・記号操作の多段推論を促せる、という実証論文。PaLM 540B では GSM8K が standard prompting 17.9% から chain-of-thought prompting 56.9% へ上がる一方、著者はこの効果が主に `~100B parameters` 級で現れる emergent ability だと主張する。

## 何を議論する論文か

- **問題設定**: 大規模言語モデルはスケールすると多くの NLP タスクで良くなるが、arithmetic reasoning、commonsense reasoning、symbolic reasoning のような多段推論では、モデルサイズを増やすだけでは高性能になりにくい。標準的な few-shot prompting は `input--output pairs` を見せるだけなので、推論を要するタスクで性能が伸びにくい、というのが出発点である。
- **対象範囲 / 仮定**: 対象は off-the-shelf language models への prompting-based inference であり、本文は `No language models were finetuned` と明記する。実験モデルは GPT-3、LaMDA、PaLM、UL2 20B、Codex。decoding は基本 greedy decoding で、LaMDA は exemplar 順序を 5 seed で平均し、他モデルは計算節約のため単一 exemplar order を報告する。
- **既存研究との差分**: 既存の rationale-augmented training / finetuning は高品質 rationale を大量に作るコストがある。Brown et al. 2020 型の standard few-shot prompting は追加学習不要だが推論タスクに弱い。この論文は、訓練データや勾配更新ではなく、prompt の出力側に自然言語の中間ステップを入れる点で差分がある。
- **この論文で答えたい問い**: 数個の chain-of-thought demonstrations だけで多段推論能力を引き出せるか。その効果はモデルスケール、タスク種類、prompt の書き方、exemplar の違いに対してどの程度頑健か。効果の理由は単なる式生成、追加 token による variable compute、事前学習知識の活性化ではなく、答えの前に自然言語の中間推論を生成すること自体なのか。

## 背景と前提

- **few-shot prompting** は、テスト入力の前に少数の exemplar を置き、同じ形式で出力させる方法である。この論文の standard prompting baseline は、質問と最終答えだけからなる `input--output pairs` である。
- **chain of thought** は、本文で `a series of intermediate natural language reasoning steps that lead to the final output` と定義される。ここでは「説明を後から付ける」自然言語説明ではなく、最終答えの前に生成される中間推論列である。
- **rationale / natural language intermediate steps** は先行研究にもあり、Ling et al. 2017 や Cobbe et al. 2021 は算数文章題で rationale を使う。ただしそれらは training from scratch や finetuning を使うのに対し、本論文は prompting だけで行う。
- **モデルスケール** が中心的な変数である。LaMDA は本文では 422M, 2B, 8B, 68B, 137B と書かれ、実験表では最小モデルが 420M と表記される。PaLM は 8B, 62B, 540B、GPT-3 系は text-ada-001, text-babbage-001, text-curie-001, text-davinci-002 を用いる。GPT-3 のサイズ対応は TeX では `presumably correspond to InstructGPT models of 350M, 1.3B, 6.7B, and 175B parameters` と慎重に書かれている。
- **評価指標** は主に accuracy / solve rate (%) である。Table `tab:flagship-table` は arithmetic について `All metrics are accuracy (%)` と明記し、図の軸は `Solve rate (%)` で統一されている。

## 提案手法

### コアアイデア

提案手法は、few-shot prompting の各 exemplar の答え側に、最終答えへ至る自然言語の中間ステップを入れるだけである。本文の中心表現は `prompt that consists of triples: 〈input, chain of thought, output〉` であり、モデル自体のパラメータ更新、タスク専用 head、外部 symbolic solver は使わない。

著者がこの形式に期待する性質は 4 つある。多段問題を intermediate steps に分解できること、生成された chain がモデル挙動を debug する窓になること、math word problems だけでなく commonsense reasoning や symbolic manipulation にも使えること、そして十分大きな既存モデルなら few-shot exemplar に chain を含めるだけで elicited できることである。

算数では、AQuA 以外の math word problem benchmark に同じ 8 個の手書き CoT exemplar を使う。AQuA は multiple choice なので 4 個の training-set exemplars and solutions を使う。commonsense では CSQA / StrategyQA は training set から exemplar を選び、BIG-bench の Date / Sports は training set がないため evaluation set の先頭 10 例を exemplar として使い残りで評価する。SayCan は本文では six examples from the training set と説明される。symbolic reasoning では last letter concatenation と coin flip の合成タスクを作り、in-domain と OOD length generalization を評価する。

### 重要な定義・数式

この論文は新しい学習目的関数や更新式を提案していない。TeX 中の中核的な形式表現は、prompt exemplar の形と、効果が現れるモデルスケールに関する記述である。

$$
\langle \text{input}, \text{output} \rangle
$$

**式の意味**: standard few-shot prompting の exemplar 形式を表す。入力に対して最終出力だけを示し、推論の途中過程は prompt に含めない。

**記号の定義**:
- $\text{input}$ ... 解かせたい質問・問題文・命令文。
- $\text{output}$ ... 直接生成させる最終答え。算数なら数値、multiple choice なら選択肢、binary task なら yes/no など。
- $\langle \cdot,\cdot \rangle$ ... TeX では `input--output pairs` と書かれる exemplar の組を、読みやすく対で表したもの。

**この論文での役割**: 比較対象 baseline の定義である。以後の性能差は、この standard prompting と chain-of-thought prompting の差として測られる。

$$
\langle \text{input}, \emph{chain of thought}, \text{output} \rangle
$$

**式の意味**: 提案する chain-of-thought prompting の exemplar 形式である。最終答えの前に、自然言語の中間推論列を入れる。

**記号の定義**:
- $\text{input}$ ... standard prompting と同じく、問題文や命令。
- $\emph{chain of thought}$ ... `a series of intermediate natural language reasoning steps that lead to the final output`。本文では答えの後の explanation ではなく、答えの前に生成される推論過程として扱われる。
- $\text{output}$ ... chain の後に出す最終答え。

**この論文での役割**: 手法そのものの定義である。算数、常識、記号操作の全実験で、この exemplar 形式が standard prompting と比較される。

$$
\sim 100\mathrm{B}\ \text{parameters}
$$

**式の意味**: 著者が chain-of-thought prompting の効果が現れるスケールとして本文で述べる目安である。TeX では `only yields performance gains when used with models of \sim100B parameters` と書かれる。

**記号の定義**:
- $\sim$ ... おおよそ、という意味。
- $100\mathrm{B}$ ... 100 billion、すなわち約 1000 億パラメータ。
- $\text{parameters}$ ... 言語モデルの学習済みパラメータ数。

**この論文での役割**: 実験結果の解釈軸である。小さいモデルは流暢だが論理的でない chain を出して性能を落とすことがあり、CoT の利得は主に大規模モデルで現れる、という emergent ability の主張につながる。

### 実装 / アルゴリズム上の要点

- step1: 各タスクについて few-shot exemplars を用意する。standard prompting では `Question -> Answer`、CoT では `Question -> intermediate natural language steps -> Answer` にする。
- step2: off-the-shelf language model に prompt を与え、greedy decoding で出力を生成する。LaMDA では exemplar order を 5 種類に shuffle し平均する。
- step3: 最終答えを accuracy / solve rate (%) で評価する。算数では GSM8K, SVAMP, ASDiv, AQuA, MAWPS、常識では CSQA, StrategyQA, Date, Sports, SayCan、記号では Last Letter Concatenation と Coin Flip を使う。
- step4: ablation として `Equation only`, `Variable compute only`, `Chain of thought after answer` を比較する。GSM8K ではこれらが CoT の利得を説明できないことを示す。
- step5: 算数では post-hoc external calculator も試す。TeX は Python `eval` を用い、generated chain 内の equation にだけ外部電卓を適用すると説明している。

## 実験・結果

- **データセット / ベンチマーク**: arithmetic は GSM8K (N=1,319), SVAMP (N=1,000), ASDiv (N=2,096), AQuA (N=254), MAWPS の SingleOp (562), SingleEq (508), AddSub (395), MultiArith (600)。commonsense は CSQA, StrategyQA, Date Understanding, Sports Understanding, SayCan。symbolic は Last Letter Concatenation と Coin Flip で、同じ step 数の in-domain と、より長い入力の OOD を見る。
- **比較対象 / baseline**: standard prompting、prior best mostly finetuning、ablation 3 種、異なる annotator / exemplar / exemplar order / exemplar 数の robustness、Sports では unaided sports enthusiast 84% との比較もある。
- **指標**: accuracy (%) または solve rate (%)。表の数値は基本的に最終答えの正誤であり、chain 自体の正しさは GSM8K の一部サンプルを手動分析している。
- **主な結果**: Table `tab:flagship-table` では PaLM 540B が GSM8K 17.9 -> 56.9、SVAMP 69.4 -> 79.0、ASDiv 72.1 -> 73.9、AQuA 25.2 -> 35.8、MAWPS 79.2 -> 93.3 と改善する。external calculator 付きでは PaLM 540B の GSM8K は 58.6、MAWPS は 93.5。GPT-3 175B は GSM8K 15.6 -> 46.9、Codex `code-davinci-002` は 19.7 -> 63.1。
- **主な結果**: commonsense の Table `tab:all-lm-commonsense` では PaLM 540B が CSQA 78.1 -> 79.9、StrategyQA 68.6 -> 77.8、Date 49.0 -> 65.3、Sports 80.5 -> 95.4、SayCan 80.8 -> 91.7。本文では StrategyQA について 75.6% vs prior state of the art 69.4% と書かれており、表の 77.8% と差がある。
- **主な結果**: symbolic の Table `tab:all-lm-symbolic` では PaLM 540B が Last Letter Concatenation の in-domain 2 words で 7.6 -> 99.4、OOD 3 words で 0.2 -> 94.8、OOD 4 words で 0.0 -> 63.0。Coin Flip は in-domain 2 flips で 98.1 -> 100.0、OOD 3 で 49.3 -> 98.6、OOD 4 で 54.8 -> 90.2。
- **著者が主張する貢献**: 追加学習なしで chain-of-thought prompting を提案したこと、arithmetic / commonsense / symbolic の 3 領域で改善を示したこと、効果が model scale の emergent property として現れること、ablation と robustness で「答えの前の自然言語中間ステップ」が重要だと切り分けたこと、記号操作で OOD length generalization を示したこと。

## 妥当性と限界

- **この主張を支える根拠**: モデル系列をまたいだスケール実験がある。LaMDA, GPT-3, PaLM で小さいモデルから大きいモデルまで比較し、Table `tab:all-lm-math` / `tab:all-lm-commonsense` / `tab:all-lm-symbolic` が standard prompting と CoT を並べる。
- **この主張を支える根拠**: ablation が比較的よく設計されている。LaMDA 137B の GSM8K では standard 6.5, CoT 14.3 に対し、equation only 5.4、variable compute only 6.4、reasoning after answer 6.1 であり、少なくとも GSM8K では「式だけ」「token 数だけ」「答え後の説明だけ」では足りない。
- **この主張を支える根拠**: robustness 実験では、Annotator B / C、concise style、GSM8K training set から sampled exemplars でも arithmetic で standard prompting を大きく上回る。FAQ では異なる exemplar order や exemplar 数でも効果が概ね残ると述べる。
- **著者が認めている limitations / future work**: chain of thought が人間の思考過程を模倣しても、ニューラルネットが実際に `reasoning` しているかは open question。few-shot では annotation cost は小さいが、finetuning 用に大量 CoT を作るなら高コスト。`no guarantee of correct reasoning paths` があり、正しい答えでも誤った reasoning path の場合がある。効果が大規模モデルでしか出にくいため real-world applications で serving cost が高い。小さいモデルに reasoning を誘導する研究が future work とされる。
- **読者として注意すべき点**: GSM8K の正解 50 例分析は、本文では「2 例を除き正しい」と書かれるが、Appendix `subsec:correct-chain-of-thought` は 50 例中 1 例が `correct by chance`、49 例が correct logic and math と述べる。Table `tab:appendix-gsm8k-correct-analysis` の caption は 7 salient cases / other 43 と書くため、細かい分類は本文と付録を照合して読む必要がある。
- **読者として注意すべき点**: prior best との比較は、Table `tab:flagship-table` 自身が prior best を `N/A (finetuning)` としており、CoT は prompting only である。これは「タスク専用 finetuning なしでも強い」という著者主張を支える一方、訓練条件をそろえた apples-to-apples 比較ではない。
- **追加で確認したい実験 / 疑問**: commonsense や binary classification では偶然正解が起きやすいと著者自身が述べるため、chain の factuality / faithfulness を大規模に評価したい。prompt engineering は完全には消えておらず、Coin Flip では Annotator A 99.6% に対して Annotator C 71.4% と大きく揺れる。GPT-3 175B は CSQA 79.5 -> 73.5、StrategyQA 65.9 -> 65.4 と CoT が効かない例もあり、モデル間転移の条件も未解決である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **chain of thought**: 最終答えへ至る自然言語の中間推論列。本文では `solutions/explanations typically come after the final answer` と区別し、CoT は答えの前に置く。
- **chain-of-thought prompting**: few-shot exemplar を `〈input, chain of thought, output〉` に拡張する方法。モデルは finetune されない。
- **standard prompting**: `input--output pairs` の few-shot baseline。最終答えを直接生成させる。
- **exemplar**: prompt 内に置く例題。算数では AQuA 以外で 8 個、AQuA で 4 個を使う。
- **emergent ability**: 小さいモデルの性能曲線から単純には予測できず、十分な scale で急に現れる能力としての位置づけ。本文は CoT の成功が `~100B parameters` 級で現れると述べる。
- **Equation only**: chain の代わりに数式だけを書かせる ablation。GSM8K では効果が小さいが、SVAMP / ASDiv / MAWPS のような少数 step 問題では改善する場合がある。
- **Variable compute only**: equation の文字数に合わせて dots を出させ、追加 token だけの効果を切り分ける ablation。本文では baseline とほぼ同等とされる。
- **Reasoning after answer / Chain of thought after answer**: 先に答えを出してから chain を書かせる ablation。答え前の sequential reasoning が必要かを調べるための比較である。
- **OOD length generalization**: few-shot exemplars より長い入力で評価すること。Last Letter Concatenation では 2-word exemplars から 3/4-word names へ、Coin Flip ではより多い flip step へ一般化できるかを見る。
- **external calculator**: generated chain 内の arithmetic equation に Python `eval` を後処理で適用する設定。モデルの推論文生成と純粋な計算ミスを分けるために使われる。
- **prompt engineering**: prompt の具体的な書き方を調整すること。著者は arithmetic では比較的 robust としつつ、FAQ で `Prompt engineering still matters` と明記する。

## 読む順番の提案

- まず Abstract と Introduction を読み、問題意識を押さえる。正規ノートでは `Summary（著者の主張）` の「問題」「手法」に対応する。
- 次に Section 2 `Chain-of-Thought Prompting` と Figure `fig:pull-figure` / `fig:dataset-examples` を見る。ここで `〈input, chain of thought, output〉` と、答え前に chain を置く理由を確認する。
- 算数の主結果は Section 3、Figure `fig:main-math`、Table `tab:flagship-table`、Table `tab:all-lm-math` を読む。正規ノートの「算数」「emergent」「ablation」の数値はここに対応する。
- 妥当性を見るには Section 3.3 `Ablation Study`、Section 3.4 `Robustness of Chain of Thought`、Appendix `tab:ablations-arithmetic` / `tab:ablations-commonsense-symbolic` を読む。正規ノートの `Critical Thoughts` の強み・弱みを検証する場所である。
- 常識・記号の結果は Section 4, Section 5 と Table `tab:all-lm-commonsense` / `tab:all-lm-symbolic` を読む。StrategyQA の本文値 75.6 と表値 77.8 の差、Sports の表値などは TeX 上で確認しておく。
- 最後に Discussion と FAQ、Appendix の correct / incorrect chain analysis を読む。limitations、prompt engineering、faithfulness の注意点は正規ノートの `Notes / Quotes` と `Critical Thoughts` に直接つながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2201.11903v6/`
- 正規ノート: `notes/arXiv-2201.11903v6.md`
