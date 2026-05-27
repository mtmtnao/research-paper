# Language Models are Few-Shot Learners（大規模自己回帰言語モデルにおける in-context learning の実証）

- arXiv: https://arxiv.org/abs/2005.14165
- 一次ソース: ../papers/arXiv-2005.14165v4/
- 正規ノート: ../notes/arXiv-2005.14165v4.md

---

## 一言で言うと

この論文は、175B パラメータの自己回帰型言語モデル GPT-3 を含む 8 サイズの同一モデル系列を訓練し、重み更新なしで zero-shot / one-shot / few-shot 評価を行うと、モデル規模の増大に伴って in-context learning 性能が滑らかに向上することを示す。著者の主張は「タスクごとの fine-tuning なしでも、多くの NLP ベンチマークで既存の fine-tuned SOTA に近づく、または一部で上回る」だが、WiC・ANLI・RACE・QuAC など明確に苦手な領域と、test-set contamination や bias / misuse / energy の問題も同時に扱っている。

## 何を議論する論文か

- **問題設定**: 近年の NLP は大規模テキストで pre-training した後、タスクごとの教師データで fine-tuning する方式が強い。しかしその方式は、タスクごとに数千から数十万のラベル付き例を必要とし、狭い fine-tuning 分布への過適合や spurious correlations の利用を招きうる、という問題を持つ。
- **対象範囲 / 仮定**: 対象は GPT-2 系の autoregressive Transformer language model であり、評価時には「without any gradient updates or fine-tuning」と明記される。タスク指定は自然言語の指示と、文脈中に入れたデモ例だけで行う。モデルはすべて $n_{\mathrm{ctx}}=2048$ tokens の context window を使う。
- **既存研究との差分**: GPT-2 でも in-context learning 的な評価は行われたが、Natural Questions で 4% など fine-tuning に大きく劣っていた。GPT-3 論文は、モデルを 125M から 175B まで 3 桁にわたり拡大し、zero-shot / one-shot / few-shot を体系的に比較する。
- **この論文で答えたい問い**: 言語モデルを大きくすると、単に language modeling loss が下がるだけでなく、few-shot のタスク適応能力、すなわち文脈中の例からタスクを実行する能力も強くなるのか。

## 背景と前提

- **Language model**: この論文では主に、左から右に token を予測する autoregressive language model を指す。GPT-3 は GPT-2 と同じ系統のモデルで、タスクごとの分類ヘッドや専用アーキテクチャを追加しない。
- **Fine-tuning**: pre-trained model の重みを、タスク固有の supervised dataset で更新する方式。論文はこれを強力な baseline と認めたうえで、タスク固有データへの依存と OOD generalization の弱さを問題にする。
- **Meta-learning / in-context learning**: introduction の脚注では、外側の学習で幅広いスキルやパターン認識能力を獲得し、推論時にその能力を使ってタスクに素早く適応または認識する構造を meta-learning と呼ぶ。特に、入力文脈内で起きる内側の過程を "in context-learning" と呼ぶ。
- **Zero-shot / one-shot / few-shot**: zero-shot は自然言語のタスク説明だけ、one-shot は説明に 1 個のデモを加える、few-shot は $K$ 個のデモを加える設定である。few-shot の $K$ は context window に収まる範囲で、通常 10 から 100 個とされる。
- **比較対象**: BERT, RoBERTa, T5, XLNet などの fine-tuned model、closed-book QA の T5-11B / T5-11B+SSM、open-domain QA の RAG、翻訳の XLM / MASS / mBART、各ベンチマークの fine-tuned SOTA が主な比較対象である。

## 提案手法

### コアアイデア

著者の方法は、新しい学習アルゴリズムを提案するというより、GPT-2 系の autoregressive Transformer を 175B パラメータまで拡大し、同じモデル系列を 8 サイズで訓練して、in-context learning が scale とともにどう変わるかを測るものである。モデルは GPT-2 の modified initialization, pre-normalization, reversible tokenization を使い、例外として Sparse Transformer に似た alternating dense and locally banded sparse attention patterns を層に入れる。

訓練データは filtered Common Crawl, WebText2, Books1, Books2, Wikipedia の混合である。Table \ref{table:dataset} では Common Crawl (filtered) が 410B tokens / training mix 60% / 0.44 epochs、WebText2 が 19B / 22% / 2.9 epochs、Books1 が 12B / 8% / 1.9 epochs、Books2 が 55B / 8% / 0.43 epochs、Wikipedia が 3B / 3% / 3.4 epochs とされる。高品質と見なすデータセットは、サイズ比例ではなく高い比率でサンプリングされる。

最大モデル GPT-3 175B は 96 layers, $d_{\mathrm{model}}=12288$, 96 heads, $d_{\mathrm{head}}=128$, batch size 3.2M tokens, learning rate $0.6 \times 10^{-4}$ である。Table \ref{table:param} の caption は、全モデルが total 300 billion tokens で訓練されたと明記する。

### 重要な定義・数式

TeX 中には、通常の言語モデル目的関数を明示する式は少ない。そのため、ここでは本文で明示される設計・評価上の式に絞る。

$$
d_{\mathrm{ff}} = 4 \ast d_{\mathrm{model}}
$$

**式の意味**: Transformer の feedforward layer の幅を、bottleneck layer の幅 $d_{\mathrm{model}}$ の 4 倍にするというモデル設計である。Table \ref{table:param} の説明で、全モデルに共通する構造として述べられる。

**記号の定義**:
- $d_{\mathrm{ff}}$ ... feedforward layer の次元数
- $d_{\mathrm{model}}$ ... bottleneck layer の unit 数、すなわち Transformer の主な hidden dimension
- $\ast$ ... TeX 中の表記に従った乗算記号

**この論文での役割**: 8 サイズの GPT-3 系列を比較する際、細部のアーキテクチャ差ではなく主に model size の効果を見るための共通設計である。

$$
n_{\mathrm{ctx}}=2048
$$

**式の意味**: すべてのモデルが一度に条件づけられる context window は 2048 tokens である、という設定である。

**記号の定義**:
- $n_{\mathrm{ctx}}$ ... モデルが推論時に参照できる文脈長
- $2048$ ... token 数

**この論文での役割**: few-shot 評価で文脈に入れられるデモ数 $K$ の上限を決める。本文では、典型的には 10 から 100 examples が入ると説明される。

$$
\frac{P(\mathrm{completion} | \mathrm{context})}{P(\mathrm{completion} | \mathrm{answer\_context})}
$$

**式の意味**: ARC, OpenBookQA, RACE のような一部の multiple choice task で、completion の条件付き尤度を、その completion 自体の無条件に近い出やすさで割って正規化する評価式である。

**記号の定義**:
- $P(\mathrm{completion} | \mathrm{context})$ ... 問題文脈を与えたとき、その選択肢 completion が出る確率
- $P(\mathrm{completion} | \mathrm{answer\_context})$ ... generic な `"Answer: "` または `"A: "` だけを answer_context としたとき、その completion が出る確率
- $\mathrm{context}$ ... デモ例と対象問題の文脈
- $\mathrm{answer\_context}$ ... completion が答えであることだけを促す汎用文字列

**この論文での役割**: multiple choice で長さや選択肢固有の出やすさに引きずられないようにする評価上の工夫である。結果の解釈では、モデルが重み更新なしで選択肢を比較している点を押さえる必要がある。

$$
\alpha = 0.6
$$

**式の意味**: free-form completion task で使う beam search の length penalty である。beam width は 4 とされる。

**記号の定義**:
- $\alpha$ ... beam search における length penalty
- beam width 4 ... 同時に保持する候補列の数

**この論文での役割**: QA や翻訳など自由生成を評価するタスクで、生成長の偏りを抑えながら F1, BLEU, exact match など標準指標で採点するための設定である。

### 実装 / アルゴリズム上の要点

- step1: GPT-2 系の autoregressive Transformer を 8 サイズ訓練する。サイズは GPT-3 Small 125M, Medium 350M, Large 760M, XL 1.3B, 2.7B, 6.7B, 13B, 175B である。
- step2: Common Crawl を quality-based filtering し、document level の fuzzy deduplication を行い、WebText2 / Books1 / Books2 / English-language Wikipedia を加える。
- step3: 各モデルを 300B tokens 訓練する。大モデルでは matrix multiply 内と layer 間の model parallelism を併用し、V100 GPU cluster を使う。
- step4: 評価時はモデル重みを更新しない。few-shot では各評価例に対して、原則としてそのタスクの training set から $K$ 個の例をランダムに引き、context と completion の形で文脈に入れる。例外として、LAMBADA と StoryCloze は development set から、original Winograd は同じ dataset から conditioning examples を引く。
- step5: multiple choice は completion の likelihood を比較し、free-form completion は beam search で生成して F1 / BLEU / exact match など標準指標で採点する。
- step6: benchmark contamination を調べるため、pretraining set との 13-gram overlap を用いて clean subset を作り、元スコアとの差を比較する。

## 実験・結果

- **データセット / ベンチマーク**: language modeling / cloze / completion では PTB, LAMBADA, StoryCloze, HellaSwag、closed-book QA では Natural Questions, WebQuestions, TriviaQA、翻訳では WMT'14 Fr-En, WMT'16 De-En, WMT'16 Ro-En、推論系では Winograd, Winogrande, PIQA, ARC, OpenBookQA、読解では CoQA, DROP, QuAC, SQuADv2, RACE、集約ベンチマークでは SuperGLUE、さらに ANLI, arithmetic, word scrambling, SAT analogies, news article generation などを使う。
- **比較対象 / baseline**: fine-tuned SOTA, fine-tuned BERT-Large, RoBERTa, T5-11B, T5-11B+SSM, RAG, XLM, MASS, mBART, UnifiedQA 系の SOTA など。比較の注意点として、GPT-3 側は原則として fine-tuning なしである。
- **指標**: accuracy, F1, BLEU, exact match, perplexity, human detection accuracy など。PTB は zero-shot perplexity、SuperGLUE は task ごとの accuracy / F1 と平均、翻訳は multi-bleu.perl の BLEU、ニュース生成は人間評価者の mean accuracy である。
- **主な結果**: Figure \ref{graph:compute} では cross-entropy validation loss の power-law trend が "additional two orders of magnitude" まで大きくは崩れないとされる。Figure \ref{figure:aggregate_performance} は 42 個の accuracy-denominated benchmarks の平均で、zero-shot も伸びるが few-shot がより速く伸びる、と説明する。
- **主な結果**: PTB は GPT-3 zero-shot perplexity 20.5 で、Table \ref{table:language} の SOTA 35.8 より低い。LAMBADA は GPT-3 few-shot accuracy 86.4%, perplexity 1.92 で、Table \ref{table:completion} の SOTA accuracy 68.0 を上回る。
- **主な結果**: closed-book QA では Table \ref{table:question} で TriviaQA が zero-shot 64.3, one-shot 68.0, few-shot 71.2 であり、RAG の 68.0 を few-shot が上回る。NaturalQS は few-shot 29.9、WebQS は few-shot 41.5 で、T5-11B+SSM の NaturalQS 36.6 / WebQS 44.7 には届かない。
- **主な結果**: 翻訳は Table \ref{table:translation} で GPT-3 few-shot が Fr$\to$En 39.2, De$\to$En 40.6, Ro$\to$En 39.5 と into-English で強い。一方 En$\to$Ro は 21.0 で、supervised SOTA 38.5 や MASS 35.2 より大きく低い。本文は GPT-2 由来の byte-level BPE tokenizer が英語中心だったことを弱点候補として挙げる。
- **主な結果**: SuperGLUE は Table \ref{table:superglue} で GPT-3 few-shot average 71.8、fine-tuned BERT-Large 69.0、fine-tuned SOTA 89.0 である。COPA は 92.0、WSC は 80.1、ReCoRD F1 は 91.1 と強いが、WiC は 49.4 で random chance と述べられる。
- **主な結果**: arithmetic は Table \ref{table:arithmetic} で GPT-3 few-shot が 2D+ 100.0, 2D- 98.9, 3D+ 80.4, 3D- 94.2, 4D+ 25.5, 4D- 26.8, 5D+ 9.3, 5D- 9.9, 2Dx 29.2, 1DC 21.3 である。本文中では 3D addition 80.2 と書かれる箇所があるが、表では 80.4 であるため、このノートでは表値を優先して両者の差に注意する。
- **主な結果**: 3 桁算術の test set と training data の一致確認では、2,000 個の addition のうち 17 matches (0.8%)、2,000 個の subtraction のうち 2 matches (0.1%) しか見つからず、著者は単純な memorization では説明しにくいと論じる。
- **主な結果**: news article generation では、短い約 200 word 記事で human accuracy が control 86% に対し GPT-3 175B は 52% (Table \ref{table:generation})、約 500 word 記事でも control 88% に対し GPT-3 175B は 52% (Table \ref{table:generation_long}) である。著者はこれを synthetic text と human-written text の区別困難化として Broader Impacts に接続する。
- **著者が主張する貢献**: 175B autoregressive LM の訓練、zero-shot / one-shot / few-shot の体系的比較、スケールと in-context learning の関係の実証、多様な NLP / synthetic tasks での評価、contamination analysis、bias / misuse / energy を含む broader impacts の予備的検討である。

## 妥当性と限界

- **この主張を支える根拠**: 同一モデル系列を 125M から 175B まで 8 サイズで訓練し、同じ zero-shot / one-shot / few-shot プロトコルで多くのタスクを評価しているため、モデルサイズと in-context learning 性能の関係を横断的に観察できる。Figure \ref{graph:compute} と Figure \ref{figure:aggregate_performance} は、loss と下流タスク性能の両方で scale に伴う滑らかな傾向を示す。
- **この主張を支える根拠**: 評価時に fine-tuning をしないことが一貫しており、few-shot で性能が上がる場合、それは重み更新ではなく文脈中のデモの利用による。SuperGLUE の Figure \ref{graph:superglue_analysis} では model size と context examples の両方で性能が伸びると説明される。
- **この主張を支える根拠**: contamination について、著者は 13-gram overlap による clean subset を作り、元スコアとの差を比較している。PIQA は 29% flagged / clean subset で 3 percentage point decrease、Winograd は 45% flagged / 2.6% decrease として、結果に asterisk を付ける。4 Wikipedia language modeling benchmarks と Children's Book Test はほぼ training data に含まれ、1BW は高い割合が training set に含まれるため報告しない。
- **著者が認めている limitations / future work**: GPT-3 は長文生成で意味的反復、coherence 低下、自己矛盾、non-sequitur を起こしうる。common sense physics、WiC / ANLI のような comparison tasks、QuAC / RACE など一部読解タスクも弱い。
- **著者が認めている limitations / future work**: autoregressive model に限定したため、bidirectional architectures や denoising objectives を含まない。著者は、WIC, ANLI, QuAC, RACE などの弱さは、比較・読み返し・短い答え生成が必要なタスクで bidirectionality が効く可能性と関係すると述べる。
- **著者が認めている limitations / future work**: few-shot learning が本当に test time に新タスクを "from scratch" で学んでいるのか、訓練時に学んだタスクを認識しているだけなのかは曖昧であり、task によって位置が異なる可能性がある。
- **著者が認めている limitations / future work**: pre-training の sample efficiency は人間より大きく劣り、推論も高価で扱いにくい。distillation は将来方向として挙げられるが、hundreds of billions parameters の規模では新しい課題がある。
- **著者が認めている limitations / future work**: Broader Impacts では intentional misuse、bias / fairness / representation、energy efficiency が扱われる。Energy 節は GPT-3 175B の training compute を several thousand petaflop/s-days とし、Table \ref{table:total_compute_calculations} は GPT-3 175B を 3.64E+03 PF-days / 3.14E+23 flops とする。
- **読者として注意すべき点**: Figure \ref{figure:aggregate_performance} の 42 benchmark 平均は著者自身が "not a rigorous or meaningful benchmark in itself" と注意している。全体平均が上がっても、WiC 49.4 や ANLI の chance 近辺のような弱点は隠れやすい。
- **読者として注意すべき点**: GPT-3 few-shot と fine-tuned SOTA の比較は、訓練条件や使えるタスク固有データ量が異なる。これは論文の問題設定そのものだが、同一 compute 予算や同一データ利用条件での比較ではない。
- **追加で確認したい実験 / 疑問**: 苦手な comparison tasks について、同規模の bidirectional model や denoising objective と直接比較すると、弱さが autoregressive objective 由来か scale 不足かを切り分けられる可能性がある（TeX 中では future direction として示唆されるが、実験はない）。
- **追加で確認したい実験 / 疑問**: few-shot が task recognition なのか de novo learning なのかを、人工タスクや反事実デモで分離して測る実験が必要である（TeX 中には具体実験は明示されていない）。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **GPT-3**: この論文では特に 175.0B パラメータの最大モデルを指すことが多いが、Table \ref{table:param} では GPT-3 Small から GPT-3 175B までの系列名としても使われる。
- **Autoregressive language model**: 左から右へ次 token を予測するモデル。著者は sampling と likelihood 計算がしやすいのでこの系統を使ったと述べる。
- **Few-shot (FS)**: 推論時に $K$ 個の task demonstrations を conditioning として与えるが、weight updates は行わない設定。通常 $K$ は 10 から 100。
- **One-shot (1S)**: 自然言語の説明に加えて 1 個のデモだけを与える設定。人間にタスクを説明するときの形式に近いとして区別される。
- **Zero-shot (0S)**: デモなしで自然言語の説明だけを与える設定。便利で spurious correlations を避けやすいが、フォーマットが曖昧なタスクでは人間にも不利な場合がある。
- **In-context learning**: 重み更新なしで、入力文脈内の説明や例からタスクを実行すること。著者は meta-learning の inner loop として位置づける。
- **Fine-tuning (FT)**: タスク固有の supervised dataset で pre-trained model の重みを更新する方式。この論文では GPT-3 自体には行わず、将来方向とする。
- **Closed-book QA**: 外部検索や補助文書を使わず、モデルのパラメータに保持された知識だけで質問に答える設定。GPT-3 はこれに加えて QA dataset での fine-tuning も使わない。
- **Test-set contamination**: pretraining data に benchmark の test / development examples が混入している問題。著者は 13-gram overlap で conservative に検出し、clean subset との差を見る。
- **Power-law scaling**: Kaplan et al. 2020 の scaling laws に基づく、compute や model size と validation loss の滑らかな関係。GPT-3 論文ではこの傾向がさらに 2 orders of magnitude 拡張されても大きくは崩れないと報告する。
- **Spurious correlations**: タスクの本質ではなく、dataset 固有の表面的な手がかり。fine-tuning が狭い分布で行われると、大モデルほどこれを利用しやすいという懸念が導入で述べられる。
- **BPE / reversible tokenization**: GPT-2 由来の tokenization。翻訳節では、英語中心データで作られた byte-level BPE tokenizer の再利用が out-of-English 翻訳の弱さの可能性として挙げられる。
- **Broader Impacts**: この論文では、deliberate misuse、bias / fairness / representation、energy usage の三つを中心に、技術の社会的影響を予備的に論じる節である。

## 読む順番の提案

- まず `content/abstract.tex` と `content/1_introduction/introduction.tex` を読む。ここで問題設定、fine-tuning への批判、meta-learning / in-context learning の脚注定義、Figure \ref{figure:aggregate_performance} の読み方を押さえる。正規ノートでは `Summary（著者の主張）` と `Takeaway` の前半に対応する。
- 次に `content/2_approach/approach.tex` と `content/2_approach/evaluation.tex` を読む。zero-shot / one-shot / few-shot / fine-tuning の違い、$K$, $n_{\mathrm{ctx}}=2048$, multiple choice の likelihood 比を確認する。正規ノートの「手法」「Notes / Quotes」の評価設定に対応する。
- モデルとデータは `content/2_approach/model_and_architectures.tex`, `training_dataset.tex`, `training_process.tex`, `tables/param.tex`, `tables/dataset.tex` を見る。175B の層数・hidden size・batch size・learning rate、300B tokens、データ混合比はここで裏取りする。
- 結果は全節を一気に読むより、まず `content/3_results/results.tex`, `graphs/compute.tex`, `tables/language.tex`, `tables/completion.tex`, `Closed_Book_Question_Answering_-_Knowledge_Based_Tasks.tex`, `tables/superglue.tex`, `tables/arithmetic.tex`, `News_Article_Generation.tex` を読む。正規ノートの代表数値と `Critical Thoughts` の根拠になる。
- 信頼性と限界は `content/4_preventing_memorization/measuring_and_preventing_memorization_of_benchmarks.tex`, `tables/overlap_master.tex`, `content/5_limitations/Limitations.tex` を読む。PIQA / Winograd の asterisk、LAMBADA contamination、WiC / ANLI / RACE / QuAC の弱さをここで確認する。
- 社会的影響は `content/6_broader_impacts/Broader_Impacts.tex`, `Potential_Misuse_Applications.tex`, `Fairness_Bias_and_Representation.tex`, `Energy.tex`, `tables/total_compute_calculations.tex` を読む。正規ノートの broader impacts と energy の記述につながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2005.14165v4/`
- 正規ノート: `notes/arXiv-2005.14165v4.md`
