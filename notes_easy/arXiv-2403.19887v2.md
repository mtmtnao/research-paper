# Jamba: A Hybrid Transformer-Mamba Language Model（Attention-SSM ハイブリッド LLM の位置づけ）

- arXiv: https://arxiv.org/abs/2403.19887
- 一次ソース: ../papers/arXiv-2403.19887v2/
- 正規ノート: ../notes/arXiv-2403.19887v2.md

---

## 一言で言うと

Jamba は、Transformer の Attention 層、Mamba に代表される state-space model (SSM) 層、Mixture-of-Experts (MoE) を組み合わせ、長文脈での KV cache と throughput の制約を下げつつ、Mixtral や Llama-2 70B に近い標準ベンチマーク性能を狙う decoder-only base LLM である。公開構成は 12B active / 52B total available parameters、256K tokens context をサポートし、著者は “first production-grade Attention-SSM hybrid model” と位置づけている（main.tex Introduction）。

## 何を議論する論文か

- **問題設定**: Transformer は長文脈で key-value (KV) cache が大きくなり、生成時も各 token で全文脈への計算が必要になる。Mamba/SSM は長距離系列を効率よく扱えるが、同規模 Transformer に性能で遅れるという前提から、Attention と SSM をどの比率で混ぜれば品質・メモリ・throughput の折り合いが取れるかを扱う。
- **対象範囲 / 仮定**: 対象は pretrained base large language model であり、公開モデルは alignment、instruction tuning、moderation mechanisms を経ていない（main.tex Important notice）。訓練データは Web、books、code を含む in-house dataset で、last update は March 2024、quality filters と deduplication を含むとだけ説明される。
- **既存研究との差分**: Transformer、Mamba、MoE は既存要素だが、Jamba はこれらを Jamba block として大規模に組み合わせる。関連研究として H3、Hyena、StripedHyena などの Attention-SSM 系が挙げられるが、著者は規模と production-grade 実装の点で Jamba が異なると主張する。
- **この論文で答えたい問い**: Attention と Mamba の比率 $a:m$、MoE の入れ方 $e,n,K$、明示的な位置情報の要否、Mamba 大規模化時の安定化が、品質・KV cache・throughput・long-context 能力にどう効くかを実験で示す。

## 背景と前提

- **Transformer / Attention**: 文脈中の token 間関係を self-attention で扱う標準的な LLM アーキテクチャ。長文脈では KV cache が制約になるため、Attention 層数を減らすことがメモリ削減に直結する。
- **Mamba / SSM**: Mamba は state-space model の一種として扱われる。RNN のように単一 summary state を持つ利点を意識しつつ、RNN より訓練を並列化しやすく長距離関係に強いが、著者は「comparably sized Transformer language models」にはまだ性能で遅れると述べる。
- **MoE**: Jamba では MLP の一部を MoE に置き換え、total available parameters を増やしながら、各 forward step の active parameters と compute を管理するために使う。router が token ごとに top $K$ experts を選ぶ。
- **baseline との関係**: 標準ベンチマークでは Llama-2 13B、Llama-2 70B、Gemma、Mixtral と比較する。メモリ・throughput・long-context では特に Mixtral-8x7B と Llama-2-70B が比較軸になる。
- **評価への姿勢**: 著者自身が Evaluation 冒頭で、benchmarks は実アプリケーションで重要なものと部分的にしか相関せず、gaming を誘うため cautiously に扱うと述べる。この論文の数値は、主張の根拠ではあるが、著者も万能な品質測定とは見ていない。

## 提案手法

### コアアイデア

Jamba は Transformer layers、Mamba layers、MoE modules を混ぜる hybrid decoder architecture である。1 つの **Jamba block** は $l$ 個の層からなり、各層は Attention module または Mamba module の後に MLP を持つ。一部の MLP は MoE layers に置き換えられる。公開実装では Jamba block を 4 個積み、各 block の中で Attention:Mamba を $a:m=1:7$ にするため、Attention 層は全体で少ない。

この設計の狙いは、Attention を完全には捨てずに in-context learning や format adherence に必要な能力を保ち、Mamba 層を多くすることで長文脈時の KV cache と throughput を改善することである。MoE は model capacity を増やすが、top-$K$ routing により active parameters を抑える役割を持つ。

### 重要な定義・数式

TeX 中には目的関数や Mamba 更新式のような中核的な明示式はほぼ無く、主な「数式」は architecture の設計変数である。以下は main.tex の Jamba block 定義と公開構成に出てくる表記を保った整理である。

$$
l,\quad a:m,\quad e,\quad n,\quad K
$$

**式の意味**: Jamba architecture の自由度を表す記号である。main.tex は “the different degrees of freedom in the Jamba architecture” としてこれらを列挙している。

**記号の定義**:
- $l$ ... 1 つの Jamba block に含まれる層数
- $a:m$ ... attention-to-Mamba layers の比率
- $e$ ... single MLP の代わりに MoE を使う頻度
- $n$ ... MoE layer あたりの total number of experts
- $K$ ... 各 token で使う top experts の数

**この論文での役割**: 品質、KV cache、throughput、model capacity、active parameters の tradeoff を調整するための設計軸である。Ablations and Insights の Tables 4-8 は、主にこの設計空間の妥当性を調べる。

$$
l = 8,\quad a:m = 1:7,\quad e = 2,\quad n = 16,\quad K = 2
$$

**式の意味**: 公開 Jamba 実装の 1 block あたりの具体設定である（main.tex, “Jamba Implementation for a Single 80GB GPU”）。

**記号の定義**:
- $l=8$ ... 1 Jamba block は 8 layers
- $a:m=1:7$ ... 8 layers のうち Attention と Mamba を 1:7 の比率で置く
- $e=2$ ... every other layer で MLP を MoE に置き換える
- $n=16$ ... MoE layer ごとに 16 experts
- $K=2$ ... token ごとに top-2 experts を使う

**この論文での役割**: 単一 80GB GPU に収める公開モデルの設計であり、Table 1 の 4GB KV cache、Figure 2 の context length、Figure 3 の throughput 比較につながる。

$$
\text{Available params}=52\text{B},\quad \text{Active params}=12\text{B},\quad \text{KV cache}=4\text{GB}
$$

**式の意味**: Table 1 における Jamba のパラメータ数と、256K context, 16bit での KV cache 量をまとめたものである。

**記号の定義**:
- Available params ... MoE experts を含む total available parameters
- Active params ... 1 forward step で実際に使われる parameters
- KV cache ... attention keys and values を context に対して保存するメモリ

**この論文での役割**: Jamba の効率主張の中心である。Table 1 では LLAMA-2 (6.7B available / 6.7B active) が 128GB、Mistral が 7.2B active / 32GB、Mixtral が 12.9B active / 32GB、Jamba が 12B active / 4GB と比較される。

$$
\text{L-Eval Avg F1: Mixtral}=0.43,\quad \text{Jamba}=0.44
$$

**式の意味**: Table 3 の long-context QA benchmarks における 3-shot F1 の平均である。

**記号の定義**:
- L-Eval ... long-context QA の評価枠組み
- Avg F1 ... LongFQA、CUAD、NarrativeQA、NQ、SFiction の F1 平均
- Mixtral / Jamba ... 比較対象モデル

**この論文での役割**: Jamba が長文脈で Mixtral と同等以上に動くという著者主張を支える。ただし差は平均 0.01 であり、読者は dataset ごとの勝敗も見る必要がある。

### 実装 / アルゴリズム上の要点

- step1: Jamba block を 4 個並べる。各 block は $l=8$ layers、$a:m=1:7$、MoE は every other layer。
- step2: 各 layer は Attention module または Mamba module の後に MLP を置く。公開構成の図では Attention MoE layer は示されるが、footnote で “our architecture does not use” とされる。
- step3: MoE layer では $n=16$ experts から token ごとに top $K=2$ experts を router が選ぶ。これにより total available parameters と active parameters を分ける。
- step4: Mamba layers には large model scale での training stabilization のため RMSNorm を内部 activations に追加する。Figure 9 は loss spikes が抑えられることを示す。
- step5: 明示的な positional embeddings や RoPE は使わない。著者は Mamba layers が implicit position information を提供すると推定するが、これは Table 8 の 1.3B / 250B tokens 比較に基づく。
- step6: その他の architecture details は GQA、SwiGLU、MoE load balancing、64K vocabulary、BPE tokenizer、each digit is a separate token、Llama/Mistral tokenizers の dummy space removal。
- step7: 訓練は NVIDIA H100 GPUs と in-house proprietary framework で実施され、FSDP、tensor parallelism、sequence parallelism、expert parallelism を用いる。Jamba implementation は context lengths of up to 1M tokens で訓練され、released model は up to 256K tokens を扱う。

## 実験・結果

- **データセット / ベンチマーク**: 標準評価は HellaSwag (10-shot)、WinoGrande (5-shot)、ARC-E (0-shot)、ARC-Challenge (25-shot)、PIQA (zero-shot)、BoolQ (10-shots)、QuAC (zero-shot)、GSM8K (3-shot CoT)、HumanEval (pass@1)、NQ (5-shot)、TruthfulQA (zero-shot)、MMLU (5-shot)、BBH (3-shot)。long-context では needle-in-a-haystack、Trec-Fine / NLU Intent / Banking77 / CLINC150 の few-shot classification、L-Eval 由来の NarrativeQA / LongFQA / NQ / CUAD / SFiction を使う。
- **比較対象 / baseline**: Table 2 では Llama-2 13B、Llama-2 70B、Gemma、Mixtral。Table 3 と Figure 5 では Mixtral。throughput と context length では Mixtral 8x7B と Llama-2-70B。ablation では pure Attention、pure Mamba、Attention-Mamba hybrid、Jamba+MoE、Jamba+RoPE を比較する。
- **指標**: 標準ベンチマークの task score、HumanEval は pass@1、L-Eval は F1、few-shot classification は exact match with greedy decoding、ablation では OLLM、HellaSwag、WinoGrande、NQ、log-prob per byte on C4 / Books / Code、throughput は tokens/second。
- **主な結果**: Table 1 では Jamba の KV cache は 256K context, 16bit で 4GB、Mixtral は 32GB、LLAMA-2 (6.7B available / 6.7B active) は 128GB。Figure 2 では single A100 80GB GPU で Jamba が Mixtral の 2x、Llama-2-70B の 7x の context length を扱えるとされる。Figure 3 では single A100 80GB GPU, int8, 8K context, batch variation と、4 A100 GPUs, no quantization, 128K context の両設定で Mixtral に対し 3x throughput と報告される。
- **主な結果**: Table 2 の Jamba は HellaSwag 87.1、WinoGrande 82.5、ARC-E 73.5、ARC-C 64.4、PIQA 83.2、NQ 45.9、TruthfulQA 46.4、BoolQ 88.2、QuAC 40.9、GSM8K 59.9、HumanEval 29.3、MMLU 67.4、BBH 45.4。Mixtral は MMLU 70.6、HumanEval 34.8、GSM8K 60.4 などで上回るが、Jamba は HellaSwag、WinoGrande、PIQA で表中最高値。
- **主な結果**: Table 3 の L-Eval 3-shot F1 は LongFQA 0.44、CUAD 0.44、NarrativeQA 0.30、NQ 0.60、SFiction 0.40、Avg 0.44。Mixtral は 0.42、0.46、0.29、0.58、0.42、Avg 0.43。
- **主な結果**: Table 4 の 1.3B / 250B tokens ablation では、Attention と Mamba より Jamba no MoE が良く、$a:m=1:3$ と $a:m=1:7$ は OLLM 37.2、HellaSwag 65.1、WinoGrande 61.7 とほぼ同じ。Table 5 の 7B / 50B tokens では Jamba no MoE が OLLM、HellaSwag、NQ、C4/Books/Code log-prob で Attention/Mamba を上回る一方、WinoGrande は Attention 59.7、Jamba 58.8、Mamba 55.8 である。
- **主な結果**: Table 6 では pure Mamba が IMDB 48.8、QuAC 20.2、NarrativeQA 27.7 と落ち、Attention-Mamba は IMDB 90.9、QuAC 26.6、NarrativeQA 43.7。著者は Mamba が “Positive/Negative” 形式に従わず “Very Good” や “3/10” などを出す例を挙げ、ICL / format adherence の問題と解釈する。
- **主な結果**: Table 7 では 7B / 50B tokens の Attention-Mamba hybrid に MoE を加えると、OLLM 36.6 から 38.1、HellaSwag 62.5 から 66.0、WinoGrande 58.8 から 61.2、NQ 15.4 から 18.9 に改善する。Table 8 では Jamba と Jamba+RoPE の log-prob は同じ値で、明示的位置情報なしでも同程度とされる。
- **著者が主張する貢献**: 公開実装を Apache 2.0 license で Hugging Face に出し、Attention-SSM hybrid architecture を production-grade に示したこと、長文脈で小さい KV cache と高 throughput を示したこと、Attention/Mamba 比、MoE、位置情報、RMSNorm の設計判断を ablation で支えたこと、pure Mamba の ICL / format adherence の弱点を観察し hybrid の induction-head 的挙動を示したこと。

## 妥当性と限界

- **この主張を支える根拠**: 効率面は Table 1 の KV cache、Figure 2 の single GPU context length、Figure 3 の throughput で支えられる。品質面は Table 2 の標準ベンチマーク、Table 3 と Figures 4-5 の long-context 評価で支えられる。設計面は Tables 4-8 と Figures 6-9 の ablation で、$a:m=1:7$、MoE、RMSNorm、位置情報なしの選択を説明している。
- **著者が認めている limitations / future work**: 公開モデルは pretrained base model で、alignment / instruction tuning / moderation mechanisms がないため、additional adaptation なしに production environments や end users に使うべきでない。Future work として、hybrid models at large scale で ICL の emergence を調べること、state-space models から attention-like scores を抽出する方向、smaller-scale training runs の checkpoints 公開による追試促進が述べられる。
- **読者として注意すべき点**: 訓練データは in-house dataset で、具体的なデータ量や最終 12B/52B モデルの正確な訓練 token 数は TeX 中には明示されない。比較ベンチマークの prompt 詳細や、比較対象スコアの取得方法の細部も TeX 中には明示されない。著者自身が benchmarks を cautiously に扱うと述べるため、Table 2 のスコアだけで実アプリケーション性能を断定しない方がよい。
- **読者として注意すべき点**: Long-context の強さは needle-in-a-haystack、few-shot classification、L-Eval で示されるが、Table 3 の Avg F1 差は Jamba 0.44 vs Mixtral 0.43 と小さい。needle-in-a-haystack は埋め込まれた statement の retrieval なので、長文脈 reasoning 全般を代表するとは限らない。
- **読者として注意すべき点**: ablation は 1.3B / 250B tokens または 7B / 50B tokens が中心で、最終 12B active / 52B total モデルで全設計を再比較した表は TeX 中にはない。位置情報なしの結論も Table 8 の 1.3B / 250B tokens 比較が根拠であり、すべてのスケール・すべての長文脈条件で保証されたものではない。
- **追加で確認したい実験 / 疑問**: instruction tuning 後の long-context 性能、より複雑な multi-hop QA や code editing での 256K context 評価、最終スケールでの $a:m$ と MoE 設定の再 ablation、公開モデルの benchmark contamination の検証、pure Mamba の format adherence が SFT や format-aware training でどこまで改善するか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Jamba block**: Transformer layers、Mamba layers、MoE modules を組み合わせる基本単位。公開実装では 4 blocks を積む。
- **$a:m$**: attention-to-Mamba layers の比率。公開実装は $1:7$。Attention を減らすほど KV cache と long-context compute は下がるが、能力低下の可能性がある。
- **$l,e,n,K$**: $l$ は block 内層数、$e$ は MoE を入れる頻度、$n$ は experts 数、$K$ は token ごとに使う experts 数。
- **Available params / active params**: MoE により total available parameters と forward step で使う active parameters が分かれる。Jamba は 52B available、12B active。
- **KV cache**: Attention の keys and values を context に対して保存するメモリ。Jamba は Attention 層を少なくすることで 256K context, 16bit の KV cache を 4GB にする。
- **Mamba / SSM**: Jamba の非 Attention 層の基盤。長文脈・throughput に有利だが、pure Mamba は一部 ICL/format adherence タスクで弱いと観察される。
- **MoE**: MLP の一部を experts に置き換える仕組み。Jamba では every other layer、16 experts、top-2 experts per token。
- **ICL / induction heads**: few-shot examples の input-output format をその場で利用する能力と、それを支える Attention head の説明。Figure 8 では最後の token “:” から few-shot labels に attention が向く例を示し、3 attention layers（layers 4, 12, 20）に 12 such heads を見つけたと書かれる。
- **RMSNorm**: 大規模 Jamba の Mamba 内部 activations が大きくなり loss spikes を起こしたため、Mamba layers 内部に追加された正規化。
- **RoPE / positional information**: Jamba では明示的な positional embeddings や RoPE を使わない。著者は Mamba layers が implicit position information を提供すると推定する。
- **OLLM**: HuggingFace OpenLLM leaderboard の summary statistic。ablation の小規模評価で使われる。
- **log-prob per byte**: C4、Books、Code の perplexity 系評価として使われる指標。表では値が高い、つまり負の値が 0 に近いほど良い。

## 読む順番の提案

- まず Abstract と Introduction を読み、Jamba が “hybrid Transformer-Mamba mixture-of-experts (MoE) architecture” として何を解こうとしているか、公開モデルが base model である注意書きを確認する。正規ノートの Summary に対応する。
- 次に Model Architecture と Figure 1、Table 1 を読む。$l,a:m,e,n,K$、available/active parameters、KV cache の関係が分かると、正規ノートの Takeaway にある「Attention は 1/8 で十分」「KV cache 4GB」が読める。
- その後、Reaping the Benefits の single GPU 実装と throughput analysis、Figures 2-3 を読む。single 80GB GPU、int8、8K/128K context、3x throughput という主張の条件を確認する。
- Evaluation では Table 2、Table 3、Figures 4-5 を先に見る。標準ベンチマークの強弱と、long-context 評価で何を測っているかを分けて読む。
- Ablations and Insights は Tables 4-8、Figures 6-9 の順に読む。正規ノートの Critical Thoughts にある設計妥当性・疑問点はこの節に対応する。
- main.bbl は、Mamba、Transformer、Mixtral、H3/Hyena/StripedHyena、L-Eval、induction heads など、正規ノートの Related Papers に挙がる先行研究名を確認するために見る。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2403.19887v2/`
- 正規ノート: `notes/arXiv-2403.19887v2.md`
