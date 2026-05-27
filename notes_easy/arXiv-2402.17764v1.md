# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits（1.58-bit ternary LLM による推論コスト削減と新しい計算パラダイム）

- arXiv: https://arxiv.org/abs/2402.17764
- 一次ソース: ../papers/arXiv-2402.17764v1/
- 正規ノート: ../notes/arXiv-2402.17764v1.md

---

## 一言で言うと

この論文は、LLM の重みを ternary $\{-1,0,1\}$ に制約した **BitNet b1.58** を提案し、同じモデルサイズ・同じ学習 token 数で事前学習した reproduced FP16 LLaMA LLM に対し、3B 規模から perplexity と end-task performance で並ぶ、と主張する。abstract では full-precision Transformer LLM を FP16 or BF16 と説明するが、主実験の baseline は FP16 LLaMA LLM である。主な狙いは、精度だけでなく latency、memory、throughput、energy consumption を同時に下げる「1-bit LLMs」の計算パラダイムを示すことである。

## 何を議論する論文か

- **問題設定**: LLM は大規模化により性能を伸ばしてきたが、推論時の memory、latency、throughput、energy consumption が配備上の制約になる。TeX では、LLM の計算コストの大部分は matrix multiplication にあり、vanilla LLM では FP16/BF16 の addition と multiplication が支配的だと説明する。
- **対象範囲 / 仮定**: 対象は Transformer LLM であり、BitNet b1.58 は `nn.Linear` を `BitLinear` に置き換え、1.58-bit weights と 8-bit activations で scratch から学習する。主比較は、著者らが RedPajama dataset で 100 billion tokens 事前学習した reproduced FP16 LLaMA LLM である。
- **既存研究との差分**: OPTQ、AWQ、SmoothQuant、QuIP/Quip# などの post-training quantization は、学習済みモデルを後から低 bit 化する。一方、この論文は BitNet 系の 1-bit model architecture を前提に、original 1-bit BitNet に 0 を追加して ternary $\{-1,0,+1\}$ を採用し、0 による feature filtering を追加の表現能力として位置づける。
- **この論文で答えたい問い**: 重みを $\{-1,0,+1\}$ に制約しても、同条件で学習した FP16 LLaMA LLM と同等の PPL・zero-shot accuracy を保てるか。さらに、その制約が latency、GPU memory、throughput、energy の面でどの程度の効率改善につながるか。

## 背景と前提

- **bit 数と量子化**: FP16/BF16 は 16-bit floating values を使う。low-bit quantization は weights や activations の精度を下げて memory と計算量を減らす手法で、TeX では 16 bits から 4-bit variants へ進む傾向があるが post-training quantization は sub-optimal だと述べる。
- **BitNet との関係**: BitNet は 1-bit Transformer の先行研究であり、matrix multiplication が integer addition 中心になるため energy cost を大きく削減できる方向として扱われる。BitNet b1.58 はこの枠組みに 0 を追加し、ternary weights にする。
- **行列積と推論コスト**: LLM では `nn.Linear` の行列積が大きな割合を占める。BitNet b1.58 について、著者は「almost no multiplication operations for matrix multiplication」と説明し、matrix multiplication を highly optimized できる計算パラダイムとして位置づける。
- **memory bandwidth の問題**: 推論では DRAM から on-chip accelerator memory、例えば SRAM、へ parameters を移すこと自体も高コストである。1-bit LLMs は capacity と bandwidth の両面で memory footprint を下げる、というのが論文の前提である。
- **LLaMA 型構成**: BitNet b1.58 は LLaMA/LLaMA 2 を open-source LLM の de-facto backbone と見なし、RMSNorm、SwiGLU、rotary embedding、no biases を採用する。Huggingface、vLLM、llama.cpp へ最小限の変更で統合できることも設計理由として書かれている。

## 提案手法

### コアアイデア

BitNet b1.58 は、Transformer の線形層を `BitLinear` に置き換え、各 weight を $\{-1,0,+1\}$ の 3 値に制約する LLM である。TeX の abstract では「every single parameter (or weight) of the LLM is ternary \{-1, 0, 1\}」と述べられている。

この 3 値化により、FP16/BF16 の密な浮動小数点演算ではなく、integer addition が中心の計算へ寄せる。original 1-bit BitNet と比べた追加点は 0 の導入であり、著者はこれを「explicit support for feature filtering」と呼び、modeling capability を強めると主張する。

活性は 1.58-bit ではなく 8-bit で扱う。BitNet の実装を踏襲しつつ、非線形関数前の $[0,Q_b]$ への scaling をやめ、per token で $[-Q_b,Q_b]$ に scaling することで zero-point quantization を避ける、という設計である。

### 重要な定義・数式

$$
\widetilde{W}_{ij}\in\{-1,0,+1\}
$$

**式の意味**: BitNet b1.58 の weight が取りうる値を表す。TeX 本文では式としてではなく「every parameter is ternary, taking on values of \{-1, 0, 1\}」および「constrain the weights to -1, 0, or +1」として説明される。

**記号の定義**:
- $\widetilde{W}_{ij}$ ... 量子化後の weight matrix $\widetilde{W}$ の $i,j$ 成分
- $\{-1,0,+1\}$ ... BitNet b1.58 の各 weight が取りうる ternary values

**この論文での役割**: 論文全体の中核制約である。original 1-bit BitNet に 0 を追加することで、1.58-bit weights と feature filtering を両立する、という主張につながる。

$$
\widetilde{W} = \mathrm{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right)
$$

**式の意味**: 元の weight matrix $W$ を平均絶対値 $\gamma$ で scale し、$-1$ から $1$ の範囲に丸め込んで ternary weight $\widetilde{W}$ を得る absmean quantization である。

**記号の定義**:
- $W$ ... 量子化前の weight matrix
- $\widetilde{W}$ ... 量子化後の ternary weight matrix
- $\gamma$ ... weight matrix の average absolute value
- $\epsilon$ ... 式中で $\gamma$ に加えられる定数。TeX 中に具体的な説明はない
- $\mathrm{RoundClip}$ ... round と clipping を組み合わせた関数

**この論文での役割**: `BitLinear` 内で weights を $-1,0,+1$ に制約する具体的な quantization function であり、手法の再現に必要な主要式である。

$$
\mathrm{RoundClip}(x, a, b) = \max(a, \min(b, \mathrm{round}(x)))
$$

**式の意味**: 入力 $x$ を最も近い整数に round し、その値を区間 $[a,b]$ からはみ出さないように clipping する関数である。この論文では $a=-1, b=1$ として使う。

**記号の定義**:
- $x$ ... round と clipping の対象になる値
- $a,b$ ... clipping の下限と上限
- $\mathrm{round}(x)$ ... $x$ を最も近い整数へ丸める操作
- $\max,\min$ ... 下限・上限を適用するための演算

**この論文での役割**: scale 済みの weight を ternary values に落とす最後の操作である。absmean quantization が単なる scale ではなく、離散値への丸めと範囲制約を含むことを明示している。

$$
\gamma = \frac{1}{nm}\sum_{ij} |W_{ij}|
$$

**式の意味**: weight matrix 全体の絶対値平均を定義する。各 weight の大きさを、この平均値で割ってから丸める。

**記号の定義**:
- $\gamma$ ... absmean quantization で使う scale
- $n,m$ ... weight matrix $W$ のサイズ
- $W_{ij}$ ... $W$ の $i,j$ 成分
- $|W_{ij}|$ ... weight の絶対値

**この論文での役割**: weight matrix の平均的な大きさに応じて ternary への丸めを行うための正規化項である。

$$
[0,Q_b]\ \text{before non-linear functions}\quad\rightarrow\quad[-Q_b,Q_b]\ \text{per token}
$$

**式の意味**: activation quantization の変更点を表す。TeX では、元の BitNet と異なり、非線形関数前の activations を $[0,Q_b]$ に scale せず、すべての activations を per token で $[-Q_b,Q_b]$ に scale して zero-point quantization をなくす、と説明する。

**記号の定義**:
- $Q_b$ ... activation の量子化範囲の上限として TeX に現れる記号。具体的な値は TeX 中では明示されない
- $[0,Q_b]$ ... 元の BitNet で言及される非線形関数前の activation scaling range
- $[-Q_b,Q_b]$ ... BitNet b1.58 で用いる per-token activation scaling range

**この論文での役割**: weight だけでなく activation 側の実装単純化に関わる。著者は、この変更が implementation と system-level optimization に便利で、実験では performance への影響が negligible だったと述べる。

### 実装 / アルゴリズム上の要点

- step1: Transformer 内の `nn.Linear` を `BitLinear` に置き換える。論文は BitNet b1.58 を BitNet architecture に基づく Transformer として定義している。
- step2: weights は absmean quantization で $\{-1,0,+1\}$ に制約する。post-training quantization ではなく、「trained from scratch, with 1.58-bit weights and 8-bit activations」と書かれている点が重要である。
- step3: activations は 8-bit とし、per token で $[-Q_b,Q_b]$ に scale する。zero-point quantization を避けるため、実装とシステム最適化が単純になる、という説明である。
- step4: LLaMA-alike components として RMSNorm、SwiGLU、rotary embedding を使い、all biases を削除する。これは Huggingface、vLLM、llama.cpp との統合を意識した設計である。
- step5: 推論コスト評価では FasterTransformer を用い、BitNet b1.58 側には Ladder の 2-bit kernel を統合する。TeX は latency と memory が 2-bit kernel で測定されており、さらなる最適化余地があると明記している。

## 実験・結果

- **データセット / ベンチマーク**: 100 billion tokens の事前学習には RedPajama dataset を使う。zero-shot tasks は ARC-Easy、ARC-Challenge、Hellaswag、Winogrande、PIQA、OpenbookQA、BoolQ。validation perplexity は WikiText2 と C4 に言及される。2T tokens 実験では StableLM-3B の data recipe に従い、Winogrande、PIQA、SciQ、LAMBADA、ARC-easy で評価する。
- **比較対象 / baseline**: 主比較は著者らが reproduced した FP16 LLaMA LLM。2T tokens 実験では StableLM-3B と比較し、StableLM 3B の値は technical report から直接取ったと書かれている。
- **指標**: Memory (GB, lower is better)、Latency (ms, lower is better)、PPL (lower is better)、zero-shot accuracy、Max Batch Size、Throughput (tokens/s)、energy consumption を見る。runtime latency は time per output token として報告される。
- **主な結果**: Table 1 では、700M と 1.3B では BitNet b1.58 が PPL で FP16 LLaMA にやや劣るが、3B で LLaMA 10.04 に対して BitNet b1.58 9.91 となり、著者は 3B から full precision baseline に match すると主張する。

| Model | Size | Memory (GB) | Latency (ms) | PPL |
|---|---:|---:|---:|---:|
| LLaMA LLM | 700M | 2.08 (1.00x) | 1.18 (1.00x) | 12.33 |
| BitNet b1.58 | 700M | 0.80 (2.60x) | 0.96 (1.23x) | 12.87 |
| LLaMA LLM | 1.3B | 3.34 (1.00x) | 1.62 (1.00x) | 11.25 |
| BitNet b1.58 | 1.3B | 1.14 (2.93x) | 0.97 (1.67x) | 11.29 |
| LLaMA LLM | 3B | 7.89 (1.00x) | 5.07 (1.00x) | 10.04 |
| BitNet b1.58 | 3B | 2.22 (3.55x) | 1.87 (2.71x) | 9.91 |
| BitNet b1.58 | 3.9B | 2.38 (3.32x) | 2.11 (2.40x) | 9.62 |

- **zero-shot accuracy**: Table 2 の平均値は、700M で LLaMA 45.5 vs BitNet b1.58 44.3、1.3B で 46.2 vs 45.4、3B で 49.7 vs 50.2、3.9B BitNet b1.58 で 51.2 である。3B の各タスク値は BitNet b1.58 が ARCe 61.4、ARCc 28.3、HS 42.9、BQ 61.5、OQ 26.6、PQ 71.5、WGe 59.3 で、平均 50.2 と報告される。
- **memory / latency**: Table 1 から、3B では BitNet b1.58 が 2.71 times faster、3.55 times less GPU memory とされる。3.9B BitNet b1.58 は LLaMA 3B より PPL が良く、2.4 times faster、3.32 times less memory と本文で説明される。
- **70B latency / throughput**: Figure 2 の説明本文では、BitNet b1.58 70B は LLaMA baseline より 4.1 times faster とされる。Table 3 では、2 つの 80GB A100 cards、pipeline parallelism、sequence length 512 の条件で、LLaMA 70B は Max Batch Size 16、Throughput 333 tokens/s、BitNet b1.58 70B は Max Batch Size 176、Throughput 2977 tokens/s である。
- **energy**: Figure 3 の本文では、BitNet b1.58 の大部分は INT8 addition calculation、LLaMA は FP16 addition と FP16 multiplication から成ると説明される。energy model in `energycost` and `pokebnn` に基づき、7nm chips で matrix multiplication の arithmetic operations energy consumption を 71.4 times 削減すると報告する。
- **2T tokens**: Table 4 では StableLM-3B と BitNet b1.58 3B を 2T tokens で比較する。Winogrande 64.56 vs 66.37、PIQA 76.93 vs 78.40、SciQ 90.75 vs 91.20、LAMBADA 66.09 vs 67.63、ARC-easy 67.78 vs 68.12、Avg. 73.22 vs 74.34 で、著者は BitNet b1.58 が all end tasks で superior performance と述べる。
- **著者が主張する貢献**: Abstract と Results では、BitNet b1.58 が同じ model size と training tokens の full-precision Transformer LLM に PPL と end-task performance で match しつつ、latency、memory、throughput、energy consumption で cost-effective だと主張する。さらに「1.58-bit LLM defines a new scaling law」とし、13B BitNet b1.58 は 3B FP16 LLM より、30B BitNet b1.58 は 7B FP16 LLM より、70B BitNet b1.58 は 13B FP16 LLM より latency、memory usage、energy consumption の点で efficient だと列挙する。

## 妥当性と限界

- **この主張を支える根拠**: 著者は、同じ RedPajama 100 billion tokens で学習した reproduced FP16 LLaMA と BitNet b1.58 を、PPL、zero-shot accuracy、GPU memory、latency で比較している。さらに 70B throughput、7nm energy model、2T tokens の StableLM-3B 比較を追加し、性能と効率の両面から主張を支えようとしている。
- **著者が認めている limitations / future work**: TeX には独立した limitations section はない。ただし Results では、latency と memory は 2-bit kernel で測定しており、さらに cost を減らす最適化余地があると明記される。Discussion and Future Work では、1-bit MoE LLMs、long sequence support、edge and mobile、new hardware for 1-bit LLMs が将来方向として挙げられる。long sequence については、activations を 16 bits から 8 bits に下げることで同じ resources で context length を doubled でき、さらに 4 bits or even lower へ losslessly compressed できる可能性を future work として残す。
- **読者として注意すべき点**: 3B 未満では PPL と平均 zero-shot accuracy の両方で BitNet b1.58 が FP16 LLaMA に勝っていないため、「match」は主に 3B 以降の主張として読む必要がある。OPTQ、AWQ、SmoothQuant、QuIP/Quip# は Introduction の citation と bibliography で確認できるが、表の直接 baseline ではない。MMLU、GSM8K、HumanEval のような評価は TeX に列挙された実験には出てこない。
- **読者として注意すべき点**: energy の 71.4 times は matrix multiplication の arithmetic operations energy consumption に対する値である。Figure 3 では 512 tokens の end-to-end energy cost も示すが、TeX 本文に図中の詳細数値は書かれていない。
- **読者として注意すべき点**: 「new scaling law」の 13B b1.58 vs 3B FP16、30B vs 7B、70B vs 13B は、本文の列挙どおり latency、memory usage、energy consumption に関する効率比較であり、同じ accuracy を保証する等価関係としては書かれていない。
- **追加で確認したい実験 / 疑問**: 同じ token budget と同じ data mix で、FP16、post-training quantization 系、original BitNet、BitNet b1.58 を直接比較すると、0 による feature filtering の効果をより切り分けられる。long sequence、4-bit 以下 activations、edge/mobile CPU、1-bit 専用 hardware は Discussion の主張に対応する自然な追加検証である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **BitNet b1.58**: 提案モデル名。BitNet architecture に基づき、Transformer の `nn.Linear` を `BitLinear` に置き換え、weights を $\{-1,0,+1\}$、activations を 8-bit として scratch から学習する。
- **1.58-bit weights**: 各 weight が ternary $\{-1,0,+1\}$ の 3 値を取ることを指す。TeX では「resulting in 1.58 bits in the binary system」と説明される。
- **ternary**: この論文では weight が $-1$、$0$、$+1$ の 3 値を取ること。0 は feature filtering を可能にする値として扱われる。
- **feature filtering**: 0 を weight values に追加したことで可能になる、と著者が説明する性質。original 1-bit BitNet との差分を支える概念である。
- **BitLinear**: BitNet 系モデルで `nn.Linear` を置き換える線形層。重みを低 bit に制約し、行列積の計算パラダイムを変えるための部品として使われる。
- **absmean quantization**: weight matrix を $\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|$ で scale してから `RoundClip` で $-1,0,+1$ に丸める量子化関数。
- **zero-point quantization**: この論文では、非線形関数前の activations を $[0,Q_b]$ に scale する元の BitNet 側の処理と対比される。BitNet b1.58 は activations を $[-Q_b,Q_b]$ に per-token scaling することで、これを取り除くと述べる。
- **LLaMA-alike components**: RMSNorm、SwiGLU、rotary embedding、bias removal の組み合わせ。BitNet b1.58 を既存の open-source LLM ecosystem に近づけるための設計である。
- **PPL / perplexity**: language modeling の評価指標。Table 1 では lower is better として扱われ、3B で LLaMA 10.04、BitNet b1.58 9.91 と報告される。
- **zero-shot accuracy**: ARC-Easy、ARC-Challenge、Hellaswag、BoolQ、OpenbookQA、PIQA、Winogrande での end-task performance。Table 2 は lm-evaluation-harness の pipeline に従ったと書かれている。
- **FasterTransformer / Ladder 2-bit kernel**: 推論 latency と GPU memory の測定に使った実装基盤。BitNet b1.58 の測定には Ladder の 2-bit kernel が統合される。
- **throughput**: ここでは 70B モデルでの tokens/s。Table 3 では 2x 80GB A100、sequence length 512、GPU memory limit まで batch size を増やす条件で比較される。
- **new scaling law**: 著者の用語。BitNet b1.58 の model size と FP16 LLM の model size を latency、memory usage、energy consumption の効率軸で比較する主張であり、accuracy 等価だけを意味しない。
- **KV caches**: long sequence inference で memory consumption を増やす要因として Discussion に出る。BitNet b1.58 は activations を 16 bits から 8 bits にすることで同じ resources で context length を doubled できる、と著者は述べる。

## 読む順番の提案

- まず `main.tex` の abstract を読み、「ternary \{-1, 0, 1\}」「matches the full-precision」「latency, memory, throughput, and energy consumption」という主張の範囲を確認する。正規ノートでは Summary の最初の 2 bullet に対応する。
- 次に §The Era of 1-bit LLMs を読み、post-training quantization、BitNet、DRAM-to-SRAM transfer、feature filtering の問題設定を押さえる。正規ノートの Critical Thoughts にある「PTQ 系との直接比較が無い」という注意点は、この節と Table の比較対象を見比べると理解しやすい。
- その後 §BitNet b1.58 の Quantization Function を読み、$\widetilde{W}$、`RoundClip`、$\gamma$、activation range $[-Q_b,Q_b]$ を確認する。正規ノートの Notes / Quotes にある absmean quantization の式へつながる。
- 実験は Table 1、Table 2、Table 3、Table 4 の順に読む。Table 1 は PPL と cost、Table 2 は zero-shot accuracy、Table 3 は 70B throughput、Table 4 は 2T tokens での StableLM-3B 比較である。
- 最後に Discussion and Future Work を読み、1-bit MoE、long sequence、edge/mobile、new hardware が実験済みの結論ではなく将来方向として書かれていることを確認する。正規ノートの Critical Thoughts と Related Papers は、ここで出てくる未検証の主張や引用関係を整理する補助になる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2402.17764v1/`
- 正規ノート: `notes/arXiv-2402.17764v1.md`
