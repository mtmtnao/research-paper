# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases（オンデバイス向け sub-billion LLM のアーキテクチャ設計研究）

- arXiv: https://arxiv.org/abs/2402.14905
- 一次ソース: ../papers/arXiv-2402.14905v2/
- 正規ノート: ../notes/arXiv-2402.14905v2.md

---

## 一言で言うと

モバイル端末で動かせる 1B 未満の LLM では、単に大規模モデルを小さくするのではなく、深く細い Transformer、embedding sharing、grouped-query attention、immediate block-wise weight sharing を組み合わせることで精度とオンデバイス実行性を両立できる、という論文。主張の中心は、sub-billion 領域では scaling law 的な「同じパラメータ数ならアーキテクチャの影響は小さい」という見方に反し、depth と weight utilization が重要になるという点にある（`main.tex` abstract / Section 2）。

## 何を議論する論文か

- **問題設定**: クラウド LLM はコスト・遅延・エネルギー消費が大きく、スマートフォン上で使うには DRAM とバッテリーが制約になる。著者は iPhone 15 の DRAM 6 GB、Google Pixel 8 Pro の DRAM 12 GB、mobile app は DRAM の 10% を超えるべきでない、という制約から sub-billion parameter LLM を動機づける（`main.tex` Introduction, Figure 2）。
- **対象範囲 / 仮定**: 主な設計探索は 125M と 350M の Transformer language model。探索実験は 120k iterations / 0.25T tokens、最終表のモデルは 480k iterations / 1T tokens で学習される。オンデバイス推論では memory movement が latency bottleneck になり、SRAM は通常 around 20MB で 1 transformer block 程度しか保持できない、という仮定が layer sharing の採用理由になっている（Section 2.3）。
- **既存研究との差分**: OPT、BLOOM、GPT-Neo、Cerebras-GPT、Pythia などは小さい variant を含むが、著者は「sub-billion parameters の制約下に最適化されていない」と位置づける。Kaplan et al. の scaling law が示す「architecture designs have negligible impact」という見方に対し、125M / 350M の grid search で depth と width の配分を調べる（Section 2.2.2, Related Work）。
- **この論文で答えたい問い**: 限られたパラメータ予算を、層数・幅・embedding・attention head・weight sharing にどう配分すれば、オンデバイス用途に十分な精度と速度を持つ sub-billion LLM を作れるか。

## 背景と前提

- Transformer LLM は token embedding、multi-head self-attention、feed-forward network (FFN) を積み重ねる。小さいモデルでは embedding 層の比率が大きく、例えば `vocab_size=32k`、`embedding_dim=512` なら入力 embedding と出力 fully-connected layer がそれぞれ 16M parameters を持つため、125M モデル全体の 20% 以上を占める（Section 2.2.3）。
- この論文での **MobileLLM** は、SwiGLU FFN、deep-and-thin architecture、input-output embedding sharing、grouped query attention (GQA) を組み合わせた baseline family を指す。**MobileLLM-LS** はそこに immediate block-wise layer sharing を加えたもの。
- **layer sharing** は、distinct weights を増やさず、同じ transformer block を繰り返し計算する方法である。Table `tab:main` の MobileLLM-LS の `#Layers` は caption にある通り distinct weights の層数で、Table `tab:latency` では MobileLLM-LS-125M が `2x30 layers, adjacent blocks sharing weights` と説明される。
- 比較対象は、zero-shot common sense reasoning では Cerebras-GPT、LaMini-GPT、Galactica、OPT、GPT-Neo、Pythia、RWKV、BLOOM など。downstream では chat benchmark と API calling も扱い、API calling では LLaMA-v2 7B も比較に入る。
- 事前学習データの種類・出典は TeX 中に明示されていない。したがって、このノートでは性能差を「アーキテクチャだけの差」と断定しない。

## 提案手法

### コアアイデア

著者は、sub-billion scale では「限られた weights をどう再利用し、どこに配分するか」が大きな設計変数になると見る。具体的には、FFN を `FC -> ReLU -> FC` から SwiGLU に変える、同程度のパラメータ数で width より depth を増やす、入出力 embedding を共有して浮いたパラメータを層や embedding dimension に回す、GQA で key-value heads の冗長性を減らす、という順に MobileLLM を構成する（Figure 3, Appendix Table `tab:appendix_roadmap`）。

さらに、深いモデルが有利という観察から、同じ block weights を隣接する 2 block で共有する immediate block-wise sharing を導入する。これはパラメータ数を増やさずに実効的な計算深さを増やすが、shared weights を SRAM/cache に置いたまま続けて 2 回計算できるため、単に 60 層の non-shared model にするより memory movement が少ない、というのがハードウェア側の根拠である（Figure 5, Table `tab:latency`）。

### 重要な定義・数式

この論文は新しい連続最適化問題を定義する論文ではなく、明示式は多くない。以下は TeX 中で手法・評価判断に直接関わる定義・式に絞る。

$$
(vocab\_size, embedding\_dim) = (32k, 512) \Rightarrow 16M \text{ parameters per embedding layer}
$$

**式の意味**: 入力 embedding と出力 fully-connected layer の weight size がどちらも `(vocab_size, embedding_dim)` になる、という Section 2.2.3 の記述を、論文中の具体例で書いたもの。`32k x 512` なので各 embedding 層が 16M parameters になる。

**記号の定義**:
- $vocab\_size$ ... 語彙数。論文中の例では 32k。
- $embedding\_dim$ ... token embedding の次元。論文中の例では 512。
- $16M$ ... 入力 embedding または出力 fully-connected layer 片方のパラメータ数。

**この論文での役割**: 125M 程度の小型 LLM では embedding が全体の大きな比率を占めるため、input-output embedding sharing によって 16M parameters、約 11.8% を削減できる。Table `tab:embeding_share` では、30-layer 125M model で `Without emb-share` 135M / Avg. 44.8、`+ emb-share` 119M / Avg. 44.6、`+ emb-share, ↑ depth` 125M / Avg. 45.0 と報告される。

$$
H_{\mathrm{KV}} = \frac{H}{n}, \qquad n \in \mathbb{Z}^{+}
$$

**式の意味**: Section 2.2.4 の GQA の説明「the number of key-value heads is $1/n$ that of query heads」を式で整理したもの。query heads を $H$、key-value heads を $H_{\mathrm{KV}}$ とすると、key/value は query より少ない head 数で共有される。

**記号の定義**:
- $H$ ... query heads の数。
- $H_{\mathrm{KV}}$ ... key-value heads の数。
- $n$ ... query heads が kv-heads の何倍かを表す正の整数。TeX では、query heads が割り切れる正の整数と説明される。

**この論文での役割**: GQA はもともと KV-cache 削減の手法だが、この論文では小型 LM における weight re-utilization として使われる。Appendix Table `tab:larger_architecture` の最終設計では MobileLLM-125M が `#Head=9, #KV-Head=3, Emb Dim=576`、MobileLLM-350M が `#Head=15, #KV-Head=5, Emb Dim=960` で、どちらも $n=3$ になっている。

$$
\mathcal{L}_{CE} = -\frac{1}{n}\sum_c\sum^n_{i=1} p_c^{\mathcal{T}}(X_i)\log(p_c^{\mathcal{S}}(X_i))
$$

**式の意味**: Appendix `Knowledge Distillation` にある KD loss。LLaMA-v2 7B teacher の logits から得た soft labels と、125M / 350M student の出力分布の cross-entropy を計算する。

**記号の定義**:
- $X_i$ ... current batch の $i$ 番目の sample。
- $n$ ... batch 内の sample 数。
- $c$ ... class の index。この論文では vocabulary size に等しい。
- $\mathcal{T}$ ... teacher network、ここでは LLaMA-v2 7B。
- $\mathcal{S}$ ... student network、ここでは 125M または 350M model。
- $p_c^{\mathcal{T}}(X_i)$, $p_c^{\mathcal{S}}(X_i)$ ... teacher / student が class $c$ に割り当てる確率。

**この論文での役割**: KD は採用手法ではなく negative result として重要である。Appendix Table `tab:appendix_kd` では、125M は Label 43.9 / 29h に対し Label+KD 43.8 / 93h、350M は Label 49.1 / 42h に対し Label+KD 48.8 / 109h で、著者は KD を使わず hard labels で学習すると結論づける。

### 実装 / アルゴリズム上の要点

- step1: baseline model の FFN を `FC -> ReLU -> FC` から SwiGLU に変更する。125M では zero-shot reasoning Avg. が 42.6 から 43.9 に上がる（Section 2.2.1, Appendix Table `tab:appendix_roadmap`）。
- step2: 同程度のパラメータ数で depth / width を grid search する。125M では 9 models、350M では 10 models、合計 19 models を学習し、deeper and thinner models が多くのタスクで有利だと報告する（Section 2.2.2, Appendix Tables `tab:appendix_depth_vs_width`, `tab:appendix_depth_more_task`）。
- step3: input embedding と output fully-connected layer の weights を共有する。30-layer 125M setting では 135M から 119M に削減され、深さを 32 layers に増やしても 125M に収まる（Table `tab:embeding_share`）。
- step4: GQA を入れ、kv-heads を query heads より減らす。head sweep では 16 query heads と 4 kv-heads が accuracy / memory trade-off の指針として示されるが、最終 MobileLLM は Appendix Table `tab:larger_architecture` の通り 125M が L30/H9/KV3/Emb576/Hidden1536/124.6M、350M が L32/H15/KV5/Emb960/Hidden2560/345.3M。
- step5: MobileLLM-LS では immediate block-wise sharing を採用する。Table `table:layer_share` では repeat-all-over share が Avg. 45.2 (125M) / 50.7 (350M) で最も高いが、Figure 5 と Section 2.3 では cache locality の理由から immediate block-wise share を選ぶ。
- step6: 最終 training は Adam optimizer、weight decay 0.1、initial learning rate 2e-3、cosine learning-rate decay、32 A100 GPUs、batch size 32 per GPU、480k iterations / 1T tokens。API calling fine-tuning は synthetic dataset 5000 training samples / 2500 testing samples、平均 8 conversation turns、4 epochs、Adam、linear-decay learning rate starting at 2e-5、weight decay 0.01。

## 実験・結果

- **データセット / ベンチマーク**: zero-shot common sense reasoning は ARC-easy、ARC-challenge、BoolQ、PIQA、SIQA、HellaSwag、OBQA、WinoGrande。question answering / reading comprehension は TQA と RACE。chat は AlpacaEval と MT-Bench。API calling は著者が生成した synthetic dataset。quantization は W8A8 PTQ、latency は iPhone 13 (iOS 17.2.1) + ExecuTorch + MPS。
- **比較対象 / baseline**: OPT、BLOOM、Galactica、Cerebras-GPT、GPT-Neo、Pythia、RWKV、LaMini-GPT など。chat では Falcon-1.3B や OPT-1.3B も比較され、API calling では LLaMA-v2 7B が入る。baseline results は open-source Hugging Face models を同じ evaluation procedure で評価したと本文にある。
- **指標**: zero-shot common sense reasoning は各 task accuracy と Avg.、TQA は F1 score、RACE は accuracy、AlpacaEval は GPT-4 evaluator による text-davinci-001 への pairwise win rate、MT-Bench は GPT-4 rating の平均 score、API calling は EM_intent / EM_structure と Rouge-1 / Rouge-L、latency は load / init / execute time。
- **主な結果**: Table `tab:main` では MobileLLM-125M が Avg. 46.3、MobileLLM-LS-125M が 47.0。Pythia-160M は 42.5、RWKV-169M は 43.6 なので、本文の主張通り +3.8 / +2.7 points で、かつ 22% / 26% smaller とされる。350M では MobileLLM-350M が 51.3、MobileLLM-LS-350M が 52.1、RWKV-430M が 47.0。
- **主な結果**: Table `tab:more_task` では MobileLLM-125M が TQA 1-shot 13.9 / RACE middle 39.7 / high 28.9、MobileLLM-LS-125M が 14.2 / 40.7 / 29.6。MobileLLM-350M は TQA 1-shot 22.0、MobileLLM-LS-350M は RACE middle 47.3 / high 33.7。
- **主な結果**: Table `tab:chat` では MobileLLM-LS-350M が AlpacaEval win 48.20% / MT-Bench 3.16、MobileLLM-350M が 47.08% / 3.28。比較として Falcon-1.3B は 30.38% / 2.54、OPT-1.3B は 38.84% / 2.24。
- **主な結果**: Table `tab:api` では MobileLLM-350M が EM_intent 65.3、EM_structure 48.8、R1 46.8、RL 44.6。LLaMA-v2 7B は 62.8、50.9、56.5、54.3 で、著者は MobileLLM-350M が intent / structure exact match で comparable と主張する。
- **主な結果**: Table `tab:latency` では MobileLLM-125M が load 39.2 ms / init 1361.7 ms / execute 15.6 ms、MobileLLM-LS-125M が 43.6 ms / 1388.2 ms / 16.0 ms、60-layer non-shared が 68.6 ms / 3347.7 ms / 29.0 ms。本文は LS の overhead を load+init 2.2%、execute 2.6%、non-shared の増加を load+init 143%、execute 86% とまとめる。
- **著者が主張する貢献**: depth が width より重要であることを sub-billion LLM で示す、embedding sharing と GQA を小型モデルで再評価する、immediate block-wise weight sharing を提案する、125M / 350M の zero-shot tasks で previous SOTA を上回る、chat / API calling の on-device use cases で同サイズモデルを大きく上回る、600M / 1B / 1.5B へ設計思想が scale することを Appendix で示す。

## 妥当性と限界

- **この主張を支える根拠**: depth vs width は 125M で 9 models、350M で 10 models の grid search に基づく。Appendix Table `tab:appendix_depth_vs_width` では 125M 相当で 24 / 30 layers が Avg. 44.8、350M 相当で 32 layers が Avg. 49.8 と高い。本文は TQA / RACE でも深いモデルの傾向が強いと述べる。
- **この主張を支える根拠**: Appendix Table `tab:appendix_roadmap` が設計選択ごとの寄与を分解している。125M では baseline 42.6、SwiGLU 43.9、deep-thin 44.8、embedding share 44.6、GQA 45.0、1T training 46.3、layer sharing 1T 47.0。350M では baseline 47.4 から layer sharing 1T 52.1 まで上がる。
- **この主張を支える根拠**: オンデバイス性は Table `tab:latency` の iPhone 13 実測で補強される。特に LS は同程度の model size なので load/init の増加が小さく、shared weights の data locality により execute time も 16.0 ms に留まる、という説明になっている。
- **著者が認めている limitations / future work**: 明示的な `Limitations` 節は TeX 中にない。関連する注意として、KD は 2.6--3.2x slower で精度が同等以下だったため採用しない、W8A8 PTQ は gap within 0.5% で compatible、Impact Statement では LLM inference の energy consumption mitigation を目的とする、と述べる。
- **読者として注意すべき点**: 事前学習データの種類・出典が TeX 中に明示されていないため、既存モデルとの差を architecture のみに帰すことはできない。本文は baseline methods を open-source Hugging Face models で再評価したと述べるが、pre-training corpus をそろえた比較とは書いていない。
- **読者として注意すべき点**: layer sharing の精度だけを見ると Table `table:layer_share` では repeat-all-over share が immediate block-wise share より高い（125M: 45.2 vs 45.0、350M: 50.7 vs 50.2）。著者は SRAM/cache 利用を理由に immediate を採るが、seed variance や統計的有意性は TeX 中に示されていない。
- **読者として注意すべき点**: chat の AlpacaEval / MT-Bench は GPT-4 evaluator に依存する。API calling dataset は language model に conversation を simulate させて生成した synthetic dataset であり、実利用分布との一致は TeX 中には示されていない。
- **追加で確認したい実験 / 疑問**: 同じ事前学習データ・tokenizer・training compute で既存小型 architecture と MobileLLM を比較すると、architecture contribution をより明確に切り分けられる。600M / 1B / 1.5B では Appendix に結果はあるが、125M / 350M と同じ粒度の depth-width grid は示されていないため、大きめのモデルで同じ傾向がどこまで続くかも追加確認したい。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **sub-billion parameter LLM**: 1B 未満の language model。モバイル端末の DRAM とエネルギー制約から、この論文の主要対象になる。
- **MobileLLM**: SwiGLU FFN、deep-and-thin architecture、input-output embedding sharing、GQA を組み合わせた著者の baseline model family。
- **MobileLLM-LS**: MobileLLM に immediate block-wise layer sharing を加えた model。LS は layer sharing を意味する。
- **deep-and-thin / lanky architecture**: 同程度のパラメータ数で layer 数を増やし、embedding dimension や head 数を相対的に抑える設計。著者は 125M で 30 または 42 layers が 12 layers より良いと述べる。
- **SwiGLU FFN**: FFN の activation / gating 設計。本文では vanilla FFN `FC -> ReLU -> FC` からの置換で 125M Avg. 42.6 から 43.9 へ上がったとされる。
- **input-output embedding sharing**: input embedding weights を output fully connected layer weights として再利用する方法。小型モデルでは embedding 層の比率が高いため、weight utilization 改善策として扱われる。
- **GQA (Grouped Query Attention)**: query heads より少ない key-value heads を使う attention。著者は small LMs でも key-value heads の redundancy を減らす手法として評価する。
- **kv-head**: attention の key/value 側の head。Section 2.2.4 では kv-heads を query heads の `1/n` にし、attention scores と output の計算で repeated すると説明される。
- **immediate block-wise sharing**: 隣接する transformer blocks が weights を共有し、shared weights を cache に置いたまま連続して計算する方式。Figure 5 の setting (b)。
- **repeat-all-over sharing / reverse sharing**: Figure 5 と Table `table:layer_share` で比較される別の sharing strategy。repeat-all-over は精度が少し高いが、著者は cache 利用の観点から immediate block-wise を採用する。
- **W8A8 PTQ**: weight 8-bit / activation 8-bit の post-training quantization。Appendix Table `tab:appendix_ptq` では BF16 に対する Avg. gap が 0.0--0.4 と報告される。
- **EM_intent / EM_structure**: API calling の exact match 指標。EM_intent はどの API を呼ぶか、EM_structure は API function 内の内容構造を正しく予測できるかを測る。
- **AlpacaEval / MT-Bench**: chat benchmark。AlpacaEval は single-turn 805 questions で text-davinci-001 への pairwise win rate、MT-Bench は 160 questions / 8 knowledge domains の multi-turn benchmark で GPT-4 が 1--10 点で評価する。

## 読む順番の提案

- まず `main.tex` abstract と Introduction を読み、なぜ sub-billion / on-device が問題になるかを確認する。DRAM 6--12 GB、app は 10% 以下、7B model は 0.7 J/token、350M 8-bit model は 0.035 J/token という数値が動機づけになる。
- 次に Section 2.1--2.3 と Figure 3--5 を読む。正規ノートの `Summary（著者の主張）` の「手法」部分につながる。特に `tab:appendix_roadmap`、`tab:appendix_depth_vs_width`、`tab:embeding_share`、`tab:appendix_kv_heads` を横に置くと、各設計選択の根拠が追いやすい。
- 実験結果は Table `tab:main`、`tab:more_task`、`tab:chat`、`tab:api`、`tab:latency` を先に見る。正規ノートの `Notes / Quotes` に並んでいる数値の多くはここに対応する。
- 限界を読むには Section 3.4 Quantization、Section 3.5 Knowledge Distillation、Appendix `Knowledge Distillation`、Appendix `API Calling Dataset` を見る。正規ノートの `Critical Thoughts（評価・疑問）` と対応するが、事前学習データや synthetic API dataset に関する注意は TeX で明示されている範囲に限定して読む。
- 参考文献は `main.bbl` で確認する。GQA は Ainslie et al. 2023 と PaLM、embedding sharing は OPT、scaling law は Kaplan et al. 2020、weight sharing の関連研究は Sliced Recursive Transformer と Subformer に接続される。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2402.14905v2/`
- 正規ノート: `notes/arXiv-2402.14905v2.md`
