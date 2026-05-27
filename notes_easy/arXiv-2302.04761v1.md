# Toolformer: Language Models Can Teach Themselves to Use Tools（自己教師ありツール使用を学習する言語モデル）

- arXiv: https://arxiv.org/abs/2302.04761
- 一次ソース: ../papers/arXiv-2302.04761v1/
- 正規ノート: ../notes/arXiv-2302.04761v1.md

---

## 一言で言うと

Toolformer は、事前学習済み LM が少数の API 使用例だけを手がかりに、どのツールをいつ・どんな入力で呼び、その結果を後続トークン予測へどう組み込むかを自己教師ありで学ぶ方法である。GPT-J 6.7B をベースに、QA、Wikipedia Search、Calculator、Calendar、Machine Translation を組み込み、複数の zero-shot downstream tasks で性能を改善しつつ、API 無効時の言語モデリング perplexity をほぼ悪化させないと主張する。

## 何を議論する論文か

- **問題設定**: 大規模 LM は zero-shot / few-shot で多くの NLP タスクを解ける一方、最新情報へのアクセス、事実 hallucination、低リソース言語、正確な算術、時間経過の認識に弱い。論文は、これらを外部ツールの API call として LM に使わせる問題を扱う。
- **対象範囲 / 仮定**: 各 API の入力と出力は text sequences として表現できる必要がある。各 API について「handful of demonstrations」を用意し、普通の language modeling dataset $\mathcal{C}$ に API call を挿入して $\mathcal{C}^*$ を作る。実験の主対象は CCNet subset と GPT-J 6.7B である。
- **既存研究との差分**: WebGPT、LaMDA、Internet-Augmented Dialogue などは large amounts of human supervision に依存する。一方、PAL、ReAct、TALM などの tool use は、タスク固有 few-shot prompt や downstream task finetuning に寄る。Toolformer は、汎用 LM が self-supervised loss で API call を選別し、zero-shot prompted setting でツール選択も自分で行う点を差分としている。
- **この論文で答えたい問い**: LM は、人間が大量に「ここでこのツールを使え」と注釈しなくても、API call の位置、API 名、入力、実行結果の使い方を学べるか。その学習は downstream task を改善し、同時に通常の language modeling ability を壊さないか。

## 背景と前提

- **Language Model / next-token prediction**: この論文での LM $M$ は、文脈 $z_1,\ldots,z_n$ に対して次トークン $z_{n+1}$ の確率 $p_M(z_{n+1}\mid z_1,\ldots,z_n)$ を与えるモデルとして扱われる。API call の有用性も、後続トークンの weighted cross entropy loss が下がるかで測る。
- **API call のテキスト化**: API call は通常のテキスト列へ挿入される。論文上は `<API>`, `</API>`, `\rightarrow` を使うが、脚注では実装上 ` [`（先頭スペース付き）、`]`、`->` を使い、LM の vocabulary を変更しないと説明している。
- **self-supervised filtering**: 人間の正解ラベルではなく、API の結果を prefix として与えたときに $M$ 自身の未来トークン予測 loss がどれだけ減るかをフィルタ信号にする。ここが Toolformer の中心である。
- **使用ツール**: Question Answering は Natural Questions で finetuned された Atlas、Calculator は Python script による四則演算、Wikipedia Search は KILT Wikipedia dump 上の BM25 retriever、Machine Translation は 600M parameter NLLB と fastText language detection、Calendar は入力なしで current date を返す API として定義される。
- **baseline との関係**: 実験の主 baseline は GPT-J、GPT-J + CC、Toolformer disabled、Toolformer であり、多くの task では OPT 66B と GPT-3 175B も比較される。GPT-3 は instruction-tuned ではない original `davinci` variant と明記されている。

## 提案手法

### コアアイデア

Toolformer は、まず LM $M$ に in-context learning で通常文書中の API call 候補を生成させる。次に実際の API を実行して結果 $r_i$ を得る。最後に、API call と結果を与えたときの後続トークン予測 loss $L_i^+$ が、呼ばない場合または結果なしで呼ぶ場合の loss $L_i^-$ より十分小さいものだけを残す。残った API call を元テキストに interleave した dataset $\mathcal{C}^*$ で $M$ を通常の language modeling objective により finetune する。

この設計により、学習データの文書内容自体は元の $\mathcal{C}$ と同じで、API call の挿入だけが追加される。論文はこれを、LM の generality と language modeling abilities を保ちながら、いつ・どのツールをどう使うかを学ぶための仕組みとして位置づけている。

### 重要な定義・数式

$$
\begin{aligned}
\text{e}(c)     & = \texttt{<API>}\, a_c \texttt{(} i_c \texttt{)}\, \texttt{</API>} \\
\text{e}(c, r)  & = \texttt{<API>}\, a_c \texttt{(} i_c \texttt{)} \rightarrow r\, \texttt{</API>}
\end{aligned}
$$

**式の意味**: API call $c$ を、結果なしの文字列 $\text{e}(c)$ と、結果 $r$ を含む文字列 $\text{e}(c,r)$ として線形化する定義である。

**記号の定義**:
- $c=({a}_c,{i}_c)$ ... API call。$a_c$ が API 名、$i_c$ が API への入力。
- $r$ ... API call $c$ に対する API の返答。
- $\texttt{<API>}$, $\texttt{</API>}$, $\rightarrow$ ... API call の開始、終了、結果挿入を表す特殊トークン。ただし実装では ` [`（先頭スペース付き）、`]`、`->` を用いる。

**この論文での役割**: すべてのツールを通常の LM 入出力と同じ text sequence に落とし込むための表現である。これにより、QA、Calculator、Wikipedia Search、MT、Calendar を同じ sampling / filtering / finetuning 手順で扱える。

$$
p_i = p_M(\texttt{<API>} \mid P(\mathbf{x}), x_{1:i-1})
$$

**式の意味**: 入力テキスト $\mathbf{x}$ の位置 $i$ で、LM が API call を開始する特殊トークン `<API>` を次に出す確率を測る。$p_i$ が閾値 $\tau_s$ を超える位置だけを API call 候補位置にする。

**記号の定義**:
- $p_i$ ... 位置 $i$ で API call を始める確率。
- $p_M$ ... LM $M$ が次トークンへ割り当てる条件付き確率。
- $P(\mathbf{x})$ ... API call 例を含む prompt。Appendix の tool prompts に具体例がある。
- $x_{1:i-1}$ ... 元テキスト $\mathbf{x}=x_1,\ldots,x_n$ の先頭から位置 $i-1$ まで。
- $\tau_s$ ... sampling threshold。Appendix では default $\tau_s=0.05$、Calculator / MT では $\tau_s=0.0$ とされる。

**この論文での役割**: API call の「どこで呼ぶか」を決める第一段階である。default では閾値を超えた位置が $k$ 個を超える場合に上位 $k=5$ 位置を残し、各位置で最大 $m=5$ call を生成する。Calculator / MT は候補が少ないため $k=20$, $m=10$ を用いる。

$$
L_i(\mathbf{z}) = -\sum_{j=i}^n w_{j-i} \cdot \log {p_M(x_j \mid \mathbf{z}, x_{1:j-1})}
$$

**式の意味**: 位置 $i$ 以降の元トークン $x_i,\ldots,x_n$ を、追加 prefix $\mathbf{z}$ を与えた状態でどれだけ予測しにくいかを測る weighted cross entropy loss である。

**記号の定義**:
- $L_i(\mathbf{z})$ ... 追加 prefix $\mathbf{z}$ を与えたときの位置 $i$ 以降に対する weighted loss。
- $\mathbf{z}$ ... 比較対象として与える prefix。空列 $\varepsilon$、結果なし API call、結果あり API call などが入る。
- $w_{j-i}$ ... API call 位置からの距離に応じた重み。
- $x_j$ ... 元テキストの $j$ 番目のトークン。

**この論文での役割**: Toolformer の filtering 信号を定義する式である。API の返答が本当に後続トークン予測に役立つかを、downstream label ではなく LM 自身の loss で判定する。

$$
\begin{aligned}
L_i^+ & = L_i(\text{e}(c_i, r_i))\\
L_i^- & =  \min \left( L_i(\varepsilon), L_i(\text{e}(c_i, \varepsilon )) \right)\\
L_i^- - L_i^+  &\geq \tau_f
\end{aligned}
$$

**式の意味**: API call と結果を与えた場合の loss $L_i^+$ が、API を呼ばない場合または結果なし call を与える場合の loss $L_i^-$ より、少なくとも $\tau_f$ だけ小さい call だけを残す。

**記号の定義**:
- $c_i$ ... 位置 $i$ に挿入される API call 候補。
- $r_i$ ... $c_i$ を実行して得られた API response。
- $L_i^+$ ... API call と response を両方与えた場合の weighted loss。
- $L_i^-$ ... API call なし、または API call 文字列だけで response なしの場合の loss の小さい方。
- $\varepsilon$ ... 空列。
- $\tau_f$ ... filtering threshold。default は $1.0$、Calculator / MT では $0.5$ とされる。

**この論文での役割**: API call を教材に残すかどうかの中核判定である。$L_i(\text{e}(c_i,\varepsilon))$ も比較に入れることで、API 入力文字列そのものが後続トークンのヒントになる効果を差し引いている。

$$
w_t  = \frac{\tilde{w}_t}{ \sum_{s \in \mathbb{N}} \tilde{w}_s}
\text{ with }
\tilde{w}_t = \max(0, 1 - 0.2 \cdot t)
$$

**式の意味**: API call の近くのトークンほど大きく評価し、離れたトークンの寄与を小さくする重み関数である。$\tilde{w}_t$ は $t$ が増えると 0.2 ずつ減り、負になったら 0 になる。

**記号の定義**:
- $w_t$ ... 正規化された重み。
- $\tilde{w}_t$ ... 正規化前の重み。
- $t$ ... API call 位置からの相対距離。
- $\mathbb{N}$ ... 非負整数の添字集合として用いられている。

**この論文での役割**: API response が役立つのは挿入位置の近くであるという設計を filtering に入れる。著者は「API calls happen close to where the information ... is actually helpful」と説明している。

### 実装 / アルゴリズム上の要点

1. **Sampling API Calls**: 各 tool について prompt $P(\mathbf{x})$ を作り、各位置の $p_i$ を計算する。default は $\tau_s=0.05$、閾値を超えた位置が $k$ 個を超える場合は top $k=5$ positions、各位置で最大 $m=5$ calls。Calculator と MT は候補を増やすため $\tau_s=0.0$, $k=20$, $m=10$。
2. **Executing API Calls**: 生成された call を実際に実行し、単一 text sequence の response $r_i$ を得る。API 実行はツール依存で、neural network、Python script、retrieval system などを含む。
3. **Filtering API Calls**: $L_i^- - L_i^+ \geq \tau_f$ を満たす call だけを保持する。default $\tau_f=1.0$、Calculator / MT は $\tau_f=0.5$。
4. **Model Finetuning**: 残った call を元テキストに挿入し、$\mathcal{C}^*$ を作る。GPT-J を batch size 128、learning rate $1\cdot10^{-5}$、linear warmup 10% で finetune する。Appendix では最大 25k examples per API、max sequence length 1,024、8 NVIDIA A100 40GB GPUs、BF16、DeepSpeed ZeRO-3、最大 2k steps と書かれている。
5. **Inference**: 生成中に $\rightarrow$（実装上は `->`）が出たら decoding を中断し、API を実行して response と `</API>` を挿入して続ける。downstream evaluation では `<API>` が top-$k$ に入ったら call を開始する修正を使い、主設定は $k=10$。無限ループ回避のため、1 input あたり最大 1 API call に制限する。

## 実験・結果

- **データセット / ベンチマーク**: 訓練用 language modeling dataset は CCNet subset。downstream は LAMA の SQuAD / Google-RE / T-REx、math benchmarks の ASDiv / SVAMP / MAWPS、QA の WebQS / NQ / TriviaQA、MLQA の Es / De / Hi / Vi / Zh / Ar、temporal datasets の TempLAMA / Dateset。通常 LM 評価は WikiText と 10,000 documents の CCNet validation subset。Dateset は Appendix で 500 current dates から作られ、合計 9,400 examples とされる。
- **比較対象 / baseline**: GPT-J、GPT-J + CC、Toolformer disabled、Toolformer が主比較。さらに多くの task で OPT 66B と GPT-3 175B を比較する。scaling laws では GPT-2 124M / 355M / 775M / 1.6B と GPT-J 6.7B を比較し、tool subset は QA、Calculator、Wikipedia Search。
- **指標**: LAMA / TempLAMA / Dateset は correct word が first five words に含まれるか。Math は first number predicted、式を含む場合は `=` 後の first number。QA は first 20 words に correct answer が含まれるか。MLQA は generation を 10 words に制限し correct answer を含む percentage。Language modeling は perplexity。
- **主な結果**: LAMA Table `tab:lama_results` では Toolformer が SQuAD / Google-RE / T-REx で 33.8 / 11.5 / 53.5、GPT-J は 17.8 / 4.9 / 31.9、GPT-3 175B は 26.8 / 7.0 / 39.8。Toolformer は QA tool を 98.1% 使用し、Wikipedia Search は LAMA で unfair advantage を避けるため禁止される。
- **主な結果**: Math Table `tab:math_results` では ASDiv / SVAMP / MAWPS で Toolformer が 40.4 / 29.4 / 44.0、GPT-J が 7.5 / 5.2 / 9.9、GPT-3 175B が 14.0 / 10.0 / 19.8。Calculator tool は 97.9% の examples で使われる。
- **主な結果**: QA Table `tab:qa_results` では WebQS / NQ / TriviaQA で Toolformer が 26.3 / 17.7 / 48.8、GPT-3 175B が 29.0 / 22.6 / 65.9。Toolformer は同サイズ GPT-J 系 baseline を上回るが、GPT-3 175B には届かない。著者は単純な BM25 search と、検索結果を見て query reformulation や browsing をできない点を理由として挙げる。
- **主な結果**: MLQA Table `tab:mt_results_percentage` では Toolformer が Es / De / Hi / Vi / Zh / Ar で 20.6 / 13.5 / 1.4 / 10.6 / 16.8 / 3.7。GPT-J は 15.2 / 16.5 / 1.3 / 8.2 / 18.2 / 8.2。API calls は全言語で Toolformer disabled を上回るが、vanilla GPT-J を一貫して上回らない。MT tool 使用率は 63.8% から 94.9%、例外として Hindi は 7.3%。
- **主な結果**: Temporal Table `tab:temporal_results` では Toolformer が TempLAMA / Dateset で 16.3 / 27.3、GPT-J は 13.7 / 3.9、GPT-3 175B は 15.5 / 0.8。TempLAMA で Calendar 使用率は 0.2% にすぎず、改善は主に Wikipedia Search と QA による。Dateset では Calendar が 54.8% 使われる。
- **主な結果**: Perplexity Table `tab:perplexities` では GPT-J が WikiText / CCNet で 9.9 / 10.6、GPT-J + CC が 10.3 / 10.5、Toolformer disabled が 10.3 / 10.5。著者は、API call を加えた training は API disabled の通常 LM perplexity を悪化させないと主張する。
- **主な結果**: Table `tab:c_star` では $\tau_f=1.0$ の API call examples 数が Question Answering 18,526、Wikipedia Search 60,974、Calculator 994、Calendar 20,587、Machine Translation 1,034。$\tau_f=0.5$ では 51,987 / 207,241 / 3,680 / 61,811 / 3,156。
- **主な結果**: Figure `fig:scaling_laws` では、tools を有効に活用する能力は around 775M parameters で現れると著者は述べる。小さい GPT-2 models では tools 有り/無しが近く、大きい model でも API calls 有りと無しの差は残る。
- **著者が主張する貢献**: self-supervised に API call の生成と選別を行う汎用枠組み、perplexity-based filtering、5 つの tool を統合した zero-shot 改善、6.7B GPT-J ベースで GPT-3 175B を複数 task で上回る結果、tool use ability がモデルサイズに応じて現れるという scaling 観察である。

## 妥当性と限界

- **この主張を支える根拠**: 同じ GPT-J 系で GPT-J、GPT-J + CC、Toolformer disabled、Toolformer を分けて比較しているため、CCNet 追加学習だけの効果と API call 使用の効果をある程度切り分けている。LAMA、Math、QA、MLQA、Temporal の各 task で、使用された tool の割合も報告し、性能改善がどの API に由来するかを検討している。
- **この主張を支える根拠**: Table `tab:top-k` は decoding modification の影響を示す。T-REx では $k=0$ が 34.9、$k=1$ が 47.8、$k=3$ が 52.9、$k=10$ が 53.5。WebQS では $k=0$ が 18.9、$k=1$ が 19.3、$k=3$ と $k=10$ が 26.3。API call rate も T-REx で 0.0 / 40.3 / 82.8 / 98.1%、WebQS で 0.0 / 8.5 / 99.3 / 100.0% と増える。
- **この主張を支える根拠**: Data Quality analysis の Table `fig:model_outputs` は $L_i^- - L_i^+$ が大きい call ほど直感的に useful である傾向を示す。一方で「Fast train success」のように関連性が低いのに perplexity を下げる例もあり、filter は完全な意味的正しさの保証ではない。
- **著者が認めている limitations / future work**: chain of tool calls ができない。理由は「API calls for each tool are generated independently」で、finetuning dataset に chained tool use examples がないためである。
- **著者が認めている limitations / future work**: interactive tool use ができない。特に search engine では、複数結果の browsing や search query refinement が必要な場合があるが、現手法では WebGPT のような interaction を扱わない。
- **著者が認めている limitations / future work**: API call をするかどうかが input の exact wording に sensitive である。これは zero-shot / few-shot prompt sensitivity と同系統の問題として述べられている。
- **著者が認めている limitations / future work**: sample inefficiency がある。特に calculator API は million documents 以上を処理しても useful calls が few thousand examples にとどまる。著者は iterative bootstrapping を potential solution として挙げるが、本論文では実施していない。
- **著者が認めている limitations / future work**: API call の tool-dependent computational cost を考慮しない。現手法は loss reduction だけで call を残し、実行コストを目的関数に入れていない。
- **読者として注意すべき点**: Toolformer の強さは tool の品質に依存する。QA では Question Answering tool を disabled にしており、その設定では Wikipedia Search が 99.3% 使われる。QA task で GPT-3 175B に届かない点は、検索器と interaction 制約が性能上の境界になっていることを示す。
- **読者として注意すべき点**: MLQA では API call 使用により Toolformer disabled より改善するが、vanilla GPT-J を常に上回らない。著者は CCNet finetuning による distribution shift を理由としているが、TeX 中にはこの原因を直接検証する追加実験は示されていない。
- **追加で確認したい実験 / 疑問**: chained API call を含む $\mathcal{C}^*$ を作ったとき TempLAMA や multi-step QA が改善するか。API cost を組み込んだ filtering にしたとき、performance / cost trade-off がどう変わるか。BM25 より強い検索器や interactive search を使った場合、QA の GPT-3 との差が縮まるか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Toolformer** ... GPT-J をベースに、API call を含む $\mathcal{C}^*$ で finetune された LM。論文では「which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results」を学ぶモデルとして定義される。
- **$\mathcal{C}$ / $\mathcal{C}^*$** ... $\mathcal{C}$ は plain texts の dataset。$\mathcal{C}^*$ は API call と response を挿入した augmented dataset。finetuning は $\mathcal{C}^*$ 上で行う。
- **API call** ... $c=({a}_c,{i}_c)$ として表される tool 呼び出し。$a_c$ が tool 名、$i_c$ が入力。結果 $r$ とともに `<API> a_c(i_c) \rightarrow r </API>` として text に入る。
- **Question Answering tool** ... Atlas を使う factoid QA system。training data generation では Atlas-large、inference では Atlas-xxl を使用する。
- **Wikipedia Search tool** ... query に対して Wikipedia snippets を返す BM25 retriever。KILT の Wikipedia dump を index する。
- **Calculator tool** ... Python script による四則演算 API。`+`, `-`, `*`, `/` のみをサポートし、結果は小数点 2 桁へ丸める。
- **Machine Translation tool** ... 600M parameter NLLB による English への翻訳 API。source language は fastText classifier で自動検出する。
- **Calendar tool** ... 入力 $\varepsilon$ に対して current date を返す API。$\mathcal{C}^*$ 作成時は URL から document creation date を抽出して近似する。
- **Toolformer disabled** ... Toolformer と同じ finetuned model だが、decoding 時に `<API>` token の確率を 0 にして API call を禁止した baseline。
- **zero-shot prompted setup** ... task instruction は自然言語で与えるが、dataset-specific in-context examples は与えない評価設定。論文は PAL / TALM などの task-specific examples と対比している。
- **$k$ in decoding** ... 通常の greedy decoding では `<API>` が最尤 token のときだけ call するが、評価では `<API>` が top-$k$ に入ると call する修正を行う。主設定は $k=10$。
- **AC / NC** ... Table `tab:top-k` の列。AC は API Call した subset、NC は No Call subset。$k=1$ では NC の性能が API call をすべて禁止した場合の平均より高いことから、ある程度の calibration があると著者は述べる。
- **Dateset** ... Calendar API の有用性を見るために作られた temporal QA dataset。500 current dates、past/future date、US federal holidays を含む templates から 9,400 examples を生成する。

## 読む順番の提案

- まず Abstract と Introduction を読み、問題意識を確認する。特に「large amounts of human annotations」と「task-specific settings only」を既存研究の制約としてどう置いているかを見る。正規ノートの Summary の「問題」「手法」に対応する。
- 次に Section 2 Approach を読む。API call の線形化、Sampling、Executing、Filtering、Model Finetuning、Inference がこの論文の核である。式は $p_i$、$L_i(\mathbf{z})$、$L_i^+$ / $L_i^-$、$L_i^- - L_i^+ \geq \tau_f$ を優先する。正規ノートの「filter 基準」「decoding」に対応する。
- Section 3 Tools と Appendix `API Details` を合わせて読む。5 つの tool の実体、Calculator / MT の heuristic filtering、Calendar の URL date extraction、Atlas-large / Atlas-xxl の使い分けがここにある。正規ノートの「データ」「API ごとの heuristic」に対応する。
- Section 4 Experiments では Table `tab:c_star`, `tab:lama_results`, `tab:math_results`, `tab:qa_results`, `tab:mt_results_percentage`, `tab:temporal_results`, `tab:perplexities` を順に見る。数値と tool 使用率を対応づけると主張の根拠が読みやすい。
- Section 5 Analysis は Table `tab:top-k` と Table `fig:model_outputs` を見る。decoding の $k$ が性能と API call rate をどう変えるか、perplexity-based filtering がどの程度直感的 useful calls と一致するかを確認する。
- 最後に Limitations と Conclusion を読む。chain 不可、interactive 不可、wording sensitivity、sample inefficiency、cost-blind という境界条件は、正規ノートの Critical Thoughts と直接つながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2302.04761v1/`
- 正規ノート: `notes/arXiv-2302.04761v1.md`
