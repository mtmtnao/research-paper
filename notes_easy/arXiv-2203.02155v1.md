# Training language models to follow instructions with human feedback（RLHF による instruction-following LM alignment の実証）

- arXiv: https://arxiv.org/abs/2203.02155
- 一次ソース: ../papers/arXiv-2203.02155v1/
- 正規ノート: ../notes/arXiv-2203.02155v1.md

---

## 一言で言うと

GPT-3 の事前学習目的「internet 上の次 token 予測」と、ユーザが望む「指示に helpful / honest / harmless に従う」目的のずれを、人間の demonstration と preference を使った RLHF で補正する論文。OpenAI API Playground 由来の実プロンプト分布上で、175B InstructGPT は 175B GPT-3 より 85 ± 3%、few-shot 175B GPT-3 より 71 ± 4% の頻度で好まれると報告する（`neurips_2021.tex` §1, §4.1）。

## 何を議論する論文か

- **問題設定**: 大規模 LM は prompt によって多様な NLP task を解けるが、事実の捏造、biased or toxic text、指示不遵守を起こす。論文はこの原因を、LM objective が「predicting the next token on a webpage from the internet」であり、ユーザ目的「follow the user's instructions helpfully and safely」と異なる点に置く（§1）。
- **対象範囲 / 仮定**: 対象は GPT-3 architecture の 1.3B / 6B / 175B モデル。主な prompt 分布は、初期 InstructGPT model を OpenAI API Playground で使った顧客の prompt と、bootstrap 用の labeler-written prompt である。train/valid/test は user ID または organization ID で分割し、training split から PII を filter する（§3.2, Appendix A）。
- **既存研究との差分**: Ziegler et al. 2019 と Stiennon et al. 2020 の RLHF 手順を、summarization など狭いタスクではなく「broad distribution of language tasks」へ適用する。FLAN / T0 のような public NLP instruction tuning との比較も行い、API 実分布ではこれらが SFT baseline より弱いと主張する（§2, §4.1）。
- **この論文で答えたい問い**: 人間の demonstration と preference による fine-tuning は、API ユーザの多様な instruction に対して GPT-3 の振る舞いをどの程度 align できるか。その改善は truthfulness / toxicity / bias / public NLP benchmark の性能低下、さらに計算コストとどう trade off するか。

## 背景と前提

- **alignment の定義**: 本文は Leike et al. 2018 に従い、モデルが user intentions に沿って振る舞うことを alignment とする。実用上は Askell et al. 2021 の helpful / honest / harmless に近い枠組みで評価する（§3.6）。
- **helpful / honest / harmless**: helpful はユーザの task 解決を助けること、honest は情報を fabricate したり mislead しないこと、harmless は人や環境へ physical / psychological / social harm を与えないこと。honesty はモデルの内部 belief が分からないため、実験では truthfulness と hallucination を proxy にする（§1, §3.6）。
- **RLHF の位置づけ**: 人間の preference を reward signal として使う fine-tuning。過去の robot / Atari / summarization の RLHF を、自然言語 instruction-following に拡張する（§2, §3.1）。
- **評価分布の重要性**: この論文の主評価は public benchmark ではなく、held-out customer の API prompt 分布上の labeler preference である。したがって「ユーザ意図に従う」の実験的意味は、OpenAI API Playground prompt、labeler instructions、研究者が設計した評価基準に強く依存する（§3.6, §5.2）。
- **baseline**: GPT-3、few-shot prefix を付けた GPT-3 (`GPT-3-prompted`)、SFT、PPO、PPO-ptx、FLAN fine-tuned 175B GPT-3、T0++ fine-tuned 175B GPT-3 が比較対象として出る（§3.5）。

## 提案手法

### コアアイデア

論文の手法は Figure 2 の 3 段階で構成される。

1. **SFT**: labeler が prompt に対する望ましい response を demonstration として書き、GPT-3 を supervised learning で fine-tune する。SFT は 16 epochs、cosine learning rate decay、residual dropout 0.2 で訓練され、validation loss では 1 epoch 後に overfit するが、RM score と human preference はより長く訓練した方がよかったと述べる（§3.5, Appendix C）。
2. **RM**: SFT model の final unembedding layer を外し、prompt と completion から scalar reward を出す reward model を訓練する。labeler には同一 prompt に対する $K=4$ から $K=9$ 個の response を ranking してもらい、そこから ${K \choose 2}$ 個の pairwise comparison を作る。175B RM は不安定だったため、本文では 6B RM のみを使う（§3.5）。
3. **PPO / PPO-ptx**: RM の scalar reward を使い、SFT policy を PPO で fine-tune する。SFT model からの per-token KL penalty を加え、reward model の over-optimization を抑える。PPO-ptx は PPO gradient に pretraining gradient を混ぜ、public NLP datasets 上の performance regressions を軽減する。本文では特に断らない限り InstructGPT は PPO-ptx model を指す（§3.5）。

この手順は「human values」一般への alignment ではなく、training labelers、研究者の instruction、API Playground の顧客 prompt によって定まる specific human reference group への alignment である、と著者は §5.2 で明示している。

### 重要な定義・数式

$$
\operatorname{loss}\left(\theta \right)=-\frac{1}{{K \choose 2}}E_{\left(x, y_{w}, y_{l}\right) \sim D}\left[\log \left(\sigma\left(r_{\theta}\left(x, y_{w}\right)-r_{\theta}\left(x, y_{l}\right)\right)\right)\right]
$$

**式の意味**: Eq. (1) の reward model loss。人間が pairwise comparison で好んだ completion $y_w$ の reward が、好まれなかった completion $y_l$ より高くなるように学習する。

**記号の定義**:
- $x$ ... prompt
- $y_w$ ... pair のうち labeler が preferred とした completion
- $y_l$ ... pair のうち preferred でない completion
- $r_\theta(x,y)$ ... parameters $\theta$ を持つ reward model の scalar output
- $D$ ... human comparisons の dataset
- $K$ ... 同一 prompt に対して labeler に ranking させる response 数。本文では $K=4$ から $K=9$
- $\sigma$ ... sigmoid function

**この論文での役割**: SFT 後の policy を PPO で最適化するための reward function を作る中心式。論文は、同一 prompt から得た ${K \choose 2}$ comparisons を独立 data point として shuffle せず、1 batch element として扱うことで overfitting を避けたと説明する（§3.5）。

$$
\operatorname{objective}\left(\phi\right)=E_{\left(x, y\right) \sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x, y)-\beta \log \left(\pi_{\phi}^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right] + \gamma E_{x \sim D_\textrm{pretrain}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]
$$

**式の意味**: Eq. (2) の RL objective。RL policy が RM reward を高くしつつ、SFT policy から離れすぎないよう KL penalty を受け、PPO-ptx では pretraining distribution の log likelihood も高く保つ。

**記号の定義**:
- $\phi$ ... learned RL policy の parameters
- $\pi_{\phi}^{\mathrm{RL}}$ ... PPO で学習される policy
- $\pi^{\mathrm{SFT}}$ ... supervised trained model
- $D_{\pi_{\phi}^{\mathrm{RL}}}$ ... 現在の RL policy が生成する prompt-response の分布
- $D_\textrm{pretrain}$ ... GPT-3 の pretraining distribution
- $\beta$ ... KL reward coefficient。Appendix C では default $\beta=0.02$
- $\gamma$ ... pretraining loss coefficient。PPO では $\gamma=0$、PPO-ptx では Appendix C で $\gamma=27.8$

**この論文での役割**: PPO-ptx の定義そのもの。public NLP datasets 上の alignment tax を減らすため、単に KL coefficient を大きくするのではなく pretraining gradients を混ぜるという設計を支える（§3.5, §4.2, Appendix D）。

$$
H = -\sum_{i \in \rm choices} P_i \log_2 P_i
$$

**式の意味**: bias evaluation で使う entropy。binary choice で最大 entropy は 1 で、高いほど model がどちらかの completion に強く偏っていないことを表す（Appendix D, “Toxicity and bias evaluation details”）。

**記号の定義**:
- $i$ ... Winogender / CrowS-Pairs の choice
- $P_i$ ... model が completion $i$ に割り当てた total probability に比例する binary distribution の確率
- $H$ ... bits 単位の entropy

**この論文での役割**: InstructGPT が bias を減らしたかを測る指標。本文は、この metric では InstructGPT は GPT-3 より less biased ではなく、respectful prompt では PPO-ptx の entropy が下がり higher bias を示す場合があると述べる（§4.2）。

### 実装 / アルゴリズム上の要点

- **データ収集**: prompt は labeler-written prompt と API Playground prompt。API prompt は deduplication、1 user ID あたり最大 200 prompt、user ID / organization ID による split、training split の PII filtering を行う（§3.2, Appendix A）。
- **データサイズ**: Table `dataset-size` によると、SFT train は labeler 11,295 + customer 1,430、RM train は labeler 6,623 + customer 26,584、PPO train は customer 31,144。PPO valid は customer 16,185。
- **prompt 分布**: API prompt dataset の use-case は Generation 45.6%、Open QA 12.4%、Brainstorming 11.2%、Chat 8.4%、Rewrite 6.6%、Summarization 4.2%、Classification 3.5%、Other 3.5%、Closed QA 2.6%、Extract 1.9%（Table `instruction-categories`）。
- **言語**: training tasks は over 96% English。Appendix A では langid.py により 110k datapoints の around 96% が English と分類され、classifier inaccuracies により実際は 99% 以上かもしれないと述べる。
- **labeler**: Upwork と ScaleAI から約 40 人の contractors を hiring し、sensitive prompt への対応や researcher labels との agreement などで screening する。training labelers の inter-annotator agreement は 72.6 ± 1.5%、held-out labelers は 77.3 ± 1.3%（§3.4）。
- **RM のサイズ選択**: すべての PPO model で single 6B reward model と 6B value function を使う。Appendix C は、175B RM は validation loss が下がる可能性はあるが training が不安定で PPO value function 初期化にも不向き、計算量も増えると説明する。
- **RL 設定**: Appendix C では RL models を 256k episodes 訓練し、about 31k unique prompts を用いる。batch size 512、minibatch size 64、PPO clip ratio 0.2、rollout sampling temperature 1。PPO-ptx では RL episode 数の 8 倍の pretraining examples を使う。

## 実験・結果

- **データセット / ベンチマーク**: 主評価は held-out customer の API prompt distribution。public NLP datasets として、TruthfulQA、RealToxicityPrompts、Winogender、CrowS-Pairs、DROP、QuAC、SQuADv2、HellaSwag、SST、RTE、WSC、WMT 2015 French to English、CNN/Daily Mail Summarization、Reddit TLDR Summarization を使う（§3.6, Appendix D）。
- **比較対象 / baseline**: GPT-3、GPT-3-prompted、SFT、PPO、PPO-ptx、FLAN fine-tuned 175B GPT-3、T0++ fine-tuned 175B GPT-3。FLAN は 1.2M datapoints、T0 は 96M datapoints から 1 million に subsample して比較可能にし、どちらも 6B RM score で checkpoint を選ぶ（§3.5, Appendix C）。
- **指標**: API 分布では winrate against 175B SFT、1-7 Likert scale、metadata binary labels（hallucination、instruction following、customer assistant appropriateness など）。public eval では TruthfulQA の true / true+info、RealToxicityPrompts の Perspective API toxicity と human toxicity、bias entropy、accuracy、F1、BLEU、ROUGE-L を使う（§3.6, Table `metadata_types`, Table `autoevals`）。
- **主な結果**: 175B InstructGPT は 175B GPT-3 に 85 ± 3%、few-shot GPT-3 に 71 ± 4% の頻度で好まれる。Figure 1 caption は、1.3B PPO-ptx output が 175B GPT-3 output より好まれると述べるが、caption 内にその exact winrate は載っていない。
- **FLAN / T0 との比較**: 175B InstructGPT output は FLAN model より 78 ± 4%、T0 model より 79 ± 4% の頻度で好まれる。Intro では InstructGPT の winrate が 73.4 ± 2%、T0 が 26.8 ± 2%、FLAN が 29.8 ± 2% と報告される（§1, §4.1）。
- **truthfulness / hallucination**: 本文は TruthfulQA で truthful and informative answers が GPT-3 の約 2 倍と述べる。Table `autoevals` では “QA prompt” の true+info が 175B GPT-3 0.251、175B PPO 0.752、175B PPO-ptx 0.689。closed-domain API tasks の hallucination rate は InstructGPT 21% vs GPT-3 41%（§1, Table `autoevals`）。
- **toxicity**: RealToxicityPrompts では respectful prompt 時に InstructGPT が GPT-3 より about 25% fewer toxic outputs と本文が述べる。Table `autoevals` の 175B では respectful の toxicity が GPT 0.233、PPO 0.205、PPO-ptx 0.196。一方 basic prompt では 175B GPT 0.231、PPO-ptx 0.234 で優位は消える。biased prompt では 175B GPT 0.285 に対し 175B PPO 0.427、175B PPO-ptx 0.400 と高くなる。
- **bias**: Winogender / CrowS-Pairs では有意な改善はない。Table `autoevals` では CrowS-Pairs respectful entropy が 175B GPT 0.362、175B PPO-ptx 0.243 で、本文は respectful instruction 下で lower entropy つまり higher bias を示すと解釈している（§4.2）。
- **alignment tax**: PPO without pretraining mix は SQuADv2、DROP、HellaSwag、WMT15 Fr to En などで性能低下を起こす。PPO-ptx はこの低下を大きく緩和し、Table `autoevals` では HellaSwag 175B zero-shot が GPT 0.781 に対し PPO-ptx 0.807、few-shot が GPT 0.791 に対し PPO-ptx 0.820。ただし本文は PPO-ptx が DROP、SQuADv2、translation では GPT-3 にまだ遅れると述べる（§4.2）。
- **held-out labelers / RM generalization**: held-out labelers も training labelers と似た preference ranking を示す。5-fold cross validation の RM accuracy は training-set labelers への prediction が 72.4 ± 0.4%、held-out group が 69.6 ± 0.9%（§4.1, Appendix D）。
- **著者が主張する貢献**: RLHF を広い API prompt distribution に適用し、InstructGPT が preference、truthfulness、toxicity で GPT-3 より改善することを示したこと。PPO-ptx により alignment tax を軽減したこと。現時点の顧客自然言語 task distribution では、175B SFT 4.9 petaflops/s-days、175B PPO-ptx 60 petaflops/s-days が GPT-3 pretraining 3,640 petaflops/s-days より小さく、alignment 投資が model scaling より cost-effective だと論じること（§5.1）。

## 妥当性と限界

- **この主張を支える根拠**: 主張の中心は、held-out customer prompt 上の pairwise human preference、Likert / metadata、TruthfulQA / RealToxicityPrompts / bias benchmarks / public NLP datasets の複数評価に支えられている。特に API 分布では GPT-3 → GPT-3-prompted → SFT → PPO の段階的改善を示し、175B direct comparison の 85 ± 3% / 71 ± 4% が強い evidence になっている。
- **著者が認めている limitations / future work**: InstructGPT は “neither fully aligned nor fully safe” で、toxic / biased outputs、made-up facts、sexual and violent content をまだ生成する。false premise を含む instruction に乗ってしまう、simple question に over-hedge する、multiple explicit constraints に弱い。有害な instruction にも従う傾向があり、biased prompt では同サイズ GPT-3 より toxic output が増える（§4.3, §5.3）。
- **誰に align しているかの限定**: 著者は、align 先が training labelers、研究者、OpenAI API Playground customers によって決まると書く。labelers は mostly English-speaking people living in the United States or Southeast Asia で、API customers も waitlist と OpenAI employees の network に偏り得る。これは “human values” 一般への alignment ではない（§5.2）。
- **読者として注意すべき点**: API prompt 分布は early InstructGPT models に投げられた prompt であり、GPT-3 baseline に不利な instruction-following style を含む可能性が本文で指摘されている。評価 labeler は prompt を書いた user ではないため、labeler が推測した intention と user の actual intention がずれる可能性もある（§3.6, §4.1）。
- **追加で確認したい実験 / 疑問**: labeler 集団を地理・言語・文化・専門性で変えた場合に preference / bias / refusal behavior がどう変わるか。harmful instruction への refusal を training objective に入れた場合、helpfulness と harmlessness の trade-off はどうなるか。PPO-ptx の pretraining mix は regressions を完全には消しておらず、pretraining data に含まれる undesirable behavior を増やす可能性があるため、filtering や synthetic instructions の効果も確認したい（§5.4）。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **InstructGPT**: GPT-3 architecture を SFT、RM、PPO で fine-tune したモデル群。本文では特に断らない限り PPO-ptx model を指す。
- **misaligned**: LM の next-token prediction objective と、ユーザが望む helpful and safe instruction following がずれている状態。
- **user intention**: explicit instruction だけでなく、truthfulness、bias / toxicity 回避など implicit intentions も含む。ただし実験では labeler が prompt から推測する。
- **SFT**: labeler demonstration を使った supervised fine-tuning。PPO の初期 policy であり、baseline でもある。
- **RM**: prompt と completion に scalar reward を付ける reward model。人間の ranking から pairwise loss で訓練する。
- **PPO**: RM reward を最大化する RL fine-tuning。SFT から離れすぎないよう per-token KL penalty を入れる。
- **PPO-ptx**: PPO gradient に pretraining gradient を混ぜる variant。alignment tax を軽減するための主手法で、$\gamma>0$ の Eq. (2) に対応する。
- **KL penalty**: RL policy と SFT policy の token 確率のずれを罰する項。reward model over-optimization を抑える目的で使う。
- **alignment tax**: alignment fine-tuning によって、SQuADv2、DROP、HellaSwag、WMT15 Fr to En など、別に重要な public NLP tasks の性能が落ちること。
- **held-out labelers**: training data を作っていない別 labeler 集団。モデルが training labelers の preference に過剰適合していないかを見るために使う。
- **hallucination**: closed-domain tasks で input にない情報を output に含めること。本文では InstructGPT 21% vs GPT-3 41% と報告する。
- **entropy in bias evaluation**: Winogender / CrowS-Pairs の binary choices に対する確率分布の entropy。高いほど一方の sentence に偏らない。
- **RealToxicityPrompts prompt types**: basic prompt、respectful prompt、biased prompt の 3 種類。respectful では毒性が下がるが、biased prompt では InstructGPT の毒性が高くなる。

## 読む順番の提案

- まず `neurips_2021.tex` の Abstract と §1 Introduction を読む。ここで misalignment、helpful / honest / harmless、主要結果 85 ± 3% / 71 ± 4%、21% vs 41%、about 25% fewer toxic outputs、FLAN / T0 比較の位置づけを押さえる。
- 次に §3.1 から §3.5 を読む。Figure 2、Eq. (1)、Eq. (2)、Table `dataset-size`、Table `instruction-categories` を合わせて見ると、SFT → RM → PPO/PPO-ptx の data flow と、API prompt 分布の偏りが分かる。
- §3.6 Evaluation を読んでから §4 Results に進む。preference winrate、TruthfulQA、RealToxicityPrompts、Winogender / CrowS-Pairs、alignment tax が別々の評価軸であることを区別する。
- Appendix C / D は実装と表の裏取りに使う。特に 6B RM、$\beta=0.02$、$\gamma=27.8$、256k episodes、Table `autoevals`、bias entropy の式を確認する。
- §5.2 “Who are we aligning to?” と §5.3 Limitations は最後に必ず読む。正規ノート `notes/arXiv-2203.02155v1.md` の Critical Thoughts は、この 2 節と §3.6 の評価設計への注意につながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2203.02155v1/`
- 正規ノート: `notes/arXiv-2203.02155v1.md`
