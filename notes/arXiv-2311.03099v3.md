# Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch

- arXiv: https://arxiv.org/abs/2311.03099
- source: ../papers/arXiv-2311.03099v3/
- authors: Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li (Alibaba Group)
- venue / year: ICML 2024
- tags: [model-merging, LLM, SFT, delta-parameters, pruning]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: SFT で得られた LM の「delta parameters（fine-tuned と pre-trained の差分）」は非常に冗長で、しかも複数 SFT モデルを単純に足し合わせて merge するとパラメータが干渉し合い性能が落ちる。GPU を使った再学習なしで、同じ pre-trained backbone から派生した複数の SFT モデルを 1 つに統合して全能力を持たせたい。
- **手法**: **DARE (Drop And REscale)** — delta parameter $\bm{\delta}^t = \bm{\theta}_{\text{SFT}}^t - \bm{\theta}_{\text{PRE}}$ に対し、(1) ドロップ率 $p$ で Bernoulli マスクをかけてゼロ化、(2) 残ったものを $1/(1-p)$ で rescale するだけ。線形変換における出力の期待値が元と一致する（$\mathbb{E}[\hat h_i] = \mathbb{E}[h_i]$）ことを理論的に示している。DARE はデータも再学習も不要で、Dropout と違って推論時に**恒久的に**delta を消す点が異なる。merging では、各 SFT モデルの delta を DARE で疎にしてから Task Arithmetic / TIES-Merging / Average Merging / Fisher Merging / RegMean に渡す plug-in として使う。
- **結果**:
  - **冗長性**: WizardMath-70B では $p=0.99$ でも性能維持。モデルが大きいほど許容ドロップ率が高い（scaling law と類似の傾向）。著者は SFT delta parameter value ranges are typically small (within 0.002) と述べている。
  - **embedding 保存**: DARE は層ごとの cosine similarity が $p=0.9$ でも >0.95。rescale 無しの DropOnly は WizardMath-7B で $p=0.5/0.9$ のとき 0.85/0.68 まで落ちる。
  - **decoder merging（Table 1, Llama-2-13b 系）**: WizardLM-13B / WizardMath-13B / llama-2-13b-code-alpaca を Task Arithmetic + DARE で merge。LM & Math & Code vs LM 単体で AlpacaEval +3.10%、LM & Math vs Math 単体で GSM8K +3.18%、LM & Code vs Code 単体で MBPP +19.57%。TIES-Merging では LM & Math & Code が AlpacaEval 72.50（単体最高 LM 67.20 を上回る）。
  - **encoder merging（GLUE）**: 平均改善は Average/Task Arithmetic/Fisher/RegMean/TIES でそれぞれ +0.58%/+0.36%/+0.37%/-0.03%/+0.84% と控えめ。
  - **7B 実応用**: supermario_v2 が Open LLM Leaderboard で **2024-01-28 時点で 7B 部門 1 位**（Average 75.49、構成元 WildMarcoroni-Variant1-7B/WestSeverus-7B-DPO-v2 の 75.29 を上回る）。supermario_v1 (構成元: NeuralBeagle14-7B + Turdus、いずれも Beagle14-7B から派生) も Average 74.85 で NeuralBeagle14-7B (74.74) を上回る（Turdus の Leaderboard 結果は不掲載のため Table 2 では代替として Beagle14-7B 74.76 が併記されている）。
  - **限界条件**: WizardCoder-Python-13B を CodeLlama-13b-Python ではなく Llama-2-13b を pre-train として DARE すると、$p=0.1$ で HumanEval/MBPP が 63.41/55.4 → 0.0/0.0 に崩壊。delta の絶対値が >0.01 になる継続事前学習系では DARE は機能しない。fine-tuned（pre+delta 全体）を直接 drop すると $p=0.1$ で WizardLM-13B AlpacaEval 67.20→8.56、WizardMath-13B GSM8K 64.22→0.38 等、壊滅的。
  - **MP との比較**: Magnitude-based Pruning（MP）に勝ち、特に高 $p$ で差が拡大。MP に rescale を足すと逆に悪化（$p=0.7$ で AlpacaEval 43.85→10.61、GSM8K 46.70→0.37）。
- **貢献**: (1) SFT delta の極端な冗長性を経験的・理論的に示した、(2) 学習・GPU 不要の超単純な疎化手法 DARE、(3) 既存 merging 手法の汎用 plug-in としての有効性、(4) 「SFT は新しい能力を導入するのではなく pre-trained の能力を unlock しているだけ」という解釈を delta vs fine-tuned のドロップ実験で裏付け、(5) Hugging Face / mergekit / peft で実用採用された Open LLM Leaderboard 1 位モデル supermario_v2 を公開。

## Takeaway（自分にとっての要点）

- **delta だけ落とせばよい**という点が肝。fine-tuned 全体を落とすと $p=0.1$ で性能ゼロに落ちるのに、delta だけなら $p=0.9$〜0.99 でもほぼ無傷。これは「SFT は pre-train の能力を解放しているだけ」という強い証拠になる。
- 理論は1次の期待値保存（$\mathbb{E}[\hat h] = \mathbb{E}[h]$）までで、著者自身も Remark で "rough proof" と書き、$p$ の上界特定を future work としている。実験上は、SFT delta の絶対値が比較的小さい（例: 0.002 以内）場合に DARE が機能し、WizardCoder-Python-13B を Llama-2-13b backbone 扱いにして delta がしばしば 0.01 を超えると破綻する。
- merging に DARE を噛ませる利点は「干渉低減」。特に TIES-Merging が DARE で恩恵を受けるのは、TIES が低 magnitude を切るステップで本来の性能を削っていた分を、ゼロ詰めされた DARE 出力に置き換えることでロスが消えるため。**DARE と TIES の役割が補完的**。
- 大きいモデルほど高 $p$ に耐える（70B は 0.99、7B/13B は手前で折れる）。著者は、モデルサイズと許容 drop rate の間に定量化可能な相関があるかもしれないと述べている。
- encoder ベース（BERT/RoBERTa, GLUE）では改善幅は +0.5% 程度で、著者は DARE による改善が encoder-based LMs より decoder-based LMs で顕著だと結論している。
- 「source モデルが well fine-tuned でないとダメ」（llama-2-13b-code-alpaca が WizardLM-13B より HumanEval で低い → merge 後にコード性能を引き下げる）という観察は、merging 前に各 source の単体評価を必ず取るべきだという実務的注意。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 手法が drop + rescale だけで、CPU・追加データ不要、既存 merging 手法に直挿せる。Impact Statement では、Hugging Face community に数百モデルが作られ、huggingface/peft と arcee-ai/mergekit に統合されたと述べている。
  - 「rescale なし vs あり」「delta drop vs fine-tuned drop」「small Δ backbone vs large Δ backbone」と、効く条件と効かない条件を**両方ちゃんと示している**のが誠実。失敗例（WizardCoder-Python を Llama-2 backbone で扱うと崩壊する）まで報告している。
  - 期待値保存という1行の理論で、Dropout の inverted scaling と同じ気分のものを delta merge 文脈に持ち込んだのが綺麗。さらに DropOnly との embedding cosine similarity の差で実証している。
  - Leaderboard 1位という外形的成果と、Table 1 の「merged > source」（AlpacaEval 72.50 vs 67.20 等）という定量的成果が両方ある。
- **弱み / 疑問**:
  - 理論は**期待値（1次モーメント）の保存**しか言っていない。著者も Remark で「rough proof」と書いており、$p$ の上界を LM capacity に対して推定することは future work として残っている。
  - DARE はランダムドロップを使うが、少なくとも Table 1 と Open LLM Leaderboard の結果にはシード差・信頼区間が示されていない（評者補足）。
  - GLUE での平均改善は Average/Task Arithmetic/Fisher/RegMean/TIES で +0.58%/+0.36%/+0.37%/-0.03%/+0.84% と小さい。著者は「decoder-based LMs are able to accommodate more abilities than encoder-based LMs due to their substantially larger sizes」という可能な理由を挙げるが、追加の統計的検証は TeX 中には示されていない。
  - 「delta 絶対値が比較的小さい（例: 0.002 以内）なら効く」という条件は、この論文中の実験からの結論である。Code Llama 以外の継続事前学習、LoRA delta、量子化済みモデルでの検証は TeX 中には示されていない。
  - merging の hyperparameter（Task Arithmetic の $\lambda \in [0.5, 1.0]$、TIES の retain ratio $\in \{0.5, 0.7, 0.9\}$、DARE の $p$）の grid search が結果に何ポイント効いているか分解されていない。「DARE 効きました」と「ハイパラ swept しました」の寄与が混ざっている。
  - 著者自身が認めている limitations: (i) encoder では merged が single を超えられないケースが残る、(ii) source モデルが十分 fine-tuned でないと merge 後の性能が引きずられる（llama-2-13b-code-alpaca 例）、(iii) Llama 系で delta が小さいときに限定、(iv) 危険な生成（バイアス・差別）は merging 後も残り得る（Impact Statement）。
- **次に試したいこと**:
  - $p$ を層ごと・モジュール（attention QKV vs FFN）ごとに変える adaptive DARE（評者補足）。
  - delta の絶対値分布から「DARE が使えるか／使える $p$ の上限」を予測するスコア関数を作る（著者の future work とも一致）。
  - 代替 rescale（例: per-layer norm 保存）と DARE を比較し、期待値保存以外の条件が効くか調べる（評者補足）。
  - LoRA delta（low-rank）に対する DARE。低ランク構造を壊さない疎化（行/列単位 drop）にすると merging で何が起きるか（評者補足）。
  - 「merged > source」になる条件の cleaner な切り分け：source 同士のタスクが独立に近いほど merge gain が大きい、という仮説を相関や勾配 alignment で検証（評者補足）。
  - 量子化（INT8/INT4）と DARE の相互作用（評者補足）。
  - 安全性: DARE 後に jailbreak 耐性や RLHF アライメントが劣化しないかの体系的評価（Impact Statement の harmful information 懸念に基づく評者補足）。

## Notes / Quotes

- "DARE first performs random drop on $\bm{\delta}^t$ based on a drop rate $p$ ... and then rescales the remaining ones by a factor of $1 / (1 - p)$" (section-3-methodology.tex)
- "By setting $\gamma = 1 / (1 - p)$, we have $\mathbb{E} [h_i] = \mathbb{E} [\hat{h}_i]$, concluding that DARE can approximate the original embeddings." (section-3-methodology.tex)
- "removing fine-tuned rather than delta parameters would cause a catastrophically decreased performance" — fine-tuned 全体を落とすと崩壊する点（Remark, section-3）。
- "SFT primarily unlocks the abilities of pre-trained LMs, rather than introducing new capabilities." (introduction)
- "the tolerance of drop rate increases with the sizes of LMs" — 70B は $p=0.99$、7B/13B は手前で折れる（section-4-experiments）。
- "DARE can work well when the absolute values of SFT delta parameters are relatively small (e.g., within 0.002). Otherwise, DARE may fail." — 使用前提条件（section-4-experiments）。
- "supermario\_v2 achieves the first rank on the Open LLM Leaderboard" (2024-01-28 時点, Table 2)。
- DropOnly（rescale なし）: WizardMath-7B で $p=0.5/0.9$ のとき cosine sim 0.85/0.68。DARE は >0.95 を維持（section-4-rescale-operation）。
- MP + rescale は MP より悪い: $p=0.7$ で AlpacaEval 43.85→10.61、GSM8K 46.70→0.37、HumanEval 21.34→3.05（section-4 MP comparison）。
- WizardCoder-Python-13B を Llama-2-13b backbone で DARE すると $p=0.1$ で HumanEval/MBPP 63.41/55.4 → 0.0/0.0（section-4 "When Can DARE Be Used?"）。
- (verified 2026-05-20) supermario_v1 の構成元を「NeuralBeagle14-7B / Beagle14-7B」から「NeuralBeagle14-7B / Turdus（共に Beagle14-7B 派生）」に訂正。Beagle14-7B は Turdus の Leaderboard 結果欠落のため Table 2 に代替掲載されているだけで、実際の構成元ではない（根拠: Appendix.tex `section-appendix-details_7b_merged_model_leaderboard`）。
- (verified 2026-05-27) TeX 中に無い Leaderboard への外部評価を削除し、シード差・LoRA/量子化など評者独自の論点を「評者補足」または「TeX 中には示されていない」に限定 (section-3-methodology.tex, section-4-experiments.tex, section-5-conclusion.tex, Appendix.tex)

## Related Papers

- Ilharco+ 2023 *Editing Models with Task Arithmetic*（ICLR）— scaling term $\lambda$ で delta を足す merging baseline。DARE の主要併用先。
- Yadav+ 2023 *Resolving Interference When Merging Models*（NeurIPS）— trim/elect-sign/disjoint merge。DARE で最も恩恵を受けた手法。
- Wortsman+ 2022 *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time*（ICML）— 単純平均 baseline。
- Matena & Raffel 2022 *Merging Models with Fisher-weighted Averaging*（NeurIPS）— Fisher 情報量で重み付け平均。
- Jin+ 2023 *Dataless Knowledge Fusion by Merging Weights of Language Models*（ICLR）— 線形回帰の閉形式解で merge。
- Han+ 2015 *Learning both Weights and Connections for Efficient Neural Network* / Magnitude-based Pruning 系 — DARE が比較対象として勝つベースライン。
- Srivastava+ 2014 *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*（JMLR）— drop+rescale という形式の元祖。DARE は inference 恒久・delta 対象という点で対比される。
- Ding+ 2023 *Parameter-efficient Fine-tuning of Large-scale Pre-trained Language Models*（Nature Machine Intelligence）— "delta parameter" 用語の出典。
- Kaplan+ 2020 *Scaling Laws for Neural Language Models* / Hoffmann+ 2022 *Training Compute-optimal Large Language Models* — モデルサイズと許容ドロップ率の相関を著者が連想。
- Xu+ 2023 *WizardLM: Empowering Large Language Models to Follow Complex Instructions*, Luo+ 2023 *WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct*, Luo+ 2023 *WizardCoder: Empowering Code Large Language Models with Evol-Instruct*, Rozière+ 2023 *Code Llama: Open Foundation Models for Code*, Touvron+ 2023 *Llama 2: Open Foundation and Fine-tuned Chat Models*, Jiang+ 2023 *Mistral 7B* — 実験で merge される SFT モデル群と backbone。
