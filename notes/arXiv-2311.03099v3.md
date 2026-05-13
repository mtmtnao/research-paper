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
- **手法**: **DARE (Drop And REscale)** — delta parameter $\bm{\delta}^t = \bm{\theta}_{\text{SFT}}^t - \bm{\theta}_{\text{PRE}}$ に対し、(1) ドロップ率 $p$ で Bernoulli マスクをかけてゼロ化、(2) 残ったものを $1/(1-p)$ で rescale するだけ。線形変換における出力の期待値が元と一致する（$\mathbb{E}[\hat h_i] = \mathbb{E}[h_i]$）ことを理論的に示している。DARE は学習データもデータも不要で、Dropout と違って推論時に**恒久的に**delta を消す点が異なる。merging では、各 SFT モデルの delta を DARE で疎にしてから Task Arithmetic / TIES-Merging / Average Merging / Fisher Merging / RegMean に渡す plug-in として使う。
- **結果**:
  - **冗長性**: WizardMath-70B では $p=0.99$ でも性能維持。モデルが大きいほど許容ドロップ率が高い（scaling law と類似の傾向）。SFT delta の絶対値はだいたい 0.002 以内に収まる。
  - **embedding 保存**: DARE は層ごとの cosine similarity が $p=0.9$ でも >0.95。rescale 無しの DropOnly は WizardMath-7B で $p=0.5/0.9$ のとき 0.85/0.68 まで落ちる。
  - **decoder merging（Table 1, Llama-2-13b 系）**: WizardLM-13B / WizardMath-13B / llama-2-13b-code-alpaca を Task Arithmetic + DARE で merge。LM & Math & Code vs LM 単体で AlpacaEval +3.10%、LM & Math vs Math 単体で GSM8K +3.18%、LM & Code vs Code 単体で MBPP +19.57%。TIES-Merging では LM & Math & Code が AlpacaEval 72.50（単体最高 LM 67.20 を上回る）。
  - **encoder merging（GLUE）**: 平均改善は Average/Task Arithmetic/Fisher/RegMean/TIES でそれぞれ +0.58%/+0.36%/+0.37%/-0.03%/+0.84% と控えめ。
  - **7B 実応用**: supermario_v2 が Open LLM Leaderboard で **2024-01-28 時点で 7B 部門 1 位**（Average 75.49、構成元 WildMarcoroni-7B/WestSeverus-7B の 75.29 を上回る）。supermario_v1 も 74.85 で構成元（NeuralBeagle14-7B 74.74, Beagle14-7B 74.76）を超える。
  - **限界条件**: WizardCoder-Python-13B を CodeLlama-13b-Python ではなく Llama-2-13b を pre-train として DARE すると、$p=0.1$ で HumanEval/MBPP が 63.41/55.4 → 0.0/0.0 に崩壊。delta の絶対値が >0.01 になる継続事前学習系では DARE は機能しない。fine-tuned（pre+delta 全体）を直接 drop すると $p=0.1$ で WizardLM-13B AlpacaEval 67.20→8.56、WizardMath-13B GSM8K 64.22→0.38 等、壊滅的。
  - **MP との比較**: Magnitude-based Pruning（MP）に勝ち、特に高 $p$ で差が拡大。MP に rescale を足すと逆に悪化（$p=0.7$ で AlpacaEval 43.85→10.61、GSM8K 46.70→0.37）。
- **貢献**: (1) SFT delta の極端な冗長性を経験的・理論的に示した、(2) 学習・GPU 不要の超単純な疎化手法 DARE、(3) 既存 merging 手法の汎用 plug-in としての有効性、(4) 「SFT は新しい能力を導入するのではなく pre-trained の能力を unlock しているだけ」という解釈を delta vs fine-tuned のドロップ実験で裏付け、(5) Hugging Face / mergekit / peft で実用採用された Open LLM Leaderboard 1 位モデル supermario_v2 を公開。

## Takeaway（自分にとっての要点）

- **delta だけ落とせばよい**という点が肝。fine-tuned 全体を落とすと $p=0.1$ で性能ゼロに落ちるのに、delta だけなら $p=0.9$〜0.99 でもほぼ無傷。これは「SFT は pre-train の能力を解放しているだけ」という強い証拠になる。
- 理論は1次の期待値しか言っていない（$\mathbb{E}[\hat h] = \mathbb{E}[h]$）ので、効くのは分散も小さい＝delta が小さい場合に限られる。実際 0.002 が経験的閾値で、CodeLlama のように継続事前学習で delta が 0.01〜0.03 に膨らむと破綻する。**「DARE が効くか」は merging 前に delta のヒストグラムを見ればだいたい予測できる**。
- merging に DARE を噛ませる利点は「干渉低減」。特に TIES-Merging が DARE で恩恵を受けるのは、TIES が低 magnitude を切るステップで本来の性能を削っていた分を、ゼロ詰めされた DARE 出力に置き換えることでロスが消えるため。**DARE と TIES の役割が補完的**。
- 大きいモデルほど高 $p$ に耐える（70B は 0.99、7B/13B は手前で折れる）。これはモデル容量とパラメータ冗長性の関係を語っており、merging する LLM のサイズが大きい方が free lunch 度が高くなる方向。
- encoder ベース（BERT/RoBERTa, GLUE）では改善幅は +0.5% 程度で、大半は decoder 側で効いている。**「冗長な余白」が大きい大規模 decoder ほど merging で得をする**という応用ガイドライン。
- 「source モデルが well fine-tuned でないとダメ」（llama-2-13b-code-alpaca が WizardLM-13B より HumanEval で低い → merge 後にコード性能を引き下げる）という観察は、merging 前に各 source の単体評価を必ず取るべきだという実務的注意。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 手法が drop + rescale だけで、CPU・追加データ不要、既存 merging 手法に直挿せる。再現性とエコシステム採用速度が異常に高く、実際に mergekit/peft に取り込まれ、Hugging Face に数百モデルが派生している（Impact Statement で言及）。
  - 「rescale なし vs あり」「delta drop vs fine-tuned drop」「small Δ backbone vs large Δ backbone」と、効く条件と効かない条件を**両方ちゃんと示している**のが誠実。失敗例（WizardCoder-Python を Llama-2 backbone で扱うと崩壊する）まで報告している。
  - 期待値保存という1行の理論で、Dropout の inverted scaling と同じ気分のものを delta merge 文脈に持ち込んだのが綺麗。さらに DropOnly との embedding cosine similarity の差で実証している。
  - Leaderboard 1位という外形的成果と、Table 1 の「merged > source」（AlpacaEval 72.50 vs 67.20 等）という定量的成果が両方ある。
- **弱み / 疑問**:
  - 理論は**期待値（1次モーメント）の保存**しか言っていない。分散は $1/(1-p)$ 倍に膨らむはず（$p=0.99$ なら 100 倍）で、なぜそれでも層を重ねた出力が崩れないのかは説明されていない。著者も Remark で「rough proof」と書いており、上界の理論的特定は future work。
  - DARE はランダムドロップなので種違いで結果が変わるはず。**シードによる分散・信頼区間がほぼ報告されていない**（特に Table 1 と Open LLM Leaderboard の数値）。1 ポイント差で「1 位」と言われても safety margin が分からない。
  - GLUE での平均改善が +0.5% 弱でほぼ誤差レベル。著者は「decoder の方が能力収容力が高いから」と説明しているが、これは事後的な後付けにも見える。encoder で効かない理由をもっと統計的に検証してほしい。
  - 「delta 絶対値 < 0.002 なら効く」という閾値は**経験則**であり、Code Llama 以外の継続事前学習・LoRA マージ・量子化済みモデルでどうなるかは未検証。実運用上、ユーザは自分のチェックポイントが「DARE 適用可能ゾーン」にいるか自分でヒストグラムを取って確認する必要がある。
  - merging の hyperparameter（Task Arithmetic の $\lambda \in [0.5, 1.0]$、TIES の retain ratio $\in \{0.5, 0.7, 0.9\}$、DARE の $p$）の grid search が結果に何ポイント効いているか分解されていない。「DARE 効きました」と「ハイパラ swept しました」の寄与が混ざっている。
  - 著者自身が認めている limitations: (i) encoder では merged が single を超えられないケースが残る、(ii) source モデルが十分 fine-tuned でないと merge 後の性能が引きずられる（llama-2-13b-code-alpaca 例）、(iii) Llama 系で delta が小さいときに限定、(iv) 危険な生成（バイアス・差別）は merging 後も残り得る（Impact Statement）。
  - Open LLM Leaderboard 1位は誇示されているが、Leaderboard 自体がベンチマーク汚染や評価安定性で批判されている時期の話で、外挿に注意。
- **次に試したいこと**:
  - $p$ を層ごと・モジュール（attention QKV vs FFN）ごとに変える adaptive DARE。FFN は冗長で高 $p$、attention は低 $p$、といった構造的事前知識を入れる。
  - delta の絶対値分布から「DARE が使えるか／使える $p$ の上限」を予測するスコア関数を作る（著者の future work とも一致）。
  - 分散の発散を抑える代替 rescale（例: $\sqrt{1/(1-p)}$ や per-layer norm 保存）と DARE を比較し、$p=0.99$ で 1次/2次両方を満たすバリアントを探す。
  - LoRA delta（low-rank）に対する DARE。低ランク構造を壊さない疎化（行/列単位 drop）にすると merging で何が起きるか。
  - 「merged > source」になる条件の cleaner な切り分け：source 同士のタスクが独立に近いほど merge gain が大きい、という仮説を相関や勾配 alignment で検証。
  - 量子化（INT8/INT4）と DARE の相互作用。delta を 99% 落とした上で残り 1% を高精度で持つ「疎 delta + 量子化 base」表現でメモリ最適化。
  - 安全性: DARE 後に jailbreak 耐性や RLHF アライメントが劣化しないかの体系的評価（特に WizardLM 系を merging するとき）。

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

## Related Papers

- Ilharco+ 2023 *Task Arithmetic*（ICLR）— scaling term $\lambda$ で delta を足す merging baseline。DARE の主要併用先。
- Yadav+ 2023 *TIES-Merging*（arXiv 2306.01708）— trim/elect-sign/disjoint merge。DARE で最も恩恵を受けた手法。
- Wortsman+ 2022 *Model Soups (Average Merging)*（ICML）— 単純平均 baseline。
- Matena & Raffel 2022 *Fisher Merging*（NeurIPS）— Fisher 情報量で重み付け平均。
- Jin+ 2023 *RegMean*（ICLR）— 線形回帰の閉形式解で merge。
- Han+ 2015 *Learning both Weights and Connections* / Magnitude-based Pruning 系 — DARE が比較対象として勝つベースライン。
- Srivastava+ 2014 *Dropout*（JMLR）— drop+rescale という形式の元祖。DARE は inference 恒久・delta 対象という点で対比される。
- Ding+ 2023 *delta tuning survey*（Nature Machine Intelligence）— "delta parameter" 用語の出典。
- Kaplan+ 2020 / Hoffmann+ 2022 *scaling laws* — モデルサイズと許容ドロップ率の相関を著者が連想。
- Xu+ 2023 *WizardLM*, Luo+ 2023 *WizardMath*, *WizardCoder*, *Code Llama*, *Llama 2*, *Mistral-7B* — 実験で merge される SFT モデル群と backbone。
