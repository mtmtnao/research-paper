# Agent Skill Acquisition for Large Language Models via CycleQD

- arXiv: https://arxiv.org/abs/2410.14735
- source: ../papers/arXiv-2410.14735v4/
- authors: So Kuroki, Taishi Nakamura, Takuya Akiba, Yujin Tang (Sakana AI)
- venue / year: ICLR 2025
- tags: [LLM, model-merging, quality-diversity, MAP-Elites, evolutionary, agent, SVD]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: LLM に複数の agent skill（コーディング・OS 操作・DB クエリなど）を同時に身につけさせる継続的 fine-tuning は、(1) タスク間のデータ比率調整が難しく一方のスキルが他方を侵食する（例: code 訓練を入れると一般的な思考タスクで性能が落ちる、`liu2023agentbench`）、(2) next token prediction はタスク固有のメトリック（pass@1、success rate）と直接整合しない、という二つの構造的問題を持つ。
- **手法**: Quality Diversity（MAP-Elites）に model merging を載せた進化的探索 **CycleQD** を提案。三つの工夫:
  1. **Alternating Quality/BCs**: K 個の task metric を「世代ごとに quality と BC を入れ替えながら」最適化。世代 $t$ では $i = t \bmod K$ 番目の archive をアクティブにし、その軸を quality、残りを BC として使う。データ比率調整や複合目的関数の設計が要らない。
  2. **Model Merging as Crossover**: ランダム交叉ではなく、両親モデルの task vector $\tau = \theta - \theta_{\mathrm{base}}$ を $\omega_1, \omega_2 \sim \mathcal{N}(\mu, \sigma^2)$ で線形結合（正規化あり、負係数も許容）して子を作る（`akiba2024evolutionary` 由来）。
  3. **SVD-based Mutation**: 子の task vector の各重み行列を SVD し、特異値ベクトルを一様分布 $w \sim U[0, w_{\max}]^r$ でスケールして摂動。ガウシアン摂動は overfit を招くが、SVD で主成分方向に制限すると expert の張る凸領域の外側に出つつ過剰探索を防げる。
  - サンプリングは Pareto frontier を広げる **Elite sampling**（全タスク性能を min–max 正規化した積に比例した確率）。最終的に各 archive の elite モデルを softmax 係数 $\beta_k = \mathrm{softmax}(f_k)$ で task vector レベルに集約して 1 モデルを出力。
- **結果**（Llama3-8B-Instruct ベース、AgentBench コーディング/OS/DB の 3 タスク、1200 世代 = 各タスク 400 世代）:
  - **Table 1**: MBPP 76.4 / DB 38.2 / OS 42.6 / 平均 **52.4** で、Coding/DB/OS 各 expert（37.4 / 45.6 / 32.2）、全データ結合 SFT（47.0）、naive merge（46.7）、CMA-ES merge（46.9）、NSGA-II merge（51.6）など全 baseline を上回る。**gpt-3.5-turbo** の平均 53.7 にほぼ並ぶ（参考: gpt-4 は 61.3、base Llama3-8B-Instruct は 32.6）。
  - MBPP では expert（70.4）を超え 76.4 まで伸び、OS でも expert（30.4）→ 42.6 と大幅改善、DB だけは DB expert（42.4）から 38.2 にわずかに低下。
  - **Table 2 アブレーション**: QD（軸固定）47.6 → CycleQD（軸入替）49.4 → +SVD mutation 51.7 → +Elite sampling **52.4**。ガウシアン摂動は 48.5 で no-mutation より悪化。
  - **Table 3 汎化**（base モデル比で正規化）: CycleQD は HumanEval+ 1.10 / BigCodeBench 1.03 / Reasoning 0.95 / GSM8K 0.88 / RC 0.98 / CommonSense 1.02 / 平均 **0.99**。MBPP expert は HumanEval+ 1.18 と OOD コーディングは強いが Reasoning 0.57 と catastrophic forgetting を起こす。
  - **Table 4 SAM への適用**: SAM-ViT-Huge を CAM/POL/SKL/LEA で 2 モデルずつ merge、6 ペア中 3 ペア（CAM+POL, POL+SKL, CAM+SKL）は両 expert の 90% 以上を保持。失敗ペア（LEA を含むもの）はモデル類似度が低く、平均スコアと類似度の相関は 0.83。
- **貢献**:
  1. data-ratio チューニング不要・タスクメトリック直接最適化な multi-skill LLM 後訓練法 CycleQD。
  2. cyclic alternation / model-merge crossover / SVD mutation / Elite sampling の各設計のアブレーション根拠。
  3. SAM 等の vision モデルへの拡張可能性、および「model similarity」を expert 訓練時の正則化に使うべきという観察。

## Takeaway（自分にとっての要点）

- 「データ比率を諦めて、各タスクの専門家を別々に SFT してから進化で混ぜる」という戦略は、AgentBench の三領域で全データ結合 SFT（47.0）を 5.4pt 上回り、しかも一般言語能力（GSM8K, Reasoning, RC, CommonSense）を base 比 0.95 以上に保てる → multi-skill 後訓練の実務ベースラインとして強い。
- **quality と BC を毎世代入れ替える**だけで +1.8pt（47.6 → 49.4）。これは "どの軸を最適化するか" 自体をスケジューリングする発想で、複数目的をスカラ化せずに済むのが本質的。
- 突然変異の設計が効く: ガウシアン摂動は no-mutation より悪い（48.5 vs 49.4）が、SVD で task vector の主成分方向に縛ると 51.7 まで上がる。"freedom を増やせば良い" ではなく "意味のある方向にだけ動かす" のがコツ。
- Elite sampling の式 $\gamma_j = \prod_i (\alpha_{\mathrm{low}} + \mathrm{norm}(f_{j,i})(\alpha_{\mathrm{high}}-\alpha_{\mathrm{low}}))$ は「全 BC で良い個体を優先サンプル」する単純な積。NSGA-II の crowding distance より素直で、それでも NSGA-II merge（51.6）を上回る。
- SAM 実験の「task vector の特異値ベクトル cos 類似 0.83 相関」は、merge が効くかどうかを事前に予測する指標として使えそう。FedProx 由来の proximal term と思想が近く、expert 同士を訓練時に近づける正則化に転用できる、という著者の示唆は具体的で続き物の研究シードとして良い。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 8B モデルで AgentBench 平均 52.4 と gpt-3.5-turbo（53.7）にほぼ並ぶ数値はインパクトがある。特に OS で base 25.2 → 42.6 と大きく動く。
  - アブレーション（Table 2）が clean: QD → cyclic → +SVD → +Elite と各 component が +1.8, +2.3, +0.7 pt 単調に効いており、設計の貢献分解が説得力ある（ただし Gaussian mutation の trial2 は no-mutation より -0.9 と非単調）。
  - SAM への横展開で「LLM だけの話ではない」と示している。class of model merging × QD 全般に通用するというメッセージが立っている。
  - Catastrophic forgetting の定量（MBPP expert: Reasoning 0.57）を示した上で CycleQD が 0.95 と落とさない、という対比は実用上重要な数値。
- **弱み / 疑問**:
  - **計算コスト**: 1200 世代を回す各ステップで child を全 K archive に対して評価する必要があり、コーディング/OS/DB を各 400 世代 = 1200 回 LLM 推論評価。fine-tuning との fair comparison（GPU-hour 換算）が論文中に明示されていない。
  - **K=3 でしか検証していない**: $K$ を 5, 10 と増やしたら archive サイズが $\prod_{k \neq i} d_k$ で指数的に膨らみ、15 bin だと $15^4 \approx 5\mathrm{万}$ セル。スケール限界の議論が無い。
  - **DB で expert に負けている**（42.4 → 38.2）。著者は "mild drop" としているが、最も BC として情報量が出にくいタスクで quality 化したときに何が起きているのかの分析は無い。
  - **モデル類似度の前提**: 著者自身が limitations で「expert が divergent な設定だと CycleQD は苦しい」と認めている。Llama3-8B 共通 base から SFT した 3 expert なので元々 task vector が近い設定で、異なる base や強い LoRA など真に異種な expert で動く保証は無い。
  - **Elite sampling のハイパー** $\alpha_{\mathrm{low}}, \alpha_{\mathrm{high}}$、SVD 摂動の $w_{\max}$、merge の $(\mu, \sigma)$ の感度がメインテキストでは触れられていない（Appendix 参照と書かれているのみで TeX 内には明示の感度図は無い）。
  - **比較対象が AgentBench の 3 タスクに閉じている**: 推論寄りタスク（MATH, HumanEval 本体, ARC など）を quality として直接最適化したらどうなるかは未検証。Table 3 はあくまで「副作用としての汎化」評価。
  - gpt-3.5-turbo と "on par" の主張は平均値ベース。DB では 38.2 vs 41.6 でまだ負けており、"approaching" の方が正確。
- **次に試したいこと**:
  - $K$ を増やした時の archive 設計（例: BC を embedding 空間に潰す、PGA-MAP-Elites を載せる）。著者も future work で言及。
  - "model similarity を expert 訓練に正則化として混ぜる" 案を実装して SAM の失敗ペア（LEA を含むもの）を救えるか。
  - SVD mutation の $w$ を一様分布ではなく特異値の大きさで重み付けしたら、主成分方向の探索が効率化するか。
  - 同じ token / GPU 予算で `gpt-3.5-turbo 蒸留 → SFT` をやった場合と並べた pareto。
  - CycleQD が生成する数百〜数千の多様 agent を archive ごと残し、debate / 投票 / MoE で使う multi-agent 拡張（著者 future work の本命）。

## Notes / Quotes

- "each task's performance metric is alternated as the quality measure while the others serve as the behavioral characteristics."（abstract）
- "the archive is rotated by 90 degrees after each generation before proceeding to the optimization"（method, alternating Q/BC の直観）
- crossover: $\theta_\mathrm{child} = \theta_\mathrm{base} + \frac{\omega_1}{\omega_1+\omega_2} \tau_{p_1} + \frac{\omega_2}{\omega_1+\omega_2} \tau_{p_2}$, $\omega_i \sim \mathcal{N}(\mu, \sigma^2)$（負係数可、正規化あり、sec3）。
- mutation: $h(\theta_\mathrm{child}) = \theta_\mathrm{base} + \mathrm{concat}([U_l(\Sigma_l w)V_l^\top]_l)$、$w \sim U[0, w_{\max}]^r$、rank=1 行列は pass-through（sec3）。
- aggregation: $\theta_\mathrm{agg} = \theta_\mathrm{base} + \sum_k \beta_k \tau_k$, $\beta_k = \mathrm{softmax}(f_k)$（sec3.4）。
- BC は最弱 expert の 85%〜最強 expert の 115% を 15 bin に均等分割、各 bin 1 モデル、1200 世代（sec4）。
- 既知の限界（sec5）: "model merging hinges on the compatibility of the source models. CycleQD may encounter challenges when the expert models originate from highly divergent settings."
- SAM 実験での model similarity 定義: $s = (1/L)\sum_i \cos(\mathrm{diag}(\Sigma_{i,A}), \mathrm{diag}(\Sigma_{i,B}))$、Avg score と相関 0.83（sec4.2）。
- code: https://github.com/SakanaAI/CycleQD （abstract footnote）
- (verified 2026-05-20) Table 1/2/3/4 の全数値 (MBPP 76.4 / DB 38.2 / OS 42.6 / Avg 52.4 ほか) と base 比 (32.6 → 52.4)、アブレーション 47.6/49.4/48.5/51.7/52.4、引用キー (akiba2024evolutionary, ilharco2022editing, pugh2016quality, mouret2015illuminating, pierrot2022multi, fedprox 等) を tables/*.tex, sec3_methods.tex, sec4_experiments.tex, sec5_conclusion.tex, iclr2025_conference.bbl で裏取り。
- (verified 2026-05-20) Critical Thoughts のアブレーション増分を +1.7 → +1.8 (47.6→49.4) に修正、Gaussian mutation で非単調な点を補足 (tables/table2_ablation_studies.tex)。

## Related Papers

- Mouret & Clune 2015, MAP-Elites — 本手法のベース QD アルゴリズム。
- Pugh+ 2016 "Quality Diversity: A New Frontier for Evolutionary Computation" — QD と BC 設計の出典。
- Akiba+ 2024, Evolutionary Optimization of Model Merging Recipes — task vector を進化で混ぜる crossover の元ネタ。
- Ilharco+ 2022, "Editing models with task arithmetic" — task vector の定義。
- Liu+ 2023 AgentBench — OS/DB タスクと code 偏重の弊害の引用元、評価基盤。
- Wei+ 2024 Magicoder（Evol-Instruct-110K, OSS-Instruct-75K）— coding expert の訓練データ。
- Chen+ 2024 Agent-FLAN — OS/DB expert の訓練データ。
- Liu+ 2023 EvalPlus / MBPP+ — coding 評価。
- Dubey+ 2024 Llama3 — base model（Llama-3-8B-Instruct）、GPT MBPP スコアの出典。
- Kirillov+ 2023 Segment Anything — SAM 実験の base。
- Deb+ 2002 NSGA-II / Fontaine+ 2020 CMA-ME / Nilsson+ 2021 PGA-MAP-Elites — 比較対象および future work で参照される多目的・進化手法。
- Pierrot+ 2022 MOQD — 多目的 QD の関連手法。
- Li+ FedProx — model similarity を proximal term として使う先例。
