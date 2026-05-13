# Evolutionary Optimization of Model Merging Recipes

- arXiv: https://arxiv.org/abs/2403.13187
- source: ../papers/arXiv-2403.13187v2/
- authors: Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, David Ha (Sakana AI)
- venue / year: arXiv preprint (v2), 2024
- tags: [model-merging, evolutionary-search, LLM, VLM, japanese, CMA-ES]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: model merging（mergekit 等で行われる Task Arithmetic / TIES / DARE / Frankenmerging）は学習不要で安価だが、選ぶ source モデル・係数・層スタックの組み合わせは「黒魔術 (black art / alchemy)」で人間の直感に依存している。手動設計には限界があり、特に Frankenmerging（層スタック）は探索コストが高くほとんどの利用者が同じ recipe を使い回しているのが現状。
- **手法**: Evolutionary Model Merge。merging を 2 つの直交する空間に分解して進化計算で同時最適化する。
  - **Parameter Space (PS)**: 各層ごとに DARE-TIES の sparsification / weight mixing パラメータを置き、CMA-ES (Optuna 実装, σ=1/6, init=0.5, pop = 4 + ⌊3 ln n⌋) で最適化。1000 trials。
  - **Data Flow Space (DFS)**: 重みは固定したまま、token が通る inference path（どの model のどの層を何番目に通すか）を進化させる。N model の全層を直列に並べたものを r 回繰り返した長さ T=M×r の indicator array $\mathcal{I}$ と、層間のアクティベーションを補正する scaling matrix $W \in \mathbb{R}^{M\times M}$（$W_{ij}=\pi_\theta(i,j,t)$ で NN 化も可）を CMA-ES on EvoJAX（pop=128, 100 generations）で同時最適化。M=64, r=3 で T=192。$\mathcal{I}$ は model A の層が前半に出るよう $2\sigma$ で初期化。
  - **PS+DFS**: 先に PS で得たモデルを model A として source プールに足し、再度 DFS を回す。
- **結果（Japanese Math, MGSM-JA acc / JP-LMEH avg）**:
  - source: shisa-gamma-7b-v1 (9.6 / 66.1), WizardMath-7B-V1.1 (18.4 / 60.1), Abel-7B-002 (30.0 / 56.5)
  - **Ours (PS) 7B: 52.0 / 70.5**, Ours (DFS) 10B: 36.4 / 53.2, **Ours (PS+DFS) 10B: 55.2 / 66.2**
  - 参照: Llama 2 70B 18.0 / 64.5, Japanese StableLM 70B 17.2 / 68.3, Swallow 70B 13.6 / **71.5**, GPT-3.5 50.4, GPT-4 78.8
  - 7B/10B にもかかわらず JP-LMEH で 70B 級と互角〜上回り、PS+DFS 10B は MGSM-JA で GPT-3.5 を超える。
- **結果（baseline 比較, Table 2）**: 未最適化 merging は TIES-Merge 4.4, DARE-TIES 35.2, Frankenmerging 0.0 (MGSM-JA)。LoRA fine-tune の最良は WizardMath ベースで 43.2（JP-LMEH は 55.9 に落ちる）。提案手法は MGSM-JA も JP-LMEH も同時に上げる唯一の手段。
- **結果（VLM, ROUGE-L）**: source は LLaVA-1.6-Mistral-7B (JA-VG-VQA-500 14.3 / JA-VLM-Bench-In-the-Wild 41.1), Japanese Stable VLM (- / 40.5)。**Ours (PS) 19.7 / 51.2**, **Ours (PS+DFS) 20.4 / 47.6**。Frankenmerging 相当 (Passthrough) は 7.3 / 26.7 で大失敗。
- **結果（13B scale）**: ELYZA-japanese-Llama-2-13b-instruct + MetaMath-13B-V1.0 で MGSM-JA を 13.2 → 34.0 (PS+DFS) に。7B より絶対値は低いが（Llama-2-13B < Mistral-7B の素性差由来）スケール可能なことを示した。
- **貢献**: (1) PS と DFS を分離して進化計算で扱う一般枠組みの提案。(2) 異ドメイン横断（言語 × 数学、言語 × Vision）の merging で SOTA。(3) `EvoLLM-JP` / `EvoVLM-JP` を OSS 公開、Apache 2.0 版 `EvoLLM-JP-A` も提供。(4) 新ベンチマーク **JA-VLM-Bench-In-the-Wild**（42 画像・50 問）と JA-VG-VQA-500 を公開。(5) アブレーション群（無関係モデル混入による distraction、$W$ の有無、source 並び順、層スキップ）。

## Takeaway（自分にとっての要点）

- **「merge は学習しないので候補評価が一瞬」が進化計算と本質的に相性が良い**。NAS は候補ごとの再学習が必要で重かったが、merging は forward だけで fitness が出るので CMA-ES がそのまま回る。これは方法論として汎用的に応用が利く視点。
- **PS と DFS は直交かつ加法的**。PS だけで 52.0、DFS だけで 36.4 だが、PS で作ったモデルを model A に組み込んで DFS を重ねると 55.2。NSGA-II 的に多目的化する余地が示唆されている。
- **scaling matrix $W$ が DFS merging の要**：除去すると 13B では PS+DFS 26.4 → ablate (model #6) と PS-merged より悪化、7B 解析でも 20% 以上の劣化。単に層を抜き挿しするだけだと層間で活性化分布がズレて壊れる、を経験的に裏付けている。
- **PS merging の最適解で「3 モデルの重み和が ≈2」になる**＝単なる補間ではなく increase-density で combination している。DARE の sparsification は 100B token で継続事前学習したモデルでは過剰削減になる、という DARE 論文 §4.6 の指摘とも整合（shisa-gamma の重みは消されにくい状態に進化が誘導した）。
- **DFS が効くメカニズムは "subtract" と "add" の両面**：13B 解析では MetaMath の layer #30 を skip するだけで MGSM-JA が 8.0 → 10% に。さらに、(1) 英語数学モデルが日本語で出力しない問題を出力分布シフトで矯正、(2) 数学モデルが日本語の理解で間違える箇所を日本語 LM 層を挿入して補正、の 2 シナリオで効いている。
- **無関係モデルを 8 個まぜても PS は 40.8 までしか落ちない (元 50.0)**。手動 source 選定の感度がそこまで高くないという実用上有用な知見。
- **MGSM-JA の最終モデルは MGSM train を見ていない**：optimize 用には GSM8k の MGSM 非重複部分 1069 サンプルを翻訳して使い、test の 250 サンプルとは厳格に分離。「benchmark を最適化しなかった結果として他 benchmark に汎化した」のがこの論文のキーメッセージ。
- 実用上、**SDXL-Lightning と通常 SDXL を merge できた (EvoSDXL)** という follow-up は、異プロトコル間統合に効く可能性の示唆として面白い。

## Critical Thoughts（評価・疑問）

- **強み**:
  - PS / DFS という 2 軸への分解が綺麗で、既存手法（TIES, DARE, Frankenmerging, NAS）の各々が枠組み内に整理される。
  - 7B–10B で 70B 日本語 LLM を上回るという結果のインパクトは大きく、しかも MGSM-JA test を最適化に使っていない点で benchmark hack 批判を回避している。
  - baseline がきちんと用意されている：未最適化 merging（TIES/DARE-TIES/Frankenmerging）と fine-tune（LoRA / full × 3 source models）の両方を同データで比較。Frankenmerging が 0.0 になる落とし穴を明示しているのは誠実。
  - distraction 実験（無関係 8 モデル混入）と $W$ ablation、source 並び順 ablation が揃っており、結果の頑健性検証として丁寧。
  - Author Contributions が役割分担まで明記（Akiba=PS, Tang=DFS, Shing=VLM/diffusion, Sun=eval, Ha=guidance）。再現の手がかりとして有用。
- **弱み / 疑問**:
  - **著者自身が認める limitations**: "merged models produced responses that lacked logical coherence"、instruction tuning / alignment を行っていないので factually flawed な出力が出うる、と Discussion で明言。
  - **計算コストの fair comparison が薄い**。CMA-ES を 1000 trial × 各候補のフル評価、DFS は 128 pop × 100 gen。同じ GPU 時間を fine-tune に振った場合のカーブが示されていない（fine-tune は 3 epoch / 1069 sample しか試していない＝極端に少ない側）。
  - **MGSM-JA test set 250 問という小ささ**。±数ポイントの差は誤差範囲のはず。信頼区間を出していない。
  - **JA-VLM-Bench-In-the-Wild が 42 画像 50 問**。著者ら自身が作ったベンチで自分のモデルが SOTA、というのは慣例上やむを得ないがサンプルが小さい。GPT-4V で QA を作って人手フィルタという作り方は LLaVA-Bench-In-the-Wild に倣っているとはいえバイアスの懸念は残る。
  - **fine-tune baseline が「全 source の学習データを混ぜて base に SFT」を実装していない**（WizardMath/Abel/japanese-stablelm-base-gamma の学習データが非公開なため "infeasible" と明記）。これは公平だが、「100B token continued pretrain したモデルの知識を merging で安価に汲み出せる」という主張の真の比較対象は実は存在しない。
  - **DFS の説明（"subtract" / "add"）が事後的な解釈**。layer #30 を抜くと改善という発見は面白いが、なぜその層が harmful かは theoretical には未解明。Discussion で「層間の permutation」仮説を述べるに留まる。
  - **CMA-ES の hyperparam 探索の robustness** が示されていない。population size の "default" で 1 回ずつ走らせた結果のみ。複数 seed の分散が無い。
  - **Sakana 自身の commercial interest**（EvoLLM-JP / EvoVLM-JP プロダクトのアピール）と研究主張の境目がやや薄い。
- **次に試したいこと**:
  - 同じ GPU 時間予算を横軸にとって、Evolutionary Model Merge vs. SFT vs. LoRA の pareto curve を引く（特に少データ側でどこまで詰められるか）。
  - DFS の $W$ を hypernetwork で出力する設定（論文では言及のみ）で M を 100 層クラスに伸ばす実験。
  - PS の最適解が「重み和 ≈ 2」になる現象を他言語ペア（韓国語 × 数学、ヒンディー × 数学）でも観察できるか。これが普遍なら正規化制約を外す設計指針になる。
  - DFS で「skip すべき layer」を進化に頼らず影響度ベースで直接特定できないか。論文の「#30 を抜くだけで 10%」は probe 系手法でも到達できる可能性。
  - factually flawed の問題に対し、merge 後に軽い RLHF / DPO を被せた版と pure merge 版の安全性比較。
  - JA-VLM-Bench-In-the-Wild を CC-by 等で他研究者が拡張、もしくは中華圏・アラビア圏など別文化に同様の bench を整備して cross-cultural な merging が同様に効くか検証。

## Notes / Quotes

- "model merging is considered by many to be a form of black art or alchemy, relying on the model maker's intuition and instincts" (Introduction)
- PS: TIES-Merging + DARE を layer-wise に張り、sparsification と weight mixing の 2 パラメータ × 3 source = 6 次元（最終モデル）を CMA-ES で最適化（§3.1, §4.3 Analysis）。
- DFS: indicator array $\mathcal{I}$（長さ T=192）+ scaling matrix $W$ を同時最適化、$\mathcal{I}$ は model A 層が初手で出やすいよう $2\sigma$ で初期化。
- "the sum of the weights exceeds 1 and is approaching 2 ... a combination method which amplifies the contributions of the models, rather than a simple interpolation, proved to be more effective." (§ Analysis)
- "eliminating them in the evolved model (e.g., by setting $W_{ij}=1$) led to a performance decline exceeding 20 percent" (DFS analysis)
- "DFS-merged model decides to skip layer #30 from MetaMath-13B-V1.0 ... without any further modifications we found the performance increased to 10%." (§5 13B analysis)
- distraction 実験: +0 irrelevant 50.0 / +1 46.8 / +2 46.8 / +4 48.4 / +8 40.8（MGSM-JA, Table 2 Distraction 部）。
- VLM: Japanese cultural content（鯉のぼり、青信号、奈良の鹿、渋谷、原爆、日本の信号機）でケーススタディ（Appendix C）。「青信号」を "青" と正しく答えられたのは EvoVLM-JP のみ、LLaVA は "green"、Japanese-Stable-VLM は不安定（緑/緑白/白）。
- Limitations（著者明記）: 出力に論理整合性を欠く例あり、instruction tuning / alignment 未実施で factually flawed 出力の可能性、merged model は source の限界も継承する。
- License 注意: WizardMath が non-commercial license のため EvoLLM-JP も同条件。Apache 2.0 版として `EvoLLM-JP-A` (= shisa-gamma + Arithmo2-Mistral-7B + Abel-7B-002) を別途公開、MGSM-JA 52.4 / JP-LMEH 69.0。

## Related Papers

- Wortsman+ 2022 *Model Soups* — 重み平均 merging の出発点。
- Ilharco+ 2022 *Task Arithmetic / editing with task vectors* — PS merging の基礎。
- Yadav+ 2023 *TIES-Merging* — 干渉解消の 3 ステップ。本研究の PS の基盤。
- Yu+ 2024 *DARE* — sparsification によるパラメータ干渉低減、本研究で TIES と組み合わせ。
- Goddard+ 2024 / Labonne 2024 *mergekit / Frankenmerging* — 本研究のコミュニティ的背景、baseline 比較対象。
- Hansen+ 2006 *CMA-ES* / Akiba+ 2019 *Optuna* / Tang+ 2022 *EvoJAX* — 最適化基盤。
- Real+ 2019, So+ 2019, Zoph+ 2016, Stanley & Miikkulainen 2002 *NEAT*, Gaier & Ha 2019 *WANN* — Neural Architecture Search / morphology evolution の系譜。
- Shi+ 2022 *MGSM* / Cobbe+ 2021 *GSM8k* — 数学評価。
- shisa-gamma-7b-v1, WizardMath-7B-V1.1, Abel-7B-002, Mistral-7B-v0.1, LLaVA-1.6-Mistral-7B — source 系。
- Labonne 2024 *Automerger* — 同時期の関連実験（leaderboard 上で SLERP/DARE-TIES をランダム適用）。著者は overfit を懸念。
- Sun+ 2024 *Transformer layer swap analysis* — DFS のモチベーション（隣接層 swap で性能低下）の予備調査。
