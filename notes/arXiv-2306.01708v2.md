# TIES-Merging: Resolving Interference When Merging Models

- arXiv: https://arxiv.org/abs/2306.01708
- source: ../papers/arXiv-2306.01708v2/
- authors: Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
- venue / year: NeurIPS 2023 (UNC Chapel Hill / IBM Research / MIT)
- tags: [model-merging, task-vector, multitask, PEFT, interference]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 同じ事前学習モデルから派生した複数の fine-tuned モデルを「重み平均 / Task Arithmetic / Fisher Merging / RegMean」のような既存手法で 1 本にまとめると、マージするモデル数が増えるほど精度が落ちる。原因は task vector（$\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{init}$）同士の干渉で、内訳は (a) 冗長パラメータ（fine-tuning でわずかしか動いていない値）が influential な値を希釈してしまう、(b) 同じパラメータでもモデル間で符号が逆向きになり打ち消し合う、の 2 種類。
- **手法**: TIES-Merging = **TrIm, Elect Sign & Merge**。task vector に対して 3 ステップを順に適用する（Algorithm 1）:
  1. **Trim**: 各 $\tau_t$ について絶対値で top-$k\%$（既定 $k{=}20$）だけ残し、残り (100-k)% は 0 にリセット。
  2. **Elect**: パラメータごとに $\gamma_m^p = \textrm{sgn}(\sum_t \hat{\tau}_t^p)$ で「総質量が大きい方の符号」を選ぶ。
  3. **Disjoint Merge**: 各パラメータについて、elected sign に一致する task の値**だけ**で平均を取る（0 と逆符号は無視）。
  最後に $\theta_m = \theta_\textrm{init} + \lambda \cdot \tau_m$（既定 $\lambda{=}1$）でマージ済みモデルを得る。validation set が無くても固定ハイパラ $(k{=}20, \lambda{=}1)$ で動く点が売り。
- **結果**: Table~\ref{tab:main}（in-domain、validation あり）で全ベースラインを上回り、(IA)$^3$/T0-3B 66.4（Task Arithmetic 63.9, +2.5）、T5-base 73.9（Task Arithmetic 73.2, +0.7）、T5-large 76.9（Task Arithmetic 73.3, +3.6）、ViT-B/32 73.6（RegMean 71.8, +1.8）、ViT-L/14 86.0（Task Arithmetic 84.5, +1.5）。Table~\ref{tab:ood} の OOD でも T5-base 35.3（RegMean 34.3, +1.0）/ T5-large 40.4（RegMean 36.0, +4.4）。Table~\ref{tab:ablation} のアブレーション（(IA)$^3$ 側）では Scale, Disjoint Mean, Elect, Trim の順に効き、特に Disjoint Mean を外すと (IA)$^3$ で 70.7→67.5 と落ちる。Table~\ref{tab:oracle}（Oracle Sign）で multitask モデルの符号を貸してやると (IA)$^3$ で 66.4→72.0 まで上がり、multitask 学習の 73.1 にほぼ届く。flipping signs の Fig.~\ref{fig:flip_signs} で「高 magnitude 側の top-20/30% の符号を反転すると性能が単調に崩壊、bottom-80/70% の符号は反転しても無害」ことを示し、符号の重要性を裏付ける。
- **貢献**: (1) model merging の interference を redundant param と sign disagreement の 2 種に分解して定量化した、(2) 3 ステップ・ハイパラ 2 個・closed-form で済む手法 TIES-Merging を提案、(3) PEFT/フル fine-tune、NLP/Vision、T5-base/large/ViT-B/L、in-domain/OOD、validation 有/無、ModelSoups 風の同タスク複数 ckpt まで横断的に有効性を検証、(4) Oracle Sign 実験で「正しい符号さえ与えられれば merging で multitask 性能に届く」ことを示し、今後の方向性として「multitask sign の推定」を据えた。

## Takeaway（自分にとっての要点）

- **Sign election が肝**: 著者は、単純平均で失われる情報の一因として **符号の打ち消し**を強調している。total mass で多数決して、負け側を平均から除外する「disjoint mean」というワンフレーズで覚えられる。
- **k=20% / λ=1 の固定 recipe**: PEFT 設定で探索した recipe を、著者は validation set なし設定として ViT・T5 のフル fine-tune に流用している。Table~\ref{tab:main} では T5-base を除き、validation なしの TIES が validation なし Task Arithmetic を上回る。
- **冗長性の根拠が clean**: Fig.~\ref{fig:reset-bottomk}（top-k% 残しで k=20% でも性能維持）で「task vector は少数の high-magnitude parameters に性能が依存する」ことを実測している。pruning 文脈との接続は本文中でも述べられている。
- **Oracle Sign 実験のメッセージ**: TIES 66.4 → Oracle Sign 72.0 → Multitask 73.1。**ギャップの大半は「正しい符号を当てられないこと」に集約**されている。次の研究の的が明確（few-shot で multitask sign を推定する、Appendix Table~\ref{tab:app_oracle} で 32 sample + mean init で +1.2〜1.3% 改善）。
- **同タスク内でも符号衝突がある**（Appendix Fig.~\ref{fig:same_task_interference}）: 過剰パラメータ化 + 異なる SGD 経路で、同じ task の 10 checkpoint でも符号がばらつく。ModelSoups 系の robust averaging にも TIES が刺さる説明になっている。
- **干渉はモデル数に対して単調増加**（Fig.~\ref{fig:num_tasks}）: 2 タスクなら TIES / Task Arithmetic がほぼロス無し（Simple Averaging は約 10% ロス）。タスク数が増えるにつれ Task Arithmetic の方が TIES より急速に劣化していく（TeX の主張）。「マージするモデル数」が手法選択のキー変数。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **シンプルさ**: trim → elect → disjoint mean の 3 ステップで書ける。ハイパラ 2 個。TeX には code available の footnote がある。
  - **検証スコープが広い**: PEFT (IA)$^3$ / T5-base / T5-large / ViT-B/32 / ViT-L/14、in-domain / OOD、validation 有無、ModelSoups (BERT GLUE)、init としての利用、と論文 1 本でカバーしている範囲が広い。
  - **interference の 2 分類が解釈として綺麗**: 「redundancy で magnitude が薄まる」「sign disagreement で打ち消し合う」という診断軸が今後の merging 系の議論の語彙になる強さがある。
  - Oracle Sign 上限を示してくれているので「TIES の改善余地」が定量的に見える。
- **弱み / 疑問**:
  - **著者自身が認めている限界**（Appendix \ref{sec:limitation}）: (1) weight interpolation がなぜ動くかの理論が乏しい、(2) 共通の初期化・アーキテクチャを前提とする、(3) merging は multitask 学習にまだ届かない、(4) マージ対象 checkpoint の選び方が分からない、(5) multitask sign を multitask モデル無しで推定する方法は未解決。
  - **k=20 の根拠が PEFT 設定のみ**: Appendix sec:app_validation で (IA)$^3$ について $k\in\{10,20,30\}, \lambda\in[0.8,3.0]$ を grid し、k=20 と λ≈1 を選んでいる。これを未調整で ViT/T5 に転用しているが、Appendix Fig.~\ref{fig:hyperparams}（vs_k 図）には、$k$ が増えると性能が下がって saturate し、この曲線は task vector の parameter value 分布に依存して変わり得る、と明記されている。
  - **T5-base × validation 無しのケースで Task Arithmetic に負けている**（73.2 vs 69.7、Table~\ref{tab:main}）。固定 recipe が常に best baseline を上回るわけではなく、ここの説明は薄い。
  - **multitask 学習自体への到達はしていない**: Oracle Sign でも 72.0 < 73.1（multitask）。ギャップは小さいが「merging で multitask を完全に代替できる」とまでは言っていない。
  - **計算コストの議論が限定的**: Appendix sec:app_compute には GPU / runtime / evaluation time はあるが、フル fine-tune モデルを多数マージする際の task vector 保持メモリについての議論は TeX 中には明示されていない（評者補足）。
  - **タスクの範囲**: 本文の NLP 評価は rank classification で、classification / multiple-choice tasks に対応できると Appendix sec:app_training_details に書かれている。要約・コード生成・対話のような生成タスクでの結果は TeX 中には示されていない。
  - **「elected sign は mass の多数決」のロバスト性**: task weighting や weighted elect の検討は TeX 中には明示されていない（評者補足）。
- **次に試したいこと**:
  - **few-shot で multitask sign を推定**するパスを Appendix Table~\ref{tab:app_oracle} より進める。32 sample / mean init が +1.2〜1.3% なら、合成サンプル（pre-trained で自己生成）でどこまで近づくか（評者補足）。
  - **LoRA で同じことを**: TeX では LoRA は PEFT method の例として Background に出るが、実験対象は (IA)$^3$。LoRA での TIES は未検証（評者補足）。
  - **タスク数を更にスケール**（20〜100 タスク）: Fig.~\ref{fig:num_tasks} は T5-Large の 7 tasks（Appendix sec:app_num_tasks）までなので、それ以上の task 数での挙動は TeX 中には示されていない（評者補足）。
  - **生成タスク（要約・コード）での merging**: rank classification ベンチではなく ROUGE/pass@k で見る（評者補足）。
  - **elect を soft 化**: 多数決ではなく $\gamma_m^p = \tanh(\beta \sum_t \hat{\tau}_t^p)$ 的な連続版にすると、僅差の符号 election での感度がどう変わるか（評者補足）。
  - **マージ checkpoint 選択**: 著者が限界として挙げている (4)。task embedding を使って merging に有益な subset を選ぶ問題は実用上重要（評者補足）。

## Notes / Quotes

- "interference can stem from two major causes ... (1) Interference from redundant parameters ... (2) Interference from sign disagreement" (introduction.tex)
- "Keeping only the top-20\% of the parameter does not degrade the performance." (fig:reset-bottomk caption, background.tex)
- "sign conflicts occur even when merging only 2 models from different tasks or when merging multiple models from the same task" (background.tex)
- Step definitions: Trim = keep top-$k\%$ by $|\tau_t|$ / Elect = $\gamma_m^p = \textrm{sgn}(\sum_t \hat{\tau}_t^p)$ / Disjoint Merge = mean over $\mathcal{A}^p = \{t : \hat\gamma_t^p = \gamma_m^p\}$（method.tex）
- Validation 無し recipe: $k=20$, $\lambda=1$, mass-based sign election, disjoint mean（experiments.tex / Appendix sec:app_validation）
- Ablation order of importance（(IA)$^3$ 側）: Scale > Disjoint Mean > Elect > Trim（tab:ablation）。T5-base 側では Scale > Disjoint Mean > Trim > Elect とやや順位が違う。
- Oracle Sign で TIES 66.4 → 72.0、multitask は 73.1（tab:oracle）。"if we can obtain the correction directions for the merged model, then we can closely bridge the gap to multitask models."
- Limitations: weight interpolation の理論不足 / 共通 init・arch 前提 / multitask 学習に未到達 / checkpoint 選択未解決 / multitask sign 推定が open（Appendix sec:limitation）
- 同タスク 10 ckpt 間でも符号衝突あり: "models are highly overparameterized and hence there are multiple subnetworks ... different finetuning runs update the same parameters in different directions"（Appendix sec:app_same_task_interference）
- (verified 2026-05-20) Table/Figure 番号を TeX の document order に合わせて修正: Oracle Sign の Table 7 → tab:oracle（5 番目の表）。Fig. 2/3/4 の図番号は誤りだったため、ラベル参照（fig:reset-bottomk, fig:num_tasks, fig:flip_signs）に置換。"appendix Table 9" → tab:app_oracle、"Fig. A.2/A.4" → fig:hyperparams / fig:same_task_interference に修正（experiments.tex, appendix.tex）。
- (verified 2026-05-20) T5-large in-domain の "RegMean 73.2 比 +3.6" は実際は best baseline が Task Arithmetic 73.3 なので "Task Arithmetic 73.3 比 +3.6" に訂正（tab:main）。
- (verified 2026-05-20) num_tasks 図に関する「5 タスクを超えると Task Arithmetic が崩れる」という具体的閾値は TeX にないため削除し、「タスク数が増えると Task Arithmetic の方が急速に劣化」という TeX の表現に揃えた（experiments.tex, fig:num_tasks）。
- (verified 2026-05-20) Appendix tab:app_oracle の +1.3% は本文中の数値、表中の表示は +1.2 なので "+1.2〜1.3%" と併記（appendix.tex sec:app_estimating_multitask_sign）。
- (verified 2026-05-27) Takeaway / Critical Thoughts の TeX 根拠を越えた推測表現を削除または評者補足として明示し、validation なし TIES は T5-base では Task Arithmetic を下回ることが分かる表現に修正（main.tex, sections/appendix.tex, sections/experiments.tex）。

## Related Papers

- Ilharco+ 2023, *Editing Models with Task Arithmetic* — task vector の原典、最強 baseline。
- Matena & Raffel 2021, *Fisher Merging* — 重要度重み付き merging baseline。
- Jin+ 2023, *RegMean* — 局所 least-squares で merging する baseline、TIES より重い。
- Wortsman+ 2022, *Model Soups* — 同タスク複数 ckpt 平均、tab:modelsoups の比較対象。
- Choshen+ 2022, *Fusing Fine-tuned Models for Better Pretraining* — 平均で init を作るアイデア（tab:fusing 設定）。
- Ainsworth+ 2022, *Git Re-Basin* / Singh & Jaggi 2020 — permutation 対応マージ（TIES は permutation を扱わない側）。
- Liu+ 2022, *(IA)$^3$ / T-Few* — PEFT 実験のベース手法。
- Sanh+ 2022, *T0/Multitask Prompted Training* — multitask baseline。
- Ortiz-Jiménez+ 2023, *Task Arithmetic in the Tangent Space* — 並行研究、tangent space で disentanglement を強める理論側。
