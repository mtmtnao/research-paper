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
- **結果**: Table 1（in-domain、validation あり）で全ベースラインを上回り、(IA)$^3$/T0-3B 66.4（Task Arithmetic 63.9, +2.5）、T5-base 73.9（+0.7）、T5-large 76.9（RegMean 73.2 比 +3.6）、ViT-B/32 73.6（+1.8）、ViT-L/14 86.0（+1.5）。Table 2 の OOD でも T5-base 35.3（+1.0）/ T5-large 40.4（RegMean 36.0 比 +4.4）。Table 6 のアブレーションでは Scale, Disjoint Mean, Elect, Trim の順に効き、特に Disjoint Mean を外すと (IA)$^3$ で 70.7→67.5 と落ちる。Table 7（Oracle Sign）で multitask モデルの符号を貸してやると 66.4→72.0 まで上がり、multitask 学習の 73.1 にほぼ届く。Fig. 4（flipping signs）で「高 magnitude 側の top-20/30% の符号を反転すると性能が単調に崩壊、bottom-80/70% の符号は反転しても無害」ことを示し、符号の重要性を裏付ける。
- **貢献**: (1) model merging の interference を redundant param と sign disagreement の 2 種に分解して定量化した、(2) 3 ステップ・ハイパラ 2 個・closed-form で済む手法 TIES-Merging を提案、(3) PEFT/フル fine-tune、NLP/Vision、T5-base/large/ViT-B/L、in-domain/OOD、validation 有/無、ModelSoups 風の同タスク複数 ckpt まで横断的に有効性を検証、(4) Oracle Sign 実験で「正しい符号さえ与えられれば merging で multitask 性能に届く」ことを示し、今後の方向性として「multitask sign の推定」を据えた。

## Takeaway（自分にとっての要点）

- **Sign election が肝**: 単純平均がダメな本当の理由は magnitude 平均ではなく **符号の打ち消し**。total mass で多数決して、負け側を平均から除外する「disjoint mean」というワンフレーズで覚えられる。
- **k=20% / λ=1 がそのまま使える**: PEFT 設定で探索した recipe を未調整のまま ViT・T5 のフル fine-tune に流用しても勝つ。validation set が無い実運用の merging ではまずこれを試すべき。
- **冗長性の根拠が clean**: Fig. 2（top-k% 残しで k=20% でも性能維持）で「task vector はスパースに圧縮できる」を実測。これは pruning 文脈とも噛み合うし、merging 以外（分散学習・連合学習でのアップロード量削減）に転用できる。
- **Oracle Sign 実験のメッセージ**: TIES 66.4 → Oracle Sign 72.0 → Multitask 73.1。**ギャップの大半は「正しい符号を当てられないこと」に集約**されている。次の研究の的が明確（few-shot で multitask sign を推定する、appendix Table 9 で 32 sample + mean init で +1.3% 改善）。
- **同タスク内でも符号衝突がある**（Fig. A.4）: 過剰パラメータ化 + 異なる SGD 経路で、同じ task の 10 checkpoint でも符号がばらつく。ModelSoups 系の robust averaging にも TIES が刺さる説明になっている。
- **干渉はモデル数に対して単調増加**（Fig. 3）: 2 タスクならどの手法でもほぼロス無し、5 タスクを超えると Task Arithmetic がはっきり崩れ TIES の勝ち幅が広がる。「マージするモデル数」が手法選択のキー変数。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **シンプルさ**: trim → elect → disjoint mean の 3 行で書ける。ハイパラ 2 個。実装は数行で済むはず（OSS あり）。
  - **検証スコープが広い**: PEFT (IA)$^3$ / T5-base / T5-large / ViT-B/32 / ViT-L/14、in-domain / OOD、validation 有無、ModelSoups (BERT GLUE)、init としての利用、と論文 1 本でカバーしている範囲が広い。再現可能性が高そう。
  - **interference の 2 分類が解釈として綺麗**: 「redundancy で magnitude が薄まる」「sign disagreement で打ち消し合う」という診断軸が今後の merging 系の議論の語彙になる強さがある。
  - Oracle Sign 上限を示してくれているので「TIES の改善余地」が定量的に見える。
- **弱み / 疑問**:
  - **著者自身が認めている限界**（Appendix \ref{sec:limitation}）: (1) weight interpolation がなぜ動くかの理論が乏しい、(2) 共通の初期化・アーキテクチャを前提とする、(3) merging は multitask 学習にまだ届かない、(4) マージ対象 checkpoint の選び方が分からない、(5) multitask sign を multitask モデル無しで推定する方法は未解決。
  - **k=20 の根拠が PEFT 設定のみ**: appendix で (IA)$^3$ で $k\in\{10,20,30\}, \lambda\in[0.8,3.0]$ を grid し、k=20 と λ≈1 を選んでいる。これを未調整で ViT/T5 に転用しても効くのは結果論で、PEFT と full fine-tune では task vector の magnitude 分布が違うはず → k=20 が普遍とは限らない。Fig. A.2 (vs_k) を見ても k 依存はある。
  - **T5-base × validation 無しのケースで Task Arithmetic に負けている**（73.2 vs 69.7、Table 1）。「validation 無し recipe は万能」とは言い切れず、ここの説明は薄い。
  - **multitask 学習自体への到達はしていない**: Oracle Sign でも 72.0 < 73.1（multitask）。ギャップは小さいが「merging で multitask を完全に代替できる」とまでは言っていない。
  - **計算コストの議論が薄い**: top-k 抽出はパラメータ数線形だが、フル fine-tune モデル × タスク数で task vector を全部メモリに乗せる前提。LLM スケール（数十 B）で n=10 タスクを同時に処理する現実性は別議論。
  - **タスクの偏り**: NLP は GLUE/SuperGLUE 系の小タスクが中心、Vision も既存 task_vectors ベンチの 8 タスク。生成タスク（要約・コード生成・対話）での挙動は未検証。
  - **「elected sign は mass の多数決」のロバスト性**: タスク数が偏ると、片側に大きな magnitude を持つ 1 タスクの符号が常勝してしまう懸念。タスク重み付け（weighted elect）は議論されていない。
- **次に試したいこと**:
  - **few-shot で multitask sign を推定**するパスを Appendix Table 9 より進める。32 sample / mean init が +1.3% なら、合成サンプル（pre-trained で自己生成）でどこまで近づくか。
  - **LoRA で同じことを**: (IA)$^3$ は scale 系 PEFT。LoRA の low-rank 行列に対する top-k は意味が変わるはず（rank 単位 vs 要素単位）。要素単位で TIES するか、特異値ベースで trim するかの比較。
  - **タスク数を更にスケール**（20〜100 タスク）: Fig. 3 の延長で TIES もどこかで折れるはず。$k$ をタスク数と連動させる必要があるか。
  - **生成タスク（要約・コード）での merging**: rank classification ベンチではなく ROUGE/pass@k で見る。
  - **elect を soft 化**: 多数決ではなく $\gamma_m^p = \tanh(\beta \sum_t \hat{\tau}_t^p)$ 的な連続版にすると、僅差の符号 election での感度がどう変わるか。
  - **マージ checkpoint 選択**: 著者が限界として挙げている (4)。task embedding を使って merging に有益な subset を選ぶ問題は実用上重要。

## Notes / Quotes

- "interference can stem from two major causes ... (1) Interference from redundant parameters ... (2) Interference from sign disagreement" (introduction.tex)
- "Keeping only the top-20% of the parameter does not degrade the performance." (Fig. 2 caption, background.tex)
- "sign conflicts occur even when merging only 2 models from different tasks or when merging multiple models from the same task" (background.tex)
- Step definitions: Trim = keep top-$k\%$ by $|\tau_t|$ / Elect = $\gamma_m^p = \textrm{sgn}(\sum_t \hat{\tau}_t^p)$ / Disjoint Merge = mean over $\mathcal{A}^p = \{t : \hat\gamma_t^p = \gamma_m^p\}$（method.tex）
- Validation 無し recipe: $k=20$, $\lambda=1$, mass-based sign election, disjoint mean（experiments.tex）
- Ablation order of importance: Scale > Disjoint Mean > Elect > Trim（Table 6）
- Oracle Sign で TIES 66.4 → 72.0、multitask は 73.1（Table 7）。"if we can obtain the correction directions for the merged model, then we can closely bridge the gap to multitask models."
- Limitations: weight interpolation の理論不足 / 共通 init・arch 前提 / multitask 学習に未到達 / checkpoint 選択未解決 / multitask sign 推定が open（Appendix sec:limitation）
- 同タスク 10 ckpt 間でも符号衝突あり: "models are highly overparameterized and hence there are multiple subnetworks ... different finetuning runs update the same parameters in different directions"（Appendix A.4）

## Related Papers

- Ilharco+ 2023, *Editing Models with Task Arithmetic* — task vector の原典、最強 baseline。
- Matena & Raffel 2021, *Fisher Merging* — 重要度重み付き merging baseline。
- Jin+ 2023, *RegMean* — 局所 least-squares で merging する baseline、TIES より重い。
- Wortsman+ 2022, *Model Soups* — 同タスク複数 ckpt 平均、Table 4 の比較対象。
- Choshen+ 2022, *Fusing Fine-tuned Models for Better Pretraining* — 平均で init を作るアイデア（Table 5 設定）。
- Ainsworth+ 2022, *Git Re-Basin* / Singh & Jaggi 2020 — permutation 対応マージ（TIES は permutation を扱わない側）。
- Liu+ 2022, *(IA)$^3$ / T-Few* — PEFT 実験のベース手法。
- Sanh+ 2022, *T0/Multitask Prompted Training* — multitask baseline。
- Ortiz-Jiménez+ 2023, *Task Arithmetic in the Tangent Space* — 並行研究、tangent space で disentanglement を強める理論側。
