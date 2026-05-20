# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

- arXiv: https://arxiv.org/abs/2405.21060
- source: ../papers/arXiv-2405.21060v1/
- authors: Tri Dao (Princeton CS), Albert Gu (CMU MLD)
- venue / year: arXiv preprint, 2024-05 (TeX に venue 明示なし)
- tags: [SSM, Mamba-2, structured-matrix, linear-attention, sequence-model, architecture]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: 構造化 SSM (S4, Mamba) は系列長線形・推論時定数状態という利点を持つが、Transformer 側で蓄積された理論的理解・系統的アーキテクチャ語彙・ハードウェア最適化（matmul ユニット活用、TP/SP 等）から切り離されて発展してきた。結果として Mamba-1 の selective scan は GPU の matmul 単位を使えず Transformer ほど効率的に学習できない。
- **手法**: SSM と masked attention の橋渡しを **構造化半分離行列 (semiseparable matrix)** で行う **Structured State Space Duality (SSD) フレームワーク** を提案。
  - (1) 一般の SSM は SSS 表現 $M_{ji}=C_j^\top A_{j:i} B_i$ で半分離行列を介した行列変換に等価（線形 = 再帰的 matmul, 二次 = 素朴 matmul）。
  - (2) Linear attention を一般化した **Structured Masked Attention (SMA)** を定義し、「効率的自己回帰 ⇔ semiseparable mask」を定理 (Theorem 5.2, `thm:ss-sma`) として証明。
  - (3) SSM を $A_t = a_t I$（scalar-identity）に制限したクラスは 1-SS SMA と完全に一致 → これが SSD（dual model）。
  - (4) 半分離行列のブロック分解により、対角ブロックは attention 風の二次形式（matmul で計算）、非対角ブロックは低ランク因数分解 → chunk 間の小さな recurrence、という **SSD アルゴリズム** を導出。$\mathtt{N}=\mathtt{P}=\mathtt{Q}$（state 拡張 = head 次元 = chunk 長）で総 FLOP $O(\mathtt{TN}^2)$・メモリ $O(\mathtt{TN})$、作業の大半が $(\mathtt{N},\mathtt{N})$ 行列の matmul。
  - (5) これらを使って **Mamba-2 ブロック**を設計：$A,X,B,C$ を block 先頭で並列に作る（Mamba-1 の sequential projection を廃止）、出力直前に extra normalization (NormFormer 流) を追加、$B/C$ は全 $X$ head で共有（Multi-Input SSM / Multi-Value Attention = MIS/MVA パターン）、grouped-value (GVA) で TP を効率化。
- **結果**:
  - **速度**: SSD は Mamba の fused selective scan より state 拡張 $N=64$ で 2–8× 高速。FlashAttention-2 と比べて系列長 2K で逆転、16K で 6× 高速（Fig. *Efficiency Benchmarks*）。
  - **言語モデル (Pile, 300B tokens)**: Mamba-2-2.7B は Pile ppl 6.09 / 平均 zero-shot 60.2、Mamba-2.8B (6.22 / 59.9)、Pythia-2.8B (6.73 / 55.7) を上回り、Pythia-6.9B（同じデータ）にも勝つ（Table 1, 本文 §1）。780M / 1.3B サイズでも一貫して Mamba を上回り、おおむね Pythia の 2 倍サイズに匹敵。
  - **Chinchilla scaling laws**: 125M–1.3B で Mamba-2 は Transformer++（rotary + SwiGLU + RMSNorm + 高 LR）と Mamba をパレート支配（perplexity と wall-clock 両方, Fig. *Scaling Laws*）。
  - **MQAR (multi-query associative recall, Arora+ 2024)**: Mamba-1 が苦戦する難設定で Mamba-2 は vanilla attention すら上回る。state $N=16\to 64\to 256$ で単調改善（Fig. MQAR）。同じ state size に揃えても Mamba-2 が Mamba-1 を大きく上回るが、著者は「アーキの何が効いているかは未解明」と明記。
  - **Hybrid (350M, 48 層, 7B token)**: SSD のみ 8.60 → attention 層 6 枚混ぜると 8.26（最良）、Transformer++（attn 24 枚）は 8.68。**全層の ~10% を attention にする**のがベスト（Table *Combining SSD and Attention*）。2.7B/300B でも Mamba-2-Attention（58 SSD + 6 attn）が pure Mamba-2 / Transformer++ より良い（Pile ppl 5.95 vs 6.09 vs 6.13、平均 61.0 vs 60.2 vs 60.2, Table *Hybrid*）。
  - **Ablation**: parallel projection + extra norm（Mamba-2 block）は Mamba-1 block (sequential, no norm) より 11.76→11.49 と改善。head pattern は **MIS/MVA (11.66) ≫ MCS/MQA (12.62) ≈ MES/MKA (12.59), 中間に MHS (12.06)** と MVA が明確に勝つ。kernel feature map は cosFormer / Performer / Based / ReBased / LayerNorm 等を試したが Swish や恒等とほぼ同等、有意な改善は無し。
- **貢献**: (a) SSM と (linear) attention を **半分離行列** で統一する SSD 理論、(b) 半分離行列のブロック分解に基づく hardware-efficient な **SSD アルゴリズム**（matmul 主体・大 state 対応・Mamba scan より 2–8× 高速）、(c) これに基づく **Mamba-2 アーキテクチャ**（parallel ABCX projection・MVA head・GroupNorm/extra norm、TP/SP/可変長対応）、(d) Mamba-2 が Pile スケールで Mamba と Pythia 同サイズ・2 倍サイズを上回ることの実証、(e) 10% attention 混在のハイブリッドが純 Mamba-2 / Transformer++ を上回ることの発見。

## Takeaway（自分にとっての要点）

- **SSM = semiseparable matrix multiplication**、というのが本論文の核心。Mamba 系を「再帰だ」「scan だ」と見るより、「構造化行列の matmul を再帰でやるか naive でやるか」と見たほうが、GPU 上の最適化と attention との関係の両方が一気に開ける。
- SSD は Mamba-1 から「$A_t$ を scalar-identity に制限する」という**わずかな表現力の削減**で hardware efficiency を得ている（Mamba-1 は diagonal $A_t$）。表現力を上げるか速度を取るかという軸が明示されており、自分で SSM を選ぶときの基準になる。
- **MVA (B,C を head で共有, X が head ごと) > MQA/MKA** は意外。attention で当然視される MQA をそのまま SSM に持ってきても合わない。SSM では「入力 X こそ多 head 化すべき主体」という方が経験的に正しい。
- **chunk 長 $\mathtt{Q}$ = state 拡張 $\mathtt{N}$ = head 次元 $\mathtt{P}$** に揃えると BMM が全部 $(\mathtt{N},\mathtt{N})$ になりテンソルコアに載りやすい — 実装上の super clean な設計選択。Listing の純 PyTorch 実装で動くというのは追試敷居も低い。
- **10% attention で十分**：Mamba-2 完全置換でなく、SSD を主・attention を retrieval として薄く挟むのが現実的な落としどころ。Jamba 系の経験則とも整合。
- 1-SS mask は **input-dependent な相対位置埋め込み**として解釈できる（RoPE/AliBi 系の代替）という指摘は再利用価値が大きい。「位置情報を data-dependent gating で与える」と見なすと SSM と attention の融合設計が見やすくなる。
- TP では Mamba-1 は $(\Delta,B,C)$ が $x_c$ の関数なので all-reduce が 2 回必要だったのを、Mamba-2 では $(\Delta,B,C)$ を直接 $u$ から作ることで attention/MLP と同じ **1 回**にした、というのは大規模学習を考えるなら効く実装変更。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「SSM と attention は半分離行列の異なる decomposition」という統一視点が強力で、Linear Attention (Katharopoulos 2020) を SMA に一般化する Thm 3.7（効率的自己回帰 ⇔ semiseparable mask）が綺麗。
  - 速度比較が theoretical FLOP, wall-clock, scaling law, MQAR, zero-shot を全部出していてフェアに見える。特に scaling law と wall-clock の両方でパレート支配を主張している点。
  - Mamba-1 の architectural choice（head dim = 1, MIS）を後付けでなく「MVA に当たる」と再定義し、ablation で本当に MVA が勝つことを示したのは説得力がある。
  - Mamba-2 block の変更（parallel projection, GroupNorm）は TP 効率と直結しており、理論と systems が一貫している。
- **弱み / 疑問**:
  - **著者自身が認める limitation**: (i) SSD は softmax attention を一般化しない（feature map $\psi$ が finite な kernel に限る）。in-context learning / copying では quadratic attention にまだ劣る場面があり、concurrent work と整合（Related §10.3 末尾）。(ii) SSD は Mamba-1 より $A_t$ の表現力が落ちる（scalar-identity vs diagonal）。著者自身「構造化行列アルゴリズムを精緻化すれば general diagonal SSM でも同等にできるかも」と将来課題として明記。(iii) Mamba-2 は短系列（2K）では Transformer++ より学習効率で負ける可能性がある（同パラメタなら Transformer は L/2 MLP + L/2 attn なのに対し Mamba-2 は L 個の SSD 層）。(iv) MQAR で同 state size に揃えても Mamba-1 を上回る理由は不明と明示。
  - 上記 (iv) は重要：「アーキの何が効いているか分からない」状態で MQAR の優位を主張しており、再現実験者から見ると confounding が多い（block 構造、norm、head pattern、kernel feature map が全部同時に変わっている）。
  - Hybrid 実験（10% attention）の結論は 350M / 7B token と 2.7B / 300B token しか見ておらず、長系列・長 context（>16K）でこの比率が保たれるかは不明。MQAR の improvement も「state を増やせば良くなる」を見せただけで、Mamba-2 が真に attention 級の retrieval をしているかは別問題。
  - Zero-shot 評価は LAMBADA / HellaSwag / PIQA / Arc / WinoGrande / OpenbookQA 等の commonsense / completion 系で、reasoning や long-context は含まれない。Mamba 系の弱点（copy, dictionary lookup）を踏み込んだ評価は MQAR のみ。
  - SSD の実装が「pure PyTorch で動く」と主張する一方、本番のスループットは特化カーネルに依存しており、再現可能性の主張と実用速度の主張が同じ figure で議論される点はやや混在。
  - kernel feature map / normalization の ablation はすべて negative result（差が無いか悪化）で、SSD の数学的整理から得られる architectural improvement が「parallel projection + GroupNorm」とアルゴリズム本体の高速化に偏っており、理論の predictive power としては微妙。
- **次に試したいこと**:
  - **同 token 予算**で Mamba-2 vs Mamba-2-Attention vs Transformer++ vs FlashAttn の pareto を 8K / 16K / 32K 系列で引き、Hybrid の最適 attention 比率が seq length でどう変わるか測る。
  - MQAR で Mamba-1 → Mamba-2 の改善を要因分解（block 変更だけ / head pattern だけ / extra norm だけ / SSD への切替だけ）。著者が未解明と明記している部分。
  - 1-SS mask を「input-dependent 相対位置」と見て、RoPE 付き Transformer の位置情報を 1-SS mask に置換した hybrid を試す。
  - SSD の制約（scalar-identity $A$）を緩めて **general diagonal $A$** にする半分離アルゴ拡張（著者が将来課題に挙げた点）。
  - Mamba-2 系 model に対する mechanistic interpretability（attention sink の SSM 版が存在するか、という Related Work §6.3 の問いを実証）。

## Notes / Quotes

- "different methods of computing state space models can be reframed as various matrix multiplication algorithms on structured matrices." (intro)
- "the Mamba architecture is a multi-input SSM (MIS) that turns out to be analogous to multi-value attention (MVA)" (intro)
- 「SSD は softmax attention を generalize しない」: *"SSD does not generalize standard softmax attention, or any other transformation on the attention kernel matrix that does not have a finite feature map ψ."* (related §10.3)
- Mamba-1 と SSD の差: *"SSD differs only in a slightly more restrictive form of diagonal $A_t$, and trades off this expressivity for improved hardware efficiency (and ease of implementation)."* (related §10.1)
- 効率の鍵: $\mathtt{N}=\mathtt{P}=\mathtt{Q}$ に揃えて全 BMM を $(\mathtt{N},\mathtt{N})$ にし、scan のコスト $O(\mathtt{T}/\mathtt{Q}\cdot\mathtt{N}^2)$ は negligible（§6.3 Total Cost）。
- 表 *Compute* (`tab:compute`): Attention は state $\mathtt{T}$ / train $\mathtt{T}^2\mathtt{N}$ / inf $\mathtt{TN}$、SSM/SSD は state $\mathtt{N}$ / train $\mathtt{TN}^2$ / inf $\mathtt{N}^2$。違いは naive memory ($\mathtt{TN}^2$ vs $\mathtt{TN}$) と matmul 利用可否（SSD ✓, SSM ✗）。
- 1-SS mask の位置情報的解釈: *"the 1-SS mask of SSD can be seen as a more principled form of relative positional embeddings."* (related §10.3)
- TP 改善: Mamba-1 は block あたり all-reduce 2 回、Mamba-2 は projection を $u$ から直接生成 + GroupNorm で **1 回に半減**（systems §8.1）。
- Hybrid 結論: *"having around 10% of the total number of layers being attention performs best."* (experiments §9.2.3)
- 著者の "what aspect helps MQAR" 未解明明示: *"We are not sure which aspect of the architecture is the predominant factor, which remains a question to explore in future work."* (experiments §9.1)
- (verified 2026-05-20) venue 表記から "ICML 2024" を削除（structure.tex の \title/\author に明示なし）。
- (verified 2026-05-20) "Thm 3.7" を実際の Theorem 5.2 (`thm:ss-sma`, structure/ssd.tex L81-85) に訂正。
- (verified 2026-05-20) 章番号 §5.X/§6.X/§7.1 を arXiv 版実番号 §9.X/§10.X/§8.1 に修正（structure.tex の入力順 intro→background→ssm→attention→ssd→efficient→architecture→systems→experiments→related→conclusion より）。
- (verified 2026-05-20) Related Papers から "Cobbe+ GSM8K" を削除（本論文 TeX に GSM8K / cobbe 出現なし、ベンチマークは LAMBADA/HellaSwag/PIQA/Arc-E,C/WinoGrande/OpenbookQA のみ。structure/experiments.tex Table `table:downstream_zeroshot`）。

## Related Papers

- Gu & Dao, *Mamba* (2023) — 直接の前身。selective SSM (S6) 層を提案、Mamba-2 は SSD 層 + parallel block でこれを置き換える。
- Katharopoulos+ 2020, *Transformers are RNNs / Linear Attention* — 本論文タイトルのオマージュ元。SMA の出発点。
- Gu+ 2022, *S4: Efficiently Modeling Long Sequences with Structured State Spaces* — 構造化 SSM の系譜の起点。
- Dao 2023, *FlashAttention-2* — 速度比較の attention baseline。
- Arora+ 2024, *Zoology* / *Simple linear attention language models …* — MQAR タスク提供と Based / linear attention 改良。
- Ainslie+ 2023, *GQA: Grouped-Query Attention* — GVA (grouped-value) head の発想元。
- Sun+ 2023, *RetNet* / Qin+ 2023, *TransNormerLLM* — decay-based linear attention、time-invariant な SSD の特殊例として位置付け。
- Katsch 2023, *GateLoop* — input-dependent decay $A_t$ + dual quadratic form を concurrent に提案、SSD と同じ "surrogate attention" を導出。
- Yang+ 2024, *Gated Linear Attention (GLA)* — chunkwise アルゴリズムが SSD と並行。
- Shleifer+ 2021, *NormFormer* — Mamba-2 の extra normalization の直接的根拠。
- Shoeybi+ 2019, *Megatron-LM* — TP の元、Mamba-2 block 設計の参照点。
- Lieber+ 2024, *Jamba* / De+ 2024, *Griffin* / Botev+ 2024, *RecurrentGemma* — SSM + 少数 attention のハイブリッド系の先行・並行例。
- Beck+ 2024, *xLSTM* / Peng+ 2024, *RWKV-5/6 (Eagle/Finch)* — state expansion + gating 系の同時代モデル。
- Biderman+ *Pythia*, *The Pile* (Gao+), *LAMBADA*, *HellaSwag*, *PIQA*, *Arc-E/C*, *WinoGrande*, *OpenbookQA* — 評価ベンチマーク群（Table `table:downstream_zeroshot`）。
