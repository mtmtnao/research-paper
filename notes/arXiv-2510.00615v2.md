# Acon: Optimizing Context Compression for Long-horizon LLM Agents

- arXiv: https://arxiv.org/abs/2510.00615
- source: ../papers/arXiv-2510.00615v2/
- authors: Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan
- venue / year: ICLR 2026 submission（preprint, 2025）
- tags: [LLM-agent, context-compression, prompt-optimization, distillation, long-horizon]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: LLM agent が長 horizon タスク（dozens〜hundreds of steps）を解く際、interaction history と observation が累積して context が際限なく膨らむ。推論コストが O(n) で増えるだけでなく、無関係情報が混じって意思決定の質も下がる。既存の context compression は (a) 対話 memory（MemGPT 系）、(b) 単一 step の document/RAG 圧縮（Selective Context, LongLLMLingua, recomp）、(c) KV-cache 圧縮（LightThinker）に分かれ、いずれも「multi-step かつ heterogeneous な agent context」には不十分。agent 専用の Mind2Act / ContextualizeWeb / openhands-condenser / SWE-agent 系も「naive prompting か narrow domain」止まり（§2）。
- **手法**: **Agent Context Optimization (Acon)** — gradient-free な圧縮 guideline（自然言語 prompt）最適化フレームワーク。
  - **2 種類の圧縮**: history compression（$|h_t| > T_{\sf hist}$ のとき適用）と latest observation compression（$|o_t| > T_{\sf obs}$ のとき適用）。直近の (action, observation) pair は常に保持。
  - **目的関数**: $\max_\psi \mathbb{E}[\mathcal{R}(s_T(\psi))] - \lambda\,\mathbb{E}[C(\mathcal{H}'(\psi))]$。reward は sparse・gold 圧縮なし・コストが離散なので RL は高コストかつ高分散。
  - **Compression Guideline Optimization**: training set で「無圧縮では成功・圧縮では失敗」する contrastive subset $\mathcal{D}_{\sf cont}$ を集め、optimizer LLM（既定 `o3`）に圧縮前後の context を渡して自然言語 feedback を生成 → guideline $\mathcal{P}^{(0)} \to \mathcal{P}^{(1)}$ を update（TextGrad 風 batch textual gradient）。5 candidate 生成して subset で best 選択。これを **utility maximization step (UT)** と呼ぶ。
  - **Compression maximization step (CO)**: 続けて「圧縮版で成功した task」の trajectory だけ見て「実際にどの情報が使われたか」を analyzer LLM に出させ、$\mathcal{P}^{(1)} \to \mathcal{P}^{(2)}$ にして短さ寄りに refine。
  - **Distillation**: 最適化済み teacher（gpt-4.1）の圧縮出力 $(x,y)$ で student（Qwen3-14B / 8B, Phi-4 等）を SeqKD + LoRA (rank 16, $\alpha$=32, lr $10^{-4}$, 3 epoch) で訓練。inference 時には小さい compressor を agent と切り離して使う。
- **結果**: agent と compressor 両方に `gpt-4.1` を使った主実験。
  - **AppWorld test-normal (168 task)**: No compression Acc 56.0 / Peak 9.93k tok。Acon UT 51.2 / 7.17k、Acon UT+CO **56.5** / 7.33k（peak −26%）。FIFO 45.8, Retrieval 27.4, LLMLingua 39.3, Prompting 43.5 を圧倒。特に hard (63 task) で No-comp 39.7 に対し他 baseline は 7.9–23.8 まで崩れる一方 Acon は history で 28.6–30.2、observation で 27.0–31.8 を維持（Table 1）。
  - **OfficeBench (95 test)**: No-comp 76.84、Acon UT **74.74** / Peak 7.27→4.93k（−32%）。
  - **8-objective QA (100 test)**: No-comp EM 0.366 / F1 0.488 / Peak 10.35k / Dep 3.32。Acon UT (history) **EM 0.373 / F1 0.494** / Peak 4.71k（−54.5%）、Dep 1.57（−52.7%）。UT のみで no-compression を F1 で上回る稀有なケース。なお paper §4.1 本文は「peak −54.5% / dep −61.5%」と書いているが、−61.5% は obs UT 行（3.32→1.28）と数値が一致しており、history UT 行との conflate と推定（要注意）。
  - **Compressor distillation**: gpt-4.1 compressor → Qwen3-14B / 8B / Phi-4 / gpt-4.1-mini に LoRA distill しても teacher の **>95%** を維持。
  - **Small agent benefits**: agent 自体を小モデルにすると効果がさらに大きい。Qwen3-14B が AppWorld 26.8% → 33.9%（+32% rel）、8-obj QA EM 0.158 → 0.197（+46% rel）、OfficeBench +20%。
  - **Optimizer ablation (AppWorld)**: `o3` + contrastive feedback **51.2**、contrastive 抜き 50.6、`gpt-4.1` optimizer 47.6、`gpt-5` 50.6。
  - **Threshold**: history 4096 / observation 1024 が最良 trade-off。
  - **限界**: history compression は KV-cache を壊して再計算が走るため、API cost ベースだと total cost が増えるケースあり（appendix Fig. api_cost）。observation 圧縮は cost も減る。
- **貢献**: (1) history と observation を統一的に扱う agent 専用 compression フレームワーク Acon、(2) failure-driven かつ task-aware な gradient-free guideline optimization（UT + CO の 2 段）、(3) 圧縮 module の distillation で 95% 保持、(4) AppWorld / OfficeBench / 8-obj QA の 3 ベンチで peak token 26–54% 削減 & 小 agent 20–46% 改善という実証。

## Takeaway（自分にとっての要点）

- **「圧縮しないと成功・圧縮すると失敗」する pair が natural-language gradient の核**。スカラー reward じゃなく "どこで情報が落ちたか" を LLM に書かせる contrastive feedback が効くという結論は、prompt 最適化全般に転用しやすい（pure failure trace 単独より +0.6pt: 50.6→51.2、+contrastive で gpt-5 optimizer も超える）。
- **agent context は world model**（causal relation / evolving state / preconditions / future decision cue の 4 種類が混ざる）という frame は実用的。「summarize しろ」では足りない理由が言語化されている。
- **history 圧縮は cost を下げない**ことがあるという正直な分析。KV-cache が壊れるので、agent の steps が増えれば total token は逆に増えうる。observation 圧縮の方が cost 効果は素直。実運用ではまず observation から入るのが筋。
- **小 agent ほど効く**（Qwen3-14B が +32–46%）。distraction を減らすと小さい LM の判断力が回復するというのは、agent 設計の equalizer 仮説として強い実証。
- **Compressor は distill 可能**。teacher = gpt-4.1, student = 14B 以下で 95% 保持なら、production 上は「大 LLM を agent に、distilled 小 LM を圧縮係に」というアーキ分離が現実的。
- prompt optimizer は `o3` 一強ではなく `gpt-5` でも -0.6pt まで詰める（47.6 の `gpt-4.1` optimizer は明確に弱い → analyzer に reasoning model が要る）。
- threshold は moderate（hist 4096 / obs 1024）が sweet spot。閾値を下げすぎると圧縮回数が増えて精度劣化する、というのも実装上の落とし穴。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 3 ベンチ（AppWorld, OfficeBench, 8-obj QA）×（agent: gpt-4.1 / gpt-4.1-mini / gpt-5-chat / Qwen3）×（compressor: gpt-4.1 / 4.1-mini / Qwen3-14B/8B / Phi-4）の組み合わせを総当たりに近く埋めており、ablation の質が高い。
  - "history 圧縮は API cost を下げない" を限界として自ら明示している（appendix の cost analysis と sec:analysis 両方）。多くの context-compression 論文がここを誤魔化す中、KV-cache 破壊コストまで議論しているのは誠実。
  - hard split（AppWorld 63 task）で baseline が壊滅（FIFO 15.9, Retrieval 7.9, LLMLingua 15.9）するのに Acon は 30.2 を維持、という勾配が大きい比較対象を意図的に取っている点が良い。
  - UT と CO を分離した alternating optimization は novel な切り口で、UT 単独より UT+CO の方が AppWorld では精度も上がる（51.2→56.5）という反直感的な結果が出ている。
- **弱み / 疑問**:
  - **限界として認められている KV-cache 破壊問題**は decisive。history 圧縮が "cost を下げないどころか上げる" なら主タイトルの "Optimizing Context Compression" は memory（peak token）の話であって compute / dollar の話ではない。peak token と $ cost を区別して読まないと過大評価する。
  - **OfficeBench は train/test を著者が 1:1 で自作 split**（92/95）。公式 split がないとは言え、optimization が test に leak しない保証は guideline level だけで、task の文体類似度等の bias を受ける可能性がある。
  - **prompt optimizer に `o3` を使うコスト**が報告されていない（5 candidate × |D_cont| 回の o3 呼び出しは agent 側よりずっと高そう）。一度作れば固定 prompt として配れる、という反論はできるが「benchmark ごとに作り直し」の前提なら無視できない。
  - **contrastive feedback 有無の差が +0.6pt しかない**（AppWorld 51.2 vs 50.6）。method 章で押している contrastive setup の貢献が、ablation 上は marginal。サンプルサイズ 168 task で偶然と区別がつくか怪しい。
  - **「distill しても 95% 保持」の 95% が何の平均か**は本文だと曖昧。Fig 5 の dot を見ないと benchmark/サイズごとの分散が分からない。
  - **history + observation 同時圧縮では大きく劣化する**（appendix 言及）。「両方使えば最強」ではない点は、汎用 framework の主張をやや弱める。
  - 8-obj QA は MEM1 設定で 8 question を 1 task に束ねた合成ベンチ。NQ ベース・BM25 検索なので long-horizon と言いつつ事実上は long-context QA の延長で、AppWorld のような真の tool-use とは性質が違う。3 ベンチの "等価性" を強調しすぎない方が誠実。
  - **Gemini / Claude 系での検証なし**（著者も限界として明記）。"model-agnostic" の主張は OpenAI 系内での agnostic にとどまる。
- **次に試したいこと**:
  - 同じ peak token 予算で **CoT-self-consistency vs Acon** の pareto curve を引く（debate 論文と同じ問題意識）。
  - **CO step を contrastive 化**：UT は「失敗→成功」、CO は「成功 trace のみ」だが、CO も「短く成功 vs 短く失敗」の pair で学習すれば、短さと正確さの境界をもっと鋭く学べそう。
  - **KV-cache を壊さない history 圧縮**：append-only な要約 token を末尾に置く / streaming attention でカット位置を sliding に取る、と組み合わせれば本論の主限界をほぼ消せる可能性。
  - **prompt budget を agent persona × compressor persona で分配**：Du+2023 の persona 多様化と組み合わせて debate-style 圧縮（multiple compressor の出力を 1 agent が読む）を試したい。
  - **distill compressor の自己改善ループ**：teacher 圧縮 → student → student の trajectory で再 UT を回し、teacher なしで guideline を進化させられるか。

## Notes / Quotes

- "the compressor LLM selects information to preserve based on its learned prior knowledge of importance. However, there is no guarantee that the salient details required for successful task completion are retained." (method §3.2) — agent context が world model だという主張の核。
- "agent context effectively serves as a world model of the environment, encompassing diverse forms of information such as causal relations (e.g., email leaves drafts), evolving states (e.g., account balance), preconditions (e.g., login required), and task-relevant decision cues (e.g., due dates)." (method §3.2)
- "trajectories under compressed contexts provide dense signals about the quality of compression." (method §3.3) — RL を避ける理由。
- AppWorld hard 63 task: No-comp 39.7 / FIFO 15.9 / Retrieval 7.9 / Acon UT+CO 30.2（Table 1）。難しい task で baseline が落ちるほど Acon の優位が際立つ。
- 限界（appendix §A）: "history compression can in some cases increase total cost, since additional steps may be required ... Moreover, it breaks the KV-cache of transformer-based LLMs, which forces re-computation of compressed histories."
- Threshold 既定値: $T_{\sf hist}=4096$ (AppWorld/OfficeBench), 2048 (8-obj QA); $T_{\sf obs}=1024$ (AppWorld), 512 (OfficeBench), 400 (8-obj QA)。
- API 価格: gpt-4.1 = \$3.00 / 1M input, \$0.75 cached, \$12 output（2025 年 9 月時点で計算）。
- LoRA: rank 16, α=32, lr 1e-4, 3 epoch, batch 4, seq 10k, AdamW + 5% warmup + wd 0.01, 1×A100 80GB。
- (verified 2026-05-20) 8-obj QA UT 行の dependency 削減率を −61.5% → −52.7% に修正（tables/3_4_officebench_qa.tex, table tab:3_qa_main）。paper §4.1 本文の 61.5% は obs UT 行（1.28）と一致しており、history UT 行（1.57）とは整合しない点を併記。
- (verified 2026-05-20) AppWorld hard split での Acon の精度範囲を「28.6–31.8」→「history 28.6–30.2 / obs 27.0–31.8」に分けて記述（tables/1_appworld_gpt-4.1.tex）。
- (verified 2026-05-20) agent-specific compression baselines に ContextualizeWeb を追加（text/2_related_works_v2.tex L13）。
- (verified 2026-05-20) Threshold / LoRA / API 価格 / Optimizer ablation / Distillation 95% 保持 / 結果数値（AppWorld, OfficeBench, 8-obj QA の主要セル）を text/4_experiments.tex, text/999_appendix.tex, tables/1_appworld_gpt-4.1.tex, tables/3_4_officebench_qa.tex, tables/11_12_one_row.tex で逐一確認、いずれも整合。

## Related Papers

- TextGrad / OPRO / APO / DSPy — natural-language gradient 系の prompt 最適化（本論の土台）。
- MEM1 (NQ + 8-obj QA 構築元), LightThinker (peak tokens / dependency metric の源流)。
- MemGPT (Packer 2023) — 対話 memory のティア化、本論が "agent には不十分" と位置付ける比較対象。
- LongLLMLingua / Selective Context / recomp / CompAct — document 圧縮系 baseline。
- Mind2Act, ContextualizeWeb, openhands-condenser, SWE-agent — agent 専用 narrow 圧縮（直接の比較相手）。
- AppWorld (Trivedi+), OfficeBench, ALFWorld, WebArena, OSWorld, METR, OdysseyBench — long-horizon agent benchmark 群。
- SeqKD (Kim+ 2016) — distillation の objective。
- LoRA (Hu+ 2022) — student 訓練手法。
- streamingattention, CascadeKV, EpiCache — appendix で future work として挙げられている KV-cache 系圧縮。
