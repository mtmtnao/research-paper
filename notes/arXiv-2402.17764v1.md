# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

- arXiv: https://arxiv.org/abs/2402.17764
- source: ../papers/arXiv-2402.17764v1/
- authors: Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research / UCAS)
- venue / year: arXiv preprint 2024 (neurips_2022 style; venue は TeX 中には明示されていない)
- tags: [LLM, quantization, 1-bit, BitNet, efficiency, hardware]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: LLM の推論コスト（メモリ・レイテンシ・スループット・エネルギー）が増大している。GPTQ/AWQ などの post-training quantization は FP16→4-bit へ進んできたが、本質的に sub-optimal。一方で BitNet のような 1-bit から学習する方式は将来性があるが、binary {-1, +1} だけだと表現力に制約がある。
- **手法**: BitNet b1.58 を提案。重みを ternary $\{-1, 0, +1\}$（情報量 $\log_2 3 \approx 1.58$ bit）、活性を 8-bit でスクラッチから学習する Transformer。BitNet の `nn.Linear` を `BitLinear` に置き換える構成は同じだが、(1) 重みは absmean 量子化 ($\widetilde{W} = \mathrm{RoundClip}(W/(\gamma+\epsilon), -1, 1)$, $\gamma = \frac{1}{nm}\sum|W_{ij}|$)、(2) 活性は zero-point quantization を排して per-token で $[-Q_b, Q_b]$ にスケール、(3) LLaMA 互換コンポーネント（RMSNorm, SwiGLU, RoPE, bias 削除）を採用して既存 OSS（HuggingFace / vLLM / llama.cpp）にそのまま載せられるようにした。RedPajama で 100B トークン事前学習。
- **結果**:
  - **PPL (Table 1)**: 700M 12.33 (LLaMA) vs 12.87 (b1.58), 1.3B 11.25 vs 11.29, 3B 10.04 vs **9.91**（ここで初めて FP16 を上回る）, 3.9B b1.58 は 9.62。
  - **Zero-shot 7タスク平均 (Table 2, ARCe/ARCc/HS/BQ/OQ/PQ/WGe)**: 700M 45.5 vs 44.3, 1.3B 46.2 vs 45.4, 3B 49.7 vs **50.2**, 3.9B b1.58 は **51.2** で LLaMA 3B を上回る。
  - **コスト**: 3B で memory 7.89→2.22 GB (3.55x), latency 5.07→1.87 ms (2.71x)。70B では latency **4.1x** 高速化。
  - **Throughput (Table 3)**: 70B 同士で 2x A100 80GB / seq 512、最大バッチ 16→176 (11x)、tokens/s 333→2977 (**8.9x**)。
  - **Energy**: 7nm で行列積の演算エネルギーを **71.4x** 削減（INT8 加算中心 vs FP16 加算＋乗算）。
  - **2T tokens (Table 4)**: StableLM-3B との比較で Winogrande 64.56→66.37, PIQA 76.93→78.40, SciQ 90.75→91.20, LAMBADA 66.09→67.63, ARC-easy 67.78→68.12, 平均 73.22→**74.34** と全項目で上回る。
  - **新スケーリング則の主張**: 13B b1.58 ≈ 3B FP16, 30B b1.58 ≈ 7B FP16, 70B b1.58 ≈ 13B FP16（latency/memory/energy で）。
- **貢献**: (1) ternary 重み {-1,0,1} による 1.58-bit LLM を初めて提案し、3B 規模から FP16 LLaMA と PPL・end-task で同等であることを示した、(2) memory/latency/throughput/energy の Pareto 改善を定量化、(3) "0" を入れたことによる feature filtering が表現力の鍵だと指摘、(4) 1-bit LLM に最適化された専用ハードウェア設計を呼びかける問題提起。

## Takeaway（自分にとっての要点）

- 「binary より ternary」が重要なポイント。0 を加えるだけで $\log_2 3 = 1.58$ bit になるが、それで「特徴を消す」操作が明示的に表現でき、3B 以上で FP16 と並ぶ。1 bit 増の代償としてはコスパが極めて高い。
- **学習時から ternary**（QAT 的にスクラッチ）であって、PTQ ではない点が決定的。GPTQ/AWQ の延長線上にある手法ではなく、別系統。読むときに混同しない。
- 行列積がほぼ加算だけになるので、energy bottleneck の chip では速度メリットに直結する（"power が compute の上限"）。GPU よりむしろ ASIC/CPU/edge 向けの議論。Groq LPU や独自 1-bit ハードウェアを呼びかける discussion はこの含意。
- LLaMA 互換アーキ（RMSNorm/SwiGLU/RoPE/bias 削除）を踏襲したのは生態系への乗りやすさが目的、と明言されている。手法と独立した実用上の意思決定として参考になる。
- 70B で throughput 8.9x、batch size 11x という数字は、推論サービング側（複数 request 並列）でこそ威力が出る話。単発レイテンシ（2.71x〜4.1x）より serving 経済性のほうがインパクトが大きい。
- 活性は 8 bit に落としてあるので、KV cache も実質半分。長文 context への展開余地（将来 4 bit へ）を discussion で明示している。
- 著者らが提示する等価表（13B b1.58 ≈ 3B FP16 等）は latency/memory/energy 軸での効率等価であって精度等価ではない点に注意（本文も "in terms of latency, memory usage and energy consumption" と書いている）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 3B 規模で初めて 1-bit 系が FP16 と PPL・zero-shot で並んだという一線を越える結果で、しかも 3.9B では LLaMA 3B より良い。閾値を明確に提示している。
  - PPL だけでなく end-task 7 個 + memory + latency + throughput + energy + 2T tokens vs StableLM と、効率と性能の両方を多面的に押さえている。
  - LLaMA 互換にしたことで再現・転用のハードルが極めて低い。vLLM / llama.cpp 連携を明示的に意識した設計判断は実装者目線で誠実。
  - "feature filtering" としての 0 の意義づけは、binary BitNet との差分を概念的に説明していて納得感がある。
- **弱み / 疑問**:
  - **比較の透明性**: 比較対象が「著者ら再現の FP16 LLaMA を RedPajama 100B token で学習」。公開された LLaMA-1/LLaMA-2 ではなく、いわば自前のベースライン。RedPajama 100B token は LLaMA-1 の 1T〜1.4T トークンに比べて遥かに少なく、ベースラインの強さに疑問が残る。
  - **3B 未満では負けている**: 700M / 1.3B では PPL も zero-shot 平均もまだ FP16 に劣る（Table 1, 2）。論文は「3B から並ぶ」と肯定的に書いているが、小規模では再現の意義が下がる。
  - **PTQ 系手法（GPTQ/AWQ/QuIP/SmoothQuant）との直接比較が無い**。Intro で言及するだけで Table には現れない。「4-bit PTQ より良い」のか「同等 token 予算で見て本当に勝つのか」は本文では確認できない。
  - **MMLU / GSM8K 等の推論系ベンチマークが無い**。ARCe/HS/PIQA/BoolQ など比較的やさしいタスクのみ。1-bit 化で複雑推論が崩れていないか分からない。
  - **学習コスト・学習安定性の議論が無い**。QAT 系は学習が不安定になりやすいが、損失曲線・学習時間・収束性についての記述は無い。本文中で著者自身が学習側の難しさに言及していない。
  - **エネルギー 71.4x は "arithmetic operations for matrix multiplication" 限定**。DRAM↔SRAM 転送、attention の softmax、LayerNorm、embedding（依然 FP）等の寄与は含まれていない。end-to-end の図はあるがチップごとの内訳がない。
  - **ハードウェアが存在しないと真価が出ない**。"2-bit kernel from Ladder" を流用しているだけで、Ladder の 2-bit カーネルを 1.58-bit に最適化するともっと速い、と本人が認めている。現状の数字は中途半端な実装に基づく。
  - **論文自体が短く limitations セクションが無い**。技術レポート的な書き味で、失敗例・崩壊条件・QAT 学習のハイパーパラメータ等の検証情報がほぼ無い。再現は LLaMA 互換のおかげで簡単そうに見えるが、実体は QAT のテクニックに依存する可能性がある。
  - 著者が認めている明示的な留保: 「latency と memory は 2-bit カーネルで測定しており最適化の余地が残っている」「活性 8 bit はさらに 4 bit 以下にできる、future work」(Discussion)。後者は損失見積もりを出していない。
- **次に試したいこと**:
  - 同じ 100B token / 同じ data mix で FP16, INT8, INT4 (GPTQ), b1.58 を全部 train/quantize して PPL × end-task × energy の Pareto を描く。論文の比較表に PTQ baseline を足した版がほしい。
  - MMLU / GSM8K / HumanEval を実測し、reasoning タスクで FP16 との gap が widen するかどうか確認。
  - "0" を入れた効果を切り分けるアブレーション: 同条件で binary {-1, +1}、ternary {-1, 0, +1}、quaternary {-1, -0.5, 0.5, 1} を比較し、本当に "feature filtering" が効いているかを sparsity 統計と性能で見る。
  - KV cache を 4-bit / 2-bit にしたときの長文タスク（needle in haystack, LongBench）での劣化を測る。
  - CPU / Apple Silicon / モバイル NPU 上で b1.58 3B を量子前後で動かして、論文の "edge/mobile" 主張を裏付けるベンチを取る。

## Notes / Quotes

- "every single parameter (or weight) of the LLM is ternary $\{-1, 0, 1\}$. It matches the full-precision ... Transformer LLM with the same model size and training tokens" (Abstract)
- "1.58-bit LLM defines a new scaling law and recipe for training new generations of LLMs" (Abstract) — 強い主張。
- absmean quantization: $\widetilde{W} = \mathrm{RoundClip}(W/(\gamma+\epsilon), -1, 1)$, $\gamma = \frac{1}{nm}\sum |W_{ij}|$ （§2）。
- 活性: BitNet と異なり非線形前のスケーリングを撤廃し $[-Q_b, Q_b]$ per-token に統一（§2、"negligible effects" と書いてある）。
- LLaMA-alike: RMSNorm, SwiGLU, RoPE, no biases。HuggingFace / vLLM / llama.cpp 統合を明示（§2）。
- PPL が FP16 を超える分岐点は 3B (Table 1: 10.04 → 9.91)。
- 70B 推論: 2× A100 80GB, pipeline parallel, seq 512, batch 16→176, throughput 333→2977 tokens/s (Table 3)。
- 等価宣言: "13B BitNet b1.58 is more efficient ... than 3B FP16 LLM" (efficiency 等価であって精度等価ではない)。
- 2T tokens で StableLM-3B を全 5 タスクで上回る (Table 4)。
- Discussion: 1-bit MoE / 長文 (8 bit → 将来 4 bit 活性) / Edge & Mobile (CPU フレンドリー) / 1-bit 専用ハード設計の呼びかけ。
- 著者明示の限定: 2-bit カーネルでの計測（最適化余地あり）、4-bit 以下活性は future work。

## Related Papers

- BitNet (Wang et al. 2023, `\cite{bitnet}`) — 直接の前身、binary 重み版。本論文はこれに 0 を足した拡張。
- LLaMA / LLaMA 2 (`\cite{llama, llama2}`) — アーキ・FP16 ベースライン基盤。
- GPTQ (`\cite{gptq}`), AWQ (`\cite{awq}`), SmoothQuant (`\cite{smoothquant}`), QuIP / QuIP# (`\cite{quip, quip_sharp}`) — PTQ ライン、intro で言及されるが直接比較はされていない。
- RMSNorm (`\cite{rmsnorm}`), SwiGLU (`\cite{swiglu}`), RoPE (`\cite{rope}`) — 採用コンポーネント。
- vLLM (`\cite{vllm}`), llama.cpp — エコシステム統合先。
- Ladder (`\cite{ladder}`) — 2-bit GPU カーネルを流用。
- StableLM-3B (`\cite{StableLM-3B-4E1T}`) — 2T tokens 比較対象。
- GPipe (`\cite{gpipe}`) — 70B 推論のパイプライン並列。
- Energy modeling: `\cite{energycost, pokebnn}` — 7nm でのエネルギー推定根拠。
- RedPajama (`\cite{redpajama}`) — 学習データ。
- 評価: ARC (`\cite{arc}`), Hellaswag, Winogrande, PIQA, OpenbookQA, BoolQ, SciQ, LAMBADA, WikiText2, C4。
