# Jamba: A Hybrid Transformer-Mamba Language Model

- arXiv: https://arxiv.org/abs/2403.19887
- source: ../papers/arXiv-2403.19887v2/
- authors: Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avshalom Manevich, Nir Ratner, Noam Rozen, Erez Schwartz, Mor Zusman, Yoav Shoham
- venue / year: TeX 中には明示なし（main.tex は neurips_2023.sty を preprint で使用）
- tags: [LLM, hybrid-architecture, Mamba, SSM, Transformer, MoE, long-context]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: Transformer は KV cache とトークン毎の全文脈計算のせいで長文脈・高スループットに弱い。一方 SSM/Mamba は長距離依存に強く推論効率も良いが、同サイズの Transformer に性能で劣る。スケール大での両者ハイブリッドおよび MoE との統合は production-grade では未確立。
- **手法**: Transformer 層と Mamba 層を `a:m` 比でインターリーブし、MLP の一部を MoE（`n` 専門家、top-`K`）で置換する Jamba ブロックを積む decoder-only アーキテクチャ。リリース設定は Jamba ブロック × 4、各ブロック `l=8`, `a:m=1:7`, `e=2`, `n=16`, `K=2`。GQA / SwiGLU / 語彙 64K BPE（数字は 1 桁 1 トークン、Llama/Mistral のダミー空白は除去）。位置エンコーディングは付けない（Mamba 層が暗黙の位置情報を担う）。7B 規模で Mamba 層内部活性が暴れて loss spike が出たので Mamba 内部に RMSNorm を追加して安定化。
- **結果**: 12B active / 52B total の Jamba を公開（Apache 2.0、`huggingface.co/ai21labs/Jamba-v0.1`）。単一 A100 80GB に int8 で収まり、KV cache は 256K 文脈で **4GB**（Mixtral 32GB, Mistral 32GB, LLAMA-2 (7B) 128GB; Table 1）。256K 文脈をサポート、訓練は最大 1M トークンまで実施。**学術ベンチ**（Table 2）: HellaSwag 87.1, WinoGrande 82.5, ARC-E 73.5, ARC-C 64.4, PIQA 83.2, NQ 45.9, TruthfulQA 46.4, BoolQ 88.2, QuAC 40.9, GSM8K 59.9, HumanEval 29.3, MMLU 67.4, BBH 45.4 — Mixtral / Llama-2 70B と概ね同等。**L-Eval（3-shot F1, Table 3）**: LongFQA 0.44 / CUAD 0.44 / NarrativeQA 0.30 / NQ 0.60 / SFiction 0.40, avg **0.44 vs Mixtral 0.43**。Needle-in-a-haystack 256K でも良好。スループットは長文脈で Mixtral の **3x**（bs=16 単一 GPU 8K 文脈、および 4 GPU で 128K 文脈いずれも; Fig.3）。
- **貢献**: (1) production-grade で初の Attention-SSM ハイブリッド LLM（実装は MoE も含む）の構築と公開、(2) `a:m`、MoE 統合、位置情報、RMSNorm 安定化に関する大規模スケール ablation の体系化、(3) pure Mamba が IMDB / QuAC / NarrativeQA で format 適合に失敗するという ICL 上の限界を示し、Attention 1 層あれば修復されることを実証、(4) 小規模訓練 run の checkpoint 公開予定によりコミュニティが追検証できる土台を提供。

## Takeaway（自分にとっての要点）

- **この ablation では「Attention は 1/8 で十分」**: `a:m=1:7` と `1:3` は perplexity / OLLM ともほぼ同性能（Table 4）。長文脈用に attention 層を間引きたい設計の根拠として強い。
- **Pure Mamba が落ちる箇所は perplexity ではなく format adherence**（IMDB 48.8 vs Attention 84.1, Mamba は "Positive/Negative" でなく "Very Good", "3/10" 等を出す）。これは「ICL を支える induction head」が SSM では emergent に出にくいせい、と著者は仮説立て、ハイブリッド機の Attention 層に実際に induction-head 様の挙動を可視化（Fig.8、3 つの attention 層に 12 個）。**ICL の format 追従を補うために attention を最小限差し込む** という設計指針。
- **MoE は 7B/50B tokens の hybrid Attention-Mamba で効く**: OLLM 36.6→38.1, HellaSwag 62.5→66.0, NQ 15.4→18.9（Table 7）。Mixtral 同等の active param 数（12B）を維持しつつ容量を 52B まで稼げる。
- **位置情報なしで OK**: RoPE 有無で結果ほぼ同じ（Table 8）。Mamba 層が attention の前にあるので暗黙に位置を伝える、という解釈。実用面では実装が単純になる。
- **KV cache 4GB（256K 文脈）の効きは想像より大きい**: Mixtral が 32GB なので、256K 級の運用では「同じ GPU で文脈 2x, batch 増える」という派生効果が出る（Fig.2 で 2x context vs Mixtral, 7x vs Llama-2-70B）。
- 大規模 Mamba は **内部活性が発散しやすい** → RMSNorm を Mamba 内部に挿す（Fig.9）。Mamba を 7B+ に持っていく実装上の地雷情報として有益。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「production-grade で動く」初の Attention-SSM ハイブリッド（実装は MoE も含む）を Apache 2.0 で公開した実績そのものが大きい。ablation 由来の小規模チェックポイントも公開予定で、コミュニティが追検証しやすい。
  - 設計選択（`a:m`, MoE, RoPE, RMSNorm）を 1.3B / 7B でちゃんと ablation しており、最終 12B/52B の構成根拠がトレース可能。
  - 「pure Mamba は ICL の format 追従が苦手」を IMDB の誤答事例＋hybrid 側の induction-head 可視化までセットで示しており、SSM 単独の限界を質的に裏付けている。
  - KV cache 4GB と 3x throughput は同 active param 帯（Mixtral 12.9B vs Jamba 12B）での比較になっており、ハイブリッドの正味の利得が見える。
- **弱み / 疑問**:
  - **学習データが in-house**（Web / books / code、2024 年 3 月時点）で、ノート作成者側からはデータ汚染を検証できない。MMLU 67.4 や GSM8K 59.9 は Llama-2-70B / Mixtral と同じレンジにあるが、ベンチ汚染の議論は TeX 中には見当たらない（著者自身も冒頭で「benchmark は gameable」と書いている）。
  - **base model のみ**で alignment / instruction tuning / moderation 無し、と著者明記。「instruction-following で本当に強いか」の評価が空白。L-Eval も 3-shot で押している。
  - **Long-context 評価が薄い**: needle-in-a-haystack は単純な retrieval、L-Eval も 5 データセット avg F1 0.44 vs Mixtral 0.43 とほぼタイ。Jamba の売りである 256K を活かす真の long-reasoning（multi-hop QA, 長文 code 編集等）の評価が無い。
  - **TruthfulQA / HumanEval が Llama-2-70B と Mixtral に負ける** (HumanEval 29.3 vs Mixtral 34.8, Llama-2-70B 29.9)。コード生成・truthfulness は弱め、と表からは読める。著者はここを掘っていない。
  - ablation はほぼ 1.3B（250B tokens）か 7B（**50B tokens**）止まりで、12B/52B での再現は無い。MoE の有無や `a:m` を 12B 規模で確かめてはいない。
  - 「位置情報不要」の主張は 1.3B / 250B tokens の 1 ペアでの比較なので、長文脈で本当に大丈夫かは別途検証が要る（Table 8 の NarrativeQA は 50.5 vs 46.2 で Jamba (RoPE 無) が勝つが、サンプル少）。
  - 著者自身が認めている limitations: (a) 公開モデルは base のみで production / end user 利用は要追加適応、(b) Mamba 単体は format adherence / ICL が弱い、(c) 7B+ で Mamba 内部活性が発散して loss spike が出る（→ RMSNorm で対処済）。
- **次に試したいこと（評者補足）**:
  - 同じ active param 予算（12B）で `a:m` を 1:3, 1:7, 1:15 に振った長文脈タスク（multi-hop QA, code agent 系）の pareto を取り直す。`1:7` が本当に sweet spot か。
  - Jamba を SFT / DPO してから L-Eval / LongBench / RULER 等の long-context bench を回し、base 公開モデルでなく aligned 状態での attention-MoE/Mamba ハイブリッドの真価を測る。
  - Induction-head が attention 層数とともにどうスケールするか定量化（4 層中何層・何ヘッドが induction か、層を 1 枚にしたら ICL がいつ壊れるか）。
  - Mamba 単体の format 追従の失敗を、RLHF や format-aware SFT でどこまで詰められるか — ICL の不可分性を切り分けたい。
  - KV cache 4GB を活かして、ローカル 80GB GPU で公開モデルのサポート範囲である 256K トークン推論を回し、Mamba の暗黙位置情報が極長文脈で破綻する兆候があるかを見る。

## Notes / Quotes

- "All of this renders Jamba the first production-grade Attention-SSM hybrid model." (introduction)
- "We found that with the Mamba layer, positional embeddings or mechanisms like RoPE are not necessary, and so we do not use any explicit positional information." (§Model Architecture)
- "The Jamba model released is a pretrained base model, which did not go through alignment or instruction tuning, and does not have moderation mechanisms. It should not be used in production environments or with end users without additional adaptation." (introduction, 強調マーク付きで明示)
- "We hypothesize that this phenomenon points to a limitation of SSMs -- a potential difficulty in in-context learning (ICL)." (§Why does the Combination Work?)
- "We have found 12 such [induction] heads in our hybrid model, in all three attention layers (which correspond to layers 4, 12, 20 in the model)." (§Why does the Combination Work?)
- リリース構成: Jamba ブロック × 4、各ブロック `l=8`, `a:m=1:7`, `e=2`, `n=16`, `K=2`（§Jamba Implementation for a Single 80GB GPU）。
- KV cache 比較（Table 1, 256K context, 16bit）: LLAMA-2 (6.7B) 128GB / Mistral (7.2B) 32GB / Mixtral (46.7B/12.9B active) 32GB / **Jamba (52B/12B active) 4GB**。
- 1.3B / 250B tokens（Table 4）: Attention OLLM 36.4, Mamba 36.1, Jamba(1:3) 37.2, Jamba(1:7) 37.2。
- 7B / 50B tokens MoE 有無（Table 7）: Jamba no-MoE OLLM 36.6 → Jamba+MoE 38.1, NQ 15.4 → 18.9。
- IMDB / QuAC / NarrativeQA（Table 6）で pure Mamba が大きく落ちる：IMDB 48.8（Attn 84.1, hybrid 90.9）。
- 訓練は 1M tokens まで実施、リリースは 256K サポート（§Jamba Implementation for a Single 80GB GPU）。
- TeX 中には明示されていない事項: 訓練データの具体量・トークン総数、最終 12B/52B モデルの正確な訓練 token 数、評価時の prompt 詳細、Mixtral / Llama-2 のスコアの取得方法（自家評価か引用か）。
- (verified 2026-05-20) Summary の KV cache 比較で「Llama-2-70B 128GB」と誤記していたのを「LLAMA-2 (7B) 128GB」に訂正 (main.tex Table 1: LLAMA-2 行は 6.7B/6.7B active)。Llama-2-70B の KV cache は Table 1 に存在しない。
- (verified 2026-05-20) 図表番号を訂正: induction 可視化は Fig.7→Fig.8, RMSNorm spike-norm は Fig.6→Fig.9 (main.tex の \begin{figure} 出現順から確認)。
- (verified 2026-05-20) 表番号を訂正: MoE 有無テーブルは Table 6→Table 7, IMDB/QuAC/NarrativeQA テーブルは Table 5→Table 6 (main.tex の \begin{table} 出現順 = params-cache, eval-benchmarks, leval, attn-mamba-amba-1.3b, attn-mamba-amba-7b, mamba-problems, jamba-moe, position から確認)。
- (verified 2026-05-27) authors から TeX に明示のない所属表記を削除し、venue/year を TeX で確認できる範囲に限定 (main.tex author block, neurips_2023 preprint style)。
- (verified 2026-05-27) 「初の Attention-SSM-MoE」「MoE は大規模で初めて効く」を、TeX の主張に合わせて Attention-SSM hybrid / 7B ablation での MoE 効果に弱めた (main.tex Introduction, Table jamba-moe)。
- (verified 2026-05-27) Related Papers の文献名・著者年表記を main.bbl に合わせて修正 (main.bbl)。
- (verified 2026-05-27) Critical Thoughts の評者提案を TeX 事実と区別し、公開モデルが 1M token 推論をサポートするように読める記述を 256K support に修正 (main.tex Jamba Implementation for a Single 80GB GPU)。

## Related Papers

- Gu & Dao 2023, Mamba — SSM 側の本幹、Jamba の Mamba 層の基盤。
- Vaswani+ 2017, Transformer — Attention 層の基盤。
- Jiang+ 2024, Mixtral-8x7B — 同 active param 帯の主要比較対象、MoE 設計の先行例。
- Touvron+ 2023, Llama-2 — 70B / 13B が比較対象、GQA / SwiGLU の参照。
- Fu+ 2022, H3 — Attention を 2 層差し込む SSM ハイブリッドの直接の先行（Jamba は scale で凌駕と主張）。
- Poli+ 2023, Hyena / StripedHyena — Attention-SSM 混合の 7B 先行、Mistral-7B には及ばずという比較線。
- Shazeer+ 2017, sparsely-gated MoE / Fedus+ 2022, Switch Transformer — MoE の基盤。
- Olsson+ 2022, Induction heads — pure Mamba が ICL に弱い理由づけに引用。
- Ratner+ 2023, parallel context windows — long-context 分類評価データ（Trec-Fine, NLU Intent, Banking77, CLINC150）の出典。
- An+ 2023, L-Eval — long-context QA の評価枠組み。
- Kamradt 2023, needle-in-a-haystack — 長文脈 retrieval 評価。
- Ali+ 2024, "Hidden Attention of Mamba" — SSM から attention 様スコアを抽出する後続研究。
