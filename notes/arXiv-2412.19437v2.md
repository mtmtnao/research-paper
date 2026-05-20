# DeepSeek-V3 Technical Report

- arXiv: https://arxiv.org/abs/2412.19437
- source: ../papers/arXiv-2412.19437v2/
- authors: DeepSeek-AI (research@deepseek.com、150 名超の連名。具体名は appendix の Contributions セクション参照)
- venue / year: arXiv preprint, 2024 (v2)
- tags: [LLM, MoE, FP8, pre-training, post-training, MLA, MTP, distillation, infrastructure]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: open-source LLM を closed-source frontier（GPT-4o, Claude-3.5-Sonnet）に追いつかせたい。スケールアップしつつ訓練コスト・推論コストを抑える必要がある。具体的な技術課題は (a) MoE の load balancing が補助損失で性能を削る、(b) cross-node MoE の all-to-all 通信がボトルネック、(c) FP8 大規模訓練の実証がほぼ無い、(d) long-CoT モデル(R1) の reasoning を non-CoT モデルに移植するレシピが未確立、の4点。
- **手法**:
  - **アーキ**: 671B 総パラ/37B 活性化の MoE。V2 から踏襲した MLA（KV を $d_c=512$、Q を $d_c'=1536$ に低ランク圧縮、decoupled RoPE $d_h^R=64$、$n_h=128$、層数 61、hidden 7168）と DeepSeekMoE（1 shared expert + 256 routed experts、各トークン 8 expert 活性化、$M=4$ ノードまで）を採用。
  - **Auxiliary-loss-free load balancing**: 各 expert にバイアス $b_i$ を持たせ、top-K ルーティングは $s_{i,t}+b_i$ で決め、gating は元の $s_{i,t}$ のまま使う。各ステップ後に overload なら $-\gamma$、underload なら $+\gamma$ で更新（$\gamma=0.001$、最後の 500B トークンで 0）。補助 sequence-wise balance は $\alpha=0.0001$ と極めて小さく副次的にのみ使う。
  - **Multi-Token Prediction (MTP)**: 深さ $D=1$ で「次の次のトークン」も追加予測。Meta MTP のような独立ヘッドではなく、causal chain を保ったまま逐次予測する設計（EAGLE に近い）。weight $\lambda=0.3$ (最初の 10T) → $0.1$ (残り 4.8T)。推論時は捨てるか speculative decoding に再利用可能。
  - **FP8 mixed precision 訓練**: 大規模学習で初めて end-to-end に成功と主張。Fprop/Dgrad/Wgrad の3 GEMM をすべて FP8(E4M3 統一) で実行。activation は 1×128 タイル、weight は 128×128 ブロックで fine-grained quantization、 $N_C=128$ ごとに Tensor Core → CUDA Core に部分和を昇格して FP32 蓄積。embedding/output head/MoE gating/normalization/attention は元の精度を維持。BF16 ベースラインに対し相対 loss 誤差 < 0.25%。
  - **インフラ**: 2048 H800、16-way PP + 64-way EP(8 ノード) + ZeRO-1 DP、TP は使わない。**DualPipe** という双方向 PP スケジューラを導入。forward と backward を attention/all-to-all dispatch/MLP/all-to-all combine に分割し、20 SM を 10 通信チャネルに割り当てて IB(50GB/s)↔NVLink(160GB/s) の warp specialization で通信を計算と完全オーバーラップ。各トークンを最大 4 ノードに制限し、ノード内では 3.2 expert/ノードまで効果的にスケール。RMSNorm と MLA up-projection は recompute、optimizer の EMA は CPU に。
  - **データ・スケジュール**: 14.8T tokens、4K 系列長で pre-train、FIM(PSM 形式) を 0.1 で混合、tokenizer は 128K vocab byte-level BPE。AdamW($\beta_1=0.9,\beta_2=0.95,\text{wd}=0.1$)、lr は 2K step で 0→$2.2\times10^{-4}$、10T までフラット、4.3T で cosine で $2.2\times10^{-5}$、残り 500B で $2.2\times10^{-5}\to7.3\times10^{-6}$ に2段下げ。バッチサイズ 3072→15360（最初 469B で）。YaRN ($s=40,\alpha=1,\beta=32$) で 4K → 32K → 128K と2段 context extension（各 1000 step）。
  - **Post-training**: 1.5M 件の SFT(2 epoch、cosine lr $5\times10^{-6}\to1\times10^{-6}$、sample masking でパッキング)。reasoning データは domain ごとの「expert model」を SFT+RL で作り、`<problem, original>` と `<system prompt, problem, R1 response>` の2形式を混ぜて RL → rejection sampling で最終 SFT データを蒸留。RL は GRPO（critic 不要、グループ内 reward の標準化で advantage）。rule-based RM(math/code) と model-based RM を併用、後者は CoT 付き preference data で reward hacking 緩和。
- **結果**:
  - **訓練コスト**: 合計 2.788M H800 GPU hours（pre-training 2664K + 32K/128K 拡張 119K + post-training 5K）、$2/GPU-hour で $5.576M。1T トークンあたり 180K H800h（2048 H800 で 3.7 日）。
  - **Base モデル** (Table 2 = `tables/base_evaluation.tex`): V2-Base / Qwen2.5-72B-Base / LLaMA-3.1-405B-Base に多くで勝利、特に code/math/multilingual。
  - **Chat モデル** (Table tab:chat): MMLU 88.5、MMLU-Redux 89.1、MMLU-Pro 75.9、DROP(3-shot F1) 91.6、IF-Eval 86.1、GPQA-Diamond 59.1、SimpleQA 24.9 (GPT-4o 38.2 に劣る)、FRAMES 73.3、LongBench v2 48.7（首位）、HumanEval-Mul 82.6、LiveCodeBench(CoT) 40.5、Codeforces percentile 51.6、SWE-Bench Verified 42.0 (Claude-3.5-Sonnet 50.8 に次ぐ)、Aider-Polyglot 49.6、**AIME 2024 39.2 / MATH-500 90.2 / CNMO 2024 43.2**（non-o1 系で SOTA）、C-SimpleQA 64.8、C-Eval 86.5。open-ended は **Arena-Hard 85.5 (open-source 初の 85% 超え)、AlpacaEval 2.0 LC-win 70.0**。RewardBench 平均 87.0、maj@6 で 89.6 と GPT-4o-0806/Claude-3.5-Sonnet-1022 に並ぶ。
  - **アブレーション** (Tab. ablation_nextn, ablation_noaux_tc): 大規模(228.7B/20.9B activated, 540B/578B tokens) で MTP は HumanEval 44.5→53.7、GSM8K 72.3→74.0 等を改善。auxiliary-loss-free は HumanEval 40.2→46.3、GSM8K 70.7→74.5。
  - **MTP 推論利用**: 2 トークン目の acceptance rate 85–90%、TPS 1.8 倍。
  - **R1 蒸留アブレーション** (Tab. distill): DeepSeek-V2.5 ベースで LiveCodeBench 31.1→37.4、MATH-500 74.6→83.2(ただし応答長 769→1510)。
  - **訓練安定性**: 14.8T を通して irrecoverable loss spike も rollback もゼロ。
- **貢献**: (1) auxiliary-loss-free load balancing と sequential MTP を組み合わせた MoE レシピ、(2) 超大規模での FP8 mixed precision 訓練の実証（loss 誤差 <0.25%）、(3) DualPipe + 専用 all-to-all kernel による cross-node MoE 通信のフルオーバーラップ、(4) long-CoT R1 → 標準 LLM への蒸留パイプライン（accuracy と長さの両立）、(5) 671B/37B-activated を 2.788M H800h で訓練し open-source SOTA を $5.576M 級コストで達成、(6) 将来チップ設計への具体的提案（FP8 累算精度、tile/block 量子化のネイティブ対応、IB-NVLink 統合 co-processor、転置 GEMM 融合 等）。

## Takeaway（自分にとっての要点）

- **「補助損失を切る」が当たり前になりつつある**: routing には bias $b_i$ だけ足し、gating には混ぜないという「ルーティング/重みづけの分離」が肝。バッチ単位の負荷バランスは expert specialization を許す（Pile の domain 別 expert load 図と 1B/3B で validation loss が seq-wise 2.258 → free 2.253 / 3B で 2.085 → 2.080 という小差だが一貫した改善）→ 「sequence-wise の balance loss は専門化を潰す」という主張は説得力がある。
- **MTP は inference を重くせず精度を上げる**: $D=1$ で causal chain を保つだけで MMLU/HumanEval/GSM8K がほぼ全 size で改善し、推論時に speculative decoding に転用すれば 1.8× TPS。実装コストの割にお得。
- **FP8 を「全 GEMM E4M3 + fine-grained scaling + CUDA core 昇格」で押し切った**: E4M3/E5M2 の混在を諦め、tile 1×128 / block 128×128 で outliers を抑え、$N_C=128$ ごとに FP32 昇格して accumulate。H800 が 14-bit しか溜まらないという Hopper の不都合を明示しているのが珍しい。
- **DualPipe は実装重いが「通信を完全に隠す」前提なら fine-grained MoE をいくらでも拡張できる**: 「計算-通信比一定なら expert を全ノードに刻んでも通信オーバーヘッドゼロに保てる」というスケーリング論。実装には 20 SM を warp specialization で 10 チャネル化、PTX 手書き、TMA + L2 制御、と相当泥臭い。
- **R1 蒸留の実用レシピ**: domain ごとに「専門家モデル」を SFT+RL で作って、`<problem, original>` と `<system prompt, problem, R1 response>` を混ぜて RL → 高温サンプリングで R1 パターンを取り込む → rejection sampling で最終 SFT。「reflection/verification を non-CoT モデルに移すには長い CoT 教師データを直接食わせるのではなく、中間 RL を挟む」のがコア。response 長と accuracy のトレードオフ（Tab. distill）を著者自身が明記。
- **C-SimpleQA で GPT-4o を 5.5pt 上回り、SimpleQA(英語) では 13pt 負ける**のは「学習データの言語配分の選択」と明言。「英語factual」と「中国語factual」が完全にトレードオフされている、という運用判断は他社モデル比較を読むとき覚えておきたい。
- **Self-rewarding (constitutional AI + voting で生成データを RL に回す)** を採用と明記。RewardBench 平均 87.0 → maj@6 で 89.6 と self-judge の voting を使う運用。
- **GRPO** はもはや DeepSeek の標準 RL recipe（DeepSeekMath 由来）。critic を捨てるのは MoE のような巨大 policy で特に効く。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 671B/37B-activated を $5.576M, 2.788M H800 hour, 2 ヶ月未満で完了したというコストの公開はインパクト大。closed-source 各社の暗黙の前提（数千万 GPU-hour 級）を強く揺さぶる。
  - FP8 を「論文として記述できる範囲で」最初から最後までやり切った数少ない実例。tile/block 量子化、$N_C$ 昇格、E4M3 統一、online scaling、BF16 EMA in CPU、…と implementation の網羅性が高く、ハードウェア提案までセットなのが好み。
  - auxiliary-loss-free, MTP, R1 蒸留すべてに**同一規模のアブレーション**（small 15.7B vs large 228.7B）を付けていて、effect size の話を逃げていない。
  - Open-ended（Arena-Hard, AlpacaEval2）で 85.5 / 70.0 と「定型 benchmark しか勝てない open model」のイメージを破った。
  - 14.8T を通じて loss spike も rollback もゼロ、と訓練安定性まで強調。再現する側にとっては重要情報。
- **弱み / 疑問**:
  - **著者自身の limitations**: (i) inference の最小デプロイ単位が大きく（prefilling 32 GPU、decoding 320 GPU）小規模チームに重い、(ii) V2 比 2× の生成速度はまだ伸び代あり、と明記。「次世代 hardware が出れば自然に解決」と書いているが、現状ハードウェア前提では open-weight でも使えるユーザーは限定される。
  - **コスト数字に "prior research and ablation experiments" は含まれていない** と注記がある。$5.576M は「公式 run 1 本のみ」の値で、実態の R&D コストはずっと大きい。比較材料として扱う際は注意。
  - SimpleQA 24.9 で GPT-4o (38.2) / Claude-3.5-Sonnet (28.4) に英語 factual で負けるのを「設計選択」と片付けているが、SimpleQA はモデルが何を知らないかを測る代表ベンチで、利用者視点では Claude 比 -3.5pt は普通に痛い。
  - MTP の ablation は **D=1 のみ**。Meta の MTP 論文のように D を増やした比較が無く、「sequential causal chain」設計の優位性は実証されていない（自社の D=1 が D 独立ヘッドより良いか不明）。
  - aux-loss-free と batch-wise aux loss が 1B/3B で同等性能と書いているのに、本番では aux-loss-free を採用した強い理由が示されていない（「実装が簡単だから」程度の論調）。
  - R1 蒸留が「reasoning を入れた代わりに応答が長くなる」と明示しているが、最終 V3 の応答長について chat eval Table の数字（max 8K 出力制限）以外の統計が無い。MATH-500 などで実際に何 token 使っているのか欲しい。
  - Codeforces percentile 51.6 は他社（Claude 20.3, GPT-4o 23.6）を大きく上回るが、これは LiveCodeBench 系のように評価方法が異なる可能性がある（"percentage of competitors" としか書かれていない）。R1 由来の bias が出ていないか気になる。
  - 安全性（red-teaming, jailbreak, refusal calibration）について本文ではほぼ触れていない。RewardBench の Safety 87.0 のみが間接情報。SFT/RL での toxicity 制御や Chinese 文脈での compliance 設計は明示されていない。
  - "Self-rewarding" として constitutional AI + voting を採用と書いているが、constitution の中身、voting の規模、reward hacking の検出方法は具体的に書かれていない。再現困難。
  - Pre-training データの中身が依然ブラックボックス。「math/code 比率を上げた」「多言語拡張した」程度。
- **次に試したいこと**:
  - aux-loss-free の bias 更新を「動的に止める」(最後 500B で $\gamma=0$) という運用が、 final loss / expert specialization に与える影響を ablate する。$\gamma$ schedule の意義は本論では言及のみ。
  - MTP depth を $D\in\{1,2,4\}$ で比較し、acceptance rate と TPS の Pareto を引く。speculative decoding に再利用するなら $D$ を増やしたい誘惑がある。
  - FP8 E4M3 統一 + tile/block scaling を BF16 と「同じ optimizer state(BF16) + master FP32」条件で純粋比較したい。本論の「<0.25% 相対 loss 誤差」は trick の塊で、構成要素ごとの寄与が読めない。
  - R1 蒸留パイプを math/code 以外のドメイン（医療、法律、エージェント計画）に拡張すると length が爆発するはずで、accuracy vs length のフロンティアがどう動くか。
  - SimpleQA(英語) の劣勢を「英語 web 比率を増やすだけ」で埋められるのか、それとも retrieval を後段に挟むほうが効くのかの比較。
  - DualPipe を non-MoE / dense モデルに転用したときの bubble 削減と 2× メモリのトレードオフ。「micro-batch の数に依存しない」というのは強い性質なので dense でも美味しい可能性。

## Notes / Quotes

- "we did not experience any irrecoverable loss spikes or perform any rollbacks." (abstract / introduction)
- "Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training." (abstract)
- "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score $s_{i,t}$." (§Architecture, aux-loss-free)
- "Different from [Meta MTP], which parallelly predicts $D$ additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth." (§MTP)
- "for the first time, validate the feasibility and effectiveness of FP8 training on an extremely large-scale model." (introduction)
- "the relative loss error of our FP8-training model remains consistently below 0.25%, a level well within the acceptable range of training randomness." (content/fp8.tex)
- "the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits" (content/fp8.tex)
- "only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink." (§Infra, all-to-all)
- "the acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics" (§Post-training discussion)
- 著者明示の limitation: "the recommended deployment unit for DeepSeek-V3 is relatively large, which might pose a burden for small-sized teams" / "there still remains potential for further enhancement" (§Conclusion)
- TeX 中には明示されていないこと: redundant experts のヒューリスティック詳細、constitutional AI の constitution 内容、pre-training データ混合の具体的比率。

## Related Papers

- DeepSeek-V2 (DeepSeek-AI 2024) — MLA / DeepSeekMoE の前身。アーキ・YaRN 設定をほぼ踏襲。
- DeepSeek-R1 series — post-training の蒸留教師。本論文の reasoning データ生成元。
- DeepSeekMath (Shao et al., 2024) — GRPO の出典。
- Wang et al. *Auxiliary-Loss-Free Load Balancing* (`noaux_tc`) — 採用された load balancing 法の原典。
- Meta MTP (Gloeckle et al., 2024) — MTP の比較対象。本論文は causal chain 維持で差別化。
- EAGLE (Li et al., 2024) — sequential MTP の構造的類似物（目的は speculative decoding）。
- ZeroBubble (Qi et al., 2023) / 1F1B (PipeDream) / Chimera — DualPipe の比較対象。
- YaRN (Peng et al., 2023) — context extension。
- Microscaling formats (Rouhani et al., 2023) — fine-grained quantization と整合し、Blackwell でハード支援される方向。
- RewardBench (Lambert et al., 2024) — generative RM 評価。
- SimpleQA / C-SimpleQA, GPQA, MMLU-Pro, MMLU-Redux, LongBench v2, FRAMES, AIME 2024, CNMO 2024, SWE-Bench Verified, Aider, LiveCodeBench, Codeforces — 評価ベンチ群。
- Constitutional AI (Bai et al., 2022) — self-rewarding の枠組み。
