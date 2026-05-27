# DeepSeek-V3 Technical Report

- arXiv: https://arxiv.org/abs/2412.19437
- source: ../papers/arXiv-2412.19437v2/
- authors: DeepSeek-AI (research@deepseek.com、150 名超の連名。具体名は appendix の Contributions セクション参照)
- venue / year: TeX 中には明示なし（main.tex は technical report 形式）
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
  - **Chat モデル** (Table tab:chat): MMLU 88.5、MMLU-Redux 89.1、MMLU-Pro 75.9、DROP(3-shot F1) 91.6、IF-Eval 86.1、GPQA-Diamond 59.1、SimpleQA 24.9 (GPT-4o 38.2 に劣る)、FRAMES 73.3、LongBench v2 48.7（首位）、HumanEval-Mul 82.6、LiveCodeBench(CoT) 40.5、Codeforces percentile 51.6、SWE-Bench Verified 42.0 (Claude-3.5-Sonnet 50.8 に次ぐ)、Aider-Polyglot 49.6、**AIME 2024 39.2 / MATH-500 90.2 / CNMO 2024 43.2**（non-o1-like models で SOTA と著者主張）、C-SimpleQA 64.8、C-Eval 86.5。open-ended は **Arena-Hard 85.5 (open-source 初の 85% 超え)、AlpacaEval 2.0 LC-win 70.0**。RewardBench は DeepSeek-V3 平均 87.0、DeepSeek-V3 (maj@6) 平均 89.6。
  - **アブレーション** (Tab. ablation_nextn, ablation_noaux_tc): 大規模(228.7B/20.9B activated, 540B/578B tokens) で MTP は HumanEval 44.5→53.7、GSM8K 72.3→74.0 等を改善。auxiliary-loss-free は HumanEval 40.2→46.3、GSM8K 70.7→74.5。
  - **MTP 推論利用**: 2 トークン目の acceptance rate 85–90%、TPS 1.8 倍。
  - **R1 蒸留アブレーション** (Tab. distill): DeepSeek-V2.5 ベースで LiveCodeBench 31.1→37.4、MATH-500 74.6→83.2(ただし応答長 769→1510)。
  - **訓練安定性**: 14.8T を通して irrecoverable loss spike も rollback もゼロ。
- **貢献**: (1) auxiliary-loss-free load balancing と sequential MTP を組み合わせた MoE レシピ、(2) 超大規模での FP8 mixed precision 訓練の実証（loss 誤差 <0.25%）、(3) DualPipe + 専用 all-to-all kernel による cross-node MoE 通信のフルオーバーラップ、(4) long-CoT R1 → 標準 LLM への蒸留パイプライン（accuracy と長さの両立）、(5) 671B/37B-activated を 2.788M H800h で訓練し open-source SOTA を $5.576M 級コストで達成、(6) 将来チップ設計への具体的提案（FP8 累算精度、tile/block 量子化のネイティブ対応、IB-NVLink 統合 co-processor、転置 GEMM 融合 等）。

## Takeaway（自分にとっての要点）

- **auxiliary-loss-free load balancing の要点**: routing には bias $b_i$ だけ足し、gating には混ぜないという「ルーティング/重みづけの分離」。著者は batch-wise balancing が sequence-wise auxiliary loss より flexible で expert specialization を許すと説明し、Pile の domain 別 expert load 図と 1B/3B で validation loss が seq-wise 2.258 → free 2.253 / 3B で 2.085 → free 2.080 という実験を示している。
- **MTP は推論時に捨てられる training objective**: $D=1$ で causal chain を保ち、MTP module は inference で discard 可能。同一推論コストの ablation では HumanEval/GSM8K などが改善し、speculative decoding に転用した評価では second-token acceptance rate 85–90%、TPS 1.8×。
- **FP8 を「全 GEMM E4M3 + fine-grained scaling + CUDA core 昇格」で成立させる設計**: E4M3/E5M2 の混在ではなく E4M3 を全 tensors に使い、activation tile 1×128 / weight block 128×128 で outliers に対応し、$N_C=128$ ごとに CUDA Core 側で FP32 accumulation する。H800 の FP8 GEMM accumulation precision が約 14 bit に限られることを著者は明示している。
- **DualPipe のスケーリング主張は computation-to-communication ratio 一定が条件**: 著者は、計算通信比を一定に保てば fine-grained experts を cross-node に置いても near-zero all-to-all communication overhead を保てると述べる。実装は 20 SM を 10 communication channels に分け、warp specialization、customized PTX、communication chunk size auto-tuning を使う。
- **R1 蒸留の実用レシピ**: domain ごとに「専門家モデル」を SFT+RL で作って、`<problem, original>` と `<system prompt, problem, R1 response>` を混ぜて RL → 高温サンプリングで R1 パターンを取り込む → rejection sampling で最終 SFT。「reflection/verification を non-CoT モデルに移すには長い CoT 教師データを直接食わせるのではなく、中間 RL を挟む」のがコア。response 長と accuracy のトレードオフ（Tab. distill）を著者自身が明記。
- **C-SimpleQA で GPT-4o を 5.5pt 上回り、SimpleQA(英語) では 13.3pt 負ける**: 著者は SimpleQA の劣後を design focus/resource allocation に帰し、Chinese knowledge により多くの training tokens を割いたため C-SimpleQA が強いと説明している。
- **Self-rewarding (constitutional AI + voting で生成データを RL に回す)** を採用と明記。RewardBench 平均 87.0 → maj@6 で 89.6 と self-judge の voting を使う運用。
- **GRPO**: 本論では DeepSeek-V2 と同様に GRPO を採用し、policy と同サイズになりがちな critic model を使わず、group scores から baseline を推定すると説明している。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 671B/37B-activated を $5.576M, 2.788M H800 hour, 2 ヶ月未満で完了したというコストを Table 1 と本文で公開している。
  - FP8 training について、tile/block 量子化、$N_C$ 昇格、E4M3 統一、online scaling、BF16 optimizer states、FP8 activation caching/communication まで本文で具体的に記述し、将来ハードウェアへの提案も付けている。
  - auxiliary-loss-free と MTP には small/large MoE の ablation があり、R1 蒸留にも DeepSeek-V2.5 ベースの ablation がある。
  - Open-ended evaluation（Arena-Hard, AlpacaEval2）で 85.5 / 70.0。著者は Arena-Hard で 85% を超えた最初の open-source model と主張している。
  - 14.8T を通じて loss spike も rollback もゼロ、と訓練安定性まで強調。再現する側にとっては重要情報。
- **弱み / 疑問**:
  - **著者自身の limitations**: (i) inference の最小デプロイ単位が大きく（prefilling 32 GPU、decoding 320 GPU）小規模チームに重い、(ii) V2 比 2× の生成速度はまだ伸び代あり、と明記。「次世代 hardware が出れば自然に解決」と書いているが、現状ハードウェア前提では open-weight でも使えるユーザーは限定される。
  - **コスト数字に "prior research and ablation experiments" は含まれていない** と注記がある。$5.576M は official training のみの値なので、R&D コスト込みの比較には使えない。
  - SimpleQA 24.9 で GPT-4o (38.2) / Claude-3.5-Sonnet (28.4) に英語 factual knowledge で負ける。著者は design focus and resource allocation と説明し、中国語知識により多くの training tokens を割いたと述べる。
  - MTP の ablation は **D=1 のみ**。Meta の MTP 論文のように D を増やした比較が無く、「sequential causal chain」設計の優位性は実証されていない（自社の D=1 が D 独立ヘッドより良いか不明）。
  - aux-loss-free と batch-wise auxiliary loss が 1B/3B で同じ validation loss を達成している一方、batch-wise load balancing methods には efficiency 上の potential challenges が 2 点ある、と著者は述べる。本番採用の決定要因は TeX 中ではそれ以上詳しく分解されていない。
  - R1 蒸留が「reasoning を入れた代わりに応答が長くなる」と明示しているが、最終 V3 の応答長について chat eval Table の max 8192 tokens 以外の統計は TeX 中には示されていない。
  - Codeforces percentile 51.6 は他社（Claude 20.3, GPT-4o 23.6）を大きく上回るが、TeX 中の評価詳細は "percentage of competitors" という説明に限られている。
  - 安全性（red-teaming, jailbreak, refusal calibration）について本文ではほぼ触れていない。RewardBench の Safety 87.0 のみが間接情報。SFT/RL での toxicity 制御や Chinese 文脈での compliance 設計は明示されていない。
  - "Self-rewarding" として constitutional AI + voting を採用と書いているが、constitution の中身、voting の規模、reward hacking の検出方法は具体的に書かれていない。再現困難。
  - Pre-training データの中身が依然ブラックボックス。「math/code 比率を上げた」「多言語拡張した」程度。
- **次に試したいこと**:
  - aux-loss-free の bias 更新を「動的に止める」(最後 500B で $\gamma=0$) という運用が、 final loss / expert specialization に与える影響を ablate する（評者補足）。$\gamma$ schedule の意義は本論では言及のみ。
  - MTP depth を $D\in\{1,2,4\}$ で比較し、acceptance rate と TPS の Pareto を引く（評者補足）。本論の MTP ablation は 1-depth MTP module の比較に限られる。
  - FP8 E4M3 統一 + tile/block scaling を BF16 と「同じ optimizer state(BF16) + master FP32」条件で比較する（評者補足）。本論の「<0.25% 相対 loss 誤差」は high-precision accumulation と fine-grained quantization を組み合わせた結果。
  - R1 蒸留パイプを math/code 以外のドメイン（医療、法律、エージェント計画）に拡張し、accuracy vs length のフロンティアがどう動くかを見る（評者補足）。
  - SimpleQA(英語) の劣勢を「英語 web 比率を増やすだけ」で埋められるのか、それとも retrieval を後段に挟むほうが効くのかを比較する（評者補足）。
  - DualPipe を non-MoE / dense モデルに転用したときの bubble 削減と 2× parameter memory のトレードオフを見る（評者補足）。DualPipe は micro-batch 数が増えても bubble/activation memory が増えない、と著者は述べる。

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
- (verified 2026-05-27) venue/year を TeX で確認できる範囲に限定し、Takeaway / Critical Thoughts から TeX 根拠のない一般化・外部比較・推測表現を削除または評者補足として明示 (main.tex, content/fp8.tex, tables/chat_evaluation.tex)
- (verified 2026-05-27) RewardBench、SimpleQA/C-SimpleQA、R1 蒸留、MTP/auxiliary-loss-free ablation の数値と留保を TeX 表・本文に合わせて修正 (main.tex, tables/chat_evaluation.tex)

## Related Papers

- DeepSeek-V2 (DeepSeek-AI 2024) — MLA / DeepSeekMoE の前身。アーキ・YaRN 設定をほぼ踏襲。
- DeepSeek-R1 series — post-training の蒸留教師。本論文の reasoning データ生成元（TeX 本文で言及、main.bbl 項目なし）。
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
