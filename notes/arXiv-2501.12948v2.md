# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- arXiv: https://arxiv.org/abs/2501.12948
- source: ../papers/arXiv-2501.12948v2/
- authors: DeepSeek-AI (core contributors: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, ほか)
- venue / year: arXiv preprint, 2025（TeX は独自 `deepseek.cls` を使用し、`\figurename` を `Supplementary Fig.`、`\tablename` を `Supplementary Table` に再定義。掲載先・採録会議は TeX 中には明示なし）
- tags: [LLM, reasoning, reinforcement-learning, GRPO, chain-of-thought, distillation, DeepSeek]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: LLM の推論能力強化は CoT prompting や SFT に依存しがちで、人手で書かれた推論トレースの質・量にスケーラビリティが縛られる。さらに人間の思考様式を模倣させる枠組みでは、人類が知らない推論パターンに到達できない可能性がある。
- **手法**:
  - ベースは DeepSeek-V3-Base（671B MoE、活性化 37B）。
  - **DeepSeek-R1-Zero**: SFT を一切経ず、`<think>...</think><answer>...</answer>` という最小限のテンプレだけ与えて、GRPO（Group Relative Policy Optimization、価値モデルなしで group 内 reward を mean/std 正規化して advantage に使う PPO 派生）で純粋 RL。報酬は rule-based のみで accuracy（数式 boxed answer マッチ、コードはコンパイラ＋テストケース）＋ format（think タグ）。learning rate 3e-6、KL 係数 0.001、temperature 1、batch 512（32 questions × 16 sample）、10,400 step（1.6 epoch）、8.2k step を境に max length 32,768 → 65,536 へ拡張。
  - **DeepSeek-R1** はマルチステージ：(1) 数千件の会話的 cold-start CoT で SFT → (2) 1 段目 RL（言語一貫性報酬 `Reward_language = 目標言語語数/総語数` 追加、GRPO clip ε=10）→ (3) rejection sampling で reasoning 600k + 非 reasoning 200k ≈ 800k 件の SFT → (4) 2 段目 RL（推論は rule、一般タスクは helpful/safety RM、温度 0.7、計 1,700 step、最後の 400 step だけ preference reward を入れて reward hacking を回避）。
  - 蒸留版（Table `distill_config`）の base は **Qwen2.5-Math-1.5B / Qwen2.5-Math-7B / Qwen2.5-14B / Qwen2.5-32B / Llama-3.1-8B / Llama-3.3-70B-Instruct**（1.5B・7B のみ Qwen2.5-Math 系列。学習率はそれぞれ 1e-4 / 8e-5 / 7e-5 / 6e-5 / 5e-5 / 2e-5）。上記 800K で SFT のみ（RL は入れず比較のため）。
- **結果**（Pass@1、temperature 0.6 / top-p 0.95、AIME・GPQA は k=64 サンプル平均）:
  - **R1-Zero**: AIME 2024 を初期 15.6% → 77.9%（cons@16 で 86.7%）。MATH-500 95.9、CNMO 2024 **88.1**、GPQA Diamond **75.8**、LiveCodeBench 50.0、Codeforces rating 1444（80.4 percentile）、MMLU 88.8。報酬を与えるだけで reflection・verification・"wait" を多用する "aha moment"（step 8000 付近）が自発的に出現し、思考トークン長も単調に増加。
  - **R1（最終版）**: AIME 79.8、MATH-500 **97.3**、CNMO 2024 78.8、LiveCodeBench 65.9、Codeforces rating **2029**（96.3 percentile、人類の 96.3% を上回る）、SWE Verified 49.2、Aider-Polyglot 53.3、MMLU 90.8、MMLU-Pro 84.0、IF-Eval 83.3、FRAMES 82.5、AlpacaEval2.0 LC-winrate 87.6、ArenaHard 92.3。OpenAI-o1-1217 と math/code で同等（AIME 79.2 vs 79.8、Codeforces 2061 vs 2029）。AIME Pass@64 は 90.0%、majority voting で 79.8→86.7。
  - **蒸留**: R1-Distill-Qwen-32B が AIME 72.6 / MATH 94.3 / GPQA 62.1 / LiveCodeBench 57.2 / Codeforces 1691。R1-Distill-Llama-70B は AIME 70.0 / MATH **94.5** / GPQA **65.2** / LCB **57.5** / Codeforces 1633。Qwen2.5-32B-Base に同様の RL（10k step 超）をかけた "Qwen2.5-32B-Zero" は AIME 47.0 にとどまり、Distill-Qwen-32B（72.6）に大きく劣る → **小モデルは蒸留が圧倒的に有利**。Qwen2-Math-7B-Zero（o1 リリース前の base）でも AIME 22.3 → reasoning が漏出していない base からでも RL で創発する。
  - **計算コスト**: H800 64×8 GPU で R1-Zero 198h、R1 80h、SFT データ作成 5K GPU 時間、合計 147K H800 GPU 時間 ≒ **約 \$294K**（\$2/GPU・h 換算）。
- **貢献**:
  1. SFT を完全に飛ばした純粋 outcome-based RL でも、frontier 級の reasoning が base model から創発することを実証（R1-Zero）。
  2. 読みやすさと一般タスク性能を取り戻すためのマルチステージ pipeline（cold-start SFT → RL → rejection-sampling SFT → 2nd RL）を提示し、各 Dev1〜Dev3 段階のベンチマーク内訳まで開示。
  3. GRPO + 言語一貫性報酬 + 大きい clip ratio（ε=10）等、再現に必要なハイパーをかなり具体的に公開。
  4. R1 / R1-Zero / R1-Distill-Qwen-{1.5,7,14,32}B / R1-Distill-Llama-{8,70}B を MIT で公開。
  5. 「小モデル単独 RL より、大モデルの蒸留の方が安く・強い」ことを Qwen2.5-32B-Zero vs Distill-Qwen-32B で定量比較。
  6. PRM・MCTS をいずれも試して失敗した経緯と理由（fine-grained step 定義の困難、reward hacking、トークン空間の指数爆発）を共有。

## Takeaway（自分にとっての要点）

- **「verifier さえあれば pure RL で十分」**という強いステートメントが、671B 級の base model 限定では成立する。論文自身、7B dense や 16B MoE では response 長を伸ばしても reasoning が改善せず、繰り返しに陥ったと明言（"importance of base checkpoint"）。RL は base capacity に対して非線形に立ち上がるので、自分が手元で再現するなら **モデル選定が最優先**。
- **R1-Zero は CNMO/GPQA で R1 を上回る場面がある**（CNMO 88.1 vs 78.8、GPQA 75.8 vs 71.5）。マルチステージ後処理は writing/IF-Eval/AlpacaEval を稼ぐ代わりに hard reasoning を一部削っている。reasoning 単能力だけ欲しいなら R1-Zero 派生の方が良い、という運用判断ができる。
- **Aha moment は "wait" 出現率の急増として観測可能**（step 8000 以降）。これは reasoning RL が "うまく回った" ことの軽量モニタリング指標として使えそう。
- **GRPO の利点**: value model が要らない＝メモリ・計算半減、かつ長 CoT で final reward から partial value を学ぶ困難を避けられる。代わりに group size G に分散が依存する。PPO も λ=1.0 にチューニングすれば追いつくが探索コストが追加。
- **蒸留 800K の内訳が公開**（reasoning 600K：math/code/STEM/logic、non-reasoning 200K：writing/factual/role-play）。これを SFT するだけで Qwen2.5-32B が QwQ-32B-Preview と Qwen2.5-32B-Zero 両方を圧倒。**再現実験としては「800K を入手して SFT」が最短経路**。
- **報酬設計の知見**:
  - reasoning には neural RM を使わない（reward hacking）。
  - 一般タスクの preference RM は **最後の 400 step だけ** 投入（早く入れると hacking）。
  - helpful RM は length bias を抑えるために chosen/rejected の長さを揃え、Δ>1 のみ採用（66K pair）。safety は point-wise（106K）。
- **few-shot prompting は R1 では性能が下がる**。チューニング時の常識（few-shot で底上げ）は反転する点に注意。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 671B / 37B-activated MoE と \$294K でフロンティア性能、という再現可能性の説得力（o1 ファミリーがほぼブラックボックスである現状において貴重）。
  - 失敗（PRM、MCTS、小モデル RL）まで率直に共有しており、後追い研究のコスト削減になる。
  - 蒸留 vs RL を **同じ base（Qwen2.5-32B）** で揃えて比較した点が公平。Distill 72.6 vs Zero 47.0 という差は明確。
  - Multilingual safety 50 言語、jailbreak 2,232 テンプレ、taxonomy 28 sub-category 1,120 件、in-house bench まで含めた safety 評価が想像以上に厚い。
- **弱み / 疑問**:
  - **データ汚染**: AIME/MATH/GPQA は公開後数年経つので、10-gram 一致のみの除染では言い換え事例を完全には拾えないと著者自身が認めている。AIME 2025 で 75% を出した点はそれなりの傍証だが、o1 80% との 5pt 差をどう解釈するかは曖昧。
  - **R1-Zero の "aha moment" の因果性**: "wait" 出現と精度向上は同時に起きているだけで、片方を介入したらどうなるかという ablation は無い。emergent claim としては定性的に過ぎる。
  - **clip ratio ε=10** という極端な値。論文は「low だと token が大量に truncate される」と説明するが、ε を上げると標準的 PPO の安全性保証は崩れる。安定するかどうかは reference policy を 400 step ごとに更新している運用に強く依存しているはずで、これが他環境で再現するのか不明。
  - **総コスト \$294K** が "RL 単体" のものなのか "DeepSeek-V3-Base の事前学習を除いた値" なのか文面上明示が弱い（後者と読むのが自然だが、純粋に RL の手軽さを示したいなら base 込みの total を併記して欲しい）。
  - **蒸留評価で RL を加えない**: 「効果を示すだけが目的」とは書いてあるが、R1-Distill + RL がさらに伸びる可能性は調べられていない。コミュニティに丸投げという姿勢。
  - **多 turn 対話に弱い**: 800K SFT が single-turn 中心と明記。実プロダクトでの長期セッションで reasoning が壊れる可能性が残る。
  - **HarmBench で著しく低い**（原因は歌詞など IP 系拒否漏れ）。著者は「他カテゴリでは安全」と言うが、HarmBench 全体のスコアが他モデルとどの程度離れているかは表でしか追えない。
  - 著者が認める limitations: **構造化出力・tool use なし／overthinking／非中英の language mixing／few-shot に弱い／SWE 系 RL データ不足**。Reward Hacking は writing 等で残課題と明言。
- **次に試したいこと**:
  - 800K 蒸留データを使って自分の手元の 7B〜32B モデルを SFT、AIME/LiveCodeBench でどこまで再現できるか確認。
  - "wait" 出現率を training-time のオンライン指標として記録し、reward と共に early stopping/シード選択に利用する。
  - cold-start を **量を変えた ablation**（数百〜数万）でやり直し、reasoning 喪失（R1-Zero→Dev1 で AIME 77.9→59.0 のドロップ）の起点を特定。
  - GRPO の `Reward_language` を多言語タスクで設計し直すと、limitations に挙がっている非中英の language mixing が解けるか。
  - PRM を「再ランカ専用」（RL の報酬には使わない）に押し戻した上で、R1 出力の Pass@64 → Pass@1 を縮める実験（majority voting 79.8→86.7 がそのまま向上余地）。

## Notes / Quotes

- 「we bypass the conventional supervised fine-tuning (SFT) phase before RL training. This design choice stems from our hypothesis that human-defined reasoning patterns may limit model exploration」（Introduction）。
- 「Wait, wait. Wait. That's an aha moment I can flag here」— R1-Zero の中間 checkpoint が自発的に発した reflection（Table aha_moment）。
- 「the count of the reflective words rises 5- to 7-fold compared to the start of training」（appendix §Self-Evolution）。
- "the effectiveness of reinforcement learning from base models is highly dependent on the underlying model capacity"（Discussion: importance of base checkpoint）。
- 「more training steps with the model based preference reward signal may lead to reward hacking」→ general RM は **最後の 400 step 限定**（§Training Details of the Second RL Stage）。
- R1-Zero の `Reward_rule = Reward_acc + Reward_format`、R1 の最終報酬は `Reward = Reward_reasoning + Reward_general + Reward_language`、`Reward_general = Reward_RM + Reward_format`。
- 蒸留 vs RL の結論: 「distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL ... may not even achieve the performance of distillation」（§Distillation vs RL）。
- 失敗談（§Unsuccessful Attempts）: PRM は「step 粒度の定義困難 + 自動アノテ精度不足 + reward hacking」、MCTS は「トークン空間が指数爆発、value model が育たない」。
- AIME 2025 で 75%（o1: 80%）、AMC 12 2024 で 143.7/150 → USAMO 出場ライン超え（§Generalization）。
- データ規模: RL データ Math 26K / Code 17K(+8K bug fix) / STEM 22K / Logic 15K / General 66K（Table tab:rl_data）。SFT 約 800K = reasoning 600K + non-reasoning 200K（§800K Supervised Data）。
- 既知の限界（§Conclusion, Limitation, Future Work）: structure output、tool use なし、token efficiency（overthinking）、language mixing（中英以外）、few-shot で精度低下、SWE 系で RL データ不足、writing 等の reward hacking。
- 公開モデル: R1, R1-Zero, R1-Distill-Qwen-{1.5,7,14,32}B, R1-Distill-Llama-{8,70}B（HuggingFace、§Open Weights）。MIT ライセンスである旨は本文 ChatbotArena 節で「open-source model under the MIT License」と一度だけ言及。
- (verified 2026-05-20) 蒸留 base model を Table `distill_config`（disill_config.tex）に合わせ Qwen2.5-Math-{1.5,7}B + Qwen2.5-{14,32}B に明記。1行目タイトル・abstract・introduction・experiments・appendix の数値（chateval.tex / distill_eval.tex / distill_vs_rl.tex / math_competitions.tex / 800k_stats.tex）と整合確認。venue 行の「Nature 投稿フォーマット」は TeX 中に明示なしのため「Supplementary Fig./Table への renewcommand があり掲載先は不明」に修正。

## Related Papers

- DeepSeek-V3 Technical Report（DeepSeek-AI 2024、`dsviii`）— base model と多くの pipeline 要素の供給元。
- DeepSeekMath（Shao+ 2024、`deepseekmath`）— GRPO の原典。
- Schulman+ 2017 *PPO*（`schulman2017proximal`）— GRPO の比較対象。
- Wei+ 2022 *Chain-of-Thought Prompting*（`wei2022chain`）— CoT の起点。
- Wang+ 2023 *Self-Consistency*（`wangself`）— cons@k の元手法。
- OpenAI o1 / o1-mini（`gpt4` シリーズと別系統、本論文の主要ライバル）。
- Lightman+ 2023 *Let's Verify Step by Step*（`lightman2023let`）— PRM の代表、本論文では "うまく行かなかった" 側として議論。
- Zelikman+ 2022 *STaR*（`zelikman2022star`）、Yuan+ 2023、Singh+ 2024 — self-bootstrapping reasoning 系列。
- Yao+ 2023 *Tree of Thoughts*、Zhou+ 2023 *Least-to-Most* — inference-time scaling 派生。
- Snell+ 2025 *Scaling LLM Test-time Compute* — test-time compute scaling の理論的位置づけ。
- Hinton+ 2015 / Busbridge+ 2025 — 蒸留の根拠論文。
- Hendrycks+ MMLU、Wang+ MMLU-Pro、Rein+ GPQA、Jain+ LiveCodeBench、MAA AIME 2024 — 主要評価ベンチ。
- Tinyzero / Oat-Zero / open-r1 — 本研究を受けた後続再現研究として明示引用。
