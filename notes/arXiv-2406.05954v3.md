# Aligning Large Language Models with Representation Editing: A Control Perspective

- arXiv: https://arxiv.org/abs/2406.05954
- source: ../papers/arXiv-2406.05954v3/
- authors: Lingkai Kong, Haorui Wang, Wenhao Mu, Yuanqi Du, Yuchen Zhuang, Yifei Zhou, Yue Song, Rongzhi Zhang, Kai Wang, Chao Zhang
- venue / year: NeurIPS 2024（`neurips_2024.sty` を使用。Georgia Tech / Cornell / UC Berkeley / Univ. of Trento）
- tags: [LLM, alignment, representation-editing, control-theory, test-time]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: LLM の alignment は (a) RLHF/DPO 等の fine-tuning が不安定かつ計算量が大きく、目的が変わる度に再学習が必要、(b) prompting や guided decoding 等の test-time 手法はモデル本体を変えないので原モデルの能力に縛られる、という二択になっている。さらに既存の representation engineering（Li+ 2024 の inference-time intervention 等）は生成全体に「固定ベクトル」を足すだけで autoregressive な動的性を活かしていない。
- **手法**: 提案手法 **Re-Control**。autoregressive LLM を離散時間確率力学系 \(y_t \sim \text{Softmax}(W o_t),\ h_{t+1},o_{t+1}=f_{\mathrm{LM}}(h_t, y_t)\) と見なし、状態 \(s_t=\{h_t,o_t\}\) に各ステップで制御信号 \(u_t=\{u^h_t,u^o_t\}\) を加える。reward は最終応答 \(r([\mathbf{x},\mathbf{y}])\) を pretrained reward model から取り、EOS 時のみ与える（中間 reward は 0）。「ゼロポリシー（\(u_t=0\)）」を 1-step policy iteration の初期値として固定し、Bellman 方程式 \(V(s_t)=\mathbb{E}[V(s_{t+1})]\)（or EOS で \(r\)）の MSE 損失で **value network を hidden state の上に学習**。テスト時は \(u_t \leftarrow u_t+\alpha\nabla_{s_t}V_\phi(s_t+u_t)\) を \(n\) 回反復する勾配上昇を行い、最後に LLM を forward して次トークンを出す。\(\alpha\) と \(n\) を小さく抑えることが \(\lambda\|u_t\|_2^2\) 正則化の代替（"implicit regularization"）になる。実装上は最後の層の出力 \(o_t\) だけに介入し、value network は隠れ次元 4096 の 2〜3 層 MLP。
- **結果**: HH-RLHF と Stanford SHP の test prompts で評価。`Vicuna-7B`/`Falcon-7B-Instruct`/`Llama3-8B-Instruct` の 3 ベースで、評価指標は Diversity / Coherence / Average Reward / GPT-4 win rate（300 prompts, 1〜10 点採点）/ Inference time。
  - HH-RLHF · Vicuna-7B: Base win-rate 57.6 → **Re-Control 75.6**（+Prompt 80.3）、Avg Reward 5.894 → **6.214**（+Prompt 6.267）、推論時間 0.60h → 0.85h。
  - HH-RLHF · Falcon-7B: Base 42.3 → **58.0**（+Prompt 62.6）。Avg Reward 3.439 → 3.512（CD prefix 4.397 が最大）。
  - SHP · Vicuna-7B: Base 40.3 → **58.0**（+Prompt 63.6）、Avg Reward −5.68 → −5.38（+Prompt −4.63）。
  - SHP · Llama3-8B: Base 56.3 → **71.0**（+Prompt 77.0）、Avg Reward −4.64 → −4.39（+Prompt −4.14）。
  - 強 baseline の Controlled Decoding (CD, Khanov+ 2024) と比べ Re-Control は ~20× 速い（CD 47.43h vs Re-Control 0.85h on Vicuna+HH-RLHF）。CD は batch 生成不可。
  - +Prompting 時に Re-Control は **the strongest baseline** に対し GPT-4 win-rate で {+7.6, +19.0, +12.4, +13.2}% と本文 §4.3 で記載（4 設定 = HH-RLHF×{Vicuna,Falcon}, SHP×{Vicuna,Llama3}）。TeX は "strongest baseline" としか書いておらず、CD prefix+Prompting に限定した値ではない。
  - LoRA 版 PPO / DPO とは Fig.~\ref{fig:ppo}（PDF: 6_2_3.pdf）で比較。Re-Control は LoRA fine-tuning と "競争力のある" 代替であると主張。
  - OOD `HarmfulQA`（value 関数は HH-RLHF で学習）でも Re-Control+Prompting が最良（Fig.~\ref{fig:ood}）。
  - Hyperparameter study: \(\alpha\) を大きくすると reward は上がるが過大化で coherence/diversity がほぼ 0 に落ちる reward hacking。\(n\) を増やしすぎても同様。検証セット上で「coherence+diversity+reward」の和最大の \(\alpha,n\) を選ぶ。HH-RLHF 設定では Vicuna \(\alpha=0.5,n=30\)、Falcon \(\alpha=0.2,n=200\)、SHP では Vicuna \(\alpha=1.0,n=50\)、Llama3 \(\alpha=1.0,n=30\)。
- **貢献**: (1) autoregressive LLM を離散時間確率力学系として定式化し、representation editing を最適制御問題に対応付けた、(2) Bellman で value 関数を hidden state 上に学習し、テスト時の勾配上昇で **動的な** 制御信号を得る test-time alignment 法 Re-Control を提案、(3) HH-RLHF / SHP / HarmfulQA で test-time baselines（Prompting, Static RE, CD, CD prefix）を上回り、CD より約 20 倍速く、LoRA-PPO/DPO に比肩することを実験的に示した。

## Takeaway（自分にとっての要点）

- **「representation editing は本質的に最適制御である」** という視点が一番の収穫。固定ベクトルを足す既存手法 (Li+ 2023 の ITI, Turner+ activation addition) は静的フィードフォワード制御に相当し、本論文はそれを **ステップごとの閉ループ制御**（state-dependent な \(u_t\)）に拡張している。
- **1-step policy iteration ＋ ゼロポリシー** という割り切り：完全な RL を回さず、原モデルの分布だけで value 関数を fit すれば、テスト時の勾配上昇でほぼ tractable に effect を出せる。RLHF のコストの大半を捨てつつ「方向の知識」だけを value 関数に蓄える設計。
- 価値関数本体が **2〜3 層 MLP（hidden 4096）** で済むという報告は実用上強い。base LLM は触らない・差し替えやすい。
- **implicit regularization**: KL を陽に書かず、step size \(\alpha\) と 反復回数 \(n\) を validation で選ぶことで「原 hidden state から離れすぎない」を担保。reward hacking が出始める領域がはっきり観測されている (Fig. parameters_0)。
- CD (ARGS) との比較で **「reward model 全体を毎ステップ走らせる」vs「2〜3 層 value net を走らせる」**で 20×差というのは、guided decoding 系の "高い test-time cost" を正面から突いている。
- OOD（HarmfulQA）で value 関数が崩れなかったのは、「hidden state 上の linear-ish な好ましさ方向」がドメインを跨いで生きていることの証拠と読める（ただし n=サンプル数の詳細は要確認）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 力学系 / Bellman / policy iteration の標準ツールで LLM alignment を素直に書き下していて、見通しが良い。
  - 介入を「最終層 \(o_t\) のみ」「2〜3 層 MLP」と minimal にして、それでも CD prefix（reward model を prefix scorer 化した強 baseline）に勝つのは効率面で説得力がある。
  - test-time alignment という枠に閉じず、PPO / DPO（LoRA）とも比較を入れている。
  - reward hacking が起きる領域を hyperparameter study で明示しており、誇張せずに限界を見せている。
  - OOD（HarmfulQA）で value 関数の再学習なしに効くという結果は、representation editing 系全般のセールスポイントを補強する。
- **弱み / 疑問**:
  - **value 関数の汎化のメカニズム**が経験的にしか示されていない。1-step policy iteration の理論保証は最適制御文献 (policy iteration は反復前提) からはそのまま導かれず、「初期ポリシー＝ゼロ」「value 関数学習を pretrain LLM の roll-out で行う」近似が optimal control 的に何を最適化しているか厳密には書かれていない。
  - Average Reward は **reward model の値**そのもので、reward overoptimization が出やすい指標。Falcon で CD prefix が Avg Reward 4.397 を出している一方 win rate は Re-Control に劣る点は reward hacking の典型。「Re-Control が hack していない」と判断する根拠は GPT-4 評価のみで、reward model 自体が同じ HH-RLHF で訓練されているので評価の独立性は弱い。
  - **GPT-4 を judge にした win rate** は preference reward と相関するため、reward model に寄せた手法が有利になりやすい構造的バイアスがある。人手評価は実施されていない。
  - hyperparameter \(\alpha,n\) は「validation set 上で 3 指標の和」を最大化するヒューリスティクスで決めており、現場では各タスクで再チューニングが要りそう（Vicuna と Falcon で \(n\) が 30 vs 200 と桁違いに違う点も気になる）。
  - 介入は **最後の層の \(o_t\) だけ**で、appendix の limitations にも (1) として「複数層 / 低ランク部分空間への一般化」を future work として明記している。
  - **多目的（helpful × harmless × honest 等）への拡張**は出来ていない。limitations (2) でも認めている。
  - 1 prompt あたり \(M=1\) のサンプルだけで value 関数を学習している（appendix）。Bellman 学習としては薄く、長系列での誤差伝播が不安。
  - PPO/DPO 比較は LoRA 設定であり「full fine-tune vs Re-Control」を見られていない。
  - 異種モデル混合や reward model 差し替えに対するロバストネス、value 関数の learning curve / sample efficiency 等は本文には無い。
- **次に試したいこと**:
  - value 関数を **最後の層 1 つではなく多層**に張って、どの層を介入対象にすると最も効くか（ITI のように）の ablation。
  - 介入を **低ランク部分空間**（Wei+ 2024 / Wu+ ReFT 流）に制約して、ノイズ方向への過剰更新を抑制。
  - reward model を意図的に GPT-4 評価から独立にした上で、人手評価を含む再評価。reward hacking がどこから始まるかをタスク横断で見たい。
  - 多目的化：Pareto 方向に勾配を分解する Re-Control（safety と helpfulness の片方が落ちない \(\alpha\) を validation で同時最適化）。
  - value 関数を **オンラインに更新**（生成中に得られる中間信号で TD ターゲットを更新）した場合、reward hacking 領域に入る前に自動で step size を絞れるか。
  - long-horizon 生成（128 トークン制限を 1024 などへ）で誤差伝播がどこから破綻するか。

## Notes / Quotes

- "Our goal is not to update the state to the global optimum but to control the state to achieve a better value score while remaining close to the original state."（method 図キャプション）
- "This update already incorporates the regularization effect. The regularization is achieved by using a small step size \(\alpha\) and a limited number of updates \(n\)."（method §2.4, Implicit Regularization）
- "controlled decoding is 20 times slower than \ours" / "CD ... lacks support for batch generation"（exp.tex §4.3）
- 学習: Adam, lr \(10^{-4}\), batch 512, fp16, 100 epoch（HH-RLHF / SHP 共通）。Vicuna 用 value net は 3 層、Falcon / Vicuna(SHP) / Llama3(SHP) は 2 層、いずれも hidden 4096（appendix Table tab:training, tab:shptraining）。
- 推論 hyperparam: HH-RLHF で Vicuna \(\alpha=0.5,n=30\)、Falcon \(\alpha=0.2,n=200\)；SHP で Vicuna \(\alpha=1.0,n=50\)、Llama3 \(\alpha=1.0,n=30\)（appendix Table tab:inference, tab:shpinference）。
- 計算機: NVIDIA A100 80GB, CUDA 12.4, PyTorch 2.2.2, Python 3.12.2（appendix.tex §C.1）。
- 著者自身が明示する limitations（appendix §A）: ①最終層のみ介入で複数層 / 低ランク化が未着手、②単一 reward の単目的最適化、③value 関数の学習は単純 1 反復 policy iteration で収束保証なし。
- Broader Impacts: 「value 関数の学習に "negative goals" を埋め込めば誤用可能」と明記（appendix §B）。
- (verified 2026-05-20) §4.3 の「+Prompting 時 {7.6, 19.0, 12.4, 13.2}%」差分の比較対象を "CD prefix+Prompting" → "the strongest baseline"（TeX 原文 exp.tex L140）に修正。TeX は具体的なベースライン名を明示していない。
- (verified 2026-05-20) 同 {7.6, 19.0, 12.4, 13.2}% は table/maintable.tex の数値から逆算すると **相対改善率** (Ours+Prompting / best non-Ours baseline − 1)。順序は HH-RLHF·Vicuna→Falcon→SHP·Vicuna→Llama3 で TeX と整合。
- (verified 2026-05-20) Related Papers の "Li+ 2024" を "Li+ 2023" に修正 (main.bbl line 175 で "Advances in Neural Information Processing Systems, 36, 2023" と記載。bibkey の "2024" は誤誘導)。
- (verified 2026-05-20) "Fig. 6_2_3" を "Fig.~\ref{fig:ppo}（PDF: 6_2_3.pdf）" に修正 (exp.tex L214-221 で label は fig:ppo, 画像ファイルが 6_2_3.pdf)。

## Related Papers

- Li+ 2023, *Inference-Time Intervention*（NeurIPS 2023, vol. 36） — 静的な hidden state 編集の代表例。本論文の Static RE baseline。
- Zou+ 2023, *Representation Engineering* — 上位概念。論文中 Related Works で位置づけ。
- Khanov+ 2024, *ARGS / Alignment as Reward-Guided Search* — Controlled Decoding (CD) baseline。
- Mudgal+ 2023, *Controlled Decoding* — CD prefix baseline（prefix scorer を学習）。
- Rafailov+ 2023, *DPO* / Schulman+ 2017, *PPO* — fine-tuning 比較対象。
- Soatto+ 2023, Bhargava+ 2023, Luo+ 2023 — 「LLM を制御理論で見る」最近の系譜。
- Wu+ 2024, *ReFT* / Geiger+ 2024 / Wei+ 2024 — low-rank 部分空間での介入。future work で参照されている。
- Bai+ 2022 *HH-RLHF*, Ethayarajh+ 2022 *Stanford SHP*, Bhardwaj+ 2023 *HarmfulQA* — 使用ベンチマーク。
