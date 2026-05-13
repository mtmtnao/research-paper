# Continuous Thought Machines

- arXiv: https://arxiv.org/abs/2505.05522
- source: ../papers/arXiv-2505.05522v4/
- authors: Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, Llion Jones (Sakana AI; 共著者所属に University of Tsukuba と IT University of Copenhagen)
- venue / year: NeurIPS 2025 投稿版（`\usepackage[final]{neurips_2025}`、preprint）
- tags: [neural-dynamics, recurrence, biologically-inspired, adaptive-compute, interpretability]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 現代の NN は計算効率のためにニューロン個体の時間的振る舞いを抽象化して捨ててしまっている。しかし生物脳ではニューロン活動のタイミングと同期が情報処理の本質であり、この欠如が「flexibility / generalization / common sense」のギャップを生んでいる、と著者は主張する。スナップショット表現 $\mathbf{z}^t$ をそのまま下流に流すと、neural dynamics の自由度が下流タスクに直結して制限される。
- **手法**: **Continuous Thought Machine (CTM)** を提案。データ系列とは切り離した「**内部時刻 (internal tick)** $t\in\{1,\dots,T\}$」上で繰り返し処理する。2 つの中核要素:
  1. **Neuron-Level Models (NLM)**: 各ニューロン $d$ が固有重みの depth-1 MLP $g_{\theta_d}$ を持ち、自分自身の **長さ $M$ の pre-activation 履歴** $\mathbf{A}_d^t$ を入力に post-activation $\mathbf{z}_d^{t+1}$ を生成する。
  2. **Neural Synchronization as Representation**: post-activation 履歴の内積 $\mathbf{S}^t = \mathbf{Z}^t (\mathbf{Z}^t)^\intercal \in \mathbb{R}^{D\times D}$ を「同期行列」として、そこからサンプリングした $\mathbf{S}^t_\text{out}$ / $\mathbf{S}^t_\text{action}$ を線形射影し、出力ロジット $\mathbf{y}^t$ と cross-attention のクエリ $\mathbf{q}^t$ にする。同期行列の各 $(i,j)$ ペアには学習可能な指数減衰 $r_{ij}\ge 0$ を持たせ、時間スケールごとに重み付けする。
  3. シナプス（neuron 間情報共有）は U-Net 型 MLP（深さ $k$、ボトルネック幅 16）。
  4. 損失は内部 tick ごとに CE 損失と確信度（1 − 正規化エントロピー）を出し、$t_1=\arg\min \mathcal{L}^t$ と $t_2=\arg\max \mathcal{C}^t$ の 2 点で $L=(\mathcal{L}^{t_1}+\mathcal{L}^{t_2})/2$。これにより停止モジュール無しで native adaptive compute が出る。
- **結果**:
  - **2D maze ($39\times 39$、最大 100 ステップ、位置埋め込みなし)**: CTM ($D{=}2048$, $T{=}75$, $M{=}25$, 32M params) が LSTM (1/2/3 層、$T{=}50$ または $75$、42M〜109M params) と FF baseline (54.8M params) を上回り、訓練ホライズン超え (100 step 以上)・$99\times 99$ への汎化（学習方策の逐次再適用）も成功。
  - **ImageNet-1K**: ResNet-152 backbone + $D{=}4096$, $T{=}50$, $M{=}25$ の CTM で uncropped **top-1 72.47% / top-5 89.89%**（SOTA ねらいではない）。0.8 確信度しきい値で大半のサンプルは 50 tick 中 10 tick 未満で停止可能。calibration plot は強くキャリブレートされる。
  - **64 長累積 parity**: CTM 75/100 ticks の一部 seed で perfect、LSTM (parameter matched) は不安定。tick 数を増やすと正解できる位置が左→右に伸びる学習動態。例えば 100 tick 版は前から後ろへ attention を走らせる「scan」戦略、75 tick 版は逆向きの「reverse search ≒ planning」戦略を学ぶ（seed 依存）。
  - **CIFAR-10 vs Human (CIFAR-10D/CIFAR-10H)**: CTM は FF/LSTM より calibration が良く、人間より calibrate されている。LSTM は人間と同じ under-confidence。難易度に対する応答が人間に近い。
  - **CIFAR-100 ablation**: 幅 $D$ を増やすほど neuron 同士の cosine 類似度がゼロ寄りになり「多様な dynamics」が出る。tick 数を増やすと「初期と後期」の 2 山の certainty 分布が出現。
  - **Maze ablation ($15\times 15$, 約 9M params, 100k iters)**: 標準 CTM が **solve rate 65.9% / accuracy 94.6%**、NLM なし 35.0%/82.9%、synchronization なし 37.5%/85.1%、LSTM+synchronization 33.8%/82.4%。NLM と synchronization の組合せが鍵。
  - **Sorting 30 reals (CTC loss)**: 出力位置ごとの待ち tick 数が、隣り合う出力値の差 (`data delta`) と相関。訓練分布外 ($\mathcal{N}(0,\sigma^2 I)$ で $\sigma$ を変える) にも汎化。
  - **Q&A MNIST**（数字を順に観測 → index/operator embedding → 答え; modular 加減算）: CTM 10-tick が最難設定 (4 digit, 4 operator) で >96%、同パラメータ LSTM 10-tick は ≤21%。観測した数字が memory window $M$ の外に出ても synchronization が記憶として働き、訓練より長い演算列にも汎化。
  - **RL (PPO, partially observable)**: CartPole / Acrobot / MiniGrid Four Rooms で LSTM baseline と同等性能。CTM のほうがリッチで多様な neuron dynamics を示す。
- **貢献**: (1) NLM と neural synchronization 表現という 2 つの新要素を持つ CTM アーキテクチャ、(2) 内部時間軸と確信度 2 点損失による native adaptive compute、(3) 「位置埋め込み無しで maze を解く / ImageNet を `look around` する / parity を逐次アルゴリズムとして解く / Q&A MNIST で synchronization を記憶として使う」など多様な創発挙動の提示、(4) 同期行列の指数減衰付き計算を $\mathcal{O}(D_\text{sub})$/tick で行う再帰式（Appendix）。

## Takeaway（自分にとっての要点）

- **「ニューロンごとに私的な MLP $g_{\theta_d}$ を持たせる」**ことで activation function を「履歴を見るネットワーク」に拡張している点が新しい。Universal Approximator として fixed nonlinearity を学習可能 MLP に置き換える発想に近いが、入力が「自分自身の過去 $M$ tick」というのが核心。これは spike-timing-dependent な処理を微分可能・GPU フレンドリに翻訳した版と読める。
- **「latent = $\mathbf{z}^t$」をやめて「latent = $\mathbf{Z}^t (\mathbf{Z}^t)^\intercal$ の部分サンプル」にする**という設計変更は単独で意味がある。スナップショット表現を使うとそれが下流タスクの勾配で歪み、neural dynamics 自体が制約を受ける、というのが著者の動機（脚注）。Synchronization なら dynamics は自由に動ける。これは「representation を作るレイヤと dynamics を作るレイヤを分離する」という設計原則として一般化できる。
- **同期行列のサンプリング戦略 (Dense / Semi-dense / Random pairing)** がタスク依存で違うのは重要。maze は Dense、ImageNet は Random + $n_\text{self}=32$（snapshot 復元の余地）、parity と Q&A MNIST は Semi-dense。$D\times(D+1)/2$ ペアを使い切る必要はないが、選択方式がボトルネックの強さを決める。
- **損失 $L = (\mathcal{L}^{t_1}+\mathcal{L}^{t_2})/2$ で停止モジュールなしに adaptive compute が出る**のは美しい。PonderNet / ACT の halting module は要らないという主張は実装簡素化の意味で価値あり。
- **同期行列の指数減衰付き計算を一次再帰で $\mathcal{O}(D_\text{sub}/\text{tick})$ に落とせる**（Appendix \ref{sec:appendix-recursion}）。これがなければ $\mathcal{O}(D^2 t)$ で爆発するので実装上の要。
- **Q&A MNIST で「観測した数字が memory window $M$ の外に出ても recall できる」**結果は、synchronization 表現が NLM の有限 FIFO を超えて記憶を保持できるという証拠。「neuron dynamics の同期パターンに記憶を埋め込む」というのは将来の memory-augmented model の方向性として面白い。
- **Maze ablation の数字 (65.9 vs 35〜37%)** が一番説得力ある。NLM・synchronization・LSTM+synch のいずれを欠いても solve rate が半減するので、「NLM と synch の両方が必要」という主張が定量的に立っている。
- **ImageNet の calibration が ResNet-152 を ResNet-152 として使うより素直に良くなる**のはオマケとして地味に大きい。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「neural dynamics をなぜ・どう使うのか」という問いを、再帰 + neuron 私的 MLP + 同期行列という具体的かつ訓練可能な形に落としきっている。微分可能で gradient-based 学習に乗るので、SNN や Liquid Time-Constant Network に比べて実装・スケールしやすい。
  - 1 つのアーキテクチャでハイパラだけ変えて maze / ImageNet / parity / CIFAR / Q&A / sort / RL を一通り回している breadth は誠実。各タスクで「LSTM/FF baseline と parameter / tick を揃える」配慮もしている。
  - Maze ablation (Tab. \ref{tab:maze_ablation}) が「NLM だけ抜く」「synchronization だけ抜く」「LSTM に synchronization を載せる」の 3 通りを試していて、新規 2 要素が独立に効くというより**組み合わせで効く**ことを示している。これは強い証拠。
  - Parity で「100 tick は scan / 75 tick は reverse search」のような seed 依存戦略が emerge する観察は interpretability の素材として価値が高い。
- **弱み / 疑問**:
  - 著者自身が limitations で認めている通り、**ImageNet 72.47% top-1 は ResNet-152 単体 (≈78%) より低い**。「SOTA ねらいではない」と明言しているが、「neural dynamics を入れるとなぜか劣る」のか「ハイパラ未探索」なのかが分離できておらず、計算オーバーヘッド ($T=50$ tick 分) に見合うかの判断材料が足りない。
  - **Parity の seed 分散が大きい**: 75 tick / memory 25 の 3 seed のうち 1 つは suboptimal に収束（Fig. \ref{fig:appendix/parity/training}）。「algorithm を学ぶ」と言うわりに学習の安定性が seed gacha なのは弱い。
  - **比較対象が LSTM / FF にほぼ限定**。Universal Transformer / PonderNet / ACT / Looped Transformer / Sparse Universal Transformer / Quiet-STaR との同条件比較は Related Work で参照されるだけで実験はない。adaptive compute と sequential reasoning という同じ土俵の手法と並べないと「CTM の優位」を切り分けにくい。
  - **同期行列 $D\times D$ のサンプリング戦略がタスクごとに違う**（Dense / Semi-dense / Random）が、なぜそうなのかの原理的説明はない。tuning artifact である可能性。
  - **NLM のパラメータ数増加**を著者は limitation として認めるが、parameter-matched 比較は「同期重ねの LSTM」など追加要素を入れた形でしかしていない。「同じパラメータ予算を素直に大きい Transformer に振った場合」との pareto は不明。
  - **2 tick 損失 ($t_1$, $t_2$)** が学習を駆動するが、$t_1=\arg\min\mathcal{L}$ は **真ラベルを使う argmin** なので推論時には存在しない。学習と推論の停止規準が一致しない設計の良し悪しは議論が薄い。
  - **「neural synchronization が記憶」**という主張は Q&A MNIST で示されているが、synch なし baseline (post-activation を直接使う) との比較で同じ recall ができないことを示していないので、本当に synchronization 由来かは未確定。
  - 「**TeX 中には明示されていない**」: 同期行列の low-rank 構造（実は $\mathbf{Z}^t$ が $D\times t$ なので rank は $\min(D,t)$ 以下）が表現能力にどう効くかという議論はなく、また同期行列を使った大きな実装上の数値安定化（diag 抜き、正規化）の細部は appendix を超えて触れられていない。
- **次に試したいこと**:
  - 同じ計算予算で **Universal Transformer / Looped Transformer + early-exit** と CTM を maze / parity / Q&A MNIST で並べた pareto。「neural dynamics 表現」ではなく「単に深く回したから解けた」可能性を切る。
  - **NLM を学習可能 activation function として既存 Transformer に差し込む** ablation。「neuron 私的 MLP の効果」だけを Transformer ハーネスで測れば、CTM 全体ではなく要素ごとの寄与が分かる。
  - **同期行列の指数減衰 $r_{ij}$ の学習結果を可視化**して、タスクごとに有効な時間スケールがどう分布するか（maze は使う / ImageNet はほぼ使わないという脚注の数値化）。multi-scale 表現としての本気度が見える。
  - **adaptive compute の `over-thinking`**（ImageNet 21202 の例で正解 → 不正解と漂流する現象、Fig. \ref{fig:imagenet-21202}）を停止規準で抑える手法。$\arg\max \mathcal{C}^t$ までで止めるのが安全だが、その後も計算が動いて確信度が下がる挙動は実用上ノイズ。
  - **言語モデリングへの応用**（著者 future work）。tick を「思考ステップ」として推論時 scaling として使えるか、特に Q&A MNIST の「観測ウィンドウ外を recall」が言語の長文脈に効くかは興味深い。

## Notes / Quotes

- "Most artificial neural networks ignore the complexity of individual neurons" (abstract)
- 設計動機: "We found that `snapshot' representations were too constraining: projecting from $\mathbf{z}^t$ strongly ties it to the downstream task and thereby limits the types of dynamics it can produce, whereas synchronization decouples it." (脚注、§Synchronization)
- $L = (\mathcal{L}^{t_1}+\mathcal{L}^{t_2})/2$ で「停止モジュール無しの native adaptive compute」($\mathsection$\ref{sec:loss})
- 同期行列の再帰計算: $\alpha^{t}_{ij}, \beta^{t}_{ij}$ の一次再帰で $\mathcal{O}(D_\text{sub})$/tick (Appendix \ref{sec:appendix-recursion})
- Maze ablation 数字: CTM 65.9% solve / 94.6% acc, no NLMs 35.0/82.9, no synch 37.5/85.1, LSTM+synch 33.8/82.4 (Tab. \ref{tab:maze_ablation})
- ImageNet ResNet-152 + CTM ($D=4096$, $T=50$): top-1 72.47% / top-5 89.89%
- 「neuron $d$ ごとの私的 MLP $g_{\theta_d}$」「同期 = $\mathbf{Z}^t (\mathbf{Z}^t)^\intercal$ の sub-sampling」が 2 つの核
- limitations（著者明記）: 内部 sequence による訓練時間増、NLM によるパラメータ増、breadth 優先で SOTA 比較の depth が浅い
- parity: seed 分散が大きく、3 seed 中 1 seed は suboptimal 収束
- ImageNet `over-thinking` 例 (Fig. \ref{fig:imagenet-21202}): 正解を通過してから不正解へ漂流

## Related Papers

- Schwarzschild+ 2021 "Can You Learn an Algorithm?" — iterative algorithmic learner、maze の比較対象。
- Graves 2016 "Adaptive Computation Time" — halting module 付き adaptive compute、parity / sort タスクの起源。
- Banino+ 2021 "PonderNet" — halting で adaptive compute する代表例。
- Hasani+ 2021 "Liquid Time-Constant Networks" — biologically-inspired dynamics、近縁。
- Reichert & Serre 2013 — 複素ニューロンによる同期、本論文が「post-hoc synchrony ではなく学習表現として synchrony を使う」として明示的に区別する位置付け対象。
- Maass 2011 "Liquid State Machines"、Zenke+ 2018 "SuperSpike" など SNN 系 — biological 同期の先行研究。
- Mnih+ 2014 "Recurrent Models of Visual Attention (RAM)" — sequential glimpses による視覚処理の先行。
- Zelikman+ 2024 "Quiet-STaR"、Goyal+ 2019 "Recurrent Independent Mechanisms" — iterative/modular reasoning。
- Ha & Schmidhuber 2018 "World Models"、Gornet & Thomson 2024 "Automated Construction of Cognitive Maps" — maze タスクで参照される世界モデル / 認知地図文献。
- Peterson+ 2019 CIFAR-10H、Ho-Phuoc 2018 CIFAR-10D — 人間ベースライン。
- Manhaeve+ 2018 "DeepProbLog"、Schlag+ 2021 — Q&A MNIST の発想源。
- Schulman+ 2017 PPO、Huang+ 2022 CleanRL — RL 実装。
- Muller+ 2018、Miyato+ 2024、Jacobs+ 2025 — traveling waves / Kuramoto oscillator など創発波の関連。
