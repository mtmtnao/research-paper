# World Models

- arXiv: https://arxiv.org/abs/1803.10122
- source: ../papers/arXiv-1803.10122v4/
- authors: David Ha, Jürgen Schmidhuber
- venue / year: arXiv 2018（TeX は ICML 2018 accepted テンプレ。インタラクティブ版 https://worldmodels.github.io ）
- tags: [model-based-RL, world-model, VAE, MDN-RNN, evolution-strategies, CMA-ES, dream]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: model-free RL は credit assignment の都合で、多数の文献では少数パラメータの小さなネットが使われがちだと著者は述べる。一方で本論文は、1990--2015 年の RNN ベース world model + controller 系研究（Schmidhuber らの \textit{Making the World Differentiable} / \textit{Learning to Think} 等）の key concepts を実験的に示す簡略フレームワークを提示する。pixel 観測だけで「環境を生成的にシミュレートする内部モデル」を学習し、それを使って小さな controller を訓練できるかが課題。
- **手法**: agent を3要素 V/M/C に分割する。
  - **V (Vision)**: ConvVAE。64×64×3 frame を潜在 $z$ に圧縮（CarRacing は $z\in\mathbb{R}^{32}$、Doom は $z\in\mathbb{R}^{64}$）。
  - **M (Memory)**: MDN-RNN。LSTM(隠れ256/512) + 5 component の Mixture Density 出力で $P(z_{t+1}\mid a_t,z_t,h_t)$ を予測。サンプリング時に温度 $\tau$ で不確実性を制御。Doom では $done$ も同時予測。
  - **C (Controller)**: 線形モデル。CarRacing は $a_t = W_c[z_t\,h_t]+b_c$（867 params）、Doom は $a_t = W_c[z_t\,h_t]$ で、入力の hidden state は LSTM の cell vector $c$ と output vector $h$（1,088 params）。CMA-ES で最適化（population 64、各 agent 16 rollout 平均を fitness）。
  - V/M は random policy で集めた 10,000 rollout の観測・行動から reward なしで学習し、その後で C を CMA-ES で訓練する分離パイプライン。
  - VizDoom 実験では M を Gym Env でラップして「dream（DoomRNN）」を作り、actual 環境を一切使わずに dream の中だけで C を訓練 → 実環境へ transfer する。
- **結果**:
  - **CarRacing-v0**（解とみなす閾値 = 100 試行平均 900）: Full World Model **906 ± 21**（100 試行）/ 900.46（1024 試行）で初の "solve"。V のみ 632 ± 251、V+hidden 788 ± 141。比較: DQN 343 ± 18、A3C(continuous) 591 ± 45、A3C(discrete) 652 ± 10、ceobillionaire(leaderboard) 838 ± 11（Table: CarRacing）。
  - **VizDoom Take Cover**（解 = 750 ts）: dream で訓練した policy が actual で **1092 ± 556**（τ=1.15、100 trial）。ランダム 210 ± 108、Gym Leader 820 ± 58。Virtual/Actual の温度応答表は τ=0.10 のとき Virtual 2086 / Actual 193、τ=1.0 で 1145/868、τ=1.15 で 918/**1092**、τ=1.30 で 732/753（Take Cover 温度表）。
  - 低 $\tau$ では mode collapse で「fireball が一切撃たれない dream」が出来てしまい、virtual 2086 → actual 193 と完全に転移失敗。$\tau$ を上げて dream を「より不確実で難しく」する方が transfer が改善するという非自明な発見。
  - V と M の各モデル訓練は単一 GPU で 1 時間未満。CMA-ES は population 64、各 agent 16 rollout で fitness を評価し、CarRacing では 1800 generations 後に 1024 rollout 平均 900.46。
- **貢献**: (1) pixel 観測のみから V(VAE)+M(MDN-RNN)+C(線形) の単純構成で CarRacing-v0 を史上初に solve、(2) MDN-RNN が生成した「dream」内だけで policy を学んで実環境へ transfer できることを VizDoom Take Cover で実証、(3) 温度 $\tau$ を「world model の exploitability ↔ 学習しやすさ」のハイパラとして提示、(4) 1990 年代の C–M (controller–model) 系の枠組みを現代 deep learning (VAE+LSTM+CMA-ES) で再構成し、iterative training の道筋を示した。

## Takeaway（自分にとっての要点）

- **「巨大な world model + 極小 controller」の分業**が思想の核。credit assignment は小さな線形 C の探索空間に寄せ、表現の重い仕事は reward なしで学習する V/M が引き受ける。
- **MDN（混合）にして確率分布を出す**点が深い。決定的モデルだと C が「世界の隙間」を突く adversarial policy（VizDoom で fireball を魔法のように消す挙動）を学びやすい、という問題意識が "Cheating the World Model" 節で明示されている。MDN-RNN と温度 $\tau$ は、realism と exploitability の tradeoff を調整する手段として使われている。
- **dream で学んで実環境へ転移**は、この実験では "dream が actual より難しい" 設定（τ=1.15）で actual score が最大になっている点が面白い。低温度（τ=0.1）では virtual score は高いが actual へ転移しない。
- **C を線形にしたから CMA-ES が回せた**のであって、これは backprop 不要・reward 履歴不要・並列化容易、を引き出す設計判断として効いている。RNN world model と evolution-based controller を小さな実験系に落としている。
- **iterative training** はまだ Take Cover/CarRacing では試していない（random policy 1 周のみ）と明記。世界が複雑になると world model のために exploration が要る、という素直な認識。intrinsic motivation や PowerPlay 系の話とつながる。
- **歴史的位置づけ**として、Schmidhuber が 1990 年代から書いてきた C–M スキーム（\textit{Making the World Differentiable}, \textit{Learning to Think}）と Ha のエンジニアリングを接続するメタな論文でもある。引用と本文の構造を見るとそう読める。

## Critical Thoughts（評価・疑問）

- **強み**:
  - pixel 観測から CarRacing-v0 の 900 を超えた初の報告（DQN/A3C は 343–652 で頭打ち）で、controller が 867 パラメータの線形モデルだけ、という結果のインパクトは大きい。
  - dream-only 学習で実環境 transfer が成立することを、virtual/actual の温度応答テーブルできれいに見せている。τ=0.10 → virtual 2086 / actual 193 という対比は world-model exploit が起きる証拠として教科書的。
  - V / M / C の 3 ブロックを完全に分離訓練できる、という再現性の高さ。1 GPU 1 時間 + CPU 並列で再現可能と明記しているのは強い。
  - "Cheating the World Model" 節で**自前モデルの脆弱性を率直に開示**しており（adversarial fireball-extinguishing policy、mode collapse）、limitation の扱いが誠実。
- **弱み / 疑問**:
  - **タスクは 2 つで、著者自身も relatively simple と位置づけている**。CarRacing はランダム生成トラック、Take Cover は左右移動/停止の行動空間。iterative training の実証はなく「複雑タスクで本当にスケールするか」は open（著者も明記）。
  - **random policy で集めた 10,000 rollout で V/M を訓練**しているため、より難しい環境では controller が探索して新しい観測を集め、world model を改善する iterative training が必要になる、と著者は述べている。
  - **C が線形（CarRacing 867 params）**であることが効くタスク選定になっており、「世界モデルが優秀だから controller が小さくて済んだ」のか「タスクが線形 controller に向いていたから済んだ」のか切り分けが弱い（評者補足）。V-only 632、V+hidden 788、Full 906 の比較はあるが、$h_t$ を壊すような追加 ablation は TeX 中には示されていない。
  - **VizDoom は actual 環境では一切 C を訓練していない**（DoomRNN だけで訓練）と明記。これは強みでもあるが、「実環境で fine-tuning した場合との比較」がないので transfer gap の本当の大きさが見えない。
  - **CMA-ES の計算コスト**（CarRacing では 1800 generations × pop 64 × 16 rollouts）は controller が小さいから扱いやすい。controller を大きくした場合の最適化比較は TeX 中には示されていない（評者補足）。
  - **著者自身が認めている限界**:
    - VAE が unsupervised なのでタスク無関係な特徴を圧縮してしまう（Doom の壁タイル詳細を再構成して、CarRacing の道路タイルを落とす、という具体例）。reward 信号で fine-tune すれば改善するが、その VAE は別タスクで再利用しづらくなる。
    - LSTM ベースの M は容量が限られ catastrophic forgetting が起きるので、iterative training のスケールには限界がある。
    - step-by-step planning であり、人間的な hierarchical / abstract reasoning には届かない（\textit{Learning to Think} 枠組みなら可能と示唆）。
    - mode collapse（低 τ で fireball が出ない dream になる）と adversarial exploit は本質的問題として残る。
- **次に試したいこと**:
  - sparse reward 環境で iterative training + curiosity（M の loss flip）を実装し、何 iteration で world model が破綻なく拡張できるか測る（評者補足）。
  - C を線形のまま固定し、V のみ / V+hidden / V+M を追加 ablation して、906 − 788 = 118 点の差分のどれだけが「未来予測」由来か計測する（評者補足）。
  - dream 内 transfer の「τ を上げると actual が改善する」現象が CarRacing で再現するか確認する（論文では Take Cover で温度応答表を出している）。
  - MDN-RNN をより高容量な系列モデルや外部メモリに置き換えて catastrophic forgetting が緩むかを iterative training の文脈で評価（著者の future work に基づく評者補足）。
  - C を非線形にしたとき CMA-ES でなく ES/PPO のどれがよいか比較する（評者補足。本論文は C の最適化に CMA-ES を使用）。

## Notes / Quotes

- "Because our world model is only an approximate probabilistic model of the environment, it will occasionally generate trajectories that do not follow the laws governing the actual environment." (4.5 Cheating the World Model)
- "We find agents that perform well in higher temperature settings generally perform better in the normal setting. In fact, increasing $\tau$ helps prevent our controller from taking advantage of the imperfections of our world model." (VizDoom 4.3)
- "$\tau=0.1$ → the monsters inside this dream environment fail to shoot fireballs ... due to mode collapse. ... Whatever policy learned inside of this dream will achieve a perfect score of 2100 most of the time, but will obviously fail when unleashed into the harsh reality of the actual world, underperforming even a random policy." — mode collapse の具体例。
- パラメータ内訳（Table）: CarRacing は VAE 4,348,547 / MDN-RNN 422,368 / **Controller 867**、Doom は VAE 4,446,915 / MDN-RNN 1,678,785 / **Controller 1,088**。
- CarRacing スコア表: Full World Model **906 ± 21** vs DQN 343±18, A3C(cont) 591±45, A3C(disc) 652±10, leaderboard 838±11, V-only 632±251, V+hidden 788±141。
- Take Cover 温度表: τ=0.10/0.50/1.00/1.15/1.30 で Actual = 193/196/868/**1092**/753、Random Policy 210±108、Gym Leader 820±58。閾値 750。
- 訓練設定: V/M は random policy 10,000 rollout、1 GPU 1 時間未満、MDN-RNN は 20 epochs、5 Gaussian mixtures。CMA-ES pop=64、各 agent 16 rollouts、CarRacing は 1800 generations で 900.46/1024。
- 既知の限界（Discussion）: VAE が task-irrelevant な特徴を学ぶ、LSTM の容量限界・catastrophic forgetting、step-by-step planning に留まり階層化していない。
- TeX に明示されていない事項: \textit{Learning to Think} (Schmidhuber 2015) で示唆されている「C が M の subroutine を呼び出す」スキームは本論文では実装していない（"Experiments with those more general approaches are left for future work."）。
- (verified 2026-05-26) Doom の Controller は bias なしの $a_t = W_c[z_t\,h_t]$、かつ hidden state に LSTM の cell/output vector を使うよう修正 (main.tex, Procedure / Appendix Controller)。
- (verified 2026-05-26) 1800 generations と 1024 rollout 評価は CarRacing の記述に限定し、Doom へ一般化しない形に修正 (main.tex, Appendix Evolution Strategies / DoomRNN)。
- (verified 2026-05-26) Critical Thoughts の外挿（難環境、追加 ablation、非線形 C、系列モデル置換）を TeX 事実と分けるため「評者補足」を明記または TeX 根拠のある表現に縮小 (main.tex, Iterative Training Procedure / Discussion)。
- (verified 2026-05-26) 問題設定の「model-free RL は小さなネットしか使えない」「知見が散逸」を、TeX の "often only use small neural networks" / "distill several key concepts" に合わせて弱めた (main.tex, Introduction)。
- (verified 2026-05-20) Cheating the World Model 引用文を TeX 通り "Because our world model..." に修正、section ラベルを 4.5 に明示 (main.tex L540, section "Cheating the World Model")。
- (verified 2026-05-20) Kingma & Welling の年を (2014) → (2013) に修正 (main.bbl L383: \bibitem[Kingma \& Welling(2013)])。
- (verified 2026-05-20) Related Papers の Schmidhuber 1990 系列の citation key を s05_cm → s05_making_the_world_differentiable に修正 (main.bbl L615: \bibitem[Schmidhuber(1990{\natexlab{a}})]{s05_making_the_world_differentiable})。

## Related Papers

- Schmidhuber, \textit{Making the World Differentiable} (1990) ほか s05_making_the_world_differentiable / s05a_cm / s05b_rl — 本論文の思想的源流である C–M スキーム。
- Schmidhuber, \textit{On Learning to Think} (2015, arXiv:1511.09249) — 用語と枠組みを直接借りている。
- Kingma & Welling, VAE (2013, main.bbl 表記) — V モデルの直接の祖。
- Bishop, Mixture Density Networks (1994) / Graves, \textit{Generating Sequences with RNNs} (2013) / Ha, SketchRNN — MDN-RNN の系譜。
- Hansen, CMA-ES — controller 最適化に使用。
- Deisenroth & Rasmussen, PILCO — 確率的 dynamics model + policy search の代表。GP ベースで高次元 pixel には向かない、と本論文が対比。
- Oh et al., \textit{Action-Conditional Video Prediction} / Chiappa et al., \textit{Recurrent Environment Simulators} — 似た dynamics model 研究、ただし dream-only 学習はしていない。
- Nagabandi et al. (2017) — model-based で初期化して model-free で fine-tune するハイブリッド、本論文が比較先として挙げる。
- Pathak et al. (2017) — intrinsic curiosity、iterative training の今後の方向として参照。
- Foster (2017), \textit{Replay Comes of Age} — hippocampal replay と iterative training の類比に引用。
