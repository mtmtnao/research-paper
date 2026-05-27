# World Models（VAE+MDN-RNN による世界モデルと小さな Controller の分離学習）

- arXiv: https://arxiv.org/abs/1803.10122
- 一次ソース: ../papers/arXiv-1803.10122v4/
- 正規ノート: ../notes/arXiv-1803.10122v4.md

---

## 一言で言うと

ピクセル観測から VAE と MDN-RNN で環境の圧縮表現と未来予測を学習し、その特徴を入力にする小さな線形 Controller だけを報酬で最適化する論文。CarRacing-v0 では Full World Model が 906 $\pm$ 21 を達成し、VizDoom Take Cover では MDN-RNN が作る dream 環境内だけで学習した方策が actual 環境で 1092 $\pm$ 556 を達成した、というのが著者の主な実験的主張である。

## 何を議論する論文か

- **問題設定**: 高次元の RGB 画像列を直接扱う強化学習で、環境の空間的・時間的構造を大きな world model に学習させ、報酬最大化を担当する Controller を小さく保てるかを調べる。Introduction では model-free RL が credit assignment problem によって大規模モデルの重みを学習しにくく、実用上は小さなネットワークが使われがちだと述べる。
- **対象範囲 / 仮定**: 実験対象は OpenAI Gym の `CarRacing-v0` と VizDoom の `Take Cover`。どちらも、まず random policy から 10,000 rollouts を集め、V と M は報酬なしで学習する。CarRacing では C を actual 環境で訓練し、VizDoom では C を virtual/dream 環境だけで訓練して actual 環境に転移する。
- **既存研究との差分**: 論文自身は model-based RL 全体のレビューではなく、1990--2015 年の RNN-based world models and controllers 系列の key concepts を、VAE、MDN-RNN、CMA-ES で簡略に実証することを目的にしている。Related Work では PILCO、video prediction、RNN hallucination、`Learning to Think` などと接続する。
- **この論文で答えたい問い**: 1) world model から得た $z_t$ と $h_t$ だけで、小さな線形 Controller が pixel-based continuous control を解けるか。2) learned dynamics model を actual 環境の代替にして、dream 内で訓練した policy が actual 環境に transfer するか。3) learned world model の不完全性を Controller が exploit する問題に、MDN-RNN と temperature $\tau$ がどう効くか。

## 背景と前提

- **World model**: この論文では、agent が直接見る高次元観測を圧縮し、次の潜在表現 $z$ の分布を予測する内部モデルを指す。Agent Model 節では、agent を Vision (V), Memory (M), Controller (C) の 3 要素に分ける。
- **Model-based RL と model-free RL**: model-based RL は環境の dynamics model を学び、それを policy 学習や planning に使う。著者は、model-free RL では credit assignment のため大きな RNN agent を直接学習しにくいという問題意識を置く。
- **VAE / latent vector**: V は Variational Autoencoder で、各画像フレームを潜在ベクトル $z$ に圧縮する。Appendix の ConvVAE 説明では、入力を 64x64x3 に resize し、$\mu$ と $\sigma$ を出し、$z$ を $N(\mu,\sigma I)$ から sample する。
- **MDN-RNN**: M は LSTM と Mixture Density Network の組み合わせで、次の latent vector の確率分布を出す。単一点予測ではなく mixture of Gaussian distribution を使うため、Doom の fireball が撃たれる/撃たれないような discrete random events を表しやすい、という説明が Cheating the World Model 節にある。
- **CMA-ES**: C は線形モデルなのでパラメータ数が少なく、black-box optimization である Covariance-Matrix Adaptation Evolution Strategy を使える。Appendix では population size 64、各 agent 16 random rollouts、fitness は 16 rollout の average cumulative reward と記載されている。
- **先行研究との関係**: `main.bbl` では VAE は Kingma & Welling (2013)、MDN は Bishop (1994)、LSTM は Hochreiter & Schmidhuber (1997)、CMA-ES は Hansen、C--M 系の源流は Schmidhuber の 1990 年代の研究として引かれている。

## 提案手法

### コアアイデア

モデルを「表現学習・未来予測」と「報酬最大化」に分ける。V は観測フレーム $x_t$ を小さな $z_t$ に圧縮し、M は $z_t$ と行動 $a_t$ と内部状態 $h_t$ から次の $z_{t+1}$ の分布を予測する。C は $z_t$ と $h_t$ を受け取り、行動 $a_t$ を出す小さな線形モデルである。

重要なのは、V と M は random policy で集めた観測・行動列から報酬なしで学習され、C だけが環境の cumulative reward を見て最適化される点である。CarRacing では world model が特徴抽出器として使われ、VizDoom では M に `gym.Env` インターフェースをかぶせて virtual environment とし、C をその dream 内だけで学習する。

この分離により、表現の大部分は数百万パラメータの V/M に置き、credit assignment は数百から千程度の C の探索空間に押し込める。TeX の短い根拠は、Introduction の "large world model and a small controller model" と、Controller 節の "most of our agent's complexity resides in the world model (V and M)" である。

### 重要な定義・数式

$$
z \sim N(\mu, \sigma I)
$$

**式の意味**: ConvVAE が画像を潜在ベクトルに圧縮するとき、encoder が出した $\mu$ と $\sigma$ から $z$ を sampling する、という Appendix の V model の定義である。TeX では "The latent vector $z$ is sampled from the Gaussian prior $N(\mu, \sigma I)$" と書かれている。

**記号の定義**:
- $z$ ... V model が各フレームから作る latent vector
- $\mu$ ... ConvVAE encoder が出す平均ベクトル
- $\sigma$ ... ConvVAE encoder が出すスケール側のベクトルとして TeX に出る記号
- $I$ ... 対角共分散を表す単位行列
- $N_z$ ... latent vector の次元。CarRacing では 32、Doom では 64

**この論文での役割**: 高次元の 64x64x3 RGB 入力を、M と C が扱える低次元表現に変換する。VAE の Gaussian prior は、M が生成する非現実的な $z$ に対して world model を robust にする、と Appendix で説明される。

$$
P(z_{t+1} \; | \; a_t, z_t, h_t)
$$

**式の意味**: M model が、現在の行動、現在の latent vector、RNN の hidden state から、次時刻の latent vector の確率分布を予測する式である。Agent Model と Car Racing Procedure の両方に出る。

**記号の定義**:
- $z_{t+1}$ ... 次時刻に V が出すと期待される latent vector
- $a_t$ ... 時刻 $t$ で取った行動
- $z_t$ ... 時刻 $t$ の観測を V が圧縮した latent vector
- $h_t$ ... 時刻 $t$ の RNN hidden state

**この論文での役割**: M が「未来の分布」を出すことで、$h_t$ が未来予測に関する特徴量になる。CarRacing では C に $z_t$ だけでなく $h_t$ も渡すことで 632 $\pm$ 251 から 906 $\pm$ 21 へ改善する、という主張の中心になる。

$$
a_t = W_c \; [z_t \; h_t]\; + b_c
$$

**式の意味**: Controller が、V の latent vector と M の hidden state を連結した入力から行動を線形写像で出す式である。TeX の Eq. `controller_equation` と Car Racing Procedure に出る。

**記号の定義**:
- $a_t$ ... 時刻 $t$ の action vector
- $W_c$ ... Controller の重み行列
- $[z_t \; h_t]$ ... $z_t$ と $h_t$ を連結した特徴ベクトル
- $b_c$ ... Controller の bias vector

**この論文での役割**: C を単純化し、CMA-ES で探索可能な小さな問題にする。CarRacing の Controller は 867 parameters。Doom Procedure では式が $a_t = W_c [z_t \; h_t]$ と書かれ、bias は明示されていない。

$$
P(z_{t+1}, d_{t+1} \; | \; a_t, z_t, h_t)
$$

**式の意味**: VizDoom Take Cover で M が、次の latent vector に加えて、次フレームで agent が死ぬかどうかの binary event $done_t$、略して $d_t$、も予測する式である。

**記号の定義**:
- $d_{t+1}$ ... 次時刻の done/death event
- $z_{t+1}$ ... 次時刻の latent vector
- $a_t, z_t, h_t$ ... 現在の行動、現在の latent vector、RNN hidden state

**この論文での役割**: M だけで OpenAI Gym 風の virtual environment を作るために必要な式である。done を予測できるため、DoomRNN の中で rollout を終了し、C の survival time を reward として最適化できる。

$$
P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t)
$$

**式の意味**: Iterative Training Procedure 節で、より複雑なタスクのために M が次の observation、reward、action、done を予測する形として提示される式である。

**記号の定義**:
- $x_t$ ... 時刻 $t$ の observation
- $x_{t+1}$ ... 次時刻の observation
- $a_t$ ... 時刻 $t$ で取った action
- $r_{t+1}$ ... 次時刻の reward
- $a_{t+1}$ ... 次時刻の action
- $d_{t+1}$ ... 次時刻の done event
- $h_t$ ... M の hidden state

**この論文での役割**: 本実験の実装済み手法というより、複雑な環境へ拡張する future work の枠組みである。著者は、本論文の単純なタスクでは training loop の one iteration で十分だったが、より難しいタスクでは iterative training と exploration が必要だと述べる。

### 実装 / アルゴリズム上の要点

1. Random policy で 10,000 rollouts を集め、観測フレームと行動 $a_t$ を保存する。
2. VAE (V) を訓練し、各フレームを $z$ に変換する。Appendix では ConvVAE は 64x64x3 入力、4 convolutional layers と 4 deconvolutional layers、VAE training は random policy データに対して 1 epoch、loss は reconstruction の $L^2$ distance と KL loss とされる。
3. MDN-RNN (M) を訓練する。Appendix では MDN-RNN は 20 epochs、CarRacing は LSTM 256 hidden units、Doom は 512 hidden units、両タスクで 5 Gaussian mixtures、相関 $\rho$ はモデル化せず factored Gaussian と書かれている。Car Racing 節の脚注では、V と M は別々に訓練しても各モデルが single GPU で 1 時間未満だったと述べる。
4. Controller (C) を線形モデルとして定義する。CarRacing では steering left/right、acceleration、brake の 3 continuous actions を出す。Doom では discrete actions を -1 から 1 の continuous action space に変換し、三分割して left / stay / right を表す。
5. CMA-ES で C を最適化する。population size は 64、各 agent は 16 random rollouts、fitness は average cumulative reward。CarRacing では 1800 generations 後に 1024 random rollouts 平均 900.46 の agent が得られた。
6. VizDoom では M を `gym.Env` のように wrap し、C を actual 環境ではなく virtual/dream 環境で訓練する。M が予測する death probability が 50% を超えると virtual environment で `done=true` にする。Appendix の DoomRNN 節は、実際の VizDoom で C を訓練しておらず、VizDoom は random policy の training data 収集に使っただけだと明記する。

## 実験・結果

- **データセット / ベンチマーク**: `CarRacing-v0` と `DoomTakeCover-v0` / VizDoom Take Cover。CarRacing は random generated tracks で、100 consecutive trials の average reward が 900 以上なら solving。Take Cover は fireballs を避けるタスクで、各 rollout は最大 2100 time steps（約 60 秒）、100 consecutive rollouts の平均 survival time が 750 time steps（約 20 秒）を超えると solved。
- **比較対象 / baseline**: CarRacing 表 `car_racing_table` では DQN、A3C (continuous)、A3C (discrete)、ceobillionaire (Gym Leaderboard)、V model、V model with hidden layer、Full World Model を比較する。Take Cover 表 `doom_virtual_table` では temperature ごとの virtual/actual score に加え、Random Policy と Gym Leader を置く。
- **指標**: CarRacing は average score。Take Cover は生存 time steps を cumulative reward / score とする。両方とも 100 random/consecutive trials の平均と標準偏差が中心で、Appendix には 1024 rollouts での評価も出る。
- **主な結果**: CarRacing の Full World Model は 906 $\pm$ 21 で、DQN 343 $\pm$ 18、A3C (continuous) 591 $\pm$ 45、A3C (discrete) 652 $\pm$ 10、ceobillionaire 838 $\pm$ 11、V model 632 $\pm$ 251、V model with hidden layer 788 $\pm$ 141 を上回る。著者は "first reported solution to solve this task" と主張する。
- **主な結果**: Parameter count は、CarRacing の表 `car_racing_param_count_table` で VAE 4,348,547、MDN-RNN 422,368、Controller 867。VizDoom の表 `doom_cover_table` で VAE 4,446,915、MDN-RNN 1,678,785、Controller 1,088。
- **主な結果**: Take Cover では $\tau=1.15$ の agent が actual 環境で 1092 $\pm$ 556 を得る。temperature 表の値は、$\tau=0.10$: Virtual 2086 $\pm$ 140 / Actual 193 $\pm$ 58、$\tau=0.50$: 2060 $\pm$ 277 / 196 $\pm$ 50、$\tau=1.00$: 1145 $\pm$ 690 / 868 $\pm$ 511、$\tau=1.15$: 918 $\pm$ 546 / 1092 $\pm$ 556、$\tau=1.30$: 732 $\pm$ 269 / 753 $\pm$ 139。Random Policy は Actual 210 $\pm$ 108、Gym Leader は 820 $\pm$ 58。
- **著者が主張する貢献**: 1) raw RGB pixel stream から spatial-temporal representation を学び、CarRacing-v0 を解く compact policy を得た。2) learned world model の hallucinated dream 内だけで学習した policy を actual VizDoom に転移できた。3) temperature $\tau$ によって realism と exploitability の tradeoff を調整できることを示した。4) C--M 系の古い枠組みを、VAE + MDN-RNN + ES の実験系として再構成した。

## 妥当性と限界

- **この主張を支える根拠**: CarRacing では $z_t$ のみ、$z_t$ + hidden layer、Full World Model の ablation があり、$h_t$ を使う効果が score と挙動説明の両方で示される。Take Cover では temperature を変えた virtual/actual score 表があり、低温度では virtual score が高くても actual transfer が失敗することを示している。
- **この主張を支える根拠**: Cheating the World Model 節では、agent が fireballs を出させない adversarial policy を見つけた初期実験を説明し、learned dynamics model を actual 環境の完全な代替にする危険を明示する。その上で MDN-RNN と $\tau$ を、不完全な M を exploit しにくくするための設計として位置づける。
- **著者が認めている limitations / future work**: VAE を standalone unsupervised model として訓練すると、task-relevant でない特徴も符号化する。Discussion では Doom の壁の brick tile patterns は再構成したが、CarRacing の road tiles はうまく再構成しない例が挙げられる。
- **著者が認めている limitations / future work**: LSTM-based world model の容量は限られ、catastrophic forgetting の問題がある。Future work として higher capacity models や external memory module が挙げられる。
- **著者が認めている limitations / future work**: 本手法は early RNN-based C--M systems と同様に possible futures を time step by time step に simulate するもので、人間的な hierarchical planning や abstract reasoning から利益を得ていない。`Learning to Think` や `One Big Net` 的な一般化は future work とされる。
- **読者として注意すべき点**: 本論文の 2 タスクは著者自身が "relatively simple" と述べる範囲で、random policy から集めた 10,000 rollouts だけで reasonable world model が訓練できている。より複雑な環境で同じ分離訓練が成立するとは、TeX 中では実証されていない。
- **読者として注意すべき点**: Doom では C を actual VizDoom 環境で訓練していないことが強い主張である一方、actual fine-tuning との比較は TeX 中には示されていない。また、C は M の hidden states にアクセスするので、ゲーム観測だけを見る通常の policy とは情報条件が異なる。
- **追加で確認したい実験 / 疑問**: より難しいタスクで iterative training を何回回す必要があるか、$\tau$ の最適値が環境ごとにどれだけ敏感か、C を非線形にした場合に CMA-ES 以外の最適化が必要になるかは、この TeX だけでは未解決である。

## 用語メモ

- **World model**: 本論文では V と M を合わせた、観測圧縮と未来予測のモデル。CarRacing では特徴抽出器、VizDoom では virtual environment として使われる。
- **Vision (V)**: ConvVAE。64x64x3 の画像を latent vector $z$ にする。CarRacing は $z \in \mathcal{R}^{32}$、Doom は $z \in \mathcal{R}^{64}$ と TeX に書かれる。
- **Memory (M)**: LSTM + MDN。$P(z_{t+1}|a_t,z_t,h_t)$、Doom では $P(z_{t+1},d_{t+1}|a_t,z_t,h_t)$ をモデル化する。
- **Controller (C)**: $z_t$ と $h_t$ から action を出す線形モデル。報酬を見るのは C の最適化だけで、V/M は reward signal を知らない。
- **Latent vector $z$**: VAE がフレームを圧縮した連続ベクトル。M は raw pixel ではなく $z$ の時系列を予測する。
- **Hidden state $h_t$**: RNN の内部状態。CarRacing では LSTM output vector $h$、Doom では cell vector $c$ と output vector $h$ の両方が Controller 入力になる。
- **MDN-RNN**: RNN の出力を mixture density にしたモデル。Appendix では 5 Gaussian mixtures、diagonal covariance、相関 $\rho$ なし。
- **Temperature $\tau$**: M から $z_{t+1}$ を sampling するときの uncertainty 調整パラメータ。低すぎると mode collapse 的に fireballs が出なくなり、高すぎると virtual 環境が難しくなりすぎる。
- **Dream / hallucinated environment**: M が生成する latent space 上の virtual environment。DoomRNN はスクリーンショットを毎 step render せず、latent space だけで動く。
- **Cheating the World Model**: C が不完全な M の誤りや hidden state を exploit して、actual 環境では通用しない policy を学ぶ問題。
- **Iterative training**: まず actual 環境でデータを集め、M と C を更新し、未完了ならまた actual 環境で新しいデータを集める手順。本論文の実験では one iteration で十分だったが、複雑タスクでは必要とされる。

## 読む順番の提案

- 最初に `main.tex` の Abstract と Introduction を読み、"large world model and a small controller model" という分業の動機を確認する。正規ノートでは Summary の「問題」と Takeaway の「巨大な world model + 極小 controller」に対応する。
- 次に Agent Model 節を読み、VAE (V)、MDN-RNN (M)、Controller (C)、Eq. `controller_equation` を押さえる。正規ノートの「手法」箇条書きの V/M/C の理解に直結する。
- その後、Car Racing Experiment の Procedure、`car_racing_param_count_table`、`car_racing_table` を読む。906 $\pm$ 21、867 parameters、V-only / hidden-layer ablation の根拠はここにある。
- 次に VizDoom Experiment の Procedure、Cheating the World Model、`doom_virtual_table` を読む。dream-only training、done prediction、temperature $\tau$、mode collapse、1092 $\pm$ 556 の根拠がつながる。
- 最後に Appendix の Variational Autoencoder、Recurrent Neural Network、Controller、Evolution Strategies、DoomRNN を読む。$N_z$、hidden units、5 mixtures、20 epochs、CMA-ES population 64、16 rollouts など、正規ノートの数値を裏取りする場所である。
- Discussion と Iterative Training Procedure は、正規ノートの Critical Thoughts の「限界」「future work」「iterative training は未実装」に対応する。ここは著者の主張と読者側の疑問を分けて読むとよい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1803.10122v4/`
- 正規ノート: `notes/arXiv-1803.10122v4.md`
