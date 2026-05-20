# Weight Agnostic Neural Networks

- arXiv: https://arxiv.org/abs/1906.04358
- source: ../papers/arXiv-1906.04358v2/
- authors: Adam Gaier, David Ha
- venue / year: NeurIPS 2019
- tags: [neural-architecture-search, neuroevolution, NEAT, inductive-bias, MDL]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 通常の NN は「アーキテクチャを人が選び、重みを学習する」枠組みで、性能の大半は学習した重みに依存している。アーキテクチャ自体がどれだけタスク解を「先天的に」エンコードできるかは検証されていない。NAS は trainable な構造を探すだけで、学習なしで動くことは想定されていない。
- **手法**: Weight Agnostic Neural Network (WANN) サーチ。NEAT ベースの位相進化を使い、**重みは学習せず全結合に単一の共有スカラー** を割り当てて評価する。具体的には：(1) 最小限の疎結合ネットワークから開始、(2) 各ネットワークを共有重み値 $[-2,-1,-0.5,+0.5,+1,+2]$ の各値で複数 rollout し平均報酬を取る、(3) 「平均性能」「最大性能」「接続数」の 3 目的で NSGA-II の非優越ソート（うち 20% は max performance を採用してランクの停滞回避）、(4) tournament selection で次世代生成。位相変異は `insert node` / `add connection` / `change activation` の 3 種。活性化関数プールは linear, step, sin, cosine, Gaussian, tanh, sigmoid, abs, invert (negative linear), ReLU の 10 種（MDL/AIT に動機づけ）。
- **結果**:
  - 連続制御 3 タスク（Table 1, 100 試行平均）。`Random Shared Weight` のみでベースライン固定位相を大きく上回り、`Tuned Shared Weight` でも実用的に動く。
    - **CartPoleSwingUp**: WANN 515±58（共有ランダム） / 723±16（共有最適） / 932±6（個別学習）。固定位相は共有ランダムで 7±2、個別学習で 918±7。
    - **BipedalWalker-v2**: WANN 51±108 / 261±58 / 332±1。固定位相 (estool) は 347±1（SOTA, ha2018designrl）。WANN は 210 接続のみで動き、SOTA の 2804 接続より一桁少ない。
    - **CarRacing-v0**: WANN 375±177 / 608±161 / 893±74。固定位相 (world models) は 906±21（SOTA, ha2018worldmodels）。WANN は VAE の 16 次元潜在のみ入力で動作（baseline は world model RNN の隠れ状態も入力）。
  - **MNIST 分類**（16×16 にダウンサンプル+デスキュー、256 入力 10 出力、1849 接続）。単一共有重みで、千個オーダーの重みを学習した単層 NN と同等程度（イントロでは「higher than chance ~92%」と表現）。共有重みの値ごとに digit 別精度が違うので、複数重みを同じネットワークに刺してアンサンブル化するとランダム重みより大幅良、最良単一重みにわずかに劣る程度。
- **貢献**: (1) 重み学習なしで RL/分類タスクをそこそこ解ける WANN という概念実証、(2) NEAT を 3 目的（mean perf / max perf / 接続数）に拡張した位相サーチアルゴリズム、(3) MNIST/3 連続制御で baseline と比較し、ネット図ごと公開して内部機構（attractor + Gaussian によるスイングアップなど）を解釈、(4) コード公開とインタラクティブ記事による再現性担保。

## Takeaway（自分にとっての要点）

- 「重みを学習しない」の本質は重みを 1 個に共有して固定するだけ、という非常に身もふたもない簡素化。重みを定数化したことで「アーキ自体の性能」だけが残り、評価が一意になる ← この発想はそのまま他の構造探索にも転用可能。
- 活性化関数の多様性（linear/step/sin/Gaussian/abs など）が WANN を成立させる肝。重みで自由度を稼げない分、関数形式で「対称性・周期・反転」などの関係を表現させている。著者自身は appendix で「ReLU や sigmoid だけでも実装は可能だっただろうが、多様な活性化のおかげで対称性・反復のような関係をよりコンパクトに表現できる」「線形活性のみでは可能と確信できない」とコメント。
- 多目的化のうち 20% を `max performance` に振り替えるトリックは、「複数追加が揃って初めて性能が上がる」場面で stepping stone を残すための実装的工夫。NSGA-II ベースの NAS 全般で参考になる。
- 共有重み 1 個 → 異なる値で複数インスタンス → アンサンブル、という self-ensemble は意外と効くので、「1 つのネットワーク、複数の重みでマルチタスク」みたいな設計（後の HyperNet / soft mask 系の前哨）として読める。
- WANN が個別重みをランダムに振ると壊れる（CartPoleSwingUp で 57±121）こと、つまり「符号と相対関係に強く依存している」点は、「アーキは関係性をエンコードしている」という主張の裏側の制約として重要。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「重みなし」という極端な制約で NN の何が本当に効いているかを切り出した思考実験として鮮やか。BipedalWalker で 210 接続という小ささ、CarRacing でフィードフォワード単独で世界モデル baseline に肉薄、というのは実用面でも示唆的。
  - 結果ネットワークが小さく可視化できるため、`x` を反転して中心引き寄せ→Gaussian on $d\theta$ でスイングアップ、という説明が後付けでなく図と整合している（CartPoleSwingUp の generation 32 解析）。
  - 多くの ablation や試行錯誤（単一固定値 0.7 で学習すると速いが汎化しない、Gaussian ノイズ実験、JAX で勾配 fine-tune したが CMA-ES に負ける、など）を appendix で正直に開示しており再現性が高い。
- **弱み / 疑問**:
  - MNIST で「~92%」という表現にとどまり、test accuracy の具体数値・SOTA との比較表は本文に無く Figure に頼っている。連続制御は Table 1 で数値があるが MNIST 側は緩い。
  - 連続制御 baseline は「ある特定の固定位相」との比較で、サンプル効率（評価回数 × 集団サイズ × 世代数）コストの公平比較は無い。MNIST で 960 個体 × 4096 世代は決して軽くなく、勾配学習との計算コスト比較が抜けている。
  - 活性化関数を 10 種混ぜることが本当に必須なのか、各種の寄与アブレーションは無い（著者も「intuition だが」と認めている）。
  - 個別重みのランダム化で性能が崩れることは、結局「学習なしで動く」というキャッチコピーがやや誇張で、正確には「単一共有スカラーでさえあれば動く」。論文タイトルが釣り気味。
  - CarRacing で WANN が SOTA に肉薄しているのは VAE の表現力に大きく依存しており、「アーキの inductive bias」とは別の話を混ぜている可能性。
  - 著者自身が認める limitations: 大規模化が困難（CNN には敵わない、「embarrassing achievement になるレベル」と Discussion で予防線）、フィードフォワード制約のため CarRacing で RNN を持つ baseline に少し負ける、勾配 fine-tune が landscape の都合で効きにくい、など。
- **次に試したいこと**:
  - 同じ計算予算（GPU/CPU hours）で random NAS / DARTS / weight-trained NEAT を並べた pareto curve。「重み学習を捨てた分の節約」が世代数膨張で相殺されていないかを定量化。
  - 活性化関数プールを段階的に削って性能曲線を取り、どの関数が必須かを切り分け（cosine/Gaussian を抜くと swingup の機構が再現できるか？）。
  - WANN を初期化として通常の SGD で学習させた場合の収束速度・最終精度を、ランダム初期化と比較（Baldwin effect の定量化）。
  - 共有重みではなく「数個のクラスタ共有重み」（K-shared）に拡張して、重み数 1 → 2 → 数個でどう性能が立ち上がるかを見れば、アーキ vs 重みの寄与分解がクリアになりそう。
  - MNIST 以外、CIFAR-10 や音声系で同じ方法がスケールするかの実証（おそらく崩れる方に賭けるが、その崩れ方を見たい）。
  - concurrent work の lottery ticket / supermask（zhou2019deconstructing）と直接合流させて、「pruning → WANN 化」「WANN → pruning」の双方向比較。

## Notes / Quotes

- "Replacing weight training with weight sampling ensures that performance is a product of the network topology alone." (21_methodOverview.tex)
- 共有重みは固定列 $[-2,-1,-0.5,+0.5,+1,+2]$。$|w|>3$ では活性化が飽和、$w\approx 0$ では情報が流れず除外（23_methodDetails.tex 脚注）。
- 多目的ランキングは 80% で {mean perf, #connections}, 20% で {mean perf, max perf}（23_methodDetails.tex）。
- BipedalWalker: WANN 210 接続 vs SOTA 2804 接続（31_control.tex）。
- 個別ランダム重みでは WANN も壊れる（SwingUp: 57±121）—「符号や一貫性」が重要、振幅は許容（31_control.tex）。
- "It would be an almost embarrassing achievement if they did [match CNNs]." (40_discuss.tex) — 自己評価としての limitation。
- appendix 60: 単一固定値 0.7 で学習すると速いが 0.6 でテストすると完全に壊れる → 範囲サンプリングが汎化に必須。
- appendix: gradient-based fine-tuning（JAX）より CMA-ES / population REINFORCE のほうが WANN の MNIST 上で良い、landscape の問題。
- ハイパラ（appendix）: SwingUp pop=192/gen=1024、Biped pop=480/gen=2048、CarRace pop=64/gen=1024、MNIST pop=960/gen=4096。
- Champion network 接続数（36_controlFig.tex）: SwingUp 52, Biped 210, CarRacing 245。
- (verified 2026-05-20) 数値・固有名詞・引用先を main.tex / 21,23_method / 30,31_control / 32_controlTable / 34_class / 37_classDiagram / 40_discuss / 60_appendix / main.bbl で再確認。Takeaway の活性化関数記述を appendix 60_appendix.tex に合わせて「ReLU だけでは厳しい」→「線形活性のみでは可能と確信できない（ReLU/sigmoid なら可能）」へ訂正。

## Related Papers

- Stanley & Miikkulainen, NEAT (2002) — 位相進化の土台。WANN は重み更新を除いた変種。
- Frankle & Carbin, Lottery Ticket Hypothesis (2018) / Lee+ SNIP (2018) — pruning 側からの「動く小サブネット」探索。
- Zhou+ 2019 "Deconstructing Lottery Tickets" / supermask — 同時期の関連研究。ランダム重みで動くサブネットを mask 探索で見つけた。
- Ulyanov+ Deep Image Prior (2018), He+ 2016 — ランダム初期化 CNN がそのまま画像処理に使える話、本研究の動機の一つ。
- Ha & Schmidhuber, World Models (2018) / Ha 2018 DesignRL — CarRacing/Biped の SOTA baseline と環境設定の出典。
- Williams 1992, population-based REINFORCE — 個別重み tuning に使用。
- Solomonoff / Kolmogorov / Rissanen — MDL/AIT の哲学的バックボーン。
- Zador 2019 "A critique of pure learning" — 動物の innate behavior と本研究の motivation の橋渡し（Discussion で長々引用）。
- Baldwin 1896 / Hinton & Nowlan 1987 — Baldwin effect、進化と学習の相互作用。
