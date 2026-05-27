# Generative Adversarial Nets

- arXiv: https://arxiv.org/abs/1406.2661
- source: ../papers/arXiv-1406.2661v1/
- authors: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- venue / year: NIPS 2014
- tags: [generative-model, adversarial, deep-learning, unsupervised]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 深層生成モデルは、最尤推定で生じる扱いにくい確率計算（partition function 等）の近似と、生成側で piecewise linear unit の恩恵を受けにくいことが原因で、判別モデル（Hinton-2012, Krizhevsky-2012）に比べて深層学習の成功を取り込めていなかった。RBM/DBM/DBN は MCMC 由来の mixing 問題を抱え、score matching や NCE は正規化定数までの解析的な密度を要求する。
- **手法**: 生成器 $G(\bm{z};\theta_g)$（ノイズ $\bm{z}\sim p_{\bm{z}}$ を入力に取る MLP）と判別器 $D(\bm{x};\theta_d)$（データ由来か否かを 0/1 で出す MLP）を2人ゼロ和の minimax ゲームで同時学習：$\min_G \max_D V(D,G)=\mathbb{E}_{p_\text{data}}[\log D(\bm{x})]+\mathbb{E}_{p_{\bm{z}}}[\log(1-D(G(\bm{z})))]$。SGD で $k$ 回 $D$ を更新するごとに $G$ を1回更新（実験では $k=1$）。学習初期に $\log(1-D(G(\bm{z})))$ が飽和する問題に対しては、等価な固定点を持つ $\log D(G(\bm{z}))$ の最大化に切り替える heuristic を導入。MCMC も近似推論も不要、backprop と dropout のみで学習し、forward propagation のみでサンプリング可能。
- **結果**: 非パラメトリックの理論解析で、$G$ 固定下の最適判別器が $D^*_G(\bm{x})=\frac{p_\text{data}(\bm{x})}{p_\text{data}(\bm{x})+p_g(\bm{x})}$ となり、$G$ の目的関数が $C(G)=-\log 4 + 2\cdot \text{JSD}(p_\text{data}\|p_g)$ に等しいことを示し、global minimum が $p_g=p_\text{data}$（値 $-\log 4$）で達成される唯一解であると証明。Algorithm 1 の更新は $p_g$ 空間の凸関数の上限の劣勾配に対応し、$D$ が最適に追従する限り収束する。定量評価は Parzen window による test set 対数尤度（Table 1）で、MNIST: $225 \pm 2$（DBN $138\pm 2$, Stacked CAE $121\pm 1.6$, Deep GSN $214\pm 1.1$）、TFD: $2057 \pm 26$（DBN $1909\pm 66$, Stacked CAE $2110\pm 50$, Deep GSN $1890\pm 29$）。MNIST/TFD/CIFAR-10（FC と conv 版）でサンプルを生成し、最近傍学習例との対比で memorization でないことを示し（Figure 2）、$\bm{z}$ 空間の線形補間で滑らかな遷移を確認（Figure 3）。
- **貢献**: (1) 2人プレイヤーの adversarial process で生成モデルを学習する一般的フレームワークの提案、(2) 非パラメトリック設定で global optimality と収束証明を与えたこと（JSD への対応）、(3) MLP + backprop + dropout だけで、MCMC や近似推論なしに動く実装、(4) 既存の深層生成モデル系統（directed/undirected/auto-encoder）との比較表（Table 2）。

## Takeaway（自分にとっての要点）

- minimax の objective を JSD に書き換える導出（$C(G)=-\log 4 + 2\cdot \text{JSD}$）が枠組み全体のキモ。
- 実装上の最重要 trick：$\log(1-D(G(\bm{z})))$ の最小化ではなく $\log D(G(\bm{z}))$ の最大化に置き換える（同じ固定点・初期勾配が強い）。
- $G$ はデータを直接見ない（$D$ の勾配経由でのみ更新）ので、入力をパラメータにコピーして overfit するルートが構造的に塞がれる—これが「統計的優位性」の主張根拠。
- noise はジェネレータの最下層にだけ入れた、と明記。「中間層でも理論上は OK」だが実験ではやっていない。
- $k$ steps of $D$ per 1 step of $G$ は SML/PCD の負例チェーンを step 間で持ち越すのと同じ発想として説明されている。$D$ を最適まで回さない理由は overfitting と計算コスト。
- 「Helvetica scenario」と著者自身が呼ぶ、$G$ が多くの $\bm{z}$ を同一 $\bm{x}$ に潰して $p_\text{data}$ を表すだけの多様性を失う問題が、$D$ の更新を怠ると起きる、と本文中で明示的に警告されている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - フレームワークが極めてシンプルで、partition function も MCMC も変分下界も要らないという「引き算」の設計が美しい。理論（global optimality）と実装（MLP + backprop）の最短距離。
  - 任意の微分可能関数 $G,D$ を許す一般性（Table 2 の "Any differentiable function is theoretically permitted"）。Conclusions でも条件付き生成や semi-supervised learning などの拡張が挙げられている。
  - 評価が困難な生成モデルに対して Parzen window で他手法と横並びの数字を出しつつ、「この評価は分散が大きく高次元では微妙」と自分で限界も明記している態度が誠実。
- **弱み / 疑問**:
  - 著者自身が明記する limitations: (a) $p_g(\bm{x})$ の explicit representation がない、(b) $D$ と $G$ の同期が必須で、特に $D$ の更新を怠ると Helvetica scenario に陥る、(c) Parzen window 推定は高次元では信頼性が低い。
  - 収束証明は非パラメトリックの $p_g$ 空間の話で、実際に動かす MLP パラメータ空間では「複数の臨界点が出るが MLP は実用上ちゃんと動く」と希望的観測で片づけている（§4.2末尾）。実務上の不安定性の理論的根拠が無い。
  - Table 1 の Parzen 比較は MNIST/TFD のみ。CIFAR-10 の数値比較はなく、サンプル画像のみ。「定量的に CIFAR で勝った」とは言っていない。
  - Helvetica scenario の頻度・防ぎ方の具体策（$k$ の調整指針など）が定量化されていない。$k=1$ を使った理由が「the least expensive option」だけで、安定性とのトレードオフが示されていない。
  - 学習率・モメンタム・MLP の幅と深さ、dropout 率などのハイパーパラメータが本文に書かれておらず、GitHub レポジトリ参照になっている（脚注1）。論文単体での再現性は低い。
- **次に試したいこと**:
  - $\log D(G(\bm{z}))$ heuristic を使わず元の minimax のままで JSD として収束する版と、heuristic 版で実際に何が起きているかを比較（評者補足）。
  - 「Helvetica scenario」の発生確率を $k$（D update 回数）と $G$ の容量で sweep して定量化。本文では警告だけで数値が無い。
  - 中間層に noise を入れる版（「理論上は OK」と書かれている）が実際に効くか実験。
  - Parzen window 以外で MNIST/TFD/CIFAR-10 の生成品質を評価する指標を後付けで適用し、当時の数字の解釈をアップデートする（評者補足）。
  - 著者が conclusions で挙げた拡張のうち、(1) 条件付き化、(4) semi-supervised がどこまで有効かを検証する（評者補足）。

## Notes / Quotes

- 直感説明: 「生成器は偽札を作る counterfeiter、判別器は警察」(Introduction)。
- $D$ の最適解: $D^*_G(\bm{x}) = \frac{p_\text{data}(\bm{x})}{p_\text{data}(\bm{x}) + p_g(\bm{x})}$（Eq. 2, Proposition 1）。
- 主結果: $C(G) = -\log(4) + 2 \cdot \text{JSD}(p_\text{data}\,\|\,p_g)$（Theorem 1 の証明中）。global minimum は $p_g=p_\text{data}$ で値 $-\log 4$。
- 実装の trick: "Rather than training $G$ to minimize $\log(1 - D(G(\bm{z})))$ we can train $G$ to maximize $\log D(G(\bm{z}))$. This objective function results in the same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning." (§3末尾)
- Generator: rectifier linear + sigmoid 混合、noise は最下層のみ入力。Discriminator: maxout activations + dropout。$k=1$。Momentum を使用（Algorithm 1 末尾）。
- データセット: MNIST, Toronto Face Database (TFD), CIFAR-10（FC 版と conv discriminator + "deconvolutional" generator 版の両方）。
- Parzen window 評価 (Table 1): Adversarial nets が MNIST $225\pm 2$（最良）、TFD $2057\pm 26$（Stacked CAE $2110\pm 50$ に次ぐ）。
- 既知の限界 (§"Advantages and disadvantages"): "no explicit representation of $p_g(\bm{x})$" / "$D$ must be synchronized well with $G$ during training (in particular, $G$ must not be trained too much without updating $D$, in order to avoid 'the Helvetica scenario' in which $G$ collapses too many values of $\bm{z}$ to the same value of $\bm{x}$ to have enough diversity to model $p_\text{data}$)"。
- 将来拡張（Conclusions）: 条件付き生成、学習済み近似推論、すべての条件付き $p(\bm{x}_S\mid \bm{x}_{\not S})$ のモデル化（MP-DBM の確率版）、semi-supervised、$G/D$ 協調と $\bm{z}$ 分布の改善による効率化。
- TeX 中には明示されていない: 学習率、バッチサイズ $m$、MLP の具体的な層構成や次元、CIFAR-10 の定量比較数値。
- (verified 2026-05-26) Takeaway / Critical Thoughts から TeX に無い後続研究名・後年の用語を削除し、評者独自の検証案は評者補足として明示 (adversarial.tex)
- (verified 2026-05-20) Related Papers の "Bengio et al. ICML 2013/2014, GSN" 表記を分割修正。ICML 2013 (`Bengio-et-al-ICML2013`, adversarial.bbl) は "Better mixing via deep representations" であり GSN 論文ではないため、ICML 2013 (Better mixing) と ICML 2014 (Deep GSN, Table 1 比較対象) を別エントリにした。

## Related Papers

- Hinton et al. 2006, Deep Belief Networks — undirected/directed ハイブリッドの代表、比較対象（DBN）。
- Salakhutdinov & Hinton 2009, Deep Boltzmann Machines — undirected 系で MCMC mixing 問題の代表。
- Hyvärinen 2005, Score Matching / Gutmann & Hyvärinen 2010, NCE — 解析的密度を要する代替学習基準。本研究は NCE の「discriminative criterion で生成モデルを fit」する精神を、固定 noise → 別の判別器に置き換えた拡張と位置付けられる。
- Bengio et al. ICML 2013, "Better mixing via deep representations" — MCMC mixing 問題の代表的引用先（Related work で引用）。
- Bengio et al. ICML 2014, "Deep generative stochastic networks trainable by backprop" — Markov chain を学習する直接の比較対象（Deep GSN として Table 1 に登場）。
- Kingma & Welling 2014, Auto-Encoding Variational Bayes / Rezende et al. 2014, Stochastic Backpropagation — 同時期の back-prop で生成モデルを学習する系統。
- Goodfellow et al. 2013, Maxout Networks — 判別器に採用された活性化。
- Hinton et al. arXiv 2012, Dropout — 判別器の正則化。
- Younes 1999 / Tieleman 2008, SML/PCD — $k$ steps の交互更新を正当化する類比。
- Breuleux & Bengio 2011 — Parzen window 評価手法の出典。
