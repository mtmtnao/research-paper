# Denoising Diffusion Probabilistic Models

- arXiv: https://arxiv.org/abs/2006.11239
- source: ../papers/arXiv-2006.11239v2/
- authors: Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
- venue / year: NeurIPS 2020
- tags: [diffusion, generative-model, score-matching, image-synthesis, variational-inference]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: Diffusion probabilistic models (Sohl-Dickstein+ 2015) は定義と訓練が素直で尤度評価も容易だが、GAN・autoregressive・flow・VAE 等と比べて高品質サンプルを生成できることが示されていなかった。また denoising score matching + Langevin dynamics (NCSN, Song & Ermon 2019) や autoregressive モデルとの理論的関係も未整理だった。
- **手法**: 固定 Gaussian forward process $q(\bx_t|\bx_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\bx_{t-1},\beta_t\bI)$ を持つ離散時間 Markov chain の reverse process $p_\theta(\bx_{t-1}|\bx_t)=\mathcal{N}(\bmu_\theta,\sigma_t^2\bI)$ を variational bound で学習する。鍵となる選択は3つ:
  1. **ε-prediction parameterization**: $\bmu_\theta$ を直接出さず、$\bx_t = \sqrt{\bar\alpha_t}\bx_0+\sqrt{1-\bar\alpha_t}\bepsilon$ の $\bepsilon$ を予測する $\bepsilon_\theta(\bx_t,t)$ を学習。これは多ノイズスケールの denoising score matching と等価で、サンプリング式は annealed Langevin dynamics に酷似（式 (11), Alg. 2）。
  2. **Simplified objective $L_\mathrm{simple}=\E_{t,\bx_0,\bepsilon}\|\bepsilon-\bepsilon_\theta(\sqrt{\bar\alpha_t}\bx_0+\sqrt{1-\bar\alpha_t}\bepsilon,t)\|^2$**: variational bound 中の $\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ 重みを捨てた非加重 MSE。小さい $t$ の項を相対的に down-weight し、大ノイズの除去に学習を集中させる。
  3. **$\sigma_t^2$ は学習せず定数固定**（$\sigma_t^2=\beta_t$ と $\sigma_t^2=\tilde\beta_t$ がほぼ同等）。
  - アーキテクチャは PixelCNN++ ベースの U-Net + group normalization、Transformer sinusoidal 位置埋め込みで時刻 $t$ を全層に注入、$16\times16$ 解像度で self-attention。$T=1000$、$\beta_t$ は $10^{-4}\to0.02$ の線形スケジュール、データは $[-1,1]$ にスケーリング、$L_0$ は離散デコーダで処理。
- **結果**:
  - **CIFAR10 unconditional**: IS $9.46\pm0.11$、**FID $3.17$（test set FID は $5.24$）**、NLL $\leq 3.75$ test (3.72 train)。当時の class-conditional 含む大半のモデルを上回り、unconditional では StyleGAN2+ADA (FID 3.26) と並ぶ。
  - **LSUN $256^2$** (Table lsun_fid): Bedroom FID **4.90**（large, 256M params） / 6.36（114M）、Church **7.89**、Cat **19.75**。ProgressiveGAN 同等〜及ばず、StyleGAN2 にはやや劣る。
  - **CelebA-HQ $256\times256$**: ProgressiveGAN 同等の品質を主張（定量比較は本文中に明示なし）。
  - **Loss ablation (Table 2)**: $\tilde\bmu$-pred + $L$(fixed iso $\Sigma$) は FID 13.22、$\bepsilon$-pred + $L$ は 13.51 で同等。だが $\bepsilon$-pred + $L_\mathrm{simple}$ で 3.17 と一気に改善。$\tilde\bmu$-pred + MSE と learned diagonal $\bSigma$ は学習不安定で "blank"。
  - **Rate-distortion 解析**: CIFAR10 で rate 1.78 bits/dim + distortion 1.97 bits/dim (RMSE 0.95/255)。Lossless codelength の半分以上が知覚不能な細部に費やされている。
  - **Progressive 生成**: $\hat\bx_0$ を時刻ごとに復元すると、大域構造が先、細部が後で出現。
- **貢献**:
  1. Diffusion ↔ multi-noise denoising score matching + Langevin dynamics の明示的同値（ε-prediction を介する）。
  2. 「重みを捨てた」simplified objective $L_\mathrm{simple}$ が標本品質を劇的に上げるという経験的発見。
  3. Diffusion を progressive lossy decoder / 一般化された autoregressive モデルとして解釈する rate-distortion フレーム。
  4. 当時の unconditional CIFAR10 SOTA (FID 3.17)、LSUN/CelebA-HQ で高品質サンプル、公開実装。

## Takeaway（自分にとっての要点）

- **「平均ではなくノイズを予測する」** という再パラメータ化が肝。$\bmu_\theta$ を直接 fit するより、$\bx_t$ 中の Gaussian ノイズ $\bepsilon$ を出させる方が score matching と整合し、しかも単純な MSE で書ける。
- **理論的に正しい重み付き ELBO を捨てて、unweighted MSE に切り替えた途端に FID が 13.51 → 3.17** に改善（Table 2）。これは "尤度最適 = 標本品質最適" ではないことのきれいな実証で、後続の DDPM 派生研究はこの「重みを設計する」観点を継承している。
- **$\sigma_t^2$ を学習させると壊れる**。固定で十分。後の研究（Improved DDPM, Nichol & Dhariwal 2021）が learned $\Sigma$ を成立させるのは別工夫が要る、という伏線。
- **rate-distortion 視点**: lossless 1.78+1.97≒3.75 bits/dim のうち distortion 側（知覚に効く部分）は RMSE で 0.95/255 にしかならない。だから NLL が悪く見えても標本は綺麗、という説明が成立する。likelihood と perceptual quality が乖離する理由の良い説明。
- **forward process が学習不要・$\bx_T$ がほぼ data と独立**という設計が、VAE/flow と diffusion を分ける本質。VAE は encoder を学習するから後段の表現が data に依存する。
- **autoregressive との接続**: diffusion を「Gaussian ノイズ順で並べた generalized bit ordering の autoregressive」と解釈すると、$T$ を data 次元と独立に選べる利点（$T=1000 \ll 32\cdot32\cdot3$）が出てくる。
- **実装メモ**: T=1000, β linear 1e-4→0.02, U-Net 35.7M (CIFAR) / 114M (256²) / 256M (LSUN-bedroom large), Adam lr 2e-4 (2e-5 for 256²), batch 128/64, EMA 0.9999, dropout 0.1 only on CIFAR, TPU v3-8。CIFAR は 800k step で 10.6 時間。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「数式上は等価な3つの設計選択（μ pred / ε pred / x0 pred）と重み付き/非加重 ELBO」をきれいに ablation で並べ、経験的最適解 ($\bepsilon$-pred + $L_\mathrm{simple}$) を特定している。後続が follow しやすい良いベースライン。
  - score matching, Langevin dynamics, autoregressive decoding, lossy compression という独立した4文脈すべてに diffusion を橋渡しした概念整理が秀逸。
  - 実装公開（github.com/hojonathanho/diffusion）+ 詳細な hyperparameter 開示で再現性が高い。
- **弱み / 疑問**（著者自身が認めているもの含む）:
  - **NLL は競合しない**（著者自ら認める）: CIFAR10 で Sparse Transformer 2.80, Gated PixelCNN 3.03 に対し本提案は ≤ 3.75。"半分以上が imperceptible details に費やされる" と説明しているが、density estimator としては劣る。
  - **Progressive compression は概念実証どまり**（著者自ら "only a proof of concept", appendix）: minimal random coding が高次元で tractable でない。
  - **Learned $\bSigma$ が不安定**（Table 2 で実際に学習失敗）—固定 $\sigma_t$ で逃げているが、最尤の意味では sub-optimal。
  - **サンプリングが遅い**: $T=1000$ ステップ × NN 評価。論文では CIFAR で 256 枚 17 秒、256² で 128 枚 **300 秒**。実応用には致命的（後の DDIM, progressive distillation 等が解決する課題）。本論文中に latency についての limitation 明示はあまり強くない。
  - **モード網羅性・多様性の定量評価が薄い**: FID/IS のみで、recall/coverage 指標や mode collapse の検証はない。GAN との比較なら本来必要。
  - **LSUN Cat FID 19.75** は StyleGAN2 (6.93) に大きく劣る。高解像度・多様分布で diffusion がまだ不利な領域があることが示唆されているが本文での議論は薄い。
  - **interpolation で eyewear だけ滑らかに変わらない**と著者自身が述べている（§4.2）。離散属性は latent diffusion 経路に乗りにくい。
  - **設計判断の sweep 範囲が限定的**: $T$ や $\beta$ の上限は sweep していない（"set $T=1000$ without a sweep"）。最適点ではない可能性。
- **次に試したいこと**:
  - **重み関数の系統的探索**: $L_\mathrm{simple}$ は重み=1 だが、$t$ 依存重み $w(t)$ を学習可能にして perceptual loss と NLL の Pareto を取る（後の Improved DDPM / VDM 方向の追検証）。
  - **$T$ を 100 / 50 / 10 まで落としたときの FID 曲線**を引いて、本論文の $T=1000$ がどれだけ over-killed か確認（DDIM 等の前段）。
  - **CIFAR10 で IS/FID と NLL の同時 Pareto** をモデルサイズ・objective・$\bSigma$ 設定を変えて引く。
  - **$\bepsilon$-pred vs $\bx_0$-pred** を改めて現代的設定で比較（論文では "$\bx_0$ pred は早期実験で悪かった" としか書かれていない）。
  - **rate-distortion 解釈を loss 設計に直接戻す**: 知覚に効くタイムステップを自動検出して優先的に学習する curriculum。
  - 自分が普段触る非画像領域（音声/タンパク/グラフ）に同じ "noise-prediction + simplified MSE" を移植したときの sample quality / NLL の挙動。

## Notes / Quotes

- "the $\bepsilon$-prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model's variational bound to an objective that resembles denoising score matching." (§3.2)
- "We find that training our models on the true variational bound yields better codelengths than training on the simplified objective, as expected, but the latter yields the best sample quality." (§4.1) — likelihood と perceptual quality の乖離を明示。
- "More than half of the lossless codelength describes imperceptible distortions." (§4.3) — rate 1.78 / distortion 1.97 bits/dim, RMSE 0.95/255。
- "We can therefore interpret the Gaussian diffusion model as a kind of autoregressive model with a generalized bit ordering that cannot be expressed by reordering data coordinates." (§4.3)
- 限界の自認: "our lossless codelengths ... are not competitive with other types of likelihood-based generative models" (§4.3)、"Our lossy compression argument ... is only a proof of concept" (Appendix)、"learning reverse process variances ... leads to unstable training and poorer sample quality compared to fixed variances" (§4.2)、interpolation で "smoothly vary attributes such as pose, skin tone, hairstyle, expression and background, **but not eyewear**" (§4.4)。
- Forward posterior: $q(\bx_{t-1}|\bx_t,\bx_0)=\mathcal{N}(\tilde\bmu_t(\bx_t,\bx_0),\tilde\beta_t\bI)$、$\tilde\beta_t=(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)\cdot\beta_t$（式 (6),(7)）。
- サンプリング1ステップ: $\bx_{t-1}=\frac{1}{\sqrt{\alpha_t}}(\bx_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\bepsilon_\theta(\bx_t,t))+\sigma_t\bz$（Alg. 2）。
- $L_T=D_\mathrm{KL}(q(\bx_T|\bx_0)\|\mathcal{N}(0,I))\approx 10^{-5}$ bits/dim と無視可能になるよう $\beta$ を設計（§4 冒頭）。

## Related Papers

- Sohl-Dickstein+ 2015, *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* — diffusion の原典、forward/reverse process と ELBO 分解の出所。
- Song & Ermon 2019, *Generative Modeling by Estimating Gradients of the Data Distribution* (NCSN) — denoising score matching + annealed Langevin dynamics の baseline。本論文の ε-prediction はこれと等価。
- Song & Ermon 2020, *Improved Techniques for Training Score-Based Generative Models* (NCSNv2) — 直接比較対象（CIFAR FID 31.75）。
- Vincent 2011, *A Connection Between Score Matching and Denoising Autoencoders* — score matching ↔ denoising の理論基盤。
- Kingma & Welling 2013, VAE / Rezende+ 2014 — latent variable model としての位置付け。
- Salimans+ 2017, PixelCNN++ / Ronneberger+ 2015, U-Net — アーキテクチャ基盤。
- Karras+ 2018/2019/2020, ProgressiveGAN / StyleGAN / StyleGAN2 / StyleGAN2+ADA — 標本品質の比較対象。
- Du & Mordatch 2019, *Implicit Generation with EBMs* / Grathwohl+ 2020 JEM — energy-based model 比較。
- Gregor+ 2016, *Towards Conceptual Compression* — progressive decoding 系の先行研究。
- Menick & Kalchbrenner 2018, *Generating High Fidelity Images with Subscale Pixel Networks* — autoregressive 順序が標本品質に与える影響。
- Havasi+ 2018 / Harsha+ 2007 — minimal random coding（progressive compression 解釈の基盤、ただし高次元で intractable）。
- Vaswani+ 2017, Transformer — sinusoidal position embedding を時刻条件付けに流用。
- Wu & He 2018, Group Normalization — U-Net 中の正規化。
