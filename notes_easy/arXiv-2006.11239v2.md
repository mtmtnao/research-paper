# Denoising Diffusion Probabilistic Models（高品質画像生成における離散時間 diffusion model の再定式化）

- arXiv: https://arxiv.org/abs/2006.11239
- 一次ソース: ../papers/arXiv-2006.11239v2/
- 正規ノート: ../notes/arXiv-2006.11239v2.md
- 著者: Jonathan Ho, Ajay Jain, Pieter Abbeel
- venue / year: NeurIPS 2020

---

## 一言で言うと

この論文は、固定した Gaussian forward process と学習する Gaussian reverse process からなる diffusion probabilistic model で、高品質な画像生成が可能であることを示す。中心は、reverse process の平均を直接予測する代わりにノイズ $\bepsilon$ を予測する parameterization と、重みを外した $L_\mathrm{simple}$ により、denoising score matching と Langevin dynamics との関係を明示しつつ CIFAR10 で FID 3.17 を得る点である。

## 何を議論する論文か

- **問題設定**: データ画像 $\bx_0$ から徐々に Gaussian noise を加える forward process $q(\bx_{1:T}|\bx_0)$ を固定し、その逆向きにノイズを取り除く reverse process $p_\theta(\bx_{0:T})$ を学習して、新しい画像サンプルを生成する。
- **対象範囲 / 仮定**: 画像データは整数値 $\{0,1,\dotsc,255\}$ から $[-1,1]$ に線形スケーリングされる。forward process は variance schedule $\beta_1,\dotsc,\beta_T$ に従う Markov chain で、実験では $T=1000$、$\beta_1=10^{-4}$ から $\beta_T=0.02$ まで線形に増やす。reverse process は Gaussian transition とし、分散は学習せず $\sigma_t^2\bI$ に固定する。
- **既存研究との差分**: GAN、autoregressive model、flow、VAE は高品質な画像・音声生成を示していた一方、Sohl-Dickstein et al. (2015) の diffusion model については高品質サンプルの実証が不足していた。さらに Song and Ermon (2019, 2020) の denoising score matching / NCSN 系とは近いが、本論文は latent variable model として variational inference で sampler 自体を直接訓練する点を強調する。
- **この論文で答えたい問い**: diffusion model は GAN などと比較できる品質の画像を生成できるのか。どの reverse process parameterization と training objective がサンプル品質を改善するのか。高品質サンプルにもかかわらず log likelihood が競合しない理由を、rate-distortion や progressive decoding の観点からどう説明できるのか。

## 背景と前提

- **Latent variable model としての diffusion model**: 論文では $\bx_1,\dotsc,\bx_T$ を $\bx_0$ と同じ次元を持つ latent とみなし、$p_\theta(\bx_0)=\int p_\theta(\bx_{0:T})d\bx_{1:T}$ と書く。他の latent variable model との差分として、この論文の approximate posterior $q(\bx_{1:T}|\bx_0)$ は学習せず、手で固定したノイズ付加過程である。
- **Forward process / reverse process**: forward process はデータからノイズへ進む固定 Markov chain、reverse process はノイズからデータへ戻る学習済み Markov chain である。TeX では forward process を "diffusion process"、reverse process を $p_\theta(\bx_{t-1}|\bx_t)$ と呼ぶ。
- **Variational bound**: 学習は negative log likelihood の variational bound $L$ を最小化する形から出発する。bound を $L_T$、$L_{t-1}$、$L_0$ に分解することで、各 timestep の Gaussian KL を閉形式で扱える。
- **Denoising score matching との関係**: $\bepsilon$-prediction によって、variational bound の中間項 $L_{t-1}$ が複数ノイズレベルの denoising score matching に似た重み付き MSE へ変形される。sampling update は annealed Langevin dynamics に似る。
- **比較対象**: CIFAR10 では EBM、JEM、BigGAN、StyleGAN2 + ADA、Gated PixelCNN、Sparse Transformer、PixelIQN、NCSN/NCSNv2、SNGAN、SNGAN-DDLS などと比較する。LSUN では ProgressiveGAN、StyleGAN、StyleGAN2 と FID を比較する。

## 提案手法

### コアアイデア

forward process は、データ $\bx_0$ に小さな Gaussian noise を $T$ 回足して、最後の $\bx_T$ がほぼ standard normal prior $\mathcal{N}(\bzero,\bI)$ になるように固定する。reverse process はこの Markov chain を逆向きにたどる learned Gaussian transition で、$p_\theta(\bx_{t-1}|\bx_t)=\mathcal{N}(\bx_{t-1};\bmu_\theta(\bx_t,t),\bSigma_\theta(\bx_t,t))$ とする。

本論文の重要な設計は、$\bmu_\theta$ を直接予測する代わりに、$\bx_t$ を作るときに加えた Gaussian noise $\bepsilon$ を neural network $\bepsilon_\theta(\bx_t,t)$ に予測させることである。この parameterization により、reverse process の平均は $\bepsilon_\theta$ から計算でき、variational bound の $L_{t-1}$ は $\|\bepsilon-\bepsilon_\theta(\cdot,t)\|^2$ に係数を掛けた形へ変形される。実験で用いる $L_\mathrm{simple}$ は、その係数を外した単純な MSE objective である。

### 重要な定義・数式

根拠となる式番号は `main.tex` の `eq:forwardprocess`, `eq:q_marginal_arbitrary_t`, `eq:vb`, `eq:mu_func_approx_langevin`, `eq:training_objective_simple` と Algorithm 2 である。

$$
\begin{aligned}
p_\theta(\bx_{0:T}) &\defeq p(\bx_T)\prod_{t=1}^T p_\theta(\bx_{t-1}|\bx_t), \qquad
p_\theta(\bx_{t-1}|\bx_t) \defeq \mathcal{N}(\bx_{t-1}; \bmu_\theta(\bx_t,t), \bSigma_\theta(\bx_t,t)), \\
q(\bx_{1:T}|\bx_0) &\defeq\prod_{t=1}^T q(\bx_t|\bx_{t-1}), \qquad
q(\bx_t|\bx_{t-1})\defeq \mathcal{N}(\bx_t;\sqrt{1-\beta_t}\bx_{t-1},\beta_t\bI).
\end{aligned}
$$

**式の意味**: 上段は学習する reverse process、下段は固定する forward process を定義している。reverse process は $\bx_T$ から $\bx_0$ へ戻る learned Gaussian Markov chain で、forward process はデータに少しずつ Gaussian noise を加える Markov chain である。

**記号の定義**:
- $\bx_0$ ... データ画像
- $\bx_1,\dotsc,\bx_T$ ... データと同じ次元を持つ latent variables
- $p(\bx_T)=\mathcal{N}(\bx_T;\bzero,\bI)$ ... reverse process の出発 prior
- $p_\theta(\bx_{t-1}|\bx_t)$ ... 学習する逆向き transition
- $q(\bx_t|\bx_{t-1})$ ... 固定した forward transition
- $\beta_t$ ... timestep $t$ の forward process variance
- $\bmu_\theta,\bSigma_\theta$ ... neural network parameter $\theta$ による reverse Gaussian の平均と分散

**この論文での役割**: diffusion model 全体の確率モデル定義である。以後の variational bound、$\bepsilon$-prediction、sampling algorithm はすべてこの forward/reverse process の組をどう学習・実行するかとして導かれる。

$$
q(\bx_t|\bx_0)=\mathcal{N}(\bx_t;\sqrt{\bar\alpha_t}\bx_0,(1-\bar\alpha_t)\bI),
\qquad
\bx_t=\sqrt{\bar\alpha_t}\bx_0+\sqrt{1-\bar\alpha_t}\bepsilon,\quad \bepsilon\sim\mathcal{N}(\bzero,\bI)
$$

**式の意味**: 1 step ずつ forward process を回さなくても、任意の timestep $t$ の noisy image $\bx_t$ を $\bx_0$ から直接サンプルできることを示す。

**記号の定義**:
- $\alpha_t \defeq 1-\beta_t$ ... 1 step で元信号を残す係数
- $\bar\alpha_t \defeq \prod_{s=1}^t\alpha_s$ ... $t$ step 後までの累積係数
- $\bepsilon$ ... standard normal noise
- $\sqrt{\bar\alpha_t}\bx_0$ ... $t$ step 後に残る元画像成分
- $\sqrt{1-\bar\alpha_t}\bepsilon$ ... $t$ step 後に加わった noise 成分

**この論文での役割**: Algorithm 1 の training で、毎回 $t\sim\mathrm{Uniform}(\{1,\dotsc,T\})$ を選んで noisy input を直接作るために使う。$\bepsilon$-prediction objective もこの再パラメータ化から書ける。

$$
\mathbb{E}_q\!\left[
\underbrace{\kl{q(\bx_T|\bx_0)}{p(\bx_T)}}_{L_T}
+\sum_{t>1}\underbrace{\kl{q(\bx_{t-1}|\bx_t,\bx_0)}{p_\theta(\bx_{t-1}|\bx_t)}}_{L_{t-1}}
\underbrace{-\log p_\theta(\bx_0|\bx_1)}_{L_0}
\right]
$$

**式の意味**: negative log likelihood の variational bound を、最終 latent の prior matching $L_T$、中間 step の Gaussian KL $L_{t-1}$、最後の discrete decoder 項 $L_0$ に分解している。

**記号の定義**:
- $L_T$ ... $q(\bx_T|\bx_0)$ が prior $p(\bx_T)$ に近いかを見る KL
- $L_{t-1}$ ... true posterior $q(\bx_{t-1}|\bx_t,\bx_0)$ と learned reverse transition の KL
- $L_0$ ... $\bx_1$ から discrete data $\bx_0$ を復元する decoder の negative log likelihood
- $\kl{\cdot}{\cdot}$ ... KL divergence
- $p_\theta(\bx_0|\bx_1)$ ... 最終 step の discrete decoder

**この論文での役割**: diffusion model の訓練目的の出発点である。論文はこの bound を理論的な基盤にしつつ、実際の高品質生成には重みを外した $L_\mathrm{simple}$ がよいことを ablation で示す。

$$
\begin{aligned}
\bmu_\theta(\bx_t,t)
&=\frac{1}{\sqrt{\alpha_t}}\left(
\bx_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\bepsilon_\theta(\bx_t,t)
\right), \\
L_\mathrm{simple}(\theta)
&\defeq
\mathbb{E}_{t,\bx_0,\bepsilon}
\left[
\left\|
\bepsilon-\bepsilon_\theta(\sqrt{\bar\alpha_t}\bx_0+\sqrt{1-\bar\alpha_t}\bepsilon,t)
\right\|^2
\right].
\end{aligned}
$$

**式の意味**: 1 つ目の式は、network が予測する $\bepsilon_\theta$ から reverse Gaussian の平均 $\bmu_\theta$ を計算する parameterization である。2 つ目の式は、実際に使う simplified objective で、加えた noise $\bepsilon$ と予測 noise $\bepsilon_\theta$ の MSE を最小化する。

**記号の定義**:
- $\bepsilon_\theta(\bx_t,t)$ ... noisy image $\bx_t$ と timestep $t$ から noise を予測する neural network
- $\bmu_\theta(\bx_t,t)$ ... reverse transition の平均
- $L_\mathrm{simple}$ ... TeX の `eq:training_objective_simple` で定義される非加重 MSE objective
- $t$ ... $1$ から $T$ の一様分布から選ばれる timestep
- $\theta$ ... neural network の学習パラメータ

**この論文での役割**: 本論文の主な設計選択である。TeX の Table 2 では、$\bepsilon$ prediction と $L_\mathrm{simple}$ の組み合わせが CIFAR10 FID 3.17 を達成し、variational bound の固定分散版 FID 13.51 より大きく改善している。

$$
\bx_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}\left(
\bx_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\bepsilon_\theta(\bx_t,t)
\right)
+\sigma_t\bz,
\qquad
\bz\sim\mathcal{N}(\bzero,\bI)
$$

**式の意味**: Algorithm 2 の sampling 1 step である。現在の noisy sample $\bx_t$ から network が予測した noise 成分を引き、固定分散 $\sigma_t^2$ に対応する stochastic noise を加えて $\bx_{t-1}$ を得る。

**記号の定義**:
- $\bx_t$ ... 現在の reverse process state
- $\bx_{t-1}$ ... 1 step 後に少しデータ側へ戻った state
- $\sigma_t$ ... reverse process の固定 standard deviation
- $\bz$ ... sampling 時に加える standard normal noise。Algorithm 2 では $t>1$ ならサンプルし、$t=1$ では $\bz=\bzero$
- $1-\alpha_t=\beta_t$ ... forward process の timestep variance

**この論文での役割**: 生成時の実行式である。$\bx_T\sim\mathcal{N}(\bzero,\bI)$ から始め、$t=T,\dotsc,1$ の順にこの式を適用して $\bx_0$ を返す。

### 実装 / アルゴリズム上の要点

- **Training Algorithm 1**: $\bx_0\sim q(\bx_0)$、$t\sim\mathrm{Uniform}(\{1,\dotsc,T\})$、$\bepsilon\sim\mathcal{N}(\bzero,\bI)$ をサンプルし、$\|\bepsilon-\bepsilon_\theta(\sqrt{\bar\alpha_t}\bx_0+\sqrt{1-\bar\alpha_t}\bepsilon,t)\|^2$ の gradient descent step を行う。
- **Sampling Algorithm 2**: $\bx_T\sim\mathcal{N}(\bzero,\bI)$ から始め、$t=T,\dotsc,1$ の順に update する。$t>1$ では $\bz\sim\mathcal{N}(\bzero,\bI)$、最後だけ $\bz=\bzero$ とする。
- **Variance の扱い**: $\bSigma_\theta(\bx_t,t)=\sigma_t^2\bI$ とし、学習しない time-dependent constants に固定する。TeX では $\sigma_t^2=\beta_t$ と $\sigma_t^2=\tilde\beta_t$ が似た結果だったと述べる。
- **Data scaling と decoder**: 画像を $[-1,1]$ にスケーリングし、$L_0$ では Gaussian から導かれる independent discrete decoder を使う。より強い conditional autoregressive decoder を入れることは "straightforward" だが future work とされる。
- **Architecture**: PixelCNN++ 風の U-Net backbone、group normalization、Transformer sinusoidal position embedding による timestep 条件付け、$16\times16$ feature map resolution で self-attention を使う。CIFAR10 model は 35.7M parameters、LSUN/CelebA-HQ は 114M parameters、large LSUN Bedroom は約 256M parameters。
- **Training details**: TPU v3-8 を使用。CIFAR10 は batch size 128、800k steps、21 steps/sec、10.6 hours。$256^2$ 画像は batch size 64、2.2 steps/sec。CelebA-HQ 0.5M steps、LSUN Bedroom 2.4M、LSUN Cat 1.8M、LSUN Church 1.2M、large LSUN Bedroom 1.15M steps。
- **Hyperparameters**: Adam、learning rate $2\times10^{-4}$、$256\times256$ 画像では $2\times10^{-5}$。EMA decay 0.9999。CIFAR10 dropout 0.1、他 dataset は dropout 0。random horizontal flips は CIFAR10 で使い、他 dataset でも LSUN Bedroom を除いて使う。
- **Sampling cost**: TeX の experimental details では、CIFAR10 で 256 images の sampling が 17 seconds、CelebA-HQ/LSUN $256^2$ で 128 images が 300 seconds と報告されている。

## 実験・結果

- **データセット / ベンチマーク**: CIFAR10 unconditional、CelebA-HQ $256\times256$、LSUN Bedroom / Church / Cat $256\times256$。CIFAR10 と CelebA-HQ は TensorFlow Datasets、LSUN は StyleGAN のコードで準備したと付録に書かれている。
- **比較対象 / baseline**: CIFAR10 Table 1 では conditional baseline として EBM、JEM、BigGAN、StyleGAN2 + ADA (v1)、unconditional baseline として Diffusion (original)、Gated PixelCNN、Sparse Transformer、PixelIQN、EBM、NCSNv2、NCSN、SNGAN、SNGAN-DDLS、StyleGAN2 + ADA (v1) を載せる。LSUN Table 3 では ProgressiveGAN、StyleGAN、StyleGAN2 と比較する。
- **指標**: Inception Score (IS)、FID、negative log likelihood (NLL, bits/dim)、rate-distortion の rate (bits/dim) と distortion (RMSE on $[0,255]$ scale)。
- **主な結果**:
  - CIFAR10 unconditional の `Ours ($L_\mathrm{simple}$)` は IS $9.46\pm0.11$、FID $3.17$、NLL test $\leq3.75$、train 3.72。FID は training set に対して計算され、test set に対する FID は 5.24 と本文で補足されている。
  - 同じ CIFAR10 で `Ours ($L$, fixed isotropic $\bSigma$)` は IS $7.67\pm0.13$、FID 13.51、NLL test $\leq3.70$、train 3.69。TeX は、true variational bound は codelength をよくするが、sample quality は simplified objective が最良だと述べる。
  - CIFAR10 baseline の例として、StyleGAN2 + ADA (v1) unconditional は IS $9.74\pm0.05$、FID 3.26、Gated PixelCNN は IS 4.60、FID 65.93、NLL 3.03 (2.90)、Sparse Transformer は NLL 2.80、NCSN は IS $8.87\pm0.12$、FID 25.32。
  - Table 2 の ablation では、$\tilde\bmu$ prediction + $L$ learned diagonal $\bSigma$ が IS $7.28\pm0.10$ / FID 23.69、$\tilde\bmu$ prediction + $L$ fixed isotropic $\bSigma$ が IS $8.06\pm0.09$ / FID 13.22、$\bepsilon$ prediction + $L$ fixed isotropic $\bSigma$ が IS $7.67\pm0.13$ / FID 13.51、$\bepsilon$ prediction + $L_\mathrm{simple}$ が IS $9.46\pm0.11$ / FID 3.17。blank entries は "unstable to train and generated poor samples with out-of-range scores" と説明される。
  - LSUN Table 3 では、Ours ($L_\mathrm{simple}$) が Bedroom FID 6.36、Church 7.89、Cat 19.75、Ours large が Bedroom 4.90。ProgressiveGAN は Bedroom 8.34、Church 6.42、Cat 37.52。StyleGAN は Bedroom 2.65、Church 4.21、Cat 8.53。StyleGAN2 は Church 3.86、Cat 6.93。
  - CelebA-HQ $256\times256$ は Fig. 1 と appendix の samples / nearest neighbors で定性的に示される。TeX 中に CelebA-HQ の FID 数値比較は明示されていない。
  - Rate-distortion 解析では、CIFAR10 最高サンプル品質モデルについて rate 1.78 bits/dim、distortion 1.97 bits/dim、RMSE 0.95 on a 0-255 scale と報告される。本文は "More than half of the lossless codelength describes imperceptible distortions." と述べる。
  - Progressive generation では、$\hat\bx_0=(\bx_t-\sqrt{1-\bar\alpha_t}\bepsilon_\theta(\bx_t))/\sqrt{\bar\alpha_t}$ を reverse process の途中で見ると、大域的特徴が先に現れ、細部が後に現れると説明される。
  - CelebA-HQ interpolation では、$t=500$ の補間が pose、skin tone、hairstyle、expression、background などを滑らかに変える一方、eyewear は滑らかに変わらないと本文で述べられる。
- **著者が主張する貢献**:
  - diffusion model と denoising score matching over multiple noise levels、annealed Langevin dynamics の関係を $\bepsilon$-prediction parameterization によって明示した。
  - $L_\mathrm{simple}$ が sample quality を大きく改善することを Table 2 で示した。
  - diffusion model を progressive lossy decompression、さらに autoregressive decoding の一般化として解釈した。
  - CIFAR10 unconditional で FID 3.17、LSUN $256\times256$ で ProgressiveGAN と同程度の sample quality、公開実装を示した。

## 妥当性と限界

- **この主張を支える根拠**:
  - 高品質生成の主張は、CIFAR10 Table 1 の IS/FID/NLL、LSUN Table 3 の FID、Fig. 1 と appendix の生成サンプルで支えられる。
  - $\bepsilon$-prediction と $L_\mathrm{simple}$ の有効性は、Table 2 の parameterization / objective ablation で支えられる。特に $\bepsilon$ prediction + $L$ fixed isotropic $\bSigma$ は FID 13.51 で、同じ $\bepsilon$ prediction でも $L_\mathrm{simple}$ は FID 3.17 になる。
  - overfitting については、CIFAR10 の train/test codelength gap が at most 0.03 bits/dim と本文で述べられ、appendix には nearest neighbor visualizations がある。
  - likelihood と perceptual quality のずれは、CIFAR10 の rate-distortion 分析で説明される。lossless codelength の多くが perceptually imperceptible な細部に使われるという著者の解釈である。
- **著者が認めている limitations / future work**:
  - 本モデルの lossless codelength は other likelihood-based generative models と競合しない。CIFAR10 では Sparse Transformer NLL 2.80、Gated PixelCNN 3.03 に対し、Ours $L_\mathrm{simple}$ は $\leq3.75$。
  - Progressive compression は proof of concept であり、Algorithm 3/4 は minimal random coding のような高次元では tractable でない手続きを仮定する。appendix は "not yet as a practical compression system" と明記する。
  - learned diagonal $\bSigma_\theta(\bx_t)$ を variational bound に入れると unstable training や poorer sample quality につながると Table 2 と本文で述べる。
  - $p_\theta(\bx_0|\bx_1)$ により強い conditional autoregressive decoder を組み込むことは可能だが future work とされる。
  - CelebA-HQ interpolation では、著者自身が pose、skin tone、hairstyle、expression、background は滑らかに変わるが eyewear は滑らかに変わらないと述べている。
  - conclusion では、other data modalities や他の generative models / machine learning systems の component としての utility を今後調べたいと述べる。
  - Broader Impact では、fake images/videos など悪用可能性と、internet 由来 dataset の bias が生成物に反映・増幅される可能性に触れる。
- **読者として注意すべき点**:
  - CIFAR10 FID 3.17 は training set FID として計算されており、本文の test set FID は 5.24 である。これは TeX が "as is standard practice" と説明するが、比較時には基準をそろえる必要がある。
  - Final experiments は trained once で、sample quality scores と log likelihood は training 中の minimum FID value の時点で報告されている。これは付録の experimental details に書かれている評価設計である。
  - Hyperparameter search は主に CIFAR10 sample quality に対して行われ、他 dataset に transfer された。$T=1000$ は "without a sweep"、learning rate、batch size、EMA decay も sweep していないと付録にある。
  - Abstract の "On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN." は LSUN 全体の要約であり、Table 3 では Bedroom と Cat は ProgressiveGAN より良いが、Church は ProgressiveGAN 6.42 に対して Ours 7.89 で悪い。
  - CelebA-HQ は画像例と nearest neighbor 図が中心で、TeX 中に FID などの定量比較は明示されていない。
  - Sampling は $T=1000$ 回の network evaluation を伴う。TeX では CIFAR10 256 images が 17 seconds、$256^2$ 128 images が 300 seconds と報告されるが、これを limitation として強く議論しているわけではない。
- **追加で確認したい実験 / 疑問**:
  - $T$、$\beta_t$ schedule、$\sigma_t^2=\beta_t$ vs $\tilde\beta_t$ の体系的 sweep と、FID/NLL/sampling time の Pareto。
  - $L_\mathrm{simple}$ と true variational bound の中間にある timestep-dependent weighting を探索した場合の sample quality と codelength の関係。
  - $\bx_0$ prediction は early experiments で worse sample quality と述べられるだけなので、Table 2 と同じ条件での体系的比較。
  - CelebA-HQ での定量 FID、また mode coverage / recall 系の評価。
  - 強い $L_0$ decoder を入れた場合に、NLL と sample quality の tradeoff がどう変わるか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に整理する。

- **diffusion probabilistic model / diffusion model**: 固定 forward process と学習 reverse process を持つ latent variable model。$\bx_1,\dotsc,\bx_T$ はデータと同じ dimensionality を持つ。
- **forward process / diffusion process $q$**: $\bx_0$ から $\bx_T$ へ Gaussian noise を徐々に加える固定 Markov chain。実験では $\beta_t$ を学習せず定数にする。
- **reverse process $p_\theta$**: $\bx_T\sim\mathcal{N}(\bzero,\bI)$ から $\bx_0$ へ戻る learned Markov chain。transition は Gaussian。
- **variance schedule $\beta_t$**: forward process の noise variance。実験では $10^{-4}$ から $0.02$ まで線形。
- **$\alpha_t,\bar\alpha_t$**: $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$。任意の $t$ の noisy image を閉形式で書くための係数。
- **$\tilde\bmu_t,\tilde\beta_t$**: forward process posterior $q(\bx_{t-1}|\bx_t,\bx_0)$ の平均と分散。Gaussian KL を閉形式にするために使う。
- **$\tilde\bmu$ prediction**: reverse mean を forward posterior mean に合わせて予測する baseline parameterization。
- **$\bepsilon$ prediction**: noisy image $\bx_t$ に入っている Gaussian noise $\bepsilon$ を予測する本論文の parameterization。$\bmu_\theta$ は $\bepsilon_\theta$ から計算する。
- **$L$ / variational bound**: negative log likelihood の上界として最小化する objective。Gaussian KL の和として書ける。
- **$L_\mathrm{simple}$**: $\bepsilon$ と $\bepsilon_\theta$ の非加重 MSE。true variational bound の weighting を捨てるが、sample quality が最良になる。
- **$L_T,L_{t-1},L_0$**: variational bound の項。$L_T$ は prior matching、$L_{t-1}$ は中間 reverse transition、$L_0$ は discrete decoder。
- **denoising score matching**: 本論文では、$\bepsilon$-prediction objective が複数 noise scale の denoising score matching に似るという関係で出てくる。
- **annealed Langevin dynamics**: sampling update がこれに似ると説明される。NCSN との接続を読むときの前提語。
- **NCSN / NCSNv2**: Song and Ermon の score-based generative model。TeX の extended related work では、architecture、data scaling、prior との matching、sampler coefficients の扱いが本論文と異なると説明される。
- **FID**: TeX では CIFAR10 と LSUN の sample quality score として報告される。Table 1/3 と本文では、小さい値を良い sample quality として比較している。
- **IS**: Inception Score。TeX では CIFAR10 の sample quality score として報告され、Table 1/2 では大きい値を良い sample quality として比較している。
- **NLL / bits per dimension**: TeX では negative log likelihoods (lossless codelengths) として報告される。小さいほど短い codelength だが、本論文の主張では sample quality と必ずしも一致しない。
- **rate-distortion**: $L_1+\dotsc+L_T$ を rate、$L_0$ を distortion として、lossless codelength がどの程度 perceptual detail に使われるかを見る分析。
- **progressive lossy decompression**: reverse process の途中の $\bx_t$ から $\hat\bx_0$ を推定し、粗い構造から細部へ復元されると解釈する枠組み。
- **generalized autoregressive decoding**: forward process を座標 mask として特殊化すると autoregressive model に対応するため、Gaussian diffusion を通常の座標順序では表せない generalized bit ordering と見る解釈。

## 読む順番の提案

- まず Abstract と Introduction を読み、論文が「diffusion model で高品質 sample を示す」「$\bepsilon$-prediction が score matching / Langevin dynamics とつながる」「NLL は競合しないが rate-distortion で説明する」という三つの軸を持つことを確認する。正規ノートでは `Summary（著者の主張）` の問題・手法・結果に対応する。
- 次に Background の `eq:forwardprocess`, `eq:q_marginal_arbitrary_t`, `eq:vb`, `eq:q_posterior_mean_var` を読む。ここが正規ノートの forward posterior、ELBO 分解、$L_T,L_{t-1},L_0$ の記述につながる。
- Method の `Reverse process and L_{1:T-1}` で `eq:mu_func_approx_langevin` と `eq:vb_term_langevin_eps` を読む。ここが「平均ではなくノイズを予測する」理由と、denoising score matching との接続の中心である。
- `Simplified training objective` と Algorithm 1/2 を読む。$L_\mathrm{simple}$ が true variational bound の weighting を捨てていること、sampling update が $\bepsilon_\theta$ から実行されることを確認する。
- Experiments の Table 1 と Table 2 を先に見る。CIFAR10 の FID 3.17、NLL $\leq3.75$、ablation の FID 13.51 から 3.17 への差が、正規ノートの Takeaway と Critical Thoughts の根拠になる。
- Appendix の LSUN Table 3、Experimental details、Progressive compression paragraph を読む。LSUN の dataset 別の差、model size / training cost、compression 解釈が proof of concept であることが分かる。
- 最後に Related Work と `main.bbl` を見て、Sohl-Dickstein et al. (2015)、Song and Ermon (2019, 2020)、Vincent (2011)、PixelCNN++、U-Net、Group Normalization、Transformer、ProgressiveGAN / StyleGAN / StyleGAN2 への参照関係を整理する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2006.11239v2/`
- 正規ノート: `notes/arXiv-2006.11239v2.md`
