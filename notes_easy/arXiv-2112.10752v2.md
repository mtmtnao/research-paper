# High-Resolution Image Synthesis with Latent Diffusion Models（高解像度画像生成における潜在拡散モデルの提案）

- arXiv: https://arxiv.org/abs/2112.10752
- 一次ソース: ../papers/arXiv-2112.10752v2/
- 正規ノート: ../notes/arXiv-2112.10752v2.md

---

## 一言で言うと

この論文は、拡散モデルを RGB ピクセル空間ではなく、事前学習した autoencoder の潜在空間で学習する Latent Diffusion Models (LDMs) を提案する。著者は、perceptual compression と generative modeling を分離することで、画像品質と柔軟な条件付けを保ちながら、pixel-based DMs より訓練・推論コストを下げられると主張する（abstract, Sec. 1）。

## 何を議論する論文か

- **問題設定**: diffusion models (DMs) は高品質な画像生成を達成する一方、通常は pixel space で動くため、訓練に「150 - 1000 V100 days」、50k samples の生成に single A100 で約 5 日かかる例がある（Sec. 1, "Democratizing High-Resolution Image Synthesis"）。この計算量が高解像度画像生成の利用可能性を制限している。
- **対象範囲 / 仮定**: 画像を一度 autoencoder で潜在表現 $z$ に写し、その 2D 構造を保った潜在空間で UNet 型の diffusion model を学習する。第一段の autoencoder は固定して再利用し、後段の DM は $p(z)$ または $p(z \vert y)$ を学習する。
- **既存研究との差分**: VQ-VAE / VQGAN + autoregressive transformer / DALL-E 系のような two-stage 画像生成は、AR modeling を feasible にするため強い空間圧縮や 1D ordering に依存しがちだと著者は述べる。LDM は convolutional UNet を潜在空間で使うため、より mild な compression を選べる、という位置づけである（Sec. 2, Sec. 3.1）。
- **この論文で答えたい問い**: 「pixel-based DM の品質・mode-covering な性質・条件付けの柔軟性を保ちつつ、perceptually equivalent で低次元な空間へ移して、訓練と sampling のコストを下げられるか」を、無条件生成、class-conditional ImageNet、text-to-image、layout-to-image、super-resolution、inpainting で検証する。

## 背景と前提

- Diffusion models は、ノイズを加える forward process と、その逆向きの denoising process を学習する生成モデルである。この論文では、Ho et al. の reweighted objective に基づく $\epsilon$-prediction 形式を使う（Eq. \ref{eq:dmloss}, supplementary Sec. "Detailed Information on Denoising Diffusion Models"）。
- 著者は likelihood-based models の mode-covering な性質が、知覚的には重要でない high-frequency details に容量と計算を使わせる、と述べる。Fig. \ref{fig:perceptualcompression} は、perceptual compression と semantic compression を分けて考える動機づけである。
- 第一段の autoencoder は、LPIPS などの perceptual loss と patch-based adversarial objective で学習される。これは pixel-space の $L_1/L_2$ だけに頼ると blur が起きるため、local realism を強制するためである（Sec. 3.1, supplementary Sec. "Details on Autoencoder Models"）。
- この論文での latent space は、単なる 1D code sequence ではなく、$z \in \mathbb{R}^{h \times w \times c}$ の 2D spatial structure を持つ表現である。この構造を UNet の convolutional inductive bias と合わせることが、VQGAN+Transformer との差分として強調される。
- baseline には、pixel-space diffusion の ADM / ADM-G、autoregressive / transformer 系の VQGAN+T, DALL-E, ImageBART, CogView, Make-A-Scene、GAN 系の BigGAN-deep, StyleGAN, StyleGAN2, ProjectedGAN, LAFITE、super-resolution の SR3、inpainting の LaMa / CoModGAN などが含まれる。

## 提案手法

### コアアイデア

LDM は二段階の設計である。第一段では、画像 $x$ を encoder $\mathcal{E}$ で潜在 $z$ に変換し、decoder $\mathcal{D}$ で再構成する autoencoder を学習する。第二段では、凍結した $\mathcal{E}$ から得る $z$ 上で diffusion model を学習し、sampling 後に $\mathcal{D}$ を 1 回通して画像へ戻す。

この分離により、DM は high-dimensional RGB image space ではなく、知覚的に重要な情報が残った低次元空間で denoising を学ぶ。著者はこれを、perceptual compression を第一段で処理し、semantic / conceptual composition を後段の generative model に担わせる構成として説明している（Sec. 1 "Departure to Latent Space", Sec. 3）。

条件付き生成では、低解像度画像や mask など spatially aligned な条件は concatenation で扱える。一方、text prompts や layout など token-based な条件には、domain-specific encoder $\tau_\theta$ と UNet 内の cross-attention を使う。これにより、class, text, layout を同一の条件付け枠組みに入れる（Sec. 3.3）。

### 重要な定義・数式

$$
z=\mathcal{E}(x), \qquad \tilde{x}=\mathcal{D}(z)=\mathcal{D}(\mathcal{E}(x)), \qquad
f = H/h = W/w
$$

**式の意味**: 画像 $x$ を encoder で潜在表現 $z$ に写し、decoder で再構成画像 $\tilde{x}$ を得る定義である。$f$ は画像の空間サイズをどれだけ downsample したかを表す。

**記号の定義**:
- $x \in \mathbb{R}^{H \times W \times 3}$ ... RGB 空間の入力画像
- $\mathcal{E}$ ... autoencoder の encoder
- $\mathcal{D}$ ... autoencoder の decoder
- $z \in \mathbb{R}^{h \times w \times c}$ ... 2D 構造を持つ潜在表現
- $f$ ... downsampling factor。本文では $f=2^m, m \in \mathbb{N}$ を調べる

**この論文での役割**: LDM が pixel space ではなく latent space で DM を学習するための基本定義である。$f$ の選択が、計算効率と再構成 fidelity の tradeoff を決める。

$$
L_{\text{Autoencoder}} =
\min_{\mathcal{E}, \mathcal{D}} \max_\psi
\Big(
L_{rec}(x, \mathcal{D}(\mathcal{E}(x)))
- L_{adv}(\mathcal{D}(\mathcal{E}(x)))
+ \log D_\psi(x)
+ L_{reg}(x; \mathcal{E}, \mathcal{D})
\Big)
$$

**式の意味**: 第一段の autoencoder を、再構成損失、adversarial loss、discriminator 項、latent regularization を組み合わせて学習する目的関数である（Eq. \ref{eq:firststageloss}）。

**記号の定義**:
- $L_{rec}$ ... perceptual reconstruction を含む再構成損失
- $L_{adv}$ ... reconstructions に対する adversarial objective
- $D_\psi$ ... patch-based discriminator
- $L_{reg}$ ... latent space の分散や正則性を制御する項
- $\psi$ ... discriminator のパラメータ

**この論文での役割**: LDM の品質上限は第一段の再構成能力に依存するため、この autoencoder が「perceptually equivalent, but computationally more suitable space」を作る土台になる。著者は KL-reg. と VQ-reg. の 2 種類を調べる。

$$
L_{LDM} :=
\mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}
\Big[
\lVert \epsilon - \epsilon_\theta(z_t,t) \rVert_2^2
\Big]
$$

**式の意味**: 通常の DM の $\epsilon$-prediction 損失を、画像 $x_t$ ではなく latent $z_t$ に対して適用したもの。Eq. \ref{eq:ldmloss} に対応する。

**記号の定義**:
- $z_t$ ... latent $z=\mathcal{E}(x)$ に diffusion forward process でノイズを加えたもの
- $\epsilon$ ... $\mathcal{N}(0,1)$ からサンプルされるノイズ
- $t$ ... diffusion timestep。本文では Markov chain length $T$ の中から選ばれる
- $\epsilon_\theta$ ... time-conditional UNet で実装される denoising model

**この論文での役割**: 提案手法の中心であり、pixel-based DM から latent diffusion へ移る箇所である。低次元な $z$ 上でこの損失を最適化することが、訓練・sampling の効率化につながる。

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)\cdot V,
\qquad
Q=W_Q^{(i)}\cdot \varphi_i(z_t),\;
K=W_K^{(i)}\cdot \tau_\theta(y),\;
V=W_V^{(i)}\cdot \tau_\theta(y)
$$

**式の意味**: UNet の中間表現を query、条件入力 $y$ の表現を key/value として、条件情報を denoising network に注入する cross-attention である（Sec. 3.3）。

**記号の定義**:
- $Q,K,V$ ... attention の query, key, value
- $y$ ... text prompt や layout などの条件入力
- $\varphi_i(z_t)$ ... UNet の $i$ 番目の中間表現
- $\tau_\theta(y)$ ... text prompt や layout などを中間表現へ写す domain-specific encoder
- $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ ... learnable projection matrices
- $d$ ... attention の scaling に使う次元

**この論文での役割**: LDM を class, text, layout などの general conditioning inputs に対応させるための中核機構である。text-to-image では BERT-tokenizer と Transformer による $\tau_\theta$ を使う。

$$
L_{LDM} :=
\mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t}
\Big[
\lVert \epsilon - \epsilon_\theta(z_t,t,\tau_\theta(y)) \rVert_2^2
\Big]
$$

**式の意味**: 条件 $y$ を持つ画像ペアから、conditional LDM を学習する目的関数である。Eq. \ref{eq:cond_loss} に対応する。

**記号の定義**:
- $y$ ... text prompt, semantic map, class label, bounding boxes, low-resolution image などの条件入力
- $z_t$ ... latent $z=\mathcal{E}(x)$ に diffusion forward process でノイズを加えたもの
- $\epsilon$ ... $\mathcal{N}(0,1)$ からサンプルされるノイズ
- $t$ ... diffusion timestep
- $\tau_\theta$ ... 条件入力を UNet に渡す表現へ変換する encoder
- $\epsilon_\theta(z_t,t,\tau_\theta(y))$ ... 条件情報を受け取る denoising network
- $\mathbb{E}$ ... training data, noise, timestep に関する期待値

**この論文での役割**: text-to-image, layout-to-image, class-conditional ImageNet など、条件付きタスクの学習を統一する式である。本文では $\tau_\theta$ と $\epsilon_\theta$ は Eq. \ref{eq:cond_loss} で jointly optimized されると説明される。

### 実装 / アルゴリズム上の要点

- 第一段の autoencoder は OpenImages で学習され、ImageNet-Val で reconstruction metrics を評価する（Tab. \ref{tab:firststagetablecomplete}）。$f=4$ VQ-reg. は R-FID 0.58 / PSNR 27.43、$f=4$ KL-reg. は R-FID 0.27 / PSNR 27.53。
- KL-reg. は learned latent に standard normal への弱い KL penalty を課す。supplement では KL 項の重みをおよそ $10^{-6}$ と書いている。VQ-reg. は decoder 内に vector quantization layer を使い、DM 訓練時には quantization 前の $z$ を取り、quantization operation を decoder に吸収する。
- DM の backbone は time-conditional UNet。ImageNet の compression tradeoff 実験では $f\in\{1,2,4,8,16,32\}$ を比較し、diffusion steps は 1000、noise schedule は linear（Tab. \ref{tab:cin_hyperparams}）。
- text-to-image では 1.45B parameter の KL-regularized LDM を LAION-400M で学習し、BERT-tokenizer と Transformer による $\tau_\theta$ を multi-head cross-attention で UNet に接続する（Sec. 4.3.1, Tab. \ref{tab:cond_hyperparams}）。
- spatially aligned conditioning では concatenation を使う。super-resolution では low-resolution conditioning $y$ を UNet 入力に concatenate し、inpainting でも mask 付き入力に基づいて conditional image generation として扱う。
- convolutional sampling により、semantic synthesis, super-resolution, inpainting では訓練時より大きい $512^2$ から $1024^2$ 程度の画像を生成できると著者は述べる。ただし latent の variance が誘導する signal-to-noise ratio が結果に影響し、KL-reg. latent では component-wise standard deviation による rescaling を用いる（Sec. 4.3.2, supplementary Sec. \ref{suppsec:rescale}）。

## 実験・結果

- **データセット / ベンチマーク**: autoencoder は OpenImages で訓練し ImageNet-Val で評価。生成タスクでは CelebA-HQ, FFHQ, LSUN-Churches, LSUN-Bedrooms, ImageNet, MS-COCO, OpenImages, COCO, Landscapes, Places を使う。super-resolution は ImageNet $\times 4$、inpainting は Places の $512\times512$ crops を使う。
- **比較対象 / baseline**: ADM / ADM-G, CDM, SR3, VQGAN+T, ImageBART, CogView, GLIDE, Make-A-Scene, BigGAN-deep, StyleGAN / StyleGAN2, ProjectedGAN, LaMa, CoModGAN, RegionWise, DeepFill v2, EdgeConnect など。
- **指標**: FID, Inception Score (IS), Precision, Recall, PSNR, SSIM, LPIPS, throughput, V100-days, user-study preference score。FID と LPIPS は小さいほど良く、IS / Precision / Recall / PSNR / SSIM / throughput は大きいほど良い。
- **主な結果**: compression tradeoff では、ImageNet class-conditional models を同じ single NVIDIA A100、同じ train steps、同程度の parameters で比較し、LDM-4 から LDM-16 が効率と fidelity の balance を取り、LDM-1 と LDM-8 の 2M steps 後の FID 差が 38 と報告される（Sec. 4.1, Fig. \ref{fig:cin_traincourse}）。
- **主な結果**: unconditional 生成では、Tab. \ref{tab:fids} で CelebA-HQ LDM-4 が FID 5.11 / Precision 0.72 / Recall 0.49、FFHQ LDM-4 が FID 4.98 / Precision 0.73 / Recall 0.50、LSUN-Churches LDM-8 が FID 4.02 / Precision 0.64 / Recall 0.52、LSUN-Bedrooms LDM-4 が FID 2.95 / Precision 0.66 / Recall 0.48。LSUN-Bedrooms では ADM の FID 1.90 には届かないが、本文は half parameters と 4-times less train resources を強調する。
- **主な結果**: class-conditional ImageNet では、Tab. \ref{tab:imagenet_main_numbers} の LDM-4 が FID 10.56 / IS 103.49 / Precision 0.71 / Recall 0.62 / 400M params。classifier-free guidance 付き LDM-4-G は FID 3.60 / IS 247.67 / Precision 0.87 / Recall 0.48 / 400M params。ADM-G は FID 4.59 / IS 186.7 / 608M params。
- **主な結果**: compute comparison では、Tab. \ref{tab:compute_vs_fid} で ImageNet LDM-4-G が 271 V100-days、throughput 0.4 samples/sec、FID 3.60。ADM-G は generator 916 + classifier 46 = 962 V100-days、throughput 0.07 samples/sec、FID 4.59。
- **主な結果**: text-to-image では、MS-COCO $256\times256$ 評価で LDM-KL-8 が FID 23.31 / IS 20.03 / 1.45B params、LDM-KL-8-G が FID 12.63 / IS 30.29 / 1.45B params / 250 DDIM steps / c.f.g. $s=1.5$。GLIDE は FID 12.24 / 6B、Make-A-Scene は FID 11.84 / 4B（Tab. \ref{tab:txt2img}）。
- **主な結果**: layout-to-image では、COCO $256\times256$ で LDM-4 が FID 40.91、OpenImages $256\times256$ で 32.02、OpenImages $512\times512$ で 35.80（Tab. \ref{tab:layout2img}）。supplement は OpenImages から COCO へ finetune すると既存手法を上回ると述べる。
- **主な結果**: super-resolution では、ImageNet $\times4$ で LDM-4 100 steps が FID 2.8 / IS 166.3 / PSNR 24.4 / SSIM 0.69 / 169M params、LDM-4 big が FID 2.4 / IS 174.9 / 552M params。SR3 は FID 5.2 / IS 180.1 / PSNR 26.4 / SSIM 0.762 / 625M params（Tab. \ref{tab:srtable}）。著者は、LDM-SR は FID で SR3 を上回るが、SR3 の方が IS は良いと述べる。
- **主な結果**: inpainting では、Places の all samples で LDM-4 big w/ ft が FID 1.50 / LPIPS 0.137、LaMa が FID 2.21 / LPIPS 0.14、CoModGAN が FID 1.82 / LPIPS 0.15（Tab. \ref{inpaintingtable}）。効率比較では LDM-1 の FID@2k が 24.74 に対し、LDM-4 VQ w/ attn は 14.99、sampling throughput @512 は 0.07 から 0.34 samples/sec に上がる（Tab. \ref{inpaintingefficiency}）。
- **著者が主張する貢献**: abstract と Sec. 1 の contributions では、image inpainting と class-conditional image synthesis で new state-of-the-art scores、text-to-image / unconditional / super-resolution で highly competitive performance、cross-attention による general-purpose conditioning、megapixel synthesis への convolutional application、pretrained LDM / autoencoder の公開が主張される。

## 妥当性と限界

- **この主張を支える根拠**: $f\in\{1,2,4,8,16,32\}$ の比較により、pixel DM に相当する LDM-1 は訓練が遅く、LDM-32 は情報損失で fidelity が頭打ちになる、という設計上の tradeoff を Fig. \ref{fig:cin_traincourse} と Fig. \ref{fig:speedplot} で示している。
- **この主張を支える根拠**: Tab. \ref{tab:firststagetablecomplete} は、mild compression の autoencoder が DALL-E $f=8$ や VQGAN $f=16$ より低い R-FID を持つことを示す。たとえば DALL-E $f=8$ は R-FID 32.01、VQGAN $f=16$ は 4.98、著者の $f=4$ VQ-reg. は 0.58。
- **この主張を支える根拠**: unconditional, class-conditional, text-to-image, layout-to-image, super-resolution, inpainting の複数タスクで同じ latent diffusion の枠組みを使い、数値表と qualitative figures を併用して評価している。
- **著者が認めている limitations / future work**: LDM は pixel-based approaches より計算要求を下げるが、sequential sampling process は GAN より遅い。さらに $f=4$ autoencoding models でも reconstruction capability が fine-grained pixel-space accuracy を要するタスクの bottleneck になり得るため、著者は superresolution models がすでに somewhat limited だと述べる（Sec. 5）。
- **著者が認めている limitations / future work**: Societal Impact では manipulated data / misinformation / spam、deep fakes、training data leakage、data bias の再生・増幅が挙げられる。特に adversarial training と likelihood-based objective を組み合わせる two-stage approach が data をどの程度 misrepresent するかは "an important research question" とされる。
- **読者として注意すべき点**: compute comparison は A100 days を V100 days に換算するために A100 vs V100 の $\times 2.2$ speedup を仮定している（supplement Sec. \ref{suppsec:compute2}, Tab. \ref{tab:compute_vs_fid}）。この仮定込みでの比較である。
- **読者として注意すべき点**: FID は実装や preprocessing に敏感で、supplement は ImageNet と LSUN-Bedrooms で torch-fidelity と Nichol and Dhariwal の script がわずかに異なる値を出すと述べ、統一された評価手順の重要性を強調している。
- **読者として注意すべき点**: convolutional sampling による $>256^2$ 生成は主に qualitative figures で示される。高解像度域での包括的な定量比較は TeX 中には多くない。
- **追加で確認したい実験 / 疑問**: 第一段 autoencoder の訓練コストを含めた end-to-end compute comparison、cross-attention 以外の conditioning 機構との直接 ablation、新しいドメインで $f=4$ / $f=8$ が同じように最適か、は TeX 中には明示的な比較が少ない。

## 用語メモ

- **Latent Diffusion Model (LDM)**: 画像空間ではなく、事前学習済み autoencoder の潜在空間 $z$ で diffusion model を学習する本論文のモデルクラス。
- **perceptual compression**: high-frequency で知覚的に重要でない details を落とす段階。著者は第一段 autoencoder にこの役割を担わせる。
- **semantic compression**: データの semantic / conceptual composition を生成モデルが学ぶ段階。著者は LDM の後段 DM がこの部分を担うと説明する。
- **$f$ / LDM-$f$**: autoencoder の downsampling factor。$f=1$ は pixel-based DM に対応し、LDM-4 は $H/h=W/w=4$ の潜在空間で DM を学習するモデルを指す。
- **KL-reg.**: learned latent に standard normal への弱い KL penalty を課す regularization。VAE に似た扱いだが、この論文では高忠実度再構成のため very small regularization を使う。
- **VQ-reg.**: vector quantization layer を decoder 側に含める regularization。DM 訓練時には quantization 前の latent を使い、quantization operation を decoder に吸収する。
- **cross-attention conditioning**: UNet の中間表現を query、条件 encoder $\tau_\theta(y)$ の出力を key/value として条件を注入する仕組み。text, class, layout など token-based 条件に使われる。
- **classifier-free guidance / c.f.g.**: unconditional guidance として本文・表で使われる sampling 時の誘導方法。ImageNet LDM-4-G では $s=1.5$、text-to-image LDM-KL-8-G でも $s=1.5$ が表に載る。
- **convolutional sampling**: convolutional UNet の性質を使い、訓練時より大きな spatial resolution へ適用する sampling。semantic synthesis, super-resolution, inpainting で $512^2$ から $\sim1024^2$ の例が示される。
- **R-FID**: reconstruction FID。autoencoder が再構成した画像と参照画像の分布差を測る指標として Tab. \ref{tab:firststagetablecomplete} で使われる。

## 読む順番の提案

- まず abstract と Sec. 1 の "Democratizing High-Resolution Image Synthesis" / "Departure to Latent Space" を読み、pixel-based DM の計算問題と perceptual / semantic compression の分離を押さえる。正規ノートの Summary 冒頭の問題設定に対応する。
- 次に Fig. \ref{fig:perceptualcompression}, Fig. \ref{fig:firststagecomparison}, Tab. \ref{tab:firststagetablecomplete} を見る。ここが「なぜ latent space でよいのか」を支える第一段の根拠で、正規ノートの "Takeaway" の $f=4,8$ の話につながる。
- Sec. 3.1 から Sec. 3.3 を読み、$z=\mathcal{E}(x)$、Eq. \ref{eq:ldmloss}、cross-attention、Eq. \ref{eq:cond_loss} を順に確認する。正規ノートの「手法」部分と対応する。
- Sec. 4.1 と Fig. \ref{fig:cin_traincourse} / Fig. \ref{fig:speedplot} で、compression factor の sweet spot を読む。その後 Tab. \ref{tab:fids}, Tab. \ref{tab:imagenet_main_numbers}, Tab. \ref{tab:txt2img}, Tab. \ref{tab:srtable}, Tab. \ref{inpaintingtable} の順で結果を見る。
- 最後に Sec. 5 と supplement の evaluation / compute details を読む。正規ノートの Critical Thoughts にある限界、compute comparison、FID 評価手順への注意とつながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2112.10752v2/`
- 正規ノート: `notes/arXiv-2112.10752v2.md`
