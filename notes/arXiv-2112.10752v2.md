# High-Resolution Image Synthesis with Latent Diffusion Models

- arXiv: https://arxiv.org/abs/2112.10752
- source: ../papers/arXiv-2112.10752v2/
- authors: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
- venue / year: CVPR 2022（v2 = arXiv 改訂版）
- tags: [diffusion, generative-models, text-to-image, autoencoder, latent-space, cross-attention]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 拡散モデル (DM) は尤度ベースで mode-covering な代わりに、知覚的にほぼ意味の無い高周波詳細にまで容量を割いてしまう。結果として ADM 級の SOTA DM は学習に 150–1000 V100 日、50k サンプル生成に A100 1枚で 5 日要するなど計算コストが膨大で、ピクセル空間で動かす限り「democratize」できない。一方で VQ-VAE/VQGAN+AR のように highly compressed な離散空間に逃げると、再構成の faithfulness と AR モデルのスケーリング不利のトレードオフに苦しむ。
- **手法**: 二段階構成の Latent Diffusion Model (LDM)。
  1. **第一段 (perceptual compression)**: encoder $\mathcal{E}$ と decoder $\mathcal{D}$ から成るオートエンコーダを LPIPS + patch-based adversarial loss で学習し、画像を空間的因子 $f=H/h\in\{2^m\}$ で縮約した潜在 $z\in\mathbb{R}^{h\times w\times c}$ に落とす。正則化は KL-reg（VAE 風の弱い KL ペナルティ）と VQ-reg（VQGAN の量子化を decoder 側に吸収）の2系統。2D 構造を保ったまま「ほどよい」圧縮率にできる点が VQ-VAE/DALL-E 等との差。
  2. **第二段 (latent diffusion)**: 通常の DDPM 損失 $L = \mathbb{E}_{\mathcal{E}(x),\epsilon,t}\|\epsilon - \epsilon_\theta(z_t,t)\|^2$ を凍結した潜在空間で UNet に対して学習。Markov 長 $T=1000$、linear schedule。
  3. **条件付け**: UNet の中間表現に cross-attention $\text{softmax}(QK^T/\sqrt{d})V$ を挿入。$Q$ は UNet 側、$K,V$ はドメイン特化 encoder $\tau_\theta(y)$ から得る。$\tau_\theta$ はテキストなら BERT-tokenizer + 非マスク Transformer、レイアウトやセマンティクスマップなら spatial にもできる。クラス・テキスト・レイアウト・低解像度・マスクなど任意モダリティに統一的に対応。畳み込み的に評価すれば学習解像度を超えてメガピクセル合成も可能。
- **結果**:
  - **無条件生成 (Tab. tab:fids)**: CelebA-HQ で **FID 5.11**（LDM-4, 500 DDIM）で先行 likelihood/GAN を上回り SOTA。FFHQ FID 4.98、LSUN-Churches 4.02（KL-LDM-8）、LSUN-Bedrooms 2.95。LSUN-Bedrooms では ADM (FID 1.9) には届かないが、**パラメータ半分・学習計算 1/4 (55 vs 232 V100 days)** で接近。
  - **クラス条件 ImageNet (Tab. tab:imagenet_main_numbers)**: LDM-4 FID 10.56、classifier-free guidance ($s=1.5$) 付きの **LDM-4-G で FID 3.60 / IS 247.67**、400M params。ADM-G (FID 4.59, 608M) を上回る。学習計算 271 V100 days vs ADM の 916。
  - **テキスト→画像 (Tab. tab:txt2img, MS-COCO val)**: LDM-KL-8 (1.45B params, LAION-400M 学習, 250 DDIM) で FID 23.31、cfg $s=1.5$ で **FID 12.63 / IS 30.29**。GLIDE (12.24, 6B) や Make-A-Scene (11.84, 4B) と同等で、**パラメータは 1/3〜1/4**。Conceptual Captions では FID 17.01（VQGAN+T 28.86, ImageBART 22.61 を凌ぐ、645M params, 3.9 samples/s）。
  - **超解像 (Tab. tab:srtable, ImageNet ×4)**: LDM-4 100 steps で **FID 2.8** (val split features)、SR3 (FID 5.2, 625M) を大きく上回り、**169M params** に縮小。PSNR/SSIM は image regression に劣るが、user study で pixel-DM 16.0% vs LDM-4 **30.4%** が GT より好まれる。
  - **インペインティング Places (Tab. inpaintingtable, 512²)**: LDM-4 big w/ ft で **FID 1.50, LPIPS 0.137**。LaMa (FID 2.21) や CoModGAN (1.82) を上回り SOTA。ピクセル拡散 (LDM-1) に対して**スループット 2.7× 以上、FID 1.6× 改善**。user study でも LDM-4 が LaMa を 68.1% vs 31.9% で凌ぐ。
  - **圧縮率の sweet spot**: LDM-1（=ピクセル DM）は学習が遅く、LDM-32 は情報損失で頭打ち。LDM-4〜LDM-16 がバランス良く、LDM-1 と LDM-8 で 2M steps 後の FID 差は約 38。
- **貢献**: (i) 二段階圧縮を「universal autoencoder + 軽量 DM」に切り分ける構成で、CelebA-HQ で likelihood ベース SOTA、ImageNet class-cond で ADM-G 超え、テキスト・レイアウト・インペインティング・超解像で SOTA 級。(ii) cross-attention に基づく汎用 conditioning。(iii) LSGM のように encoder と prior を joint training せずに済み、reconstruction vs generative の重み付け問題を回避。(iv) 畳み込み的サンプリングで 1024² 級画像へ汎化。(v) 学習済み LDM/AE を公開し、CLIP guided synthesis などの再利用基盤を提供。

## Takeaway（自分にとっての要点）

- 「**perceptual compression と semantic compression を分離**」というメッセージが核。実用上 $f=4, 8$ が sweet spot で、$f=4$ なら R-FID 0.58（VQ）/0.27（KL）と reconstruction はほぼ無損失。これがあるので latent 側の DM は high-frequency の細部に容量を割かなくて済む。
- text-to-image を 1.45B params、A100 1枚で動くレベルで COCO FID 12 台まで持っていけたのは「**潜在を浅く取って DM を厚くする**」設計の勝利。後続の Stable Diffusion はまさにこれを動かしている。
- cross-attention は単なる実装詳細ではなく、「**条件モダリティを Q/K/V 行列だけで柔軟に差し替えできる**」点が肝。task-specific architecture を作らずに、テキスト・bbox・セマンティクスマップ・低解像度画像が全部同じ枠で動く。
- LSGM のように encoder と prior を joint training すると重み付けに苦しむが、**固定された AE + 後付け DM** にすると独立に最適化できるので学習が安定。これは「VAE を固定して prior を別に学習」する古典的アプローチの強化版とも読める。
- VQ-reg な latent の方が LDM のサンプル品質が時に良い（reconstruction では負ける）という観察は、**離散化が事前情報として効く**ことを示唆していて面白い。
- 畳み込みサンプリング + classifier-free guidance + latent rescaling で 256² 学習モデルから 512²–1024² を出すノウハウは super-resolution や inpainting の高解像度版にも転用されており、実用 hack として価値が高い。
- 学習コストの絶対量（ImageNet で 271 V100 days）は decent でこそあれ「small lab でも回せる」というほど安くはない点には注意。democratization は推論側でより劇的。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「ピクセル空間で DM を回すと無駄が多い」という観察を rate-distortion 図 (Fig. perceptualcompression) で定量的に動機付けてから、$f$ を網羅的にスイープして sweet spot を見せる流れが説得力ある。実証研究としてキレイ。
  - **5 つの異なる生成タスク**（無条件・クラス条件・テキスト・レイアウト・超解像・インペインティング）に同一フレームで取り組み、それぞれ SOTA か同等を出したという守備範囲の広さ。
  - cross-attention 機構が後の multimodal DM（Stable Diffusion, ControlNet 等）の de facto baseline になった点で歴史的価値が高い。
  - autoencoder を公開し再利用可能にした副次的価値が大きい（CLIP guidance, downstream tasks）。
  - inpainting で**ピクセル DM の 2.7× 以上の高速化と 1.6× の FID 改善**を同時達成したのは、効率改善が品質を犠牲にしないという主張を裏付ける良い実験。
- **弱み / 疑問**:
  - **著者自身が認める limitations**: (i) sampling が逐次的で GAN より遅い、(ii) $f=4$ でも reconstruction が完全ではなく、**ピクセル精度が要る応用（医用画像、超解像など）ではボトルネック**になり得る、と明記。後続の SD でも高周波テクスチャの再現性は実際に課題。
  - **AE 自体は perceptual + GAN loss** で学習されているため、likelihood-based DM の理論的旨味（mode coverage）が AE 側で adversarial に毀損されている可能性。著者も discussion で "the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective misrepresents the data remains an important research question" と認めている。
  - LSUN-Bedrooms で ADM の FID 1.90 に対して 2.95 と**負けているケース**がある。"パラメータ半分・計算 1/4" は説得力あるが、計算等価点で比較すれば差が縮まるとは限らない。
  - text-to-image は **LAION-400M**（学習に著作権・バイアス問題のあるデータ）。Societal impact のセクションでは触れているが、データ由来のバイアス・記憶（training data extraction）の定量評価は無い。
  - 圧縮率 $f$ の選び方は実質ヒューリスティック（FID プロットを見て決める）。データセット依存で、新しいドメインに移す時に再探索が必要。理論的指針が欲しい。
  - super-resolution の bicubic 前提が現実画像に効かないため LDM-BSR を作ったという報告は良いが、これは「BSR-degradation で学習しただけ」で本提案の貢献というより SR 一般の事情。
  - "convolutional sampling で 1024² まで合成"の主張は qualitative example が中心で、定量評価は限定的（高解像度域での FID 比較が無い）。
  - **deep fake・misinformation・training data leakage**の懸念を societal impact で正直に列挙しているが、対策は議論されていない（同時期の論文の標準的態度ではあるが、いまの基準だと不十分）。
- **次に試したいこと**:
  - 同じ学習 budget で（pixel DM vs LDM-4 vs LDM-8）を比較する pareto curve を ImageNet と FFHQ で引き、AE の事前学習コストも含めた fair な比較を見たい。
  - **AE を完全に likelihood-based**（KL-VAE / NVAE / VDVAE 等）に置き換えたときに sample quality がどう変わるか。adversarial loss を抜くと再構成が甘くなる vs DM が補う、のトレードオフを定量化したい。
  - cross-attention 以外の conditioning（FiLM、adaLN、Perceiver 風）と比較した ablation。今ある図表だけだと「cross-attention で十分」までしか分からない。
  - latent 空間の SNR/分散を rescale する話（Sec. suppsec:rescale）を**学習データに依存しない自動 calibration** にできるか。
  - inpainting で SOTA 化したモデルを使った training data extraction 実験（Carlini らの手法）で、LAION 学習モデルにどの程度の漏洩があるか定量化。

## Notes / Quotes

- "training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [ADM]) and ... producing 50k samples takes approximately 5 days on a single A100 GPU." (intro)
- "Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility." (intro)
- 「democratizing high-resolution image synthesis」が論文の political framing。
- 損失: $L_\text{LDM} := \mathbb{E}_{\mathcal{E}(x),\epsilon,t}[\|\epsilon - \epsilon_\theta(z_t,t)\|_2^2]$（Eq. eq:ldmloss）、条件付きは $\tau_\theta(y)$ を cross-attention 経由で注入（Eq. eq:cond_loss）。
- 第一段の正則化: **KL-reg**（VAE 風）と **VQ-reg**（量子化層を decoder に吸収。VQGAN 解釈）。
- AE の R-FID/PSNR（Tab. tab:firststagetable, OpenImages 学習, ImageNet-Val 評価）: $f=4$ VQ → R-FID 0.58, PSNR 27.43、$f=4$ KL → 0.27, 27.53。$f=8$ VQ → 1.14, 23.07。$f=16$ → 5.15, 20.83。
- ImageNet クラス条件: LDM-4 271 V100 days, ADM-G 962, ADM 916（Tab. tab:compute_vs_fid）。
- text-to-image 1.45B params model: BERT-tokenizer + Transformer 実装の $\tau_\theta$、UNet に multi-head cross-attention。
- 著者自身の limitations: "their sequential sampling process is still slower than that of GANs"、"the use of LDMs can be questionable when high precision is required"、"superresolution models are already somewhat limited in this respect"。
- Societal impact: deep fakes（特に女性が不均衡に影響を受ける）、training data extraction、データバイアスの増幅、を列挙。

## Related Papers

- Ho et al. 2020 DDPM — 基本となる reweighted DDPM 損失。
- Dhariwal & Nichol 2021 ADM / ADM-G — ピクセル空間 DM の SOTA、本論文の比較基準。Classifier guidance も流用。
- Ho & Salimans 2021 Classifier-Free Guidance — 後段の LDM-G の品質ブースタ。
- Esser et al. 2021 VQGAN（DBLP:abs-2012-09841）— 第一段の直接の先行研究。本論文は「量子化を decoder に吸収した VQGAN」を AE として再利用。
- Vahdat et al. 2021 LSGM — encoder と score prior を joint training する競合。本論文は意図的に分離して回避。
- Ramesh et al. 2021 DALL-E（DBLP:abs-2102-12092）— text-to-image AR ベースライン。
- Nichol et al. 2021 GLIDE — text-to-image 拡散モデルの当時の対抗馬。LDM は 1/4 のパラメータで同等。
- Saharia et al. 2021 SR3 — 拡散ベース super-resolution、Tab. srtable のベースライン。
- Suvorov et al. 2021 LaMa — Fast Fourier Conv ベースのインペインティング SOTA、Tab. inpaintingtable のベースライン。
- Karras et al. StyleGAN/StyleGAN2 — GAN ベースの強い比較対象、特に FFHQ / LSUN。
- Schuhmann et al. 2021 LAION-400M — text-to-image 学習データ。
- Sohl-Dickstein et al. 2015 — 拡散モデルの理論的源流。
