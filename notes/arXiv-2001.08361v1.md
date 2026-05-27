# Scaling Laws for Neural Language Models

- arXiv: https://arxiv.org/abs/2001.08361
- source: ../papers/arXiv-2001.08361v1/
- authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- venue / year: TeX 中には明示なし（main.tex は nips_2018_wider_nonotice.sty を preprint で使用）
- tags: [scaling-laws, language-models, transformer, power-law, compute-optimal]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 言語モデルの性能は「モデル形状」「サイズ $N$」「データ量 $D$」「計算量 $C$」のどれにどれだけ依存するのか、ハイパラ・アーキテクチャ細部とどう切り分けられるのか、実証的に整理されていない。固定計算予算をどう $N$ と $D$（とステップ数 $S$）に割り振るのが最適か、原則的な答えがない。
- **手法**: WebText2（20.3M 文書、96GB、$2.29\times 10^{10}$ tokens、テスト $6.6\times 10^8$ tokens、BPE vocab=50257）上で decoder-only Transformer を、非埋め込みパラメータ数 $N$ を 768〜1.5B、データ $D$ を 22M〜23B tokens、形状（depth/width/heads/$d_\mathrm{ff}$）、context 長（基本 $n_\mathrm{ctx}=1024$）、batch size ($2^{19}$ tokens 中心) を独立に振って学習。Adam（>1B パラは Adafactor）、$2.5\times 10^5$ steps、batch 512×1024、3000-step warmup + cosine→0。compute は非埋め込み $C\approx 6NBS$ で見積もり、critical batch size $B_\mathrm{crit}(L)=B_\ast/L^{1/\alpha_B}$（[1812.06162]）で $S_\mathrm{min}, C_\mathrm{min}$ に換算してフィット。
- **結果**: 他 2 要素にボトルネックされない場面で性能はべき乗則に従う:
  - $L(N) = (N_c/N)^{\alpha_N}$,  $\alpha_N\!\sim\!0.076$, $N_c\!\sim\!8.8\times 10^{13}$ params
  - $L(D) = (D_c/D)^{\alpha_D}$,  $\alpha_D\!\sim\!0.095$, $D_c\!\sim\!5.4\times 10^{13}$ tokens
  - $L(C_\mathrm{min}) = (C_c^\mathrm{min}/C_\mathrm{min})^{\alpha_C^\mathrm{min}}$, $\alpha_C^\mathrm{min}\!\sim\!0.050$, $C_c^\mathrm{min}\!\sim\!3.1\times 10^8$ PF-days
  - $\alpha_B\!\sim\!0.21$, $B_\ast\!\sim\!2\times 10^8$ tokens（$L$ が 13% 減るごとに $B_\mathrm{crit}$ がほぼ倍）
  - 同時依存: $L(N,D) = \left[(N_c/N)^{\alpha_N/\alpha_D} + D_c/D\right]^{\alpha_D}$ と $L(N,S) = (N_c/N)^{\alpha_N} + (S_c/S_\mathrm{min})^{\alpha_S}$（$\alpha_S\!\approx\!0.76$, $S_c\!\approx\!2.1\times 10^3$）。trend は $C_\mathrm{min}$ で 8 桁、$N$ で 6 桁、$D$ で 2 桁にわたり成立。形状（depth/width/heads）依存は数%、$(n_\mathrm{layer},d_\mathrm{model})=(6,4288)$ は $(48,1600)$ より loss が 3% 以内。LSTM は文脈の早い位置では Transformer と同等、後半 token で負ける。汎化: Books Corpus / Common Crawl / Wikipedia / Internet Books で WebText2 性能と一定オフセットで相関、訓練段階に依らない。
  - **compute-efficient な配分**: $N\propto C_\mathrm{min}^{0.73}$, $B\propto C_\mathrm{min}^{0.24}$, $S\propto C_\mathrm{min}^{0.03}$, $D\propto C_\mathrm{min}^{0.27}$（具体係数 $N_e=1.3\!\times\!10^9, B_e=2.0\!\times\!10^6$ tokens, $S_e=5.4\!\times\!10^3$ steps, $D_e=2\!\times\!10^{10}$ tokens）。最適は「converged loss から $\alpha_N/\alpha_S\!\approx\!10\%$ 上で止める」こと。典型的に使われる $f'\!=\!2\%$ 設定と比べ、同じ loss を出すのに 2.7x 大きいモデル・7.7x 少ない steps・35% の compute で済む。
  - overfitting を抑えるには $D\gtrsim 5\times 10^3\, N^{0.74}$。モデルを 8x 大きくしてもデータは 5x で足りる。
  - $L(C_\mathrm{min})$ と $L(D)$ を外挿すると $C^\ast\!\sim\!10^4$ PF-days, $N^\ast\!\sim\!10^{12}$ params, $D^\ast\!\sim\!10^{12}$ tokens, $L^\ast\!\sim\!1.7$ nats/token で交差し、それ以前に scaling law が破綻するはず、と予想。
- **貢献**: (1) $N$, $D$, $C_\mathrm{min}$ いずれに対しても LM cross-entropy がべき乗則に従うことを 6〜8 桁にわたり実証、(2) $L(N,D), L(N,S)$ という 2 変数同時の閉形式 fit、(3) 固定 compute 予算での最適配分 $N\!\propto\!C^{0.73}, B\!\propto\!C^{0.24}, S\!\propto\!C^{0.03}$ を理論・実測の両面で導出、(4)「大モデルほどサンプル効率が高い」「最適には収束させずに止める」という運用指針、(5) LSTM／Universal Transformer との比較、汎化のロスオフセット。

## Takeaway（自分にとっての要点）

- **"big models > big data"**: 計算予算が増えたとき、データ量や step 数より圧倒的にパラメータ数 $N$ に振るのが最適。$D\propto C^{0.27}$, $S\propto C^{0.03}$ という指数の小ささが衝撃。シリアル step 数はほぼ伸ばさない。
- **収束まで訓練しないのが正解**: compute-efficient 訓練は converged loss の $\sim 10\%$ 上で止める。研究者がよくやる $f'\!=\!2\%$ ($\sim$収束) は $2.7\times$ 過小サイズ・$7.7\times$ 過剰 step、結果として同じ loss に 2.8x の compute を浪費している。
- **形状は弱い**: $n_\mathrm{layer}$, $n_\mathrm{heads}$, $d_\mathrm{ff}/d_\mathrm{model}$ は数% しか効かない。アスペクト比を 40 倍振っても loss は 3% 内。ハイパラ探索より $N$ を増やすほうが効くという強いメッセージ。
- **埋め込みを含めると trend が汚れる**: $N$ は必ず non-embedding で測る。embedding 行列はスケーリングの「ノイズ」なので、ALBERT 的に削ってよい示唆。
- **critical batch size の道具化**: $B_\mathrm{crit}(L)=B_\ast/L^{1/\alpha_B}$ で「実測の $C, S$」を「最小 compute / 最小 step」両方向に補正できる。これで実験条件の差を吸収して trend を綺麗にできるのがこの論文の地味な要点。
- **オーバーフィットはほぼ $N^{0.74}/D$ だけで決まる**: あらゆる $(N,D)$ 設定で $\delta L$ がこの 1 変数に潰れる。dropout 10% を入れた前提だが、「データを増やす速度はモデルの 0.74 乗でよい」は実務 rule of thumb として使いやすい。
- **転移はオフセット**: out-of-distribution は in-distribution validation との間に「一定の追加 loss」がつくだけで、伸び方は同じ。訓練分布性能を上げれば転移性能も同じ勢いで上がる。
- **$L^\ast\!\sim\!1.7$ nats/token は「自然言語のエントロピー推定」かもしれない**、という discussion は強気だが面白い。WebText2 は 1.4 tokens/word, 4.3 chars/token なので、1 word あたり約 2.4 nats の粗い推定に対応する。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 8 桁の $C_\mathrm{min}$、6 桁の $N$、2 桁超の $D$ にわたり power law を示した実証論文。
  - 配分 $N\!\propto\!C^{0.73}, S\!\propto\!C^{0.03}$ を**実測**と**$L(N,S)$ からの解析的予測** $\alpha_C^\mathrm{min}/\alpha_N\!\approx\!0.71$ の両方で出して整合させた点が説得力を上げている。
  - 形状・LSTM・転移・early stopping・critical batch まで、関連する話題を 1 本の枠組みに統合してある。
  - WebText2 という単一データセット・vocab に対する係数だと自覚的で、「$N_c, D_c$ には fundamental meaning はない」「指数だけが意味を持つ」と明言している点が良い。
- **弱み / 疑問**:
  - **データの質・分布を変えていない**: 訓練は WebText2 のみ。trend がトークナイザや分布で再スケールされると述べており、データ品質と $\alpha_D$ の関係は未検討。
  - **正則化/データ拡張を最適化していない**と著者自身が caveat に書いている（§Caveats）。dropout 固定 10%、$L(N,D)$ fit は $D\!\approx\!2\times 10^7$ tokens では悪い（40 updates/epoch）。小データ域は別 regime の可能性。
  - **理論的根拠なし**: 著者自身「we do not have a solid theoretical understanding for any of our proposed scaling laws」と認めている。$1/D$ 展開を仮定する 3 番目の原則は "speculative"。
  - **$B_\mathrm{crit}(L)$ の外挿に自信なし**と明記。これがズレると $\alpha_C^\mathrm{min}$ もずれる。
  - **$C\!\approx\!6NBS$ 近似が $n_\mathrm{ctx}\gtrsim 12d_\mathrm{model}$ で破綻する**と自覚されており、long-context 時代には注意。
  - **学習率以外のハイパラ（init scale, momentum 等）を tune し切れていない**と認めている。最適 learning rate は target loss に依存し、短い run では大きい LR が良い可能性も未探索。
  - **$L^\ast\!\sim\!1.7$ nats/token を「自然言語のエントロピー推定」と読める、という conjecture** は指数のわずかな違いで桁が動く（自分で highly uncertain と書いている）。哲学的主張に近く、現状そのまま信じる根拠は弱い。
  - 「サンプル効率が良い大モデル」は **訓練側** の議論で、inference cost を含む目的関数は扱っていない。Appendix A.3 では小さいモデルが inference cost を考えると有用な場合がある、とだけ述べている。
  - $\alpha_N\!=\!0.076$ の小ささから、N を 10x 増やしても loss は $10^{-0.076}\!\approx\!0.84$ 倍にしかならない。著者は discussion で、loss の滑らかな改善が関連する language tasks の改善に変換されるかは重要な未検討点だと述べている。
- **次に試したいこと**:
  - 同じ枠組みを image / audio / video / RL の生成モデルで再現し、$\alpha_N, \alpha_D, \alpha_C$ がドメイン横断で universal か検証。
  - データ品質・重複・curation を変えて $D_c, \alpha_D$ がどう動くか測り、compute-optimal の $N(C)$ がどこまでデータ依存か可視化。
  - $\alpha_S\!=\!0.76$ の universality を Hessian スペクトルから説明する noisy quadratic / NTK 連結。
  - $L^\ast$ が自然言語の entropy-per-token 推定として読めるか、WebText2 以外のデータ分布で確認する。
  - 訓練ロス改善が downstream language tasks の改善にどう対応するかを調べる。
  - inference cost を目的関数に入れた場合、Appendix A.3 の suboptimal model size 議論がどう変わるかを調べる。

## Notes / Quotes

- "Performance has a power-law relationship with each of the three scale factors $N, D, C$ when not bottlenecked by the other two, with trends spanning more than six orders of magnitude." (introduction)
- "Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence." (abstract)
- "The performance penalty depends predictably on the ratio $N^{0.74}/D$, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty." (Summary §1.1)
- "Big models may be more important than big data." (Discussion §6)
- compute-efficient 訓練は「converged loss から $\alpha_N/\alpha_S\!\approx\!10\%$ 上」で止めると最適（Appendix A.2）。
- $C^\ast\!\sim\!10^4$ PF-days, $N^\ast\!\sim\!10^{12}$ params, $L^\ast\!\sim\!1.7$ nats/token で $L(C_\mathrm{min})$ と $L(D)$ が交差 → そこで scaling laws は破綻する見込み（§5.3）。
- 著者明示の限界（§Caveats）: 理論なし／$B_\mathrm{crit}$ の外挿に不安／小データ域と正則化を tune していない／$C\!\approx\!6NBS$ は $n_\mathrm{ctx}\gtrsim 12 d_\mathrm{model}$ で破綻／init・momentum 等の hyperparam tuning が不十分／LR は target loss 依存で短い run は未探索。
- 「核となる scaling laws の数値は WebText2 / BPE tokenizer 依存。$N_c, D_c$ に fundamental meaning はなく、指数 $\alpha$ だけが意味を持つ」(§1.1)。
- (verified 2026-05-20) Related Papers の Hestness+ 2019 の会議名を SC'19 → PPoPP'19 に修正（main.bbl: "Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming, PPoPP '19"）。Summary／Takeaway／Critical Thoughts の数値・固有名は main.tex（abstract, §1.1, §3, Tables of $L(N,D)$ / $L(N,S)$ fits, App. A Tables, §Caveats, §5.3, footnote on WebText2 stats）と再照合し、その他は問題なし。
- (verified 2026-05-27) venue/year の断定を TeX で確認できる preprint style 使用に限定し、TeX/main.bbl に無い Chinchilla/DeepSeek/GPT-3/MMLU/Hoffmann+ 2022 等の後年比較を削除 (main.tex, main.bbl)。

## Related Papers

- McCandlish+ 2018 "An Empirical Model of Large-Batch Training" (arXiv 1812.06162) — critical batch size と gradient noise scale。本論文の $B_\mathrm{crit}$ の理論基盤。
- Radford+ 2019 "Language Models are Unsupervised Multitask Learners" — WebText の元データセット・BPE tokenizer・(48, 1600) baseline モデル。
- Hestness+ 2017/2019 (arXiv 1712.00409, "Beyond Human-Level Accuracy" PPoPP'19) — 最も近い先行研究。ただし彼らは super-linear な $D$-$N$ スケーリングを報告。本論文は sub-linear と逆の結論。
- Rosenfeld+ 2019 (arXiv 1909.12673) — 同種の $L(N,D)$ ansatz を独立に提案（脚注で同時期と言及）。
- Tan & Le 2019 EfficientNet (arXiv 1905.11946) — 画像でのスケーリング比較対象。
- Vaswani+ 2017 (Transformer), Dehghani+ 2018 Universal Transformer (arXiv 1807.03819) — アーキテクチャ比較。
