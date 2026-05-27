# Transformer-Squared: Self-adaptive LLMs

- arXiv: https://arxiv.org/abs/2501.06252
- source: ../papers/arXiv-2501.06252v3/
- authors: Qi Sun, Edoardo Cetin, Yujin Tang (Sakana AI / Institute of Science Tokyo, equal contribution)
- venue / year: ICLR 2025
- tags: [LLM, self-adaptation, PEFT, SVD, RL, MoE-like, LoRA-alternative]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 既存の LLM post-training は、多様な能力を 1 回の大規模 fine-tuning に詰め込むため計算コスト・training time が大きく、データの幅を広げると overfitting と task interference の両立が難しい。LoRA 等で複数 expert module を作る案も、trainable / storage parameter の累積、狭い task domain での overfitting、runtime での柔軟な composition が未解決課題になる。
- **手法**: 二段構え。
  - (a) **SVF (Singular Value Fine-tuning)**: 各重み行列 $W = U\Sigma V^\intercal$ について、特異値だけをスケールするベクトル $z \in \mathbb{R}^r$ を学習する（$\Sigma' = \Sigma \otimes \mathrm{diag}(z)$）。学習は REINFORCE + KL ペナルティ（$\lambda$）で、reward は答えの正誤の $\{-1, +1\}$。これで各タスクごとに「expert vector」$z^k$ を得る。
  - (b) **Transformer² (実装名 $\text{Transformer}^2$)**: 推論時に 2 パス。1 パス目で入力タスクを観測して expert vector の合成 $z'$ を決め、2 パス目で $W' = U\Sigma' V^\intercal$ に置き換えた重みで実回答を生成。adaptation 戦略を 3 種提案: (A) **Prompt** で LLM に自己分類させて 1 個の $z^k$ を選ぶ、(B) **Cls-expert** で task 分類専用の SVF expert $z^c$ を学習して使う、(C) **Few-shot** で $z' = \sum_k \alpha_k z_k$ の係数 $\alpha_k$ を CEM で少数の held-out プロンプト上で最適化（タスクごとに 1 回だけ）。
- **結果**:
  - **訓練タスク（Table 1）**: SVF は多くの設定で base / LoRA を上回るが、全セルで LoRA より上ではない。Llama3-8B-Instruct: GSM8K 75.89→79.15（LoRA 77.18）、MBPP-Pro 64.65→66.67（LoRA 67.68）、ARC-Easy 88.59→89.56。Mistral-7B-Instruct-v0.3: GSM8K 42.83→49.74（+16%）、MBPP-Pro 49.50→51.52（LoRA 51.52）、ARC-Easy 81.65→85.14。Llama3-70B-Instruct: GSM8K 85.29→88.32、MBPP-Pro 80.81→80.81、ARC-Easy 89.10→88.47、LoRA は GSM8K で 77.26 まで崩れる。VLM（Llama3-Llava-Next-8B）の TextVQA でも SVF で base から「39% 以上」改善。
  - **未見タスクへの adaptation（Table 2）**: Llama3-8B では MATH 24.54→25.47、Humaneval 60.98→62.99、ARC-Challenge 80.63→82.61（Few-shot が 3 タスクで最良）。Mistral では MATH 13.02→13.39、Humaneval 43.29→47.40、ARC-Challenge 71.76→75.47。Llama3-70B は MATH のみ 40.64→40.44 と僅か劣化（GPU 制約で半分の層だけ SVF した断り付き）、Humaneval/ARC-C は改善。同条件で LoRA は MATH/Humaneval を大きく悪化（Llama3-70B で MATH 40.64→25.40）。OKVQA は、base VLM の performance が $\text{Transformer}^2$ 適用後にのみ改善したと本文が述べている。
  - **3 戦略の関係**: 著者は「より involved な戦略・追加 test-time information ほど self-adaptation が有効になる」という monotonic trend を主張する。ただし Table 2 の個別セルでは Prompt が Cls-expert を上回る例（Llama3-8B の MATH / ARC-Challenge）や、Mistral の MATH で Prompt / Cls-expert が base を下回る例がある。
  - **推論コスト（Table 3）**: 1st/2nd パス比は MATH 13%、Humaneval 19%、ARC-Challenge 47%（ARC は単一選択で 2nd パスが短いため比率が大きい）。
  - **アブレーション（Table 4）**: MLP+attn 両方で 0.58M params、GSM8K 79.23 / MATH 25.04。LoRA(PG, attn) は 6.82M で 57.92/15.72 と崩壊（RL 不安定）。SVF (next-token-pred) も 60.50 と落ちる → RL 目的関数が効いている。
  - **Cross-model transfer（Table 5）**: Llama で学習した $z$ をそのまま Mistral に適用しても 2/3 タスクで改善（Humaneval 43.29→45.12 等）。$\sigma_i$ をシャッフルすると劣化、cross-model few-shot adaptation でさらに改善（ARC-Challenge 71.76→75.64）。
- **貢献**: (1) 特異値スケーリングのみで full-rank の重み修正をかける PEFT 手法 SVF を提案し、LoRA 比でパラメータ <10%・RL と相性が良いことを示した。(2) SVF expert を building block にした self-adaptive 推論フレームワーク $\text{Transformer}^2$ と 3 つの adaptation 戦略を提案。(3) VLM への流用や Llama↔Mistral の cross-model expert 流用など、組成性・移植性を実証。

## Takeaway（自分にとっての要点）

- **LoRA の代替として SVF はかなり強い**: LoRA が rank $r'$ を選び増設行列 $A,B$ を学習するのに対し、SVF は $\min(m,n)$ 次元のスカラー列を学習するだけで full-rank な調整ができる。本論文の LoRA baseline は next-token prediction で学習され、著者は「説明テキストが無く最終答えだけの GSM8K のようなデータでは LoRA fine-tuning の要求が高い」と述べる。一方、SVF + REINFORCE は正誤シグナルで学習する。
- **expert vector の「組成性」**: LoRA の $A,B$ は permutation 自由度が高く、複数 LoRA を線形補間しても意味が壊れる、と著者は述べる。SVF は $U,V$ を固定して $\sigma$ だけスケールするので、$z$ 同士の線形補間が意味を保つという設計になっており、Few-shot adaptation では CEM でサンプル単位の係数最適化を行う。
- **Few-shot adaptation が「prompt 長を増やさない few-shot」になっている**: タスクごとに $\alpha_k$ を 1 回だけ CEM で決めれば、推論時のプロンプトは元のままで few-shot 効果が得られる。in-context learning との対比が鋭い。
- **Few-shot adaptation が最も強い傾向**: Table 2 では Few-shot が Llama3-8B と Mistral の 3 タスク中ほぼ最良で、著者は追加の test-time information が有効だと述べている。ただし Prompt → Cls-expert → Few-shot が全セルで厳密に単調増加するわけではない。
- **Cross-model transfer が部分的に成立する**: Llama で学習した SVF vector を Mistral に適用すると 2/3 タスクで base より改善し、shuffled $\sigma_i$ では悪化する。著者は、この compatibility が 2 モデルの architecture 類似性に tied している可能性と、different scales での再現性は open research question だと述べる。
- **2 パス推論の overhead**: 1st パスは長さ $O(n)$ なので生成系タスク（MATH 13%、Humaneval 19%）では実用的、選択系（ARC-C 47%）は相対的に高くつく。

## Critical Thoughts（評価・疑問）

- **強み**:
  - SVF のパラメータ効率が極端（attn のみで 0.16M、LoRA 6.82M）かつ、Table 4 で「LoRA + RL」がほぼ崩壊するのに SVF + RL は安定、という強いコントラストを示している。正則化として特異成分スケーリングが効くという主張に裏付けがある。
  - Llama3-8B / Mistral-7B / Llama3-70B / VLM（Llava-Next-8B）を含む複数モデル・複数タスクで評価している点は広い。ただし Table 1 には Llama3-70B の ARC-Easy 低下、Table 2 には Mistral の Prompt / Cls-expert が MATH で base を下回る例もある。
  - cross-model transfer のシャッフル比較（ordered vs shuffled $\sigma_i$）が「順序情報自体が転移している」ことを示すうまい統制実験。
  - Few-shot adaptation がプロンプト長を増やさず、タスクあたり 1 回の CEM で済むという設計判断が実用的。
- **弱み / 疑問**:
  - **改善幅が小さいタスクが多い**: $\text{Transformer}^2$ の未見タスク改善は Llama3-8B で +0.9〜+2.0 ポイント、Mistral でも MATH は +0.37。seed 数や信頼区間の記述が本文表からは読み取れず、これらの小幅改善の安定性は判断できない（評者補足）。
  - **Llama3-70B + MATH では劣化**（40.64→40.44 (Prompt)）。著者は GPU 制約で半分の層しか SVF していないと断っている。Cls-expert/Few-shot の 70B 結果は表に無く（Prompt のみ）、70B で最も強い few-shot adaptation を確認できない。
  - **expert は SVF 学習タスクの「latent capability」に縛られる**（著者が conclusion で limitation として明記）。base model が解けないタスクでは RL の reward が常に -1 になりやすく、sparse reward 問題が出る（methods §3.2 でも触れている）。
  - **CEM のスケーラビリティ**: expert 数 $K$ が増えると $\alpha \in \mathbb{R}^K$ の探索コストが膨らむ。これも conclusion で著者自身が認めている。
  - **公平比較の粒度**: LoRA baseline は「全 training checkpoint の中で test ごとの最高値」を取ると本文に明記されており、LoRA に有利な報告になっている。SVF 側は Figure 2 caption と appendix で validation score による checkpoint / $\lambda$ 選択と説明される。LoRA rank・学習率・clip norm の探索範囲は appendix の hyper-parameter table 参照で、本文表だけでは読みづらい。
  - **タスク分類の前提**: prompt/cls-expert は「予め定義された K カテゴリ + others」に投げ込む形式で、本当に未知のドメイン（複合タスク等）でルーティングがどう振る舞うかの定量評価は弱い（評者補足）。Confusion matrix は task classes を ground truth として predicted categories を表示する。
  - **CEM の few-shot プロンプト数や held-out 量**、SVF expert 学習時の「数百サンプル」という主張の具体的なデータ量、$\lambda$ 値などはセクション 4 本文には書かれず appendix 参照。再現性は appendix 次第。
  - **Reward が ±1 のバイナリ** という構成は MATH / Humaneval / ARC のような discrete-correctness タスクには合うが、open-ended な生成タスクや preference 学習では reward 設計が追加で必要になる（評者補足）。
- **次に試したいこと**:
  - 同じ trainable param 数で揃えた LoRA-XS / DoRA / VeRA / SVFT との比較を、同じ RL レシピで取り直す（本文では LoRA とだけ正面比較 / 評者補足）。
  - SVF expert 数 $K$ をスケールしたとき、CEM の代わりに学習ベース router（hypernet 風）を入れたら few-shot adaptation を超えられるか（評者補足）。
  - cross-model 転移を 8B↔70B やアーキテクチャ家族をまたぐ範囲で評価し、著者が open question とする different scales での再現性を調べる（sections/sec4_experiments.tex に基づく評者補足）。
  - SVF を SFT/RLHF 後の policy にかぶせて、preference reward でも reward sparsity 問題が抑えられるか（評者補足）。
  - VLM 側で expert を VLM タスクで学習した場合と、本論文のように言語タスクの expert だけで OKVQA に転移させた場合の差を切り分ける（評者補足）。

## Notes / Quotes

- "SVF learns a simple vector $z \in \mathbb{R}^r$ that provides targeted modifications to each singular component of $W$ independently, yielding a new weight matrix $W' = U \Sigma' V^\intercal$, where $\Sigma' = \Sigma \otimes \mathrm{diag}(z)$." (sec3_methods.tex)
- "While \svdacro only needs $r = \min(m,n)$ parameters, we show it empirically does not display the same shortcomings thanks to working on a highly-meaning space provided by the latent expressiveness compressed in the weights of modern LLMs." (sec3_methods.tex)
- 目的関数: $J(\theta_z) = \mathbb{E}[\log \pi_{\theta_{W'}}(\hat y_i|x_i) \cdot r(\hat y_i, y_i)] - \lambda D_\mathrm{KL}(\pi_{\theta_{W'}} \| \pi_{\theta_W})$ （REINFORCE + KL）。reward は $\{-1,+1\}$ のバイナリ。
- 推論 2 パス: 1st でタスク同定、2nd で $W' = U(\Sigma \otimes \mathrm{diag}(z')) V^\intercal$ に置き換えて回答生成。
- 3 戦略について著者は "with more involved strategies and additional information about the test-time condition, self-adaptation appears to be increasingly effective." と述べるが、Table 2 の個別値は全セルで厳密な Prompt < Cls-expert < Few-shot ではない。(sec4_experiments.tex, tables/svf_ada_tasks.tex)
- 著者自身の limitations: ①「SVF expert の能力は base model の潜在成分に縛られる」、②「CEM ベース adaptation は specialized domain 数が増えると one-time cost が増える」、③ sparse reward の懸念（弱い base model だと reward がほぼ常に -1 になる）。(sec5_conclusions.tex, sec3_methods.tex)
- Cross-model: 「Llama の $z$ をそのまま Mistral に当てると 2/3 タスクで改善、$\sigma_i$ をシャッフルすると劣化」。順序自体に意味があるという統制。
- VLM への適用は SVF training だけで「39% 以上」改善、$\text{Transformer}^2$ の adaptation 側は言語タスクで学んだ expert のみで OKVQA に転移。
- (verified 2026-05-27) 問題設定を introduction の post-training / expert-module 課題に合わせ、「黒箱でない」など TeX にない表現を削除 (sections/sec1_introduction.tex)
- (verified 2026-05-27) Table 1 の訓練タスク結果を修正し、「SVF は LoRA より一貫して上」という誤った一般化を削除。Llama3-8B MBPP-Pro、Mistral MBPP-Pro、Llama3-70B ARC-Easy の例外を反映 (tables/svf_train_tasks.tex)
- (verified 2026-05-27) adaptation 3 戦略の記述を修正し、著者の monotonic trend 主張と Table 2 の個別例外を区別 (sections/sec4_experiments.tex, tables/svf_ada_tasks.tex)
- (verified 2026-05-27) LoRA baseline / SVF checkpoint selection の記述を本文と appendix の記載に合わせて修正 (sections/sec4_experiments.tex, sections_ws_app/Aimplementation.tex, tables_ws/hyper_p_app.tex)
- (verified 2026-05-27) Takeaway / Critical Thoughts の評者推論を TeX 事実と区別し、必要箇所に「評者補足」を明記 (sections/sec3_methods.tex, sections/sec4_experiments.tex, sections/sec5_conclusions.tex)

## Related Papers

- Hu+ 2021, **LoRA** — 直接比較される PEFT 標準。本論文では rank 自由度の弊害（permutation 多重性）を指摘。
- Bałaży+ 2024, **LoRA-XS** / HuggingFace SVD training — top-$r$ 特異成分だけ使う SVD 系 PEFT。本論文は「top-$r$ 切り捨ては情報を失う」と批判して full-rank スケールを採用。
- Lingam+ 2024, **SVFT** — concurrent な SVD ベース PEFT。本論文との違いは self-adaptation でも RL でもないこと。
- Schmidhuber 1992 / 1993, fast-weight memories, self-modifying networks — 自己適応の思想的源流。
- Ha+ 2017, **HyperNetworks** — 別ネットが重みを生むという文脈。
- Williams 1992, **REINFORCE** — SVF の学習則。
- Ouyang+ 2022, InstructGPT — KL ペナルティ付き RL 微調整の方法論。
- Rubinstein 2004, **Cross-Entropy Method (CEM)** — few-shot adaptation の $\alpha_k$ 探索に使用。
- Fedus+ 2022 (Switch), Jiang+ 2024 (Mixtral) — MoE 比較。本論文は token-level routing でなく sample-level、専門化を RL で陽に促す点が違うと主張。
- Du+ 2023 (multi-agent debate), Zhuge+ 2023 (mindstorms) — macroview の self-adaptive LLM 系、本論文は microview。
- Cobbe+ 2021 (GSM8K), Hendrycks+ 2021 (MATH), Chen+ 2021 (HumanEval), Clark+ 2018 (ARC), Austin+ 2021 (MBPP), Singh+ 2019 (TextVQA), Marino+ 2019 (OKVQA) — 評価データセット。
