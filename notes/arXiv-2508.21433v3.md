# The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management

- arXiv: https://arxiv.org/abs/2508.21433
- source: ../papers/arXiv-2508.21433v3/
- authors: Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov (JetBrains Research / TUM)
- venue / year: NeurIPS 2025 Workshop "Deep Learning for Code in the Agentic Era"
- tags: [LLM-agent, software-engineering, SWE-bench, context-management, summarization, efficiency]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: LLM ベースの SE エージェント（SWE-agent, OpenHands, Cursor 等）は ReAct/CodeAct ループで毎ターン環境観測（ファイル内容・テストログ・recursive ls 等）を context に積み上げるため、軌跡が伸びるとコストが二倍以上に膨らむうえ "lost in the middle" で性能も下がる。OpenHands/Cursor などは LLM による要約 (LLM-Summary) を使っているが、単に古い観測を placeholder で消すだけの Observation Masking と比べてその複雑性が実利益を生んでいるかが未検証だった。SWE-agent の予備実験では平均ターンの ~84% が観測トークンであり (Fig. 1)、観測だけ削ればコスト構造的に十分効くはず、という観察が出発点。
- **手法**:
  - 3 戦略を同一 scaffold (SWE-agent) で揃えて比較。
    - **Raw Agent**: 無管理。τ_{t-1}=(o_sys, o_user, T_1,…,T_{t-1}) がそのまま伸びる。
    - **Observation Masking** (式 3): rolling window M=10。i < t-M の観測 o_i のみを placeholder（例: "Previous 8 lines omitted for brevity."）に置換。推論 r_i・action a_i は全て残す。
    - **LLM-Summary** (式 4-5): 累積 N+M=31 ターンになったら、先頭側 N=21 ターンを summarizer LLM π′ に渡して要約 s_t を作り、τ′=(o_sys, o_user, s_t, T_{t-M},…,T_{t-1}) に再構成。プロンプトは OpenHands のものを SE 専用にトリムして使用 (Fig. prompt:llm-summary)。
  - 5 モデル構成: Qwen3-32B（thinking / non-thinking, ctx 122K via YaRN）、Qwen3-Coder-480B-A35B-Instruct-FP8（ctx 256K）、Gemini 2.5 Flash（thinking 800 tok / non-thinking, ctx 1M）。ターン上限 250。SWE-bench Verified 500 件で本実験。
  - 評価指標は Solve Rate (%) と Instance Cost ($)。Qwen は Alibaba 公式 API 価格で post-hoc 計算、Gemini は Vertex API の返却値。95% 信頼区間はペア化ノンパラメトリック bootstrap (B=10,000)、p<0.05 で † 付与。
  - OpenHands (v0.43.0) でも SWE-bench Verified-50 で Gemini 2.5 Flash (no-think) を使い汎化を確認。
- **結果** (Table 1, n=500):
  - 文脈管理の普遍的恩恵: 5 構成中 4 構成で Observation Masking / LLM-Summary が >50% のコスト削減を達成（残り 1 つは Qwen3-32B thinking で軌跡が短すぎるため）。"任意の管理戦略 > 無管理"。
  - Observation Masking が 5 構成中 4 構成でコスト最安、かつ Qwen3-Coder 480B では solve rate も 53.4 → 54.8% と微増（+2.6%）、コストは $1.29 → $0.61（-52.7% †）。LLM-Summary は 53.8% / $0.64。$0.03/instance の差は 500 件で $15。
  - Gemini 2.5 Flash (no-think): Raw 32.8/$0.41 → ObsMask 35.6/$0.18 (-56.1% †) → LLM 36.0/$0.24 (-41.5% †)。ObsMask の方が安く solve rate もほぼ同等。
  - Gemini 2.5 Flash (thinking): Raw 40.4 → ObsMask 36.4 (-4.0pp / -9.9% †) → LLM 31.4 (-9.0pp / -22.3% †)。両戦略とも solve rate 有意低下（thinking 中に context を切るのが特に痛い）。
  - Qwen3-32B (non-thinking): Raw 17.0 → ObsMask 15.0 (-2.0pp / -11.8%, ns) → LLM 16.0 (-1.0pp / -5.9%, ns)。コストは両者 -50.9%/-55.4% †。
  - LLM-Summary は Observation Masking を consistent or significant には上回らない。
  - **Trajectory elongation effect** (Fig. trajectory_lengths_boxplots): LLM-Summary は軌跡を伸ばす。Gemini 2.5 Flash で平均 52 ターン（ObsMask 44 比 +15%, Raw 50 比 +4%）、Qwen3-Coder 480B で対 Raw +15% / 対 ObsMask +13%。要約による「整理済み感」がエージェントに継続を促す reinforcing signal になっていると推測。
  - **Summary 生成コスト** (Table summary_instance_costs): summary 呼び出し自体が instance 総額の 0.65–7.20% を占める。要約は毎回ユニークな turn 列を入力するため KV cache hit が効かず、Gemini 等の cache-hit が cache-miss の最大 1/10 価格になる API では特に不利。この生成コストを引くと ObsMask と LLM-Summary の差は大半消える。
  - **Hybrid 戦略** (§5.3, SWE-bench Verified-50, Qwen3-Coder 480B): N=43, M=W=10。ObsMask を warm-up 中に走らせ、N=43 で初めて LLM-Summary を発火（context が Raw の N=21 相当 ~30K tok に達する点で揃える）。コストは ObsMask 比 -7%, LLM-Summary 比 -11%、solve rate は Raw 比 +2.6pp。naïve に N=21,M=W=10 にすると KV cache 効率悪化と要約コスト二重計上で逆に悪化。
  - **OpenHands 汎化** (§5.1): SWE-agent で最適だった M=10 をそのまま使うと OpenHands では大幅劣化。M=58 にチューニングし直すと再現。SWE-agent は syntax retry を履歴から消すが OpenHands は残す等の scaffold 差が原因と推測。
  - **Critic-enhanced LLM-Summary** (§D.3, SWE-bench Verified 150 件 / SWE-agent): 要約に critic 的振り返りを同時生成させても solve rate は改善せず、軌跡延伸を悪化させる。
- **貢献**:
  1. SWE-agent 上で Observation Masking と LLM-Summary を 5 モデル構成 × 500 件で揃えて初の体系比較。
  2. LLM-Summary の trajectory elongation 効果と summary 生成コストの 2 要因に分解し、複雑な要約が必ずしも報われないことを定量的に示した。
  3. ObsMask warm-up + LLM-Summary を遅延発火する新規 Hybrid (N=43, M=W=10) を提案、コスト -7%/-11%、solve rate +2.6pp。
  4. OpenHands での initial generality 確認（ただし scaffold 依存ハイパーパラメータあり）。
  5. コード・データ・全 trajectory を JetBrains-Research/the-complexity-trap で公開。

## Takeaway（自分にとっての要点）

- SE エージェントの context の ~84% は観測トークン。なので「観測だけ古いものを placeholder で潰す」だけで cost のほぼ半分を回収できる。要約 LLM を呼ぶ前にまずこれを試すべき、というのが実装上の最大のインサイト。
- LLM-Summary には見落とされがちな 2 種類の hidden cost がある: (a) 要約生成自体の API コスト（最大 7.2%、しかも cache miss）、(b) 要約された context がエージェントに「まだやれる感」を与えて軌跡を伸ばすこと（+15% turn）。前者だけ気にしていると後者を見逃す。
- "warm-up 期間中は安い戦略 (ObsMask)、本当に context が膨らんだら高い戦略 (LLM-Summary)" という遅延発火 hybrid が普通に効くのは応用しやすい。N の決め方が「ObsMask 下で Raw の N=21 相当の context 量に達する点」というのも実装指針として明快。
- Rolling window サイズ M は scaffold-specific ハイパーパラメータ。SWE-agent では M=10 だが OpenHands では M=58 まで開けないとダメ。同じ「Observation Masking」でも scaffold が retry/失敗ターンをどう扱うかでチューニングが必要、というのは移植時の注意点。
- Gemini 2.5 Flash thinking で両戦略とも solve rate が大きく落ちる (-9.9pp / -22.3pp) のは「reasoning が前の context に依存しているモデルでは context を切ると壊れる」ことを示唆。思考トレースが長いモデルには別戦略が必要そう。
- Critic を要約に混ぜると軌跡が更に伸びるという negative result は、「要約 = 反省」という素朴な intuition の罠を示している。Self-Refine 系を context 管理に混ぜるのは要注意。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 同じ scaffold・同じハイパーパラメータ哲学 (M=10 で揃える) で 3 戦略を fair に並べた点。多くの先行研究は単一実装ベースで「LLM-Summary は良い」と主張するが、本論文は ObsMask という非常に弱い baseline でほぼ並ぶことを実証した。
  - Trajectory elongation という「コストが要約呼び出しでなく軌跡長で増える」メカニズムの発見は新規性が高く、Table summary_instance_costs で要約コスト分を差し引いてもなお残る差を説明している。
  - paired bootstrap n=500 で significance を出している（先行研究は SWE-bench Lite-50 など小サンプルが多い）。
  - Hybrid の naïve 設定 (N=21, M=W=10) で逆に悪化することを ablate しており、「N をどう選ぶか」が hybrid の本質という結論まで詰めている。
  - コード・データ・全 trajectory を公開しており、cost 構造を別 pricing で再計算できる。
- **弱み / 疑問**:
  - 著者自身が Limitations で挙げているとおり、SE 単一ドメインで結論を出している。観測が冗長で reasoning がコンパクトという SE の token 分布 (84% 観測) に Observation Masking は本質的に依存しており、deep research や agentic web 等の reasoning-heavy ドメインに直接外挿はできない（ただし §2 末で Tang+ 2025 が CUA/deep research で類似の知見を出していると引用）。
  - Hybrid の本実験は SWE-bench Verified-50 + Qwen3-Coder 480B のみで、500 件 × 全モデルでの確認は無い。"slight gain of 2.6pp" が 50 サンプルで何件分か考えると、bootstrap CI の幅次第では noise の可能性。
  - Observation Masking の placeholder は固定文字列で、relevance を一切見ない。著者も Limitation で「ファイルを編集した後に古いファイル内容を消してしまうかも」と認めている。SWE-bench 上で偶然それが致命的にならないだけかもしれない。
  - "trajectory elongation 自体が原因か、それとも要約品質が悪くて agent が同じ道をやり直すのか" が分離されていない。reasoning trace を要約で潰すと debug ループに陥りやすい可能性。
  - OpenHands 汎化が 50 サンプル × Gemini 2.5 Flash (no-think) のみ。論文中で「scaffold 依存で M を再チューニング必要」と認めている時点で "汎化した" と言い切るのは弱い。
  - Qwen3-32B (non-thinking) では ObsMask の solve rate が 17→15 (-11.8%) と落ちており、cost 削減と引き換えに性能が削れる場合がある。著者は ns と言うが pp で見ると無視できない。どのモデル・どの軌跡長で安全に使えるかの decision rule が欲しい。
  - Gemini thinking で LLM-Summary が Raw 比 -9.0pp (-22.3%) も solve rate を落とす現象に対する分析が薄い。"context を切ると thinking が破綻する" 仮説の検証実験 (thinking trace を保護した変種など) は欲しいところ。
  - cost は API 価格に強く依存。Alibaba pricing は cache hit/miss を区別しないので Qwen の cost は inflated、Vertex は区別するので Gemini は ObsMask が特に有利になる。"どの戦略が安いか" は pricing scheme に部分依存する。
- **次に試したいこと**:
  - Adaptive Observation Masking: ファイル編集 action があったらその直前の同ファイル read 観測を優先的に保護、recursive ls などは早めに潰す、というアクションタイプ条件付き window。
  - Trajectory elongation の因果検証: summary の有無を一致させたまま「要約済み context を渡されたとき」と「同じ token 数の Raw context」で agent の終了判断がどう変わるかを controlled に測定。
  - 同 token 予算下での self-consistency / best-of-N との pareto curve。$0.61/instance で 54.8% より、$1.29/instance で 5 軌跡 best-of-N の方が解けるならそちらが優位。
  - Hybrid の N をモデルごとに学習可能パラメータ化（context 利用効率の RL）。
  - SE 外への移植: WebShop / Mind2Web / GAIA 系で 84% 観測仮定が崩れるドメインでの再実験。
  - Critic-enhanced が失敗した原因切り分け: critic を要約に "同梱" するのではなく、別の short note として渡すと elongation を抑えられるか。

## Notes / Quotes

- "observation tokens make up around 84% of an average SWE-agent turn ... in our preliminary experiments on SWE-bench Lite-50" (§1)
- "any of the discussed management strategies are preferable to none, as they consistently reduce costs and often improve performance" (§1)
- "the only experiment for which the LLM-Summary strategy proved more cost-efficient is Qwen3-32B. In this case, the Observation Masking strategy ... led to a 13% increase in mean trajectory length compared to the Raw Agent." (§4.4)
- "context summaries act as a reinforcing signal, encouraging the agent to keep going" (§4.4)
- "the direct API cost of generating summaries accounts for up to 7.2% of the total instance cost ... summarization calls are particularly expensive because each requires processing a unique sequence of turns, limiting cache reuse" (§5.2)
- Hybrid hyper-param 選び: "we set N=43, because at this number of turns the context accumulated under the Observation Masking regime approximately matches the context accumulated under the Raw Agent at N=21 turns (≈30K tokens)" (§5.3)
- OpenHands 汎化注意: "If we simply re-use the optimal value from SWE-agent, Observation Masking performance degrades drastically. However, after tuning, we can reproduce our results on this agent scaffold." (§5.1)
- 著者自身の Limitations: (1) SE 単一ドメイン、(2) 非適応的なヒューリスティック trigger（rolling window は relevance/staleness 無視、LLM-Summary は意味的境界無視）、(3) scaffold 汎化は initial evidence のみ。(§6)
- Critic-enhanced summary: "no improvement in solve rate over standard LLM-Summary ... critic-enhanced runs producing even longer trajectories than standard summarization" (§D.3 / Appendix `appendix:ablation:critic`)
- (verified 2026-05-20) Gemini 2.5 Flash (thinking) と Qwen3-32B (non-thinking) の solve rate 変化を「pp」と「% (相対)」に分解して両方記載 (Table 1 と Table `tab:appendix_pvals_bootstrap` の整合) — 元の "-9.9pp / -22.3pp" は Table 1 の relative % を pp として誤記していた。
- (verified 2026-05-20) Critic-enhanced LLM-Summary の節番号を §B.3 → §D.3 に訂正（\appendix 後の section ordering: A=Experimental Config, B=Short Trajectories, C=Detailed Main Results, D=Additional Studies。Critic-Enhanced は D.3）。150 サンプルでの実験条件も明記 (neurips_2025.tex `appendix:ablation:critic`)。

## Related Papers

- Yang+ SWE-agent 2024 — 本実験の主 scaffold、Observation Masking の元実装。
- Wang+ OpenHands 2025 — LLM-Summary プロンプトの元実装、汎化実験対象。
- Antoniades+ SWE-Search 2025 — Observation Masking を採用する先行 SE エージェント。
- Xiao+ "Improving …" 2025 — 並行研究。LLM-Summary 変種を提案するが ObsMask と比較していない。"Delete" baseline は近い発想。
- Tang+ "Beyond …" 2025 — CUA / deep research で Observation Masking を RL 学習に使い好成績。本研究の SE 外汎化への根拠。
- Lu+ "Scaling …" 2025 — RL 学習時の context 制限緩和に LLM-Summary を使う。
- Zhou+ MEM1 2025 — 動的状態管理。multi-hop QA / web nav が対象で軌跡が桁違いに短い、ObsMask と比較していない点が本論文との差分。
- Liu+ "Lost in the Middle" 2024, Modarressi+ NoLiMa 2025, Hong+ 2025 — 長 context で性能が落ちる根拠論文。
- Jimenez+ SWE-bench 2024, OpenAI SWE-bench Verified 2024, Badertdinov+ SWE-bench-50 2024 — 評価データセット。
- Yao+ ReAct 2023, Wang+ CodeAct 2024 — エージェントの基本ループ形式。
- Shinn+ Reflexion 2023 — critic-enhanced summary の比較対象的位置づけ。
