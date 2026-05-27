# A Survey of Context Engineering for Large Language Models

- arXiv: https://arxiv.org/abs/2507.13334
- source: ../papers/arXiv-2507.13334v2/
- authors: Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu
- venue / year: TeX 中には明示なし（main.tex の Date は July 21, 2025）
- tags: [survey, LLM, context-engineering, RAG, memory, tool-use, multi-agent]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: LLM の挙動は与える情報（context）にほぼ全て依存するのに、prompt engineering / RAG / memory / tool calling / multi-agent が独立に進化して断片化している。「prompt = 静的文字列」という見方では現代の LLM システム（多段 RAG、永続 memory、tool 呼び出し、複数 agent 協調）を記述できない。さらに LLM は context 理解では強いのに、同等に洗練された long-form 出力を生成する能力に乏しいという非対称性がある（comprehension-generation gap）。
- **手法**: 1400+ 本の論文をレビューし、「Context Engineering」を新規分野として形式化する。context $C$ を単一文字列ではなく、複数の構成要素を assembly 関数 $\mathcal{A}$ で組み立てる動的構造物 $C = \mathcal{A}(c_1, \dots, c_n)$（$c_i$ を $c_{\text{instr}}, c_{\text{know}}, c_{\text{tools}}, c_{\text{mem}}, c_{\text{state}}, c_{\text{query}}$ にマップ）と定義し、context 設計を「タスク分布 $\mathcal{T}$ 上で reward 期待値を最大化する関数族 $\mathcal{F}^*$ を $|C|\le L_{\max}$ 制約下で探索する最適化問題」と定式化（why_ce.tex の式 (1)-(6)）。Bayesian context inference と情報理論的最適性（mutual information を最大化する Retrieve）も提示。その上で全体を二層 taxonomy に整理: **Foundational Components** = ①Context Retrieval & Generation（prompt 設計、CoT/ToT/GoT、external knowledge 取得）②Context Processing（long sequence、self-refinement、structured 情報統合）③Context Management（memory 階層、圧縮、KV cache）；**System Implementations** = ①RAG（modular / agentic / graph-enhanced）②Memory Systems（MemoryBank、MemGPT、MemOS など）③Tool-Integrated Reasoning（Toolformer、ReAct、ToolLLM、MCP）④Multi-Agent Systems（AutoGen、MetaGPT、CAMEL、CrewAI、KQML/FIPA ACL/MCP/A2A/ACP/ANP 等の通信プロトコル）。最後に Evaluation・Future Directions の章で各カテゴリのベンチマークと未解決課題を網羅する。
- **結果**: 新しい実験結果は持たない survey。引用先の数値を要約として並べる形：GAIA で人間 92% vs GPT-4 15%、GTA で GPT-4 <50%、WebArena leaderboard 1位は IBM CUGA 61.7%（2位 OpenAI Operator 58.1%）、商用 AI アシスタントは長い対話で精度が 30% 落ちる、Self-Refine/Reflexion で GPT-4 が約 20% 改善、GraphWiz が graph reasoning で 65% 平均精度（GPT-4 の 43.8% を上回る）、RAG/superposition prompting で text navigation 精度が 18 倍、few-shot で bug fixing の exact match +175.96% 等（why_ce.tex / evaluation.tex / future.tex）。tool 用ベンチマークの規模も整理：BFCL 2,000 ケース、T-Eval 553 ケース、API-Bank 73 API・314 対話、ToolHop 995 query × 3,912 tool、LongMemEval 500 問。
- **貢献**: (1) Context Engineering を prompt engineering の上位概念として **数式付きで定義**（$C = \mathcal{A}(\cdot)$、$\mathcal{F}^*$、Bayesian/info-theoretic な目的関数）。(2) 二層 taxonomy（Foundational Components × System Implementations × Evaluation × Future Directions）を taxonomy forest（intro.tex の forest 図）と timeline 図 (tree.pdf)、framework 図 (ce.pdf) で可視化。(3) 1400+ 論文の体系的レビューと、評価ベンチマーク・通信プロトコル・未解決課題のリスト化。(4) **理解 vs 生成の非対称性** を defining priority な open problem として明示（abstract）。

## Takeaway（自分にとっての要点）

- 著者の中心命題は「prompt = 静的文字列」から「context = 動的に組み立てられる情報パイプライン」への視点転換。$C = \mathcal{A}(c_{\text{instr}}, c_{\text{know}}, c_{\text{tools}}, c_{\text{mem}}, c_{\text{state}}, c_{\text{query}})$ という分解は、自分が普段書いている agent コードのモジュール境界（system prompt / retriever / tool spec / chat history / state / user query）にそのまま対応するので、設計レビューのチェックリストに使える。
- Retrieve を「$Y^*$ との mutual information $I(Y^*; c_{\text{know}} | c_{\text{query}})$ を最大化する写像」と書く定式化は、semantic similarity だけで retriever を評価しがちな実務に対する強い指針。「似ているが答えに寄与しない文書」を切る根拠になる。
- 全体構造として「**Foundational Components** が部品、**System Implementations** がそれらを組み合わせた architecture」という二層分離はクリーンで、自分が論文を漁る時の引き出しになる。例えば Agentic RAG は RAG (system) ⊃ Context Retrieval (component) + Self-refinement (component) + Tool calling (system) という重ね合わせとして読める。
- 著者が重要課題として据えたのが **comprehension-generation asymmetry**：理解側はベンチマーク満点に近づくのに、long-form 生成は coherence / factuality / planning で大きく崩れる。これは今後論文を読むときの観察軸として強力。
- 評価まわりで BLEU / ROUGE / perplexity が context-engineered system の評価には根本的に不適、と明言（evaluation.tex §6.3.1 Methodological Limitations）。代わりに self-refinement の改善幅、tool 呼び出しの trajectory 全体、orchestration の transactional integrity を測れと言っている。
- Multi-agent 通信プロトコルが Anthropic MCP / Google A2A / IBM ACP / ANP と既に乱立しており、相互運用性 + セキュリティが production の鍵になる、という記述は実務寄りの重要情報（future.tex §7.3.2 Large-Scale Multi-Agent Coordination）。
- 「graph を language に変換するか、専用エンコーダを足すか」は未解決の二項対立として整理されており、GraphRAG / GraphGPT / GraphWiz / G1 の位置付けが俯瞰できる（future.tex §7.2.3 Complex Context Organization and Solving Graph Problems）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「Context Engineering」を単なる用語提唱で終わらせず、$\mathcal{F}^*$ の argmax と情報理論的 / Bayesian な目的関数まで書き下した点で、survey でありながら ontology + 数式の両輪を提供している。フィールド入門の最初の 1 本として強い。
  - taxonomy（intro.tex の forest tree 図）が Components / Implementations / Evaluation / Future の 4 レーンに統一されており、論文の各 § と図がきっちり対応している。1400 本という規模も担保感がある。
  - comprehension-generation gap を "defining priority for future research" と踏み込んで提示した点は、ありがちな「全部頑張りましょう」型の survey 結論より誠実。
  - related_work.tex (§2) で既存の縦割り survey との差分（RAG ↔ memory ↔ tool-use ↔ prompt engineering を別々にレビューしてきた）を明示して、自分のポジションを正当化している。
- **弱み / 疑問**:
  - 数式は「最適化問題として書ける」と宣言する役を果たすが、$\mathcal{F}^*$ の具体的な解法・近似アルゴリズムは本文中に提示されない。式 (3)（$\mathcal{F}^*$）と式 (4)（mutual information 最大化 Retrieve）は **definitional であって prescriptive ではない**。survey として許容範囲だが、「だからどうすれば良いのか」は読者に委ねられる。
  - 引用された数値（GAIA 92% vs 15%、Self-Refine +20%、GraphWiz 65% など）はすべて元論文からの孫引きで、比較条件（モデル、prompt、シード、データ split）の整合性は survey 内では保証されていない。WebArena leaderboard 表（evaluation.tex tab:x_webarena_leaderboard）も Open Source ✓/× が混在で読むときは慎重に。
  - 「1400+ 本レビュー」と言いながら、taxonomy forest に挙がっている代表手法は各 leaf 10〜15 件程度。1400 本の選別基準・カバー率は本文では明示されない（Acknowledgments で著者自身 "may have been inadvertently overlooked" と明確に limitations として認めている）。
  - Context Engineering が prompt engineering の **真の上位概念** なのか、それとも単に「prompt + RAG + memory + tool + agent をまとめて呼び直しただけ」のリブランディングなのかは議論の余地がある。新しい数式 (1)-(6) は autoregressive LM の標準的記述で、$c_i$ への分解も実務では既知。学術的 novelty は taxonomy にあり、formalism 単体ではない。
  - 評価章で「BLEU/ROUGE/perplexity は不適」と切る一方で、代わりとなる **統一されたメトリクス** は提示されない。SagaLLM / MCP-RADAR / LongMemEval など個別フレームワークを並列に紹介するに留まる。
  - 安全性・alignment の章（future.tex §7.4.2 Safety, Security, and Robustness）は重要度の宣言が中心で、具体的な評価プロトコルは薄い。
  - 著者自身の limitations（Acknowledgments）: 「分野が急速に進化しているため最近の研究を見落としているかもしれない」とだけ書かれており、taxonomy の選択バイアスや、中国研究機関中心の著者構成（ICT CAS + UCAS + Peking + Tsinghua、UC Merced と Queensland のみ非中国）に起因しうるカバレッジ偏向については触れていない（評者補足）。
- **次に試したいこと**:
  - $C = \mathcal{A}(c_{\text{instr}}, c_{\text{know}}, c_{\text{tools}}, c_{\text{mem}}, c_{\text{state}}, c_{\text{query}})$ という分解を実装上のチェックリストに落とし、自分が触っている agent コードに対して「6 種のうち欠けている $c_i$ は何か」「$\mathcal{A}$ の順序は最適化されているか」を機械的に監査する。
  - $I(Y^*; c_{\text{know}}|c_{\text{query}})$ をプロキシで定量化する小実験（例: retriever をスコアで並べた時、上位文書を ablation したときの answer log-prob 落差を MI 代用にする）。これで「semantic similarity だけの retriever」を超えられるか検証。
  - comprehension-generation gap を切り分ける mini-bench: 同じドキュメントを与えて (a) QA、(b) 同等の length の解説生成、を同モデルで比較し、coherence / factuality がどこで割れるかを観察。
  - MCP / A2A / ACP / ANP の仕様を読み比べて、4 規格の **真の差分** がプロトコル上にあるのか、ユースケースの違いだけなのかを整理。
  - 自分で書く agent 評価フレームワークに、本論文 evaluation 章（§6.3 Evaluation Challenges and Emerging Paradigms）の self-refinement 改善幅・tool trajectory 完遂率・transactional integrity の 3 軸を取り込めるか試す。

## Notes / Quotes

- "Context Engineering re-conceptualizes the context $C$ as a dynamically structured set of informational components, $c_1, c_2, \dots, c_n$." (why_ce.tex §3.1 Definition of Context Engineering)
- 6 つの context 構成要素: $c_{\text{instr}}, c_{\text{know}}, c_{\text{tools}}, c_{\text{mem}}, c_{\text{state}}, c_{\text{query}}$（why_ce.tex §3.1）。
- 最適化目的: $\mathcal{F}^* = \arg\max_{\mathcal{F}} \mathbb{E}_{\tau \sim \mathcal{T}}[\text{Reward}(P_\theta(Y|C_\mathcal{F}(\tau)), Y^*_\tau)]$、制約 $|C| \le L_{\max}$（式 (3) eq:ce_optimization）。
- 情報理論的 Retrieve: $\text{Retrieve}^* = \arg\max I(Y^*; c_{\text{know}} | c_{\text{query}})$（式 (4) eq:info_theoretic_retrieval）。
- "a fundamental asymmetry exists between model capabilities. While current models ... demonstrate remarkable proficiency in *understanding* complex contexts, they exhibit pronounced limitations in *generating* equally sophisticated, long-form outputs." (abstract / conclusion §8)
- GAIA: 人間 92%、GPT-4 15%（evaluation.tex §6.3.1、future.tex §7.2.2）。
- GTA benchmark: GPT-4 がタスクの 50% 未満しか完遂できない（evaluation.tex §6.1.2）。
- WebArena leaderboard 上位（evaluation.tex tab:x_webarena_leaderboard, §6.2.2）: IBM CUGA 61.7%、OpenAI Operator 58.1%、Jace.AI 57.1%、ScribeAgent+GPT-4o 53.0%、AgentSymbiotic 52.1%。
- Self-Refine / Reflexion / N-CRITICS で GPT-4 が約 20% 改善（evaluation.tex §6.1.1、future.tex §7.2.4）。
- GraphWiz 65% 平均精度 vs GPT-4 43.8%（graph reasoning タスク群、future.tex §7.2.3）。
- 商用 AI アシスタントは長い対話で精度が **30% 低下**（LongMemEval、evaluation.tex §6.1.2 / §6.3.1）。
- Tool 評価ベンチマーク規模: BFCL 2,000 testing cases、T-Eval 553 tool-use cases、API-Bank 73 APIs・314 dialogues、ToolHop 995 queries × 3,912 tools、LongMemEval 500 questions（evaluation.tex §6.1.2 / §6.2.2）。
- "Traditional evaluation metrics prove fundamentally inadequate ... Static metrics like BLEU, ROUGE, and perplexity ... fail to assess complex reasoning chains, multi-step interactions, and emergent system behaviors." (evaluation.tex §6.3.1 Methodological Limitations and Biases)
- Multi-agent 通信プロトコル 4 種: MCP（Anthropic、"USB-C for AI"）、A2A（Google、Agent-to-Agent）、ACP（IBM、Agent Communication Protocol）、ANP（Agent Network Protocol）（future.tex §7.3.2）。
- 著者自認の limitation: "we acknowledge that despite our best efforts, some recent works or emerging trends may have been inadvertently overlooked or underrepresented." (Acknowledgments)
- (verified 2026-05-20) セクション番号引用を main.tex 上の実際の章番号と整合させ修正 (evaluation §5→§6、future §6→§7、conclusion §7→§8、why_ce §2.1→§3.1、related_work §1→§2、§4.3.1→§6.3.1)。
- (verified 2026-05-20) 「Figure 2: ce.pdf / Figure 1: tree.pdf」の誤った図番号付与を削除し、taxonomy forest（intro.tex）/ tree.pdf timeline / ce.pdf framework の 3 図構成として記述。
- (verified 2026-05-20) "Table 5" の番号引用を削除し、TeX ラベル tab:x_webarena_leaderboard で参照に変更。
- (verified 2026-05-20) "field-defining priority" → 原文 abstract の "defining priority for future research" に合わせて訂正。
- (verified 2026-05-27) venue/year を TeX で確認できる範囲（main.tex の Date: July 21, 2025、掲載先明示なし）に限定。
- (verified 2026-05-27) Related Papers の Xia et al. / Singh et al. の表記を main.bbl の実タイトルに合わせて修正。

## Related Papers

- Brown et al. 2020 / Vaswani et al. 2017 — LLM / Transformer の出発点（intro）。
- Lewis et al. 2020 *RAG* — Context Retrieval の代表（複数章で引用）。
- Wei et al. 2022 *Chain-of-Thought* / Kojima et al. 2022 *Zero-shot CoT* / Yao et al. 2023 *Tree-of-Thoughts* / Besta et al. 2023 *Graph-of-Thoughts* — Context Generation の代表系統。
- Dao et al. 2022 *FlashAttention* / Xiao et al. 2023 *StreamingLLM* / Munkhdalai et al. 2024 *Infini-attention* / Peng et al. 2023 *YaRN* — long context 処理。
- Madaan et al. 2023 *Self-Refine* / Shinn et al. 2023 *Reflexion* — Context Processing の self-refinement。
- Zhong et al. 2023 *MemoryBank* / Packer et al. 2023 *MemGPT* / Li et al. 2025 *MemOS* — Memory Systems。
- Schick et al. 2023 *Toolformer* / Yao et al. 2022 *ReAct* / Patil et al. 2023 *Gorilla* / Qin et al. 2023 *ToolLLM* — Tool-Integrated Reasoning。
- Wu et al. 2023 *AutoGen* / Hong et al. 2023 *MetaGPT* / Li et al. 2023 *CAMEL* / Chang et al. 2025 *SagaLLM* — Multi-Agent Systems。
- Mialon et al. 2023 *GAIA* / Wang et al. 2024 *GTA* / Zhou et al. 2023 *WebArena* / Deng et al. 2023 *Mind2Web* / Jang et al. 2024 *VideoWebArena* / Bosse et al. 2025 *Deep Research Bench* — エージェント評価ベンチマーク。
- Patil et al. 2025 *BFCL* / Chen et al. 2023 *T-Eval* / Li et al. 2023 *API-Bank* / Ye et al. 2025 *ToolHop* / Gao et al. 2025 *MCP-RADAR* — tool 利用評価。
- Xia et al. 2025 *Minerva*（本文では LongMemEval 500 questions として説明）/ Kwon et al. 2025 *Embodied Agents Meet Personalization* / Kočiský et al. 2017 *NarrativeQA* — memory / 長期記憶評価。
- Anthropic *MCP* / Google *A2A* / IBM *ACP* / *ANP* — agent 間通信プロトコル。
- Gao et al. 2024 *Modular RAG* / Singh et al. 2025 *Agentic Reasoning and Tool Integration* / Guo et al. 2024 *LightRAG* / Edge et al. 2024 *GraphRAG* / Sarthi et al. 2024 *RAPTOR* / Gutiérrez et al. 2024 *HippoRAG* — RAG / agentic reasoning architecture の進化系。
