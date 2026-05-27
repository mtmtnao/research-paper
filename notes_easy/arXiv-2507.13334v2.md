# A Survey of Context Engineering for Large Language Models（Context Engineering を LLM システム設計の統一分類として定式化するサーベイ）

- arXiv: https://arxiv.org/abs/2507.13334
- 一次ソース: ../papers/arXiv-2507.13334v2/
- 正規ノート: ../notes/arXiv-2507.13334v2.md

---

## 一言で言うと

LLM の性能は推論時に与える context に強く依存するという前提から、prompt engineering, RAG, memory, tool use, multi-agent systems を **Context Engineering** という統一枠組みに整理し、1400 本超の研究を taxonomy と数式で位置づけるサーベイである。新しいモデルや実験結果を提示する論文ではなく、「context をどう生成・処理・管理・実装・評価するか」を研究分野として定義することが主な貢献である。

## 何を議論する論文か

- **問題設定**: 現代の LLM システムでは、入力は単なる prompt ではなく、system instruction、検索された外部知識、tool 定義、memory、状態、ユーザー query などを組み合わせた情報 payload である。しかし、prompt engineering / RAG / memory systems / tool-integrated reasoning / multi-agent systems は別々の研究領域として発展しており、関係が見えにくい。
- **対象範囲 / 仮定**: TeX の定式化では、LLM は自己回帰モデル $P_{\theta}(Y|C)$ として扱われ、context $C$ は $|C| \leq L_{\text{max}}$ という長さ制約を受ける。論文は Date: July 21, 2025 のサーベイで、TeX 中に会議・誌名は明示されていない。
- **既存研究との差分**: 既存の survey は prompt engineering, RAG, memory, tool use, agents などの縦割り分野を個別に整理してきた。この論文は、それらを「Foundational Components」と「System Implementations」に分ける横断的 taxonomy としてまとめる点を差分とする（`related_work.tex` の "Our Contribution"）。
- **この論文で答えたい問い**: LLM に渡す context はどの部品から成るのか、それらをどのように生成・処理・管理するのか、RAG / memory / tool / multi-agent システムではどう統合されるのか、そして現在の評価と未解決課題は何か。

## 背景と前提

- LLM は、与えられた context $C$ に条件づけて出力列 $Y$ を生成する。したがって、context はモデルの知識補完、行動制御、推論手順、外部環境との接続を決める中心的なインターフェースになる。
- この論文での **context** は、質問文だけではない。`why_ce.tex` では $c_{\text{instr}}$, $c_{\text{know}}$, $c_{\text{tools}}$, $c_{\text{mem}}$, $c_{\text{state}}$, $c_{\text{query}}$ が context の主要部品として列挙される。
- **Prompt Engineering** は $C=\text{prompt}$ という静的文字列の設計として位置づけられる。一方、**Context Engineering** は $C=\mathcal{A}(c_1,\dots,c_n)$ という動的な組み立て問題として扱われる（`why_ce.tex`, Table `tab:pe_vs_ce`）。
- **RAG** は外部知識 $c_{\text{know}}$ を取得する代表的実装、**Memory Systems** は過去情報 $c_{\text{mem}}$ を永続化する実装、**Tool-Integrated Reasoning** は $c_{\text{tools}}$ を介して外部関数や環境を使う実装、**Multi-Agent Systems** は $c_{\text{state}}$ と agent 間通信を含む状態管理・協調実装として読むと整理しやすい。
- 従来の BLEU, ROUGE, perplexity のような静的な生成指標は、複雑な reasoning chain、tool trajectory、multi-agent の emergent behavior を評価しきれない、というのが評価章の前提である（`evaluation.tex` §6.3.1）。

## 提案手法

### コアアイデア

著者は Context Engineering を、LLM に渡す情報 payload を **設計・最適化・管理する formal discipline** として定義する。中心は、context を単一文字列ではなく、複数の情報部品を assembly function $\mathcal{A}$ で組み立てた構造物として扱う点である。

taxonomy は二層で構成される。第一層の **Foundational Components** は、(1) **Context Retrieval and Generation**: prompt-based generation, external knowledge retrieval, dynamic context assembly、(2) **Context Processing**: long sequence processing, self-refinement, multimodal / relational / structured context、(3) **Context Management**: context window constraints, memory hierarchies, context compression である。第二層の **System Implementations** は、(1) RAG、(2) Memory Systems、(3) Tool-Integrated Reasoning、(4) Multi-Agent Systems である。

この構造は `intro.tex` の taxonomy figure、`main.tex` の timeline figure `fig:context_engineering_timeline`、framework figure `fig:context_engineering_framework`、および各 system figure `fig:rag_framework`, `fig:memory_framework`, `fig:tool_context_framework`, `fig:mas_framework` に対応している。

### 重要な定義・数式

$$
P_{\theta}(Y | C) = \prod_{t=1}^{T} P_{\theta}(y_t | y_{<t}, C)
$$

**式の意味**: LLM が context $C$ を条件として出力列 $Y=(y_1,\dots,y_T)$ を逐次生成するという標準的な自己回帰モデルである（`why_ce.tex`, 式 `eq:llm_generation`）。

**記号の定義**:
- $P_{\theta}$ ... パラメータ $\theta$ を持つ LLM の条件付き確率分布
- $Y=(y_1,\dots,y_T)$ ... 生成される出力列
- $y_t$ ... $t$ 番目に生成される token
- $y_{<t}$ ... $t$ より前に生成済みの token 列
- $C$ ... 推論時にモデルへ与えられる context

**この論文での役割**: Context Engineering の議論を、LLM の出力確率を変える入力側の設計問題として接続する出発点である。prompt engineering との差分も、この $C$ をどう見るかに置かれる。

$$
C = \mathcal{A}(c_1, c_2, \dots, c_n)
$$

**式の意味**: context $C$ は、複数の情報部品 $c_i$ を高レベルの assembly function $\mathcal{A}$ が組み立てたものだと再定義している（`why_ce.tex`, 式 `eq:context_assembly`）。

**記号の定義**:
- $C$ ... 最終的に LLM に渡される context
- $c_i$ ... context の構成部品
- $\mathcal{A}$ ... formatting, selection, concatenation などを含む組み立て関数
- $c_{\text{instr}}$ ... system instructions and rules
- $c_{\text{know}}$ ... RAG や knowledge graph などで得る external knowledge
- $c_{\text{tools}}$ ... available external tools の definitions and signatures
- $c_{\text{mem}}$ ... prior interactions からの persistent information
- $c_{\text{state}}$ ... user, world, multi-agent system の dynamic state
- $c_{\text{query}}$ ... user's immediate request

**この論文での役割**: taxonomy 全体の核になる式である。各章は $c_i$ をどう得るか、どう処理するか、どう圧縮・保存するか、どう system implementation に統合するかを説明している。

$$
\mathcal{F}^* = \arg\max_{\mathcal{F}} \mathbb{E}_{\tau \sim \mathcal{T}} [\text{Reward}(P_{\theta}(Y | C_{\mathcal{F}}(\tau)), Y^*_{\tau})]
$$

**式の意味**: task distribution $\mathcal{T}$ 上で、LLM の出力品質の期待値を最大化する context 生成関数群 $\mathcal{F}$ を探す、という Context Engineering の最適化問題である（`why_ce.tex`, 式 `eq:ce_optimization`）。TeX ではこの最適化は $|C| \leq L_{\text{max}}$ という context length limit の制約を受けると書かれている。

**記号の定義**:
- $\mathcal{F}=\{\mathcal{A}, \text{Retrieve}, \text{Select}, \dots\}$ ... context を生成・選択・組み立てる関数群
- $\mathcal{F}^*$ ... 期待 reward を最大化する理想的な関数群
- $\tau$ ... 具体的な task instance
- $\mathcal{T}$ ... task の分布
- $C_{\mathcal{F}}(\tau)$ ... task $\tau$ に対して $\mathcal{F}$ が生成した context
- $Y^*_{\tau}$ ... ground-truth or ideal output
- $\text{Reward}$ ... 出力品質を評価する関数
- $L_{\text{max}}$ ... モデルの context length limit

**この論文での役割**: Context Engineering を「良い prompt を手で書く技術」ではなく、検索、選択、圧縮、memory、tool 定義、状態管理を含む system-level optimization として位置づける根拠になる。ただし、TeX 中にはこの最適化問題を一般に解く具体的アルゴリズムは提示されていない。

$$
\text{Retrieve}^* = \arg\max_{\text{Retrieve}} I(Y^*; c_{\text{know}} | c_{\text{query}})
$$

**式の意味**: 外部知識検索は、query と表面的に似た文書を取るだけでなく、理想出力 $Y^*$ について最大限情報を与える $c_{\text{know}}$ を取る問題として表される（`why_ce.tex`, 式 `eq:info_theoretic_retrieval`）。

**記号の定義**:
- $\text{Retrieve}$ ... 外部知識を取得する関数または手続き
- $\text{Retrieve}^*$ ... 条件付き相互情報量を最大化する理想的 retrieval
- $I(Y^*; c_{\text{know}} | c_{\text{query}})$ ... query が与えられた条件で、外部知識が理想出力について持つ mutual information
- $c_{\text{know}}$ ... retrieved external knowledge
- $c_{\text{query}}$ ... user's immediate request
- $Y^*$ ... 理想出力

**この論文での役割**: RAG や knowledge retrieval を単なる semantic similarity ではなく、task-relevant information の選択問題として読むための式である。RAG 章の modular / agentic / graph-enhanced retrieval の議論に接続する。

TeX には続けて、Bayesian Context Inference の式 `eq:bayesian_inference` と decision-theoretic objective の式 `eq:decision_theoretic_objective` もある。これらは、不確実性の下で context posterior $P(C | c_{\text{query}}, \dots)$ を考え、候補 context の中から期待 reward が高い $C^*$ を選ぶという見方を補足する式である。

### 実装 / アルゴリズム上の要点

この論文は単一の実装アルゴリズムを提案するものではなく、Context Engineering を設計・評価するための分類を与えるサーベイである。実装側の読みどころは、以下の論点に分けられる。

- context 部品: $c_{\text{instr}}$, $c_{\text{know}}$, $c_{\text{tools}}$, $c_{\text{mem}}$, $c_{\text{state}}$, $c_{\text{query}}$ を区別し、どの部品が不足・過剰かを見る。
- Context Retrieval and Generation: prompt、few-shot examples、CoT / ToT / GoT、外部文書、knowledge graph、tool documentation などを取得または生成する部分として読む。
- Context Processing: long sequence processing、self-refinement、structured / relational / multimodal context の統合など、取得済み情報を変換・適応させる部分として読む。
- Context Management: finite context window、lost-in-the-middle、KV cache、memory hierarchy、context compression など、限られた context 予算内で情報を保持・圧縮・取り出す部分として読む。
- assembly function $\mathcal{A}$: formatting, priority-based selection, adaptive composition などで context 部品を最終入力にまとめる役割として読む。
- system implementation と evaluation: RAG, Memory Systems, Tool-Integrated Reasoning, Multi-Agent Systems で統合され、component-level と system-level の両方で評価課題が整理される。

## 実験・結果

- **データセット / ベンチマーク**: この論文自体は新規実験を行うのではなく、既存 benchmark を整理する。TeX に出る主な benchmark は、GAIA, GTA, WebArena, Mind2Web, VideoWebArena, Deep Research Bench, DeepShop, LongMemEval, NarrativeQA, QMSum, QuALITY, MEMENTO, BFCL, T-Eval, API-Bank, StableToolBench, NesTools, ToolHop, MCP-RADAR, GraphArena, NLGraph, GraphDO などである。
- **比較対象 / baseline**: 単一の実験 baseline ではなく、prompt engineering と Context Engineering、Naive / Advanced RAG と Modular / Agentic / Graph-Enhanced RAG、GPT-4 と human performance、WebArena leaderboard の各 agent、GraphWiz と GPT-4、single-agent と multi-agent systems などが比較対象として現れる。
- **指標**: RAG では precision, recall, relevance, factual accuracy、tool use では tool selection accuracy, parameter extraction precision, execution success rates, error recovery、BFCL では call accuracy, pass rates, win rates、web agent では success rate、memory では correctness, recall@5, response time, adaptation time、従来 NLP では BLEU-4 や exact match が使われる。
- **主な結果**: 著者の体系化としては、1400 本超の研究を Foundational Components / System Implementations / Evaluation / Future Directions に整理したことが結果である。引用先の数値として、Zero-shot CoT は MultiArith accuracy を 17.7% から 78.7% に改善し、Tree-of-Thoughts は Game of 24 success rate を 4% から 74% に上げ、Graph-of-Thoughts は ToT と比べて quality を 62% 改善し cost を 31% 下げたと紹介される（`context_acquisition.tex`）。
- **主な結果**: long context では、Mistral-7B の入力を 4K から 128K tokens に増やすと計算量が 122 倍、Llama 3.1 8B は 128K-token request あたり最大 16GB を要するとされる一方、StreamingLLM は sliding window recomputation に対して最大 22.2 倍高速、H$_2$O は throughput を最大 29 倍改善し latency を最大 1.9 倍削減するとされる（`context_processing.tex`）。
- **主な結果**: memory evaluation では LongMemEval が 500 questions を用い、commercial AI assistants は extended interactions で 30% accuracy degradation を示すとされる（`evaluation.tex`, `memory.tex`）。
- **主な結果**: tool-integrated reasoning では GTA benchmark で GPT-4 が tasks の 50% 未満しか完了できず、人間は 92% とされる。GAIA では humans 92% に対して GPT-4 15% とされる（`evaluation.tex`, `future.tex`）。
- **主な結果**: tool benchmark の規模として BFCL は 2,000 testing cases、T-Eval は 553 tool-use cases、API-Bank は 73 APIs と 314 dialogues、ToolHop は 995 queries と 3,912 tools を持つ（`evaluation.tex`, `tool_augmented_systems.tex`）。
- **主な結果**: WebArena leaderboard では、IBM CUGA 61.7%、OpenAI Operator 58.1%、Jace.AI 57.1%、ScribeAgent + GPT-4o 53.0%、AgentSymbiotic 52.1% が表 `tab:x_webarena_leaderboard` に示される。
- **主な結果**: graph reasoning では GraphWiz が diverse graph tasks で 65% average accuracy、GPT-4 は 43.8% とされる（`future.tex` §7.2.3）。
- **著者が主張する貢献**: (1) Context Engineering を prompt design を超える formal discipline として定義する、(2) Foundational Components と System Implementations を分ける taxonomy を示す、(3) evaluation と future challenges を含む technical roadmap を作る、(4) context 理解は強いが long-form output 生成は弱いという **comprehension-generation gap** を重要課題として明示する（abstract, conclusion）。

## 妥当性と限界

- **この主張を支える根拠**: 定義面では `why_ce.tex` の式 `eq:llm_generation`, `eq:context_assembly`, `eq:ce_optimization`, `eq:info_theoretic_retrieval` が根拠になる。分類面では `intro.tex` の taxonomy figure と `main.tex` の framework figure が、Foundational Components と System Implementations の対応を示す。評価面では `evaluation.tex` の benchmark 整理と WebArena 表が根拠になる。
- **著者が認めている limitations / future work**: `future.tex` は、unified theoretical foundations の欠如、情報理論的な context allocation の未整備、O(n²) attention scaling、comprehension-generation gap、multimodal / graph context の表現、intelligent context assembly、large-scale multi-agent coordination、production deployment、safety / security / robustness / ethics を未解決課題として列挙する。Acknowledgments では、分野が急速に変化しているため recent works or emerging trends が overlooked or underrepresented かもしれないと述べる。
- **読者として注意すべき点**: この論文の数値は多くが引用先研究の結果であり、TeX 中で統一条件の再実験として検証されているわけではない。したがって、数値は「この survey が引用している代表的 evidence」として読み、モデル、prompt、split、評価環境の厳密比較は元論文に戻って確認する必要がある。
- **読者として注意すべき点**: 式 (3) の $\mathcal{F}^*$ は Context Engineering の目標を明確にするが、一般的な解法や近似アルゴリズムを提供するものではない。実装上は retrieval, selection, compression, tool selection, memory update などの個別問題に分解して扱う必要がある。
- **追加で確認したい実験 / 疑問**: $I(Y^*; c_{\text{know}} | c_{\text{query}})$ を実際の retriever 評価でどう近似するか、context 部品 $c_i$ の寄与を ablation でどう分離するか、multi-agent orchestration の transactional integrity をどの benchmark で測るか、comprehension-generation gap を同一条件でどう切り分けるかは、本文だけでは解決されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Context Engineering** ... LLM のための information payload を設計・最適化・管理する discipline。prompt engineering の上位概念として位置づけられる。
- **Context** ... LLM に渡す情報全体。instruction、外部知識、tool 定義、memory、state、query を含む。
- **Assembly function $\mathcal{A}$** ... context 部品を formatting, selection, concatenation などで LLM 入力にまとめる関数。
- **Foundational Components** ... Context Retrieval and Generation、Context Processing、Context Management の三分類。システムを構成する基礎技術群。
- **System Implementations** ... RAG、Memory Systems、Tool-Integrated Reasoning、Multi-Agent Systems。基礎部品を統合した応用アーキテクチャ群。
- **Context Retrieval and Generation** ... prompt-based generation、external knowledge retrieval、dynamic context assembly を含む context の取得・生成部分。
- **Context Processing** ... 長文処理、self-refinement、structured / relational / multimodal context 統合など、取得済み情報を LLM が使いやすい形に変換する部分。
- **Context Management** ... context window、lost-in-the-middle、memory hierarchy、KV cache、compression など、有限資源内で context を維持・整理する部分。
- **RAG** ... parametric knowledge と external knowledge source を組み合わせる実装。論文では modular, agentic, graph-enhanced の発展形が整理される。
- **Memory Systems** ... LLM の stateless interaction を補うため、短期・長期 memory、外部 storage、retrieval、forgetting、reflection などを扱う実装。
- **Function Calling** ... LLM が structured output を通じて外部関数を選択し、引数を作り、実行結果を受け取る仕組み。
- **Tool-Integrated Reasoning** ... reasoning と external tool execution を組み合わせ、計算、検索、コード実行、API 呼び出しなどを推論過程に組み込む枠組み。
- **Multi-Agent Systems** ... 複数 agent が communication protocols, orchestration, coordination strategies を通じて協調する実装。
- **MCP / A2A / ACP / ANP** ... agent や tool の相互運用を支える通信プロトコル群。`future.tex` では標準化、security、scalability の課題と結びつけられる。
- **Comprehension-generation gap** ... context を理解する能力に比べて、同程度に洗練された long-form output を生成する能力が弱いという著者の重要課題。
- **Lost-in-the-middle** ... 長い context の中央に置かれた情報を LLM が取り出しにくい現象。Context Management の制約として扱われる。
- **MCP-RADAR** ... tool use capabilities を多次元に評価する framework。`evaluation.tex` では software engineering と mathematical reasoning domain の objective metrics として説明される。
- **LongMemEval** ... long-term memory capabilities を 500 questions で評価する benchmark。information extraction, temporal reasoning, multi-session reasoning, knowledge updates などを扱う。

## 読む順番の提案

- まず `main.tex` の abstract を読み、論文が「1400+ papers の survey」であり、新規実験論文ではないことを確認する。
- 次に `intro.tex` の taxonomy figure `fig:context-engineering-taxonomy` を見る。正規ノート `notes/arXiv-2507.13334v2.md` の Summary にある「Foundational Components × System Implementations」の全体像に対応する。
- その後、`why_ce.tex` §3.1 の式 (1)-(6) と Table `tab:pe_vs_ce` を読む。正規ノートの Takeaway にある $C=\mathcal{A}(\cdot)$、$\mathcal{F}^*$、mutual information retrieval、Bayesian Context Inference の議論につながる。
- 次に `main.tex` の framework figure `fig:context_engineering_framework` と、`context_acquisition.tex`, `context_processing.tex`, `context_management.tex` の冒頭を読む。すべてを精読する前に、三つの Foundational Components が何を担当するかを押さえる。
- system 側は `rag.tex`, `memory.tex`, `tool_augmented_systems.tex`, `multi_agent_systems.tex` の各冒頭と figure を先に見る。正規ノートの Related Papers は、ここで登場する代表手法を文献名でたどるために使う。
- 評価を重視するなら `evaluation.tex` §6.1-§6.3 を読む。WebArena 表 `tab:x_webarena_leaderboard`、GAIA / GTA / LongMemEval / BFCL / ToolHop の数値は正規ノートの Notes / Quotes にも対応している。
- 最後に `future.tex` と `conclusion.tex` を読み、comprehension-generation gap、理論不足、O(n²) scaling、memory / tool / multi-agent 評価、security と deployment の課題を確認する。正規ノートの Critical Thoughts は、この段階で読むと論文の限界を整理しやすい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2507.13334v2/`
- 正規ノート: `notes/arXiv-2507.13334v2.md`
