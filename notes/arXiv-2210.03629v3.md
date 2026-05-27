# ReAct: Synergizing Reasoning and Acting in Language Models

- arXiv: https://arxiv.org/abs/2210.03629
- source: ../papers/arXiv-2210.03629v3/
- authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- venue / year: ICLR 2023
- tags: [LLM, reasoning, agent, tool-use, prompting, decision-making]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: LLM の「推論」（CoT 系）と「行動」（action plan 生成・WebGPT 系）は別々に研究されてきた。CoT は static black box で外界に grounded していないため hallucination と error propagation が起きる。一方で SayCan/Inner Monologue 等の acting 系は high-level な reasoning や working memory を持たず、語彙的に状態を再記述する程度に留まる。
- **手法**: ReAct — 行動空間を $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ と拡張し、環境を変えない「thought（言語空間の action）」と通常の action を few-shot prompt で交互に生成させる。thought はサブゴール分解・常識注入・観測の要約・例外処理・検索 query 再定式化など何でもよい。reasoning 中心のタスクでは thought–action–observation を密に、decision-making のタスクでは thought をスパースに（モデル自身が出す位置を決める）。base model は frozen な PaLM-540B（GPT-3 text-davinci-002 でも追試）。HotpotQA は 6-shot、FEVER は 3-shot、ALFWorld は task type ごとに 3 つ手動アノテーション、WebShop は 1–2-shot。
- **結果**:
  - **HotpotQA (EM, PaLM-540B prompting)**: Standard 28.7 / CoT 29.4 / CoT-SC 33.4 / Act 25.7 / **ReAct 27.4** / CoT-SC→ReAct 34.2 / **ReAct→CoT-SC 35.1**（Supervised SoTA は 67.5）。ReAct 単体は CoT に僅か劣るが Act より良い。Hybrid（heuristic 切替）が最強。
  - **FEVER (Acc)**: Standard 57.1 / CoT 56.3 / CoT-SC 60.4 / Act 58.9 / **ReAct 60.9** / **CoT-SC→ReAct 64.6**（best）。
  - **ALFWorld (134 unseen games, success rate %)**: ReAct best-of-6 **71** / avg 57、Act best-of-6 45、ReAct-IM (Inner Monologue 風 dense feedback) best-of-6 53、BUTLER (10^5 expert traj で IL) 37。worst ReAct trial (48) でも Act/BUTLER の best を超える。Act 比の相対改善は 33–90%（平均 62%）。
  - **WebShop (500 test instructions)**: ReAct **Score 66.6 / SR 40.0**、Act 62.3 / 30.1、IL 59.9 / 29.1、IL+RL 62.4 / 28.7、Human Expert 82.1 / 59.6。one/two-shot で IL+RL を絶対値 10pt 超え。
  - **Human study (HotpotQA 各 50 traj × 4)**: CoT の失敗の 56% が hallucination、ReAct は 0%。代わりに ReAct は reasoning error 47%（特に同じ thought/action のループ）と search 結果が non-informative 23% が主要失敗モード。
  - **Finetuning (HotpotQA, 3,000 traj)**: prompting だと PaLM-8/62B では ReAct が四手法中最低だが、finetune 後は最強。PaLM-8B finetuned ReAct > PaLM-62B prompting all、PaLM-62B finetuned ReAct > PaLM-540B prompting all。
  - **GPT-3 追試 (Appendix)**: HotpotQA EM 30.8、ALFWorld 78.4 と PaLM-540B（29.4 / 70.9）を上回る。
- **貢献**: (1) reasoning と acting を交互に出す general prompting パラダイム ReAct を提案、(2) 4 つの異種タスク（HotpotQA / FEVER / ALFWorld / WebShop）で reasoning-only / acting-only / IL / IL+RL 等の baseline と比較、(3) acting in reasoning tasks と reasoning in interactive tasks の ablation・分析、(4) prompting setup の limitation と HotpotQA finetuning 初期実験を提示。

## Takeaway（自分にとっての要点）

- ReAct の核は「action space を言語空間 $\mathcal{L}$ で拡張し、環境を変えない thought を context に足す」こと。主実験では frozen PaLM-540B に few-shot trajectory を与えるだけで、domain-specific action と free-form thought を生成させている。
- 「reasoning structure は CoT が強く、factual grounding は ReAct が強い」という著者の観察から、heuristic で両者を切り替える hybrid が HotpotQA / FEVER の表中で最高値を出している。
- ALFWorld で `ReAct-IM` (Inner Monologue 風の dense external feedback thought) との比較 71 vs. 53 は、環境フィードバック中心の thought だけでは不十分で、サブゴール分解・次サブゴール決定・commonsense による探索先推定が効いていることを示す ablation。
- prompting で最弱だった ReAct が finetuning で最強に転じるのは、ReAct が PaLM-8/62B の in-context examples だけでは学びにくい一方、3,000 examples の finetuning では最も伸びる、という論文中の結果として重要。
- WebShop で human expert（SR 59.6）にまだ大差をつけられている点について、著者は expert humans が prompting-based methods より significantly more product explorations and query re-formulations を行うと述べている。
- ReAct の主要失敗モードは reasoning error 47%（同じ thought/action の反復を含む）と search result error 23%。著者は反復について greedy decoding の sub-optimal さを疑い、better decoding（例: beam search）を future work として挙げている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - prompt フォーマットがシンプルで、著者は使用 prompt を Appendix に載せ、GPT-3 追試と prompting code も reproducibility のために提示している。
  - reasoning-heavy（HotpotQA/FEVER）と action-heavy（ALFWorld/WebShop）を同じ paradigm で扱い、reasoning-only / acting-only / IL / IL+RL などの baseline と比較している。
  - 「ReAct-IM」という Inner Monologue 風の ablation を入れて、dense external feedback thought と sparse/flexible reasoning traces を分けて比較している点。ALFWorld の All success rate では ReAct best-of-6 71、ReAct-IM best-of-6 53。
  - prompting と finetuning の傾向が逆転（PaLM-8/62B prompting で最弱 → finetune で最強）するという発見は、ReAct が追加 training data で改善する可能性を示す根拠になっている。
  - 人手で 200 trajectory にラベル付けした failure mode 分析を載せており、hallucination 0% を主張する根拠が定量化されている。
- **弱み / 疑問**:
  - HotpotQA 単体では CoT-SC (33.4) に ReAct (27.4) は負けており、ReAct が prompting 表中で最高になるのは hybrid（ReAct→CoT-SC 35.1）のとき。少なくとも HotpotQA では、ReAct 単体の優位ではなく内部知識と外部知識の組み合わせが効いている。
  - ALFWorld の 71% は best-of-6 prompt の選択結果で、avg は 57%（Table 1）。ReAct 自体で avg→best-of-6 が 57→71 と 14pt 開いている（一方 ReAct-IM の avg→best-of-6 は 48→53、Act は Table 1 に best-of-6 のみで avg 値が報告されていない）。
  - 著者自身 limitations として認めている点: (a) 複雑タスクで demonstration が in-context window を超える、(b) prompting だけだと ReAct は学びにくく finetuning が要る、(c) reasoning error 47%（同じ thought/action を繰り返すループ）が頻発し greedy decoding に起因と示唆、(d) supervised SoTA から大差（HotpotQA 35.1 vs. 67.5）。
  - WebShop の比較は IL/IL+RL に対するもので、**token budget や計算コストの公正比較**にはなっていない（TeX 中には明示されていない / 評者補足）。
  - ReAct には同じ thought/action を繰り返す失敗パターンがあり、HotpotQA では 7 steps、FEVER では 5 steps で ReAct→CoT-SC に back off している。step 上限は「more steps will not improve ReAct performance」という実験観察に基づく。
  - HotpotQA は Wikipedia passage を前提にした multi-hop QA で、ReAct も Wikipedia API を使う。一般知識/未知ドメインで同じ hallucination 抑制効果が保てるかは本論文単体では示されていない（評者補足）。
- **次に試したいこと**:
  - 同じ token budget で CoT-SC（majority over 21）と ReAct→CoT-SC を pareto curve 上で並べる（評者補足）。
  - thought/action ループ問題に対し、beam search / temperature sampling / self-consistency over ReAct trajectory（trajectory レベルの voting）で何 pt 取り返せるか。著者も footnote で beam search を future work と言っている。
  - ReAct trajectory を追加 training data として使う方向を、論文の PaLM-8B/62B finetuning 実験より広い設定で検証する（評者補足）。
  - HotpotQA → 一般 web ドメイン（Wikipedia 以外）に拡張したとき、ReAct と CoT の hallucination 差が保てるか（評者補足）。
  - thought の "edit" による人手介入（Appendix の human-in-the-loop behavior correction）を定量化して評価する（著者も more systematic study を future work としている）。

## Notes / Quotes

- "We augment the agent's action space to $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ ... a thought $\hat{a}_t$ aims to compose useful information by reasoning over the current context." (method.tex)
- Wikipedia API は `search[entity]` / `lookup[string]` / `finish[answer]` の 3 種のみ。"significantly weaker than state-of-the-art lexical or neural retrievers. The purpose is to simulate how humans would interact with Wikipedia." (experiments_language.tex)
- "Hallucination is a serious problem for CoT, resulting in much higher false positive rate than ReAct (14% vs. 6%) in success mode, and make up its major failure mode (56%)." (experiments_language.tex)
- "While interleaving reasoning, action and observation steps improves ReAct's groundedness and trustworthiness, such a structural constraint also reduces its flexibility in formulating reasoning steps" (experiments_language.tex) — 著者が認める ReAct のトレードオフ。
- "one frequent error pattern specific to ReAct, in which the model repetitively generates the previous thoughts and actions ... We suspect that this could be due to the sub-optimal greedy decoding procedure" (experiments_language.tex)
- ALFWorld の ReAct-IM 比較: "ReAct substantially outperforms IM-style prompting (71 vs. 53 overall success rate)" (experiments_rl.tex)
- Finetuning: "PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods, and PaLM-62B finetuned ReAct outperforming all 540B prompting methods." (experiments_language.tex)
- Conclusion の limitation: "complex tasks with large action spaces require more demonstrations to learn well, which unfortunately can easily go beyond the input length limit of in-context learning." (discussion.tex)

## Related Papers

- Wei+ 2022, Chain-of-Thought Prompting — 推論側の直接の前身・主 baseline。
- Wang+ 2022, Self-Consistency / Rationale-Augmented Ensembles — CoT-SC として ReAct と組み合わされる。
- Kojima+ 2022, Zero-shot CoT — CoT 系の同時期論文。
- Zelikman+ 2022, STaR — 自己生成 trajectory で finetune するブートストラップ法。ReAct の finetune 実験のレシピ源。
- Nakano+ 2021, WebGPT — 「LM がブラウザを操作する」先行研究。RL + 人手 feedback に依存する点で ReAct と対照される。
- Ahn+ 2022, SayCan — LLM + affordance による robot 行動計画。
- Huang+ 2022, Inner Monologue — ReAct と最も近い先行研究。ReAct-IM はこれをエミュレートした ablation。
- Shridhar+ 2020, ALFWorld / BUTLER — 評価環境と IL ベースライン。
- Yang+ 2018, HotpotQA / Thorne+ 2018, FEVER / Yao+ 2022, WebShop — 評価データセット。
- Chowdhery+ 2022, PaLM / Brown+ 2020, GPT-3 — base LM。

- (verified 2026-05-20) Critical Thoughts の ALFWorld variance に関する記述を修正。Table 1（iclr2023/table/rl.tex）では Act は best-of-6 のみで avg が無く、ReAct-IM の avg→best-of-6 ギャップは 5pt（48→53）。「Act/ReAct/IM 全てが 10pt 以上のギャップ」という記述は誤りだったので、ReAct 単体の 14pt ギャップ（57→71）に絞って書き直し。
- (verified 2026-05-20) Summary / Takeaway / Critical Thoughts の数値・固有名詞・引用文を text/abstract, text/intro, text/method, text/experiments_language, text/experiments_rl, text/discussion, table/reasoning, table/rl, table/human_study, table/gpt3 と突き合わせ、それ以外の記述は TeX 根拠が取れている事を確認。
- (verified 2026-05-27) Takeaway / Critical Thoughts から TeX 外の現代エコシステム・現代モデルに関する断定を削除し、評者補足は明示した (iclr2023_conference.tex, text/method.tex, text/experiments_language.tex, text/experiments_rl.tex, text/discussion.tex, text/appendix.tex)
