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
- **貢献**: (1) reasoning と acting を交互に出す general prompting パラダイム ReAct を提案、(2) 4 つの異種タスク（HotpotQA / FEVER / ALFWorld / WebShop）で CoT・Act・IL・IL+RL 等の baseline を網羅比較、(3) sparse な thought / 内部 vs. 外部知識のハイブリッド / human-in-the-loop な thought 編集など decision making での分析、(4) prompting と finetuning でのスケーリング挙動を分離して提示。

## Takeaway（自分にとっての要点）

- ReAct の核は「`Thought:` という出力フォーマットだけで action space を言語空間に拡張する」こと。新規モジュールも RL も不要で、prompt 構造だけで agent が立ち上がる。今の tool-use エコシステム（function calling、LangChain 系）の祖先がこの論文で、フォーマットの単純さこそが普及力の根源。
- 「reasoning に強い CoT」と「外部 grounding に強い ReAct」を heuristic で切り替えるハイブリッドが最強、という結果は重要。ReAct ≥ CoT ではなく、**両者は相補的で内部知識が確信できないときだけ取りに行く**という設計指針が transferable。
- ALFWorld で `ReAct-IM` (Inner Monologue 風に環境観測の再記述だけする thought) との比較 71 vs. 53 は、「思考＝観測の言い換え」ではダメで、サブゴール分解・常識・自己反省を入れて初めて効く、という ablation が綺麗。"thought とは何か" を定義する材料に使える。
- prompting で最弱だった ReAct が finetuning で最強に転じるのは、in-context での容量不足が主原因という解釈で、現在の long-context モデルや SFT データセット作りの方向性（trajectory SFT）と整合する。
- WebShop で human expert（SR 59.6）にまだ大差をつけられているのは、**explore と query 再定式化**の不足が原因と著者が明記。今の deep research 系エージェントの主戦場とほぼ同じ論点。
- ReAct の主要失敗モードは reasoning error 47%（特に thought/action ループ）と検索が空 23%。これは greedy decoding の sub-optimal さと検索ツールの脆弱さに帰着しており、根本的な手法の欠陥ではないという主張。逆に言うと**ツール側の品質と decoding 戦略がこの手の agent の伸びしろ**。

## Critical Thoughts（評価・疑問）

- **強み**:
  - prompt フォーマットがシンプルで、後段の追試・派生が極めて容易（実際に reproducibility が高く、エコシステムの基礎になった）。
  - reasoning-heavy（HotpotQA/FEVER）と action-heavy（ALFWorld/WebShop）を同じ paradigm で扱い、それぞれの baseline 群（CoT 系・IL/IL+RL 系）に対して fair 比較を組み立てた構成力。
  - 「ReAct-IM」という Inner Monologue 風の控えめな ablation を入れて、**何が effective component なのか**を分離した点。abstract reasoning 無しでは ALFWorld が −18pt 落ちると示せている。
  - prompting と finetuning の傾向が逆転（prompting で最弱 → finetune で最強）するという発見は、「ReAct が学べないのではなく in-context に詰め込めない」だけと言い切る上で説得力がある。
  - 人手で 200 trajectory にラベル付けした failure mode 分析を載せており、hallucination 0% を主張する根拠が定量化されている。
- **弱み / 疑問**:
  - HotpotQA 単体では CoT-SC (33.4) に ReAct (27.4) は負けており、ReAct が "勝つ" のは hybrid（35.1）のとき。タイトルの "synergy" は **ReAct と CoT のシステム的 hybrid** で初めて成立する、というのは abstract から少し控えめにしか読めない。
  - ALFWorld の 71% は best-of-6 prompt の選択結果で、avg は 57%。prompt 選択の variance がかなり大きく（Table 1 の Act/ReAct/IM ともに avg vs. best of 6 のギャップが 10pt 以上ある）、prompt 設計依存性は強そう。
  - 著者自身 limitations として認めている点: (a) 複雑タスクで demonstration が in-context window を超える、(b) prompting だけだと ReAct は学びにくく finetuning が要る、(c) reasoning error 47%（同じ thought/action を繰り返すループ）が頻発し greedy decoding に起因と示唆、(d) supervised SoTA から大差（HotpotQA 35.1 vs. 67.5）。
  - WebShop の比較は IL/IL+RL に対するもので、**token budget や計算コストの公正比較**にはなっていない（in-context prompting と数千～万 traj の学習が直接比較されている）。
  - ReAct の thought はモデルが内部で自由に書くので、`finish[]` を呼ばずに延々と探索したり、`search` を繰り返すループに陥る挙動が頻繁。ハードな step 上限・FEVER で 5 step、HotpotQA で 7 step というのは経験的に決めただけで、原理的解決ではない。
  - 比較する CoT の hallucination 失敗 56% は、ReAct が外部 grounding でカバーしている前提だが、HotpotQA は **そもそも Wikipedia ベースのタスク**で、ReAct に有利すぎる test bed。一般知識/未知ドメインでの hallucination 抑制効果は本論文単体では言えない。
- **次に試したいこと**:
  - 同じ token budget で CoT-SC（majority over 21）と ReAct→CoT-SC を pareto curve 上で並べる。ReAct のコスト効率の本当の優位を測りたい。
  - thought/action ループ問題に対し、beam search / temperature sampling / self-consistency over ReAct trajectory（trajectory レベルの voting）で何 pt 取り返せるか。著者も footnote で beam search を future work と言っている。
  - ReAct trajectory を SFT データに distill して small model（7–8B クラス）が prompting ReAct-540B にどこまで近づくか。論文自身の finetuning 実験を現代モデルで再現する自然な延長。
  - HotpotQA → 一般 web ドメイン（Wikipedia 以外）に拡張したとき、ReAct と CoT の hallucination 差が保てるか。
  - thought の "edit" による人手介入（論文 §4 で軽く触れている）を定量化して、agent supervision の interface としてどれだけ workable か測る。

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
