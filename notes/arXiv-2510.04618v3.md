# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

- arXiv: https://arxiv.org/abs/2510.04618
- source: ../papers/arXiv-2510.04618v3/
- authors: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun (Stanford / SambaNova / UC Berkeley)
- venue / year: ICLR 2026 submission (preprint, v3)
- tags: [LLM, context-engineering, agent, prompt-optimization, test-time-adaptation, self-improvement]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: LLM 応用（agent・ドメイン推論）は重み更新ではなく「コンテキスト適応 (context adaptation)」に依存する方向にシフトしている。しかし既存の prompt optimizer / memory 系手法には 2 つの病理がある: (1) **brevity bias** — GEPA など多くの最適化器は「短く一般化された prompt」に収束し、ドメイン固有のヒューリスティクスや failure mode が削れる。(2) **context collapse** — Dynamic Cheatsheet のような「LLM が context を一括書き直し」する設計だと、ステップが進むうちに長文 context が急に短い要約に潰れて性能が崩落する（AppWorld で step 60 → 61 のあいだに 18,282 tokens / acc 66.7 → 122 tokens / acc 57.1、ReAct baseline の 63.7 をも下回る、background §2.2）。
- **手法**: **ACE (Agentic Context Engineering)**。context を「進化する playbook」として扱い、3 役割で分担する agentic フレームワーク（design §3）:
  - **Generator**: ReAct 等で query を解き、reasoning trajectory と「どの bullet が役立った/誤導したか」をフィードバック。
  - **Reflector**: 成功・失敗のトレースから具体的な lesson を抽出（最大 5 round の iterative refinement）。
  - **Curator**: lesson を「**delta entry** = id + helpful/harmful カウンタを持つ箇条書き bullet」にまとめ、決定的（非 LLM）ロジックで既存 context にマージ。
  - **Incremental delta updates**: 全書き換えではなく局所差分のみ → collapse 回避・並列マージ可能。
  - **Grow-and-refine**: 新 id は append、既存は in-place 更新、定期的に semantic embedding で重複 prune。
  - 評価は offline（system prompt 最適化、train で学習し test 評価、pass@1）と online（test を順に処理し、各 sample で予測→context 更新）両方。
- **結果**:
  - **AppWorld (DeepSeek-V3.1-671B + ReAct, Table 1)**: ReAct baseline 平均 42.4。offline は ICL 46.0 / GEPA 46.4 に対し ACE **59.4** (+17.0)。online は DC(CU) 51.9 に対し ACE **59.5** (+17.1)。GT label 無しでも +14.8 平均改善。test-challenge TGC は 41.5 → 66.0。
  - **AppWorld leaderboard (2025-09-20)**: ReAct+ACE 59.4% で IBM-CUGA (GPT-4.1 製、60.3%) と平均でほぼ並び、test-challenge では TGC で +8.4、SGC で +0.7 上回る（より小さい open-source モデル DeepSeek-V3.1 使用）。
  - **金融 (FiNER + Formula, Table 2)**: base 69.1。offline で ACE 81.9 (+12.8、ICL 69.6 / MIPROv2 70.9 / GEPA 72.5 を凌駕)。Formula は 67.5 → **85.5** (+18.0)。online でも ACE +7.5。一方、GT label もなく実行フィードバックも弱い設定では DC も ACE も劣化することがある (FiNER online no-label で ACE -3.4)。
  - **アブレーション (Table 3)**: Reflector 無 / multi-epoch 無 → 55.1、Reflector あり multi-epoch 無 → 56.8、フル ACE → 59.4。online も offline warmup ありで 56.1 → 59.5。
  - **コスト**: AppWorld offline で GEPA に対し adaptation latency **-82.3%** (53,898→9,517 s)、rollout 数 **-75.1%** (1,434→357)。FiNER online で DC に対し latency **-91.5%** (65,104→5,503 s)、token \$ コスト **-83.6%** ($17.7→$2.9)。長い context でも GPT-5.1 では **91.8% が KV cache 再利用**され、billed input token cost は raw 比 -82.6%。
  - **汎化**: GPT-OSS-120B / GPT-5.1 / Llama-3.3-70B でも一貫してゲイン（5–12 pt）。Llama-70B は相対的に gain が小さく、Reflector 質に依存することを再確認。
- **貢献**: (1) brevity bias / context collapse を「現象として定式化」し AppWorld の具体的な数値で示した、(2) Generator/Reflector/Curator + delta bullet + grow-and-refine という再利用可能なフレームワーク、(3) agent benchmark で open-source モデル + ACE が GPT-4.1 ベース production agent と並ぶ実証、(4) 実行フィードバックのみでラベルなし自己改善できることを示した、(5) cost / latency が下がる「長い context = 高コスト」ではないという主張（KV cache 観点込み）。

## Takeaway（自分にとっての要点）

- **「prompt は短く一般化が良い」という直感を真っ向から否定**している。LLM 向けには人間向け要約と違って、ドメインヒューリスティクス・失敗例・コードスニペットを全部詰めた「playbook」のほうが効くという立場。長 context モデル + KV cache 時代だからこそ取れる戦略。
- 設計上の肝は「**LLM に context を書き直させない**」点。Curator は LLM だが、マージ自体は決定的ロジックで bullet を id 単位で append / counter 更新する。これによって rewrite 起因の崩落を遮断している。`delta entry = id + helpful/harmful counter + content` という構造は、それ自体が memory entry のフォーマットとして真似しやすい。
- offline / online を同じ枠組みで扱える点が実装上ありがたい。offline warmup で初期 playbook を作ってから online で継続更新、という流れは AppWorld online で best (+17.1)。
- ラベルなしでも執行フィードバック（コード実行成否）で動く＝ agent 系では実用的。逆に「実行で正解判定できない」ドメイン (FiNER no-label) では ACE 自身も劣化する、と論文が明示している。**「feedback signal の質」が ACE の上限を決める**、というのが正直な絵。
- coverage が広く playbook が長くなる → 普通なら推論コストが膨らむ、が ACE は KV cache 前提で「reuse 91.8%」と数字を出してきている。同じ context を何度も叩く ICL ワークロードでは現実的にコストが乗らない、という議論はベンチに載せにくい点を丁寧に拾っている。
- アブレーションで「Reflector + multi-epoch + offline warmup」が積み上げで効いていて、どれを抜いても -3〜+5 pt 変動するのが見える。Dynamic Cheatsheet の主要欠点 (rewrite + Reflector 無し) をピンポイントで補強する設計、という読み方ができる。
- 「prompt 工学を捨てて RL/finetune に行くべき」という潮流に対して、cost と解釈性で押し返してくる種類の論文。selective unlearning や継続学習との接続も discussion で示唆されている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - context collapse を **具体的な step-level の token 数 / accuracy** で示している（18,282→122 tokens, 66.7→57.1）。失敗例を定量化しているので主張に信頼が乗る。
  - 同一 LLM (DeepSeek-V3.1) を Generator/Reflector/Curator 全部に使うことで「強い Reflector が弱い Generator を引き上げているだけ」の疑念を切っている (results §4.2)。
  - cost 比較が latency / rollout / token \$ の 3 軸で揃っており、しかも KV cache 再利用率まで実測 (91.8%) しているのは誠実。
  - GT label の有/無を Table の中に並べて、ablation 的に supervision の効き方まで見せている。online no-label で AppWorld は強いが FiNER は弱い、という非対称も隠していない。
  - leaderboard 比較で open-source モデルがクローズドな production agent と並ぶ、という「ぱっと響く」結果を持っている。
- **弱み / 疑問**:
  - 著者自身が limitations (§5) で書いている通り、**Reflector の質に強く依存**する。実行フィードバックが得にくいドメインでは context が "noisy or even harmful" になりうる、と明言（FiNER online no-label の -3.4 がまさにそれ）。
  - "context は長いほうが良い" を主張するが、HotPotQA や Game of 24 のように「短い指示で十分」なタスクでは ACE は不要だと自分でも認めている。**どこからが ACE 適用領域なのか、apriori に判別する基準は提示されていない**。
  - playbook が膨張する一方で、grow-and-refine の de-duplication が semantic embedding 一発であり、**embedding が誤って同義と判定して有用 bullet を消す**ケースのアブレーションは本文では見えない（incremental update の効果は appendix）。
  - AppWorld 以外の agent benchmark での評価は appendix と StreamBench（医療 DDXPlus / BIRD-SQL）に限定。**ベンチがドメイン特化系に偏っている**ので、open-ended な web agent や多モーダル系には外挿しにくい。
  - "KV cache 再利用で長 context は高くならない" は OpenAI API の prompt caching に依存した主張で、**self-host (vLLM 等) でも同じ 80% 級の billed コスト削減が出るかは未検証**。
  - GEPA との比較で auto="heavy" を使っているが、GEPA を ACE と同じ rollout 数で揃えた条件で打つとどうなるか（**iso-budget 比較**）が無い。GEPA は本来 sample-efficient を売っているので、彼らに「より多く回せた条件」を与えた場合の上限が気になる。
  - "production-level agent IBM CUGA と並ぶ" の比較は本文 footnote 自身が「方法論的比較ではなく文脈参照」と但し書きしており、見出しほどフェアな勝利ではない（agent engineering の差を込みにした参考値）。
  - context は人間可読、と謳うが Figure 3 の例は部分表示で、実際に何百〜何千 bullet になった playbook の可読性・運用性（誰がレビューするのか）は議論されていない。
- **次に試したいこと**:
  - 同じ delta bullet 設計で **Reflector を蒸留**する（強い Reflector のフィードバックを weak Generator+Curator に flow させ続けるとどこまで base が引き上がるか）。
  - playbook を **「diff 列」として保存**しておき、ある bullet を消したら spec の何が壊れるかを post-hoc に traceback できる仕組み（selective unlearning の具体実装）。
  - iso-budget で GEPA / TextGrad / Reflexion を並べた **rollout 数 vs accuracy の pareto** を AppWorld 上で引く。
  - reward が無い設定で、Generator の自信度 (logprob) や agent-self-judge を Reflector の代理シグナルとして使い、FiNER online no-label の劣化が回復できるかを検証。
  - bullet ベースの context を **RAG index と同じインフラに載せる**（id + counter + content をベクトル DB に）と運用がどう変わるか。

## Notes / Quotes

- "We argue that contexts should function not as concise summaries, but as comprehensive, structured playbooks that are detailed, inclusive, and rich with domain insights." (introduction)
- "at step 60 the context contained 18,282 tokens and achieved an accuracy of 66.7, but at the very next step it collapsed to just 122 tokens, with accuracy dropping to 57.1—worse than the baseline accuracy of 63.7 without adaptation." (background §2.2, context collapse の定量例)
- "Inspired by the agentic design of Dynamic Cheatsheet, ACE introduces a structured division of labor across three roles ... Generator ... Reflector ... Curator." (design §3)
- "bullet ... consists of (1) metadata, including a unique identifier and counters tracking how often it was marked helpful or harmful; and (2) content..." (design §3.1)
- "ACE achieves 82.3% reduction in adaptation latency and 75.1% reduction in the number of rollouts as compared to GEPA" (results §4.5, AppWorld offline)
- "91.8% of input tokens are served from cache during evaluation stage, which reduces billed input-token cost by 82.6% relative to counting raw context tokens." (results §4.5, GPT-5.1 prompt caching study)
- "A limitation of ACE is its reliance on a reasonably strong Reflector: if the Reflector fails to extract meaningful insights from generated traces or outcomes, the constructed context may become noisy or even harmful." (discussion §5)
- "ACE is most beneficial in settings that demand detailed domain knowledge, complex tool use, or environment-specific strategies that go beyond what is already embedded in model weights or simple system instructions." (discussion §5、ACE 不要なケースの裏返し)
- 主要ハイパーパラ: Reflector refinement の最大 round = 5、offline epoch 最大 = 5、batch size = 1 (1 sample で 1 delta) (results §4.2)。

## Related Papers

- Suzgun & Krause, *Dynamic Cheatsheet* (2025) — ACE が直接の比較対象かつ着想源。rewrite 起因の context collapse を見せるための代表例。
- Agrawal+ *GEPA* (2025) — reflective prompt evolution + genetic Pareto。brevity bias の代表例として批判される一方、offline の主 baseline。
- Opsahl-Ong+ *MIPROv2* (2024, DSPy) — 共通 baseline (ベイズ最適化系)。
- Shinn+ *Reflexion* (2023) / Yuksekgonul+ *TextGrad* (2024) — natural language feedback 系の系譜。
- Yao+ *ReAct* (2023) — AppWorld の公式 agent 実装、ACE もこの上に乗る。
- Trivedi+ *AppWorld* (2024) — メイン agent benchmark。
- Loukas+ *FiNER* (2022) / Wang+ *FinLoRA / Formula* (2025) — 金融 domain benchmark。
- Wu+ *StreamBench* (2024) / Fansi+ *DDXPlus* (2022) / Li+ *BIRD-SQL* (2023) — 追加ドメイン (医療 / text-to-SQL)。
- Xu+ *A-MEM* (2025) / Suzgun+ *Dynamic Cheatsheet* — bullet 構造の memory framework として比較。
- Gim+ 2024 / Yao+ *CacheBlend* 2025 / Liu+ *KIVI* 2024 — KV cache 再利用・圧縮の根拠付け文献。
- Marreed+ *IBM CUGA* (2025) — leaderboard 比較用の production agent。
