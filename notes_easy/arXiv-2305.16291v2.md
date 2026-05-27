# Voyager: An Open-Ended Embodied Agent with Large Language Models（GPT-4 を用いた Minecraft の open-ended embodied lifelong learning agent）

- arXiv: https://arxiv.org/abs/2305.16291
- 一次ソース: ../papers/arXiv-2305.16291v2/
- 正規ノート: ../notes/arXiv-2305.16291v2.md

---

## 一言で言うと

Minecraft のような固定ゴールを持たない 3D 環境で、GPT-4 に次のタスクを提案させ、JavaScript コードとしてスキルを獲得・保存・再利用させる embodied lifelong learning agent、Voyager を提案する論文。著者は、Voyager が 160 prompting iterations で 63 unique items を発見し、既存 LLM-agent baseline より $3.3 \times$ 多い item、$2.3 \times$ 長い移動距離、最大 $15.3 \times$ 速い tech tree 到達を示したと主張している（abstract, Experiments / Evaluation Results, Table 1）。

## 何を議論する論文か

- **問題設定**: open-ended world で、エージェントが自分で適切なタスクを提案し、環境フィードバックからスキルを改善し、成功したスキルを記憶して、長時間にわたり探索を続けられるかを扱う。Introduction では、人間プレイヤーが Minecraft で木材採取から diamond tools へ進むように、エージェントにも「propose suitable tasks」「refine skills」「continually explore the world」が必要だと述べる。
- **対象範囲 / 仮定**: 実験環境は MineDojo 上の Minecraft で、制御には Mineflayer JavaScript APIs を使う。Voyager は Minecraft screen pixels から低レベル操作を直接出す手法ではなく、high-level Mineflayer API とテキスト化された feedback / meta / inventory を前提にする（Experiments / Baselines, Appendix の system-level comparison）。
- **既存研究との差分**: RL / imitation learning 系は primitive actions 上で動くため、systematic exploration・interpretability・generalization が難しいと位置づけられる。ReAct, Reflexion, AutoGPT などの LLM agent はあるが、著者はそれらを「progressively acquire, update, accumulate, and transfer knowledge over extended time spans」できる lifelong learner ではないと見る。Voyager の差分は、automatic curriculum、self-generated skill library、3 種類の feedback を用いる iterative prompting mechanism を同時に持つ点にある（Fig. 2, Appendix の system-level comparison）。
- **この論文で答えたい問い**: GPT-4 を black-box API として使うだけで、Minecraft 内で新しいタスクを自律的に選び、実行可能コードのスキルを蓄積し、新しい world / unseen tasks に転用できるか。さらに、automatic curriculum、skill library、environment feedback、execution errors、self-verification、GPT-4 code generation の各要素が実際に性能を支えているかを ablation で検証する。

## 背景と前提

- **Open-ended embodied agent**: この論文での embodied agent は、Minecraft 内の位置、inventory、equipment、nearby blocks / entities、biome、health、hunger などの状態を持ち、環境内で行動する agent を指す。open-ended は、固定された end goal や storyline がなく、agent 側が新しい目的を選び続ける状況を指す。
- **Minecraft が難しい理由**: Minecraft では procedurally generated 3D terrains を探索し、resource gathering と crafting により wooden tool $\rightarrow$ stone tool $\rightarrow$ iron tool $\rightarrow$ diamond tool のような tech tree を進む必要がある。論文は、この構造が systematic and compositional skills を要求すると説明する（Experiments / Evaluation Results, Table 1）。
- **LLM を使う前提**: Voyager は GPT-4 を prompting / in-context learning で使い、model parameter access、gradient-based training、finetuning を不要にする。実験では `gpt-4-0314`、`gpt-3.5-turbo-0301`、`text-embedding-ada-002` を使い、temperature は automatic curriculum だけ 0.1、その他は 0 に設定する（Experiments / Experimental Setup）。
- **Code as action space**: 論文は低レベル motor commands ではなく、Mineflayer を呼び出す JavaScript program を action space にする。program は temporally extended で compositional な action を表せるため、long-horizon Minecraft tasks に向くというのが著者の設計理由である（Introduction, Method / Skill Library）。
- **Baseline との関係**: ReAct は reasoning traces と action plans を生成する LLM agent、Reflexion は ReAct に self-reflection を加えたもの、AutoGPT は high-level goal を subgoals に分解して ReAct-style loop で実行するものとして再実装される。これらは元々 Minecraft 用ではないため、著者は MineDojo 実験設定に合わせて re-interpret したと明記している（Experiments / Baselines, Appendix / Baselines）。
- **直接比較しない手法**: VPT、DreamerV3、DECKARD、DEPS、Plan4MC などの prior Minecraft agents は関連研究や Appendix の system-level comparison で扱われる。ただし本文の直接比較対象は ReAct / Reflexion / AutoGPT であり、Voyager は high-level Mineflayer API を使うため、screen pixels から low-level controls を出す既存手法とは apple-to-apple でないと著者は断っている（Experiments / Baselines, Related work, Appendix の system-level comparison）。

## 提案手法

### コアアイデア

Voyager は 3 つのモジュールからなる（Fig. 2）。1 つ目は **automatic curriculum** で、GPT-4 が「discover as many diverse things as possible」という上位目標、agent state、completed / failed tasks、GPT-3.5 による追加 context を受け取り、次の単一タスクを提案する。2 つ目は **skill library** で、成功した行動を executable code として保存し、program description の embedding を key、program を value とする vector database として管理する（Fig. 4）。3 つ目は **iterative prompting mechanism** で、generated program を実行し、environment feedback、execution errors、self-verification の critique を次ラウンドの prompt に戻してコードを改善する（Method / Iterative Prompting Mechanism）。

重要なのは、Voyager がスキルを自然言語の計画ではなく JavaScript 関数として保持する点である。例えば論文中では `craftStoneShovel()` や `combatZombieWithSword()` のような関数が例示され、複雑なスキルは既存の簡単な program を compose して作ると説明される。これにより、skill は再利用可能で、人間にも読め、古い skill を model weights に上書きしないため catastrophic forgetting を緩和できる、というのが著者の主張である。

### 重要な定義・数式

TeX 中に、この手法の中核となる目的関数・更新式・評価式はほぼ明示されていない。したがって、ここでは数式を作らず、本文と Appendix で定義されている主要な設計要素を整理する。

- **Automatic curriculum**: GPT-4 による bottom-up task proposal。prompt には、diverse behavior を促す directives、agent の current state、previously completed and failed tasks、GPT-3.5 の self-ask / self-answer による additional context が入る（Method / Automatic Curriculum, Appendix / Automatic Curriculum）。prompt の条件では、次タスクは single phrase で、難しすぎず、novel and interesting で、視覚確認が必要な placing / building / planting / trading tasks は避けるよう指定される（`appendix/prompts/curriculum_prompt.txt`）。
- **Skill library**: skill は「specific task proposed by the automatic curriculum」を完了する executable code として表現される。新しい skill は GPT-4 が生成・self-verification が確認した後に追加され、program description の embedding で index される。retrieval 時は self-generated task plans と environment feedback の embedding を query context とし、top-5 relevant skills を prompt に入れる（Method / Skill Library, Fig. 4, Appendix / Skill Library）。
- **Iterative prompting mechanism**: 各ラウンドで generated program を実行し、Minecraft simulation から observations / chat log、interpreter から execution errors、critic GPT-4 から success 判定と critique を得る。self-verification が task completion を確認すれば skill library に追加し、4 rounds で行き詰まれば curriculum に別タスクを要求する（Method / Iterative Prompting Mechanism, Appendix pseudocode）。
- **Self-verification**: 各タスクに手書き success checker を作る代わりに、別の GPT-4 agent を critic として使う。入力は agent state と task で、出力は JSON の `reasoning`, `success`, `critique` である。著者は、これは成功判定と mistake reflection の両方を行うため Reflexion の self-reflection より comprehensive だと述べる（Method / Iterative Prompting Mechanism, `appendix/prompts/critic_prompt.txt`）。
- **Prompting iteration**: 表では明示式ではなく実験単位として使われる。Table 1 と Table 2 の caption は、成功した trial 数を 3 回中の fraction で示し、数値は prompting iterations の平均であり、少ないほど efficient と説明している。

### 実装 / アルゴリズム上の要点

- step1: `environment.reset()` で agent state を得る。curriculum agent は completed tasks と failed tasks から exploration progress を作り、agent state と合わせて `propose_next_task` を呼ぶ（Appendix pseudocode）。
- step2: 各タスクについて最大 4 rounds 実行する。round ごとに skill manager が task と environment feedback から関連 skill を retrieval し、action agent が task、前回 code、environment feedback、execution errors、critique、retrieved skills を使って JavaScript code を生成する。
- step3: `environment.step(code)` でコードを実行し、更新後の agent state、environment feedback、execution errors を得る。その後 critic agent が `check_task_success(task, agent_state)` を実行する。
- step4: success なら `skill_manager.add_skill(code)` で skill library に追加し、curriculum に completed task として登録する。失敗のまま 4 rounds を終えた場合は failed task として登録し、次の task proposal に進む。
- step5: action prompt では `exploreUntil`, `mineBlock`, `craftItem`, `placeItem`, `smeltItem`, `killMob`, chest 操作用 API などの control primitive と Mineflayer API が与えられる。GPT-4 には `bot.chat` で intermediate progress を出すこと、infinite loops や event listeners を避けること、関数名を意味のあるものにすることが指定される（Appendix / Skill Library, `appendix/prompts/action_prompt.txt`）。

## 実験・結果

- **データセット / ベンチマーク**: MineDojo 上の Minecraft。主な評価は open-ended exploration、tech tree mastery、map coverage、newly instantiated world における zero-shot generalization to unseen tasks。探索と tech tree は最大 160 prompting iterations、zero-shot tasks は最大 50 prompting iterations で評価される（Experiments / Evaluation Results, Table 1, Table 2）。
- **比較対象 / baseline**: ReAct、Reflexion、AutoGPT。追加比較として `Voyager w/o Skill Library`、`AutoGPT w/ Our Skill Library`、ablation variants（Manual Curriculum、Random Curriculum、w/o Environment Feedback、w/o Execution Errors、w/o Self-Verification、GPT-3.5 など）が使われる（Experiments / Baselines, Experiments / Ablation Studies, Appendix / Ablations）。
- **指標**: unique items discovered、prompting iterations to unlock tech tree levels、map traversal distance、unseen task の success fraction out of three trials、skill retrieval accuracy。Table 1 / Table 2 では N/A (0/3) は最大 iteration 内に失敗したことを意味する。
- **主な結果**: Voyager は 160 prompting iterations で 63 unique items を発見し、baseline より $3.3 \times$ 多いと報告される。map coverage では baselines より $2.3 \times$ 長い距離を移動する（Experiments / Evaluation Results, `figures/main_experiment.tex`, `figures/map.tex`）。
- **Tech tree mastery**: Table 1 では、Voyager は Wooden Tool を $6 \pm 2$ iterations (3/3)、Stone Tool を $11 \pm 2$ (3/3)、Iron Tool を $21 \pm 7$ (3/3)、Diamond Tool を 102 (1/3) で unlock する。AutoGPT は Wooden $92 \pm 72$ (3/3)、Stone $94 \pm 72$ (3/3)、Iron $135 \pm 103$ (3/3)、Diamond は N/A (0/3)。ReAct と Reflexion は全レベル N/A (0/3)。著者は、Voyager が wooden level を $15.3 \times$、stone level を $8.5 \times$、iron level を $6.4 \times$ 速く unlock し、diamond level に到達した唯一の手法だと述べる（Experiments / Evaluation Results, Table 1）。
- **Zero-shot generalization**: inventory を clear し、新しい world に reset して unseen tasks を評価する。Voyager は Diamond Pickaxe $19 \pm 3$ (3/3)、Golden Sword $18 \pm 7$ (3/3)、Lava Bucket $21 \pm 5$ (3/3)、Compass $18 \pm 2$ (3/3) を達成する。ReAct、Reflexion、AutoGPT は 4 tasks すべて N/A (0/3)。AutoGPT w/ Our Skill Library は Diamond Pickaxe 39 (1/3)、Golden Sword 30 (1/3)、Lava Bucket N/A (0/3)、Compass 30 (2/3) で、skill library が plug-and-play asset として他手法にも効くという著者の主張を支えている（Table 2）。
- **Ablation**: random curriculum に置き換えると discovered item count が $93\%$ 落ちる。self-verification を除くと $-73\%$。code generation を GPT-3.5 に置き換えると、GPT-4 版 Voyager は $5.7 \times$ 多く unique items を得る。skill library を外すと later stages で plateau しやすいと説明される（Experiments / Ablation Studies, `figures/ablation.tex`）。
- **Multimodal feedback from humans**: 著者は、当時利用可能な GPT-4 API が text-only であるため Voyager は visual perception を現在サポートしないと書く。その上で、人間が critic または curriculum の役割を担うと Nether Portal や house のような 3D structures を構築できることを示す（Experiments / Multimodal Feedback from Humans, `figures/human.tex`）。
- **著者が主張する貢献**: Introduction と abstract では、Voyager を「the first LLM-powered embodied lifelong learning agent in Minecraft」とし、automatic curriculum、ever-growing executable-code skill library、environment feedback / execution errors / self-verification を組み込む iterative prompting により、Minecraft で継続的に探索・skill acquisition・new discoveries を行えると主張する。

## 妥当性と限界

- **この主張を支える根拠**: exploration、tech tree、map coverage、zero-shot generalization の複数軸で ReAct / Reflexion / AutoGPT と比較している。特に Table 2 では新 world・empty inventory・unseen tasks で評価し、skill library の転用性を AutoGPT w/ Our Skill Library でも確認している。さらに ablation で automatic curriculum、skill library、self-verification、GPT-4 code generation の寄与を個別に落としている。Appendix では 309 samples の skill retrieval 評価で Top-5 Acc $96.5 \pm 0.3$ も示される。
- **著者が認めている limitations / future work**: GPT-4 API は GPT-3.5 より $15 \times$ expensive であり、GPT-4 の code generation quality が必要だと述べる。iterative prompting があっても正しい skill を生成できず stuck する場合がある。self-verification は spider string を spider を倒した成功 signal と認識しない例がある。automatic curriculum は存在しない `copper sword` や `copper chestplate` を提案することがあり、code generation でも cobblestone を fuel として使う、提供されていない API を呼ぶなどの hallucination が起こる（Limitations and Future Work）。
- **読者として注意すべき点**: 直接比較は high-level Mineflayer API を使う LLM-based agents 同士であり、Minecraft screen pixels から low-level controls を出す既存手法とは apple-to-apple ではない。したがって、Voyager の強さは「3D perception や sensorimotor control を解いた」ことではなく、GPT-4 とコード API を使う open-ended planning / skill accumulation の設計にある。
- **読者として注意すべき点**: Table 1 と Table 2 は 3 trials であり、Diamond Tool は Voyager でも 102 iterations (1/3) である。著者は最高値や倍率を強調するが、成功率・分散・統計的有意性は読者側で慎重に見る必要がある。
- **読者として注意すべき点**: cost の limitation は $15 \times$ という相対価格には触れるが、1 trial あたりの token 数や USD、baseline と同じ token / USD budget での比較は TeX 中には示されていない。
- **読者として注意すべき点**: Broader Impacts では Minecraft は safe and harmless 3D video game environment とされる一方、physical robots など他ドメインに適用する場合は、人間による safety constraints が必要だと書かれている。
- **追加で確認したい実験 / 疑問**: 同一 token / USD 予算での比較、より多い random seeds / trials、160 iterations を超える長期運用での skill library サイズと retrieval 精度、failed tasks の記録が後続の task proposal に与える影響、vision-language model を用いて human feedback なしで 3D structures を評価できるかを確認したい。これらは読者側の疑問であり、TeX 中で実施済みとは書かれていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Voyager**: GPT-4 を black-box query で使う Minecraft agent。automatic curriculum、skill library、iterative prompting mechanism からなる。
- **Embodied lifelong learning agent**: Minecraft world 内で行動し、探索しながら skill を増やし、過去の知識を将来の task に再利用する agent。
- **Automatic curriculum**: agent state と exploration progress に応じて GPT-4 が次タスクを提案するモジュール。著者は in-context form of novelty search とも位置づける。
- **Skill library**: 成功した行動 program を保存する ever-growing library。program description の embedding で index し、new task では関連 skill を検索して prompt に入れる。
- **Executable code**: Voyager の skill 表現。Mineflayer JavaScript APIs と control primitive APIs を呼ぶ async function として生成される。
- **Iterative prompting mechanism**: code generation、実行、feedback 取得、critic による確認、再生成を繰り返す仕組み。最大 4 rounds で打ち切る。
- **Environment feedback**: `bot.chat()` などを通じて得る進捗や失敗理由のテキスト。例として iron chestplate に iron ingots が 7 個足りないという feedback が示される。
- **Execution errors**: JavaScript interpreter から返る invalid operations や syntax errors。code debugging の材料として prompt に戻される。
- **Self-verification**: 別の GPT-4 を critic として使い、task が成功したかを判定し、失敗時に critique を出させる仕組み。
- **Prompting iteration**: この論文の実験単位。Table 1 / Table 2 では、少ない prompting iterations で成功するほど efficient とされる。
- **MineDojo**: Voyager の simulation environment の土台。Minecraft AI 研究用 framework として使われる。
- **Mineflayer**: Minecraft bot を JavaScript から制御する high-level API。Voyager はこれを motor controls として使う。
- **Tech tree mastery**: Wooden Tool、Stone Tool、Iron Tool、Diamond Tool へ順に到達できるかを測る評価。compositional skills の獲得を試す。
- **Zero-shot generalization**: skill library を学習した後、inventory を空にし、新しく生成した world で unseen tasks を from scratch に解く評価。
- **ReAct / Reflexion / AutoGPT**: 比較対象の LLM agent。論文では Minecraft にそのまま使えるわけではないため、MineDojo setting に合わせて re-interpret / re-implement している。
- **Catastrophic forgetting**: continual learning で新しいことを学ぶと古い能力が失われる問題。Voyager は skill を code library として外部記憶に追加するため、この問題を緩和すると著者は述べる。
- **Hallucination**: LLM が存在しない item や API をもっともらしく出す問題。この論文では `copper sword`、`copper chestplate`、cobblestone fuel、存在しない API 呼び出しが例示される。

## 読む順番の提案

- まず abstract と Introduction の第 2-4 段落を読み、Minecraft を open-ended lifelong learning の testbed として使う理由、agent に必要な 3 能力、Voyager の 3 モジュールを押さえる。正規ノートの Summary の「問題」「手法」と対応する。
- 次に Fig. 2 と Method を読む。Automatic Curriculum、Skill Library、Iterative Prompting Mechanism の入力・出力を追うと、正規ノートの「3 モジュール構成」の説明が読めるようになる。
- Appendix の pseudocode `appendix/pseudocode/voyager.py` を見ると、4 rounds の retry、skill retrieval、critic 判定、success / failed task 登録の制御フローが具体化する。これは正規ノートの「iterative prompting は最大 4 round」の根拠に対応する。
- 実験は Experiments の Experimental Setup と Baselines で model / temperature / MineDojo / Mineflayer / baseline の前提を確認してから、Table 1、Table 2、`figures/main_experiment.tex`、`figures/map.tex`、`figures/ablation.tex` を読む。正規ノートの「結果」「Critical Thoughts」の数値はここに対応する。
- Limitations and Future Work は最後に必ず読む。cost、inaccuracies、hallucinations は、正規ノートの「弱み / 疑問」や「比較の注意点」を TeX 根拠つきで読むための中心箇所である。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2305.16291v2/`
- 正規ノート: `notes/arXiv-2305.16291v2.md`
