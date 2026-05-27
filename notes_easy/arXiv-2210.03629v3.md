# ReAct: Synergizing Reasoning and Acting in Language Models（推論トレースと行動を統合するプロンプト型エージェント研究）

- arXiv: https://arxiv.org/abs/2210.03629
- 一次ソース: ../papers/arXiv-2210.03629v3/
- 正規ノート: ../notes/arXiv-2210.03629v3.md

---

## 一言で言うと

この論文は、LLM に task-specific action だけでなく verbal reasoning traces を同じ trajectory 内で交互に生成させる `ReAct` を提案し、外部環境から情報を得る acting と、計画・検索方針・例外処理を支える reasoning を結合する。HotpotQA / Fever / ALFWorld / WebShop で、reasoning-only や acting-only の baseline と比較し、特に外部情報が必要な推論や長い horizon の意思決定で、groundedness・解釈可能性・成功率が改善することを示す。

## 何を議論する論文か

- **問題設定**: LLM を、環境から observation $o_t$ を受け取り action $a_t$ を選ぶ agent として使うとき、単に次の action を予測するだけでは、長い文脈から高水準の goal や subgoal を追跡しにくい。一方、Chain-of-Thought (`CoT`) は reasoning trace を作るが、著者はこれを外界に grounded していない "static black box" と位置づけ、fact hallucination や error propagation が起こると述べる（`text/intro.tex`）。
- **対象範囲 / 仮定**: 主設定は frozen な PaLM-540B に few-shot in-context examples を与える prompting である（`text/method.tex`）。HotpotQA と Fever では question/claim だけを入力し、support paragraphs は与えず、内部知識または simple Wikipedia API との interaction に頼る。ALFWorld と WebShop では、テキストベースの環境で長い行動列を生成する。
- **既存研究との差分**: `CoT` は reasoning-only、`Act` は acting-only として扱われる。WebGPT, SayCan, Inner Monologue などは action planning や external feedback を扱うが、この論文は「language-space の thought」を action space に追加し、thought / action / observation を同一の逐次的 context に入れる点を中核にする。
- **この論文で答えたい問い**: reasoning と acting を分けずに interleaved trajectory として生成させると、(1) knowledge-intensive reasoning で hallucination を減らせるか、(2) interactive decision making で subgoal 分解や探索が改善するか、(3) reasoning-only / acting-only / imitation learning / reinforcement learning baseline より有効か、を実験で問う。

## 背景と前提

- **Agent と policy**: この論文では、agent は現在までの observation/action の列 $c_t$ を見て次の action を選ぶ。通常の action は environment を変え、その結果として observation が返る。
- **Chain-of-Thought (`CoT`)**: LLM に中間推論を書かせてから答えさせる prompting。著者は `CoT` を「reasoning structure」は作れるが、外部情報と interaction しないため hallucinated facts/thoughts を起こしうる baseline として扱う。
- **Self-Consistency (`CoT-SC`)**: `CoT` trajectory を 21 個 sampling し、majority answer を採用する baseline。HotpotQA / Fever の比較表では `CoT` より強い。
- **Acting-only (`Act`)**: `ReAct` trajectory から thoughts を除いた prompt。HotpotQA / Fever では Wikipedia API を操作し、ALFWorld / WebShop では環境 action を出すが、明示的な reasoning trace は持たない。
- **Dense thought と sparse thought**: HotpotQA / Fever のように reasoning が中心のタスクでは thought-action-observation を密に交互生成する。ALFWorld / WebShop のように action 数が多いタスクでは、必要な位置だけに thoughts を出す sparse な形式を使い、thought/action の出現タイミングは LLM に決めさせる（`text/method.tex`）。
- **先行研究との関係**: CoT, least-to-most, zero-shot-CoT, self-consistency, STaR, Scratchpad などは reasoning 側の文脈で引用される。WebGPT, SayCan, Inner Monologue は acting / decision making 側の近い研究として引用される。著者は Inner Monologue を最も近い先行研究の一つとしつつ、IM の "inner monologue" は environment state や未完了 goal の feedback に限られる、と対比する（`text/experiments_rl.tex`, `text/related.tex`）。

## 提案手法

### コアアイデア

`ReAct` の基本は、通常の task-specific action に加えて、環境を直接変えない language action を導入することにある。TeX では、language space に属する action $\hat{a}_t$ を thought または reasoning trace と呼び、これは observation feedback を生まないが、現在の context $c_t$ について推論し、次の reasoning / acting を支える情報を context に追加する、と定義している（`text/method.tex` の "augment the agent's action space"）。

thought の役割は一種類ではない。著者は、task goal の分解、action plan の作成、commonsense knowledge の注入、observation から重要部分を抽出すること、progress tracking、plan の切り替え、exception handling などを挙げる。HotpotQA / Fever では「何を検索すべきか」「検索結果から何が分かるか」「次に lookup/search をどう変えるか」を thought が担う。ALFWorld では「対象物をどこに探しに行くか」「subgoal が終わったか」「次の subgoal は何か」を thought が担う。

実装上は、主に frozen PaLM-540B に few-shot trajectory を prompt として与える。各 in-context example は、人手で作った actions, thoughts, environment observations の trajectory である。HotpotQA は 6 cases、Fever は 3 cases を training set から random select して手動で `ReAct` format にする（`text/experiments_language.tex`）。ALFWorld は task type ごとに 3 trajectories を annotate し、6 prompts を作る。WebShop は `Act` prompt に sparse reasoning を加えた形で評価される。

### 重要な定義・数式

$$
o_t \in \mathcal{O}, \quad a_t \in \mathcal{A}, \quad \pi(a_t | c_t), \quad c_t = (o_1, a_1, \cdots, o_{t-1}, a_{t-1}, o_t)
$$

**式の意味**: 一般的な agent-environment interaction の定式化である。時刻 $t$ に agent は observation を受け取り、過去の文脈 $c_t$ に条件づけた policy $\pi$ に従って action を選ぶ。

**記号の定義**:
- $o_t$ ... 時刻 $t$ に environment から受け取る observation。
- $\mathcal{O}$ ... observation space。
- $a_t$ ... 時刻 $t$ に agent が取る action。
- $\mathcal{A}$ ... task-specific な action space。
- $\pi(a_t | c_t)$ ... context $c_t$ の下で action $a_t$ を選ぶ policy。
- $c_t$ ... それまでの observation と action を並べた agent の context。

**この論文での役割**: `ReAct` の前提となる通常の acting agent の書き方である。著者は、この $c_t \mapsto a_t$ の mapping が implicit で extensive computation を要するとき、単なる action prediction が難しいと述べ、その解決として thought を導入する。

$$
\mathcal{\hat{A}} = \mathcal{A} \cup \mathcal{L}
$$

**式の意味**: `ReAct` は action space を、元の task-specific action space $\mathcal{A}$ と language space $\mathcal{L}$ の union に拡張する。language space に属する action が thought / reasoning trace である。

**記号の定義**:
- $\mathcal{\hat{A}}$ ... `ReAct` で拡張された action space。
- $\mathcal{A}$ ... 環境に作用する通常の action の集合。例: Wikipedia API の `search`, `lookup`, `finish`、ALFWorld の `go to`, `take`, `clean` など。
- $\mathcal{L}$ ... language space。自由形式の言語 thought が属する空間。
- $\cup$ ... 集合の和集合。

**この論文での役割**: 論文の中心的な定義である。reasoning を別モジュールではなく action space 内の一種として扱うことで、thought と external action を同じ trajectory 上で生成・保持できる。

$$
\hat{a}_t \in \mathcal{L}, \quad c_{t+1} = (c_t, \hat{a}_t)
$$

**式の意味**: language space 内の action $\hat{a}_t$ は environment を変えないため observation feedback を生まないが、agent の context には追加される。

**記号の定義**:
- $\hat{a}_t$ ... 時刻 $t$ に生成される thought / reasoning trace。
- $\mathcal{L}$ ... thought が属する language space。
- $c_t$ ... thought 生成前の context。
- $c_{t+1}$ ... thought を追加した後の context。

**この論文での役割**: thought が「環境 action ではないが、将来の reasoning / acting を支える state update である」ことを表す。HotpotQA では検索 query の再定式化、ALFWorld では subgoal tracking、WebShop では product/options の検討などがこの context update によって後続 action に効く。

### 実装 / アルゴリズム上の要点

- step1: 各タスクの few-shot exemplars を、`Thought`, `Action`, `Observation` を含む trajectory として prompt に入れる。HotpotQA / Fever は dense thought、ALFWorld / WebShop は sparse thought を使う。
- step2: LLM が thought を出した場合、それは environment に送らず、context に追記する。observation は返らない。
- step3: LLM が task-specific action を出した場合、Wikipedia API、ALFWorld environment、WebShop environment などに送って observation を得る。
- step4: observation を context に追加し、次の thought または action を生成する。
- step5: HotpotQA / Fever では `finish[answer]` が出たら終了する。Wikipedia API の action は `search[entity]`, `lookup[string]`, `finish[answer]` の 3 種で、著者はこの API を state-of-the-art retriever より弱く、人間の Wikipedia interaction を模すものとして説明する。
- step6: `ReAct` と `CoT-SC` の hybrid も使う。`ReAct -> CoT-SC` は HotpotQA で 7 steps、FEVER で 5 steps 以内に answer を返せない場合に `CoT-SC` へ back off する。`CoT-SC -> ReAct` は majority answer が $n/2$ 未満の回数しか出ない場合に `ReAct` へ back off する（`text/experiments_language.tex`）。

## 実験・結果

- **データセット / ベンチマーク**: knowledge-intensive reasoning として HotpotQA と Fever、interactive decision making として ALFWorld と WebShop を使う。HotpotQA は two or more Wikipedia passages をまたぐ multi-hop QA、Fever は claim を `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` に分類する fact verification。ALFWorld は 6 task types を持つ text-based household game で、134 unseen evaluation games で評価する。WebShop は 1.18M real-world products と 12k human instructions を持つ online shopping environment で、500 test instructions で評価する。
- **比較対象 / baseline**: `Standard prompting`, `CoT`, `CoT-SC`, `Act`, `ReAct`, `CoT-SC -> ReAct`, `ReAct -> CoT-SC` を比較する。ALFWorld では `Act`, `ReAct`, `ReAct-IM`, `BUTLER_g`, `BUTLER`、WebShop では `Act`, `ReAct`, `IL`, `IL+RL`, `Human Expert` を比較する。
- **指標**: HotpotQA は EM、Fever は Acc。ALFWorld は task-specific success rates (%)。WebShop は Score と SR。WebShop の Score は desired attributes の充足率平均、SR は全要求を満たした episode の割合として説明される（`text/experiments_rl.tex`）。
- **主な結果**: HotpotQA / Fever の PaLM-540B prompting 結果は `table/reasoning.tex` にある。HotpotQA EM は `Standard` 28.7, `CoT` 29.4, `CoT-SC` 33.4, `Act` 25.7, `ReAct` 27.4, `CoT-SC -> ReAct` 34.2, `ReAct -> CoT-SC` 35.1, `Supervised SoTA` 67.5。Fever Acc は `Standard` 57.1, `CoT` 56.3, `CoT-SC` 60.4, `Act` 58.9, `ReAct` 60.9, `CoT-SC -> ReAct` 64.6, `ReAct -> CoT-SC` 62.0, `Supervised SoTA` 89.5。
- **主な結果**: HotpotQA では `ReAct` 単体は `CoT` より低いが、`Act` より高く、hybrid の `ReAct -> CoT-SC` が 35.1 で最良である。Fever では `ReAct` 単体が `CoT` と `CoT-SC` を上回り、`CoT-SC -> ReAct` が 64.6 で最良である。著者は、`ReAct` はより factual and grounded、`CoT` は reasoning structure を作る柔軟性が高いが hallucinated facts/thoughts を起こしやすい、と解釈する。
- **主な結果**: ALFWorld の All success rate は `Act (best of 6)` 45, `ReAct (avg)` 57, `ReAct (best of 6)` 71, `ReAct-IM (avg)` 48, `ReAct-IM (best of 6)` 53, `BUTLER_g (best of 8)` 22, `BUTLER (best of 8)` 37。著者は `ReAct` の worst trial 48% でも `Act` と `BUTLER` の best trial を上回ると述べ、`Act` に対する relative performance gain は 33% から 90%、平均 62% と報告する（`text/experiments_rl.tex`, `table/rl.tex`）。
- **主な結果**: WebShop は `Act` Score 62.3 / SR 30.1、`ReAct` Score 66.6 / SR 40.0、`IL` 59.9 / 29.1、`IL+RL` 62.4 / 28.7、`Human Expert` 82.1 / 59.6。著者は `ReAct` が previous best success rate から absolute 10% improvement を得たと述べる。ただし human expert とはまだ差があり、expert humans はより多くの商品探索と query reformulation を行うと説明される。
- **主な結果**: HotpotQA の human study は、`ReAct` と `CoT` から correct/incorrect answers を各 50 trajectories ずつ random sample して、合計 200 examples を人手で分類する。`CoT` の failure では hallucination が 56%、`ReAct` は hallucination 0% だが、reasoning error 47% と search result error 23% が主要な失敗である（`table/human_study.tex`）。
- **主な結果**: HotpotQA finetuning では、3,000 trajectories with correct answers で PaLM-8B / 62B を finetune する。prompting では PaLM-8/62B の `ReAct` は 4 methods の中で worst だが、finetuning 後は best になる。TeX は「PaLM-8B finetuned `ReAct` が all PaLM-62B prompting methods を上回り、PaLM-62B finetuned `ReAct` が all 540B prompting methods を上回る」と述べる（`text/experiments_language.tex`）。
- **著者が主張する貢献**: (1) reasoning と acting を synergize する prompt-based paradigm `ReAct`、(2) diverse benchmarks で reasoning-only / action generation-only より有利であること、(3) reasoning tasks における acting と interactive tasks における reasoning の重要性の ablations / analysis、(4) prompting setup の limitations と finetuning の初期結果、である（`text/intro.tex`）。

## 妥当性と限界

- **この主張を支える根拠**: `Act` は `ReAct` trajectory から thoughts を除く baseline として構成され、特に ALFWorld では同じ annotated trajectories から thoughts だけを除くため、sparse thoughts の寄与を比較しやすい設計になっている。HotpotQA / Fever では `Act` と `ReAct` が同じ Wikipedia API を使うため、検索 access そのものではなく thought の有無を比較しやすい。
- **この主張を支える根拠**: `ReAct-IM` は IM-style dense external feedback thought の ablation であり、`ReAct` 71 vs. `ReAct-IM` 53 という ALFWorld All success rate の差から、単なる current state / current subgoal の反復ではなく、goal decomposition、subgoal completion 判定、next subgoal 決定、commonsense による探索先推定が重要だと著者は主張する。
- **この主張を支える根拠**: human study により、`CoT` の hallucination failure 56% と `ReAct` の hallucination failure 0% が比較されている。ただしこれは HotpotQA の randomly selected 200 examples に対する分析であり、全タスク・全分布での一般的証明ではない。
- **著者が認めている limitations / future work**: conclusion では、large action spaces を持つ complex tasks は多くの demonstrations を必要とし、それが in-context learning の input length limit を超えやすいと述べる。HotpotQA で finetuning は promising だが、さらなる性能改善には more high-quality human annotations が望ましいとする。multi-task training や reinforcement learning との組み合わせも future direction として挙げる。
- **著者が認めている limitations / future work**: `ReAct` には structural constraint による柔軟性低下がある。HotpotQA の分析では、同じ thoughts/actions を繰り返す failure が reasoning error に含まれ、著者は sub-optimal greedy decoding が原因かもしれず、beam search などの better decoding が助けになるかもしれないと footnote で述べる。
- **著者が認めている limitations / future work**: appendix の human-in-the-loop correction は、thought edit により ALFWorld trajectory を成功へ変えられる例を示すが、著者は "more systematic study" を future work として残している。
- **読者として注意すべき点**: HotpotQA では `ReAct` 単体 27.4 は `CoT` 29.4 と `CoT-SC` 33.4 より低い。したがって「ReAct が常に CoT より強い」という主張ではなく、外部情報による groundedness と内部知識による reasoning structure を組み合わせた hybrid が強い、という読み方が必要である。
- **読者として注意すべき点**: Wikipedia API は `search`, `lookup`, `finish` の 3 action で、exact passage name にかなり依存し、著者自身が state-of-the-art retriever より "significantly weaker" と述べる。これは実験設計上、人間の Wikipedia interaction を模し、explicit reasoning による retrieval を促すための制約である。
- **追加で確認したい実験 / 疑問**: `CoT-SC` は 21 samples、hybrid は 3-5 samples でも `CoT-SC` 21 samples に到達すると本文で述べられるが、token budget や wall-clock cost を揃えた詳細比較は TeX 中には明示されていない。実運用上は、external actions の回数、prompt 長、sampling 回数を揃えた比較を追加確認したい。
- **追加で確認したい実験 / 疑問**: WebShop で human expert 82.1 / 59.6 との差が残るため、著者が述べる product explorations と query re-formulations の不足を、どの程度 prompt design、decoding、finetuning、RL で改善できるかは追加検証が必要である。

## 用語メモ

- **`ReAct`** ... `Reasoning + Acting` を指す手法名。LLM が thought と task-specific action を interleaved manner で生成する prompting paradigm。
- **thought / reasoning trace** ... $\mathcal{L}$ に属する language action。環境を変えず observation feedback を生まないが、context に追加され、将来の reasoning / acting を支える。
- **action** ... environment に送られる操作。HotpotQA / Fever では `search[entity]`, `lookup[string]`, `finish[answer]`。ALFWorld では `go to`, `take`, `clean` など。WebShop では `search`, `click[item]`, `click[option]`, `click[Buy Now]` に相当する操作。
- **observation** ... action の結果として environment から返る情報。Wikipedia の sentence、ALFWorld の部屋・物体の状態、WebShop の検索結果や商品 option など。
- **context $c_t$** ... それまでの observation/action、`ReAct` では thought も含む trajectory の履歴。LLM はこの context を prompt として次の出力を生成する。
- **dense thought** ... HotpotQA / Fever で使う、thought-action-observation を密に交互生成する形式。
- **sparse thought** ... ALFWorld / WebShop で使う、長い action sequence の中の必要な位置にだけ thought を挿入する形式。
- **`CoT`** ... Chain-of-Thought prompting。actions と observations を使わず、reasoning-only で answer を生成する baseline。
- **`CoT-SC`** ... self-consistency を使う CoT baseline。TeX では 21 `CoT` trajectories を temperature 0.7 で sampling し、majority answer を採用する。
- **`Act`** ... `ReAct` trajectories から thoughts を取り除いた acting-only prompt。外部 action は使うが、明示的な reasoning trace はない。
- **`ReAct-IM`** ... Inner Monologue 風の ablation。ALFWorld で dense external feedback thoughts に制限され、current goal と current subgoal に関する thought だけに近い。
- **Wikipedia API** ... HotpotQA / Fever で使う外部環境。`search[entity]` は対応ページの first 5 sentences または top-5 similar entities、`lookup[string]` はページ内の次の一致文、`finish[answer]` は task 終了を返す。
- **HotpotQA** ... two or more Wikipedia passages にまたがる multi-hop question answering benchmark。この論文では question-only setup。
- **Fever** ... claim を `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` に分類する fact verification benchmark。この論文では claim のみを入力し、support paragraphs は与えない。
- **ALFWorld** ... embodied ALFRED benchmark と align する synthetic text-based game。household 内で navigation と interaction を行い、高水準 goal を達成する。
- **WebShop** ... 1.18M real-world products と 12k human instructions を持つ online shopping website environment。user instruction に合う商品を web interaction で購入する。
- **EM / Acc / SR / Score** ... HotpotQA の EM は exact match、Fever の Acc は accuracy、WebShop の SR は all requirements を満たした episode の割合、Score は desired attributes covered の平均割合。
- **IL / IL+RL** ... WebShop baseline。IL は 1,012 human annotated trajectories で訓練され、IL+RL はさらに 10,587 training instructions で reinforcement learning を行う。
- **BUTLER** ... ALFWorld baseline。各 task type で $10^5$ expert trajectories による imitation learning agent として比較される。
- **greedy decoding** ... 各ステップで最も確率の高い出力を選ぶ decoding。ALFWorld 表では BUTLER 以外の methods が greedy decoding と注記される。著者は ReAct の反復 failure に関して greedy decoding の sub-optimal さを疑っている。

## 読む順番の提案

- まず `text/abstract.tex` と `text/intro.tex` を読み、著者が reasoning と acting を別々に扱う既存研究の限界をどう整理しているかを確認する。正規ノートでは `Summary（著者の主張）` の「問題」「貢献」に対応する。
- 次に `text/method.tex` の action space 拡張 $\mathcal{\hat{A}} = \mathcal{A} \cup \mathcal{L}$ と context update $c_{t+1} = (c_t, \hat{a}_t)$ を読む。正規ノートでは `Takeaway` の「action space を言語空間で拡張する」がここにつながる。
- その後、`text/experiments_language.tex` と `table/reasoning.tex`, `table/human_study.tex` を読む。HotpotQA / Fever の数値、`ReAct` と `CoT` の失敗モード、hybrid の意味が分かる。正規ノートでは `Summary` の HotpotQA / FEVER / Human study と、`Critical Thoughts` の HotpotQA に関する注意点に対応する。
- 続いて `text/experiments_rl.tex` と `table/rl.tex` を読む。ALFWorld / WebShop で sparse thought が何を改善しているか、`Act`, `ReAct-IM`, `BUTLER`, `IL`, `IL+RL`, `Human Expert` との差を確認する。正規ノートでは `Takeaway` の `ReAct-IM` 比較と WebShop の human expert 差に対応する。
- 最後に `text/discussion.tex`, `text/appendix.tex` の GPT-3, finetuning, human-in-the-loop correction, failure examples を読む。正規ノートでは `Critical Thoughts`, `Notes / Quotes`, `Related Papers` に接続する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2210.03629v3/`
- 正規ノート: `notes/arXiv-2210.03629v3.md`
