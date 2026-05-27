# Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate

- arXiv: https://arxiv.org/abs/2305.19118
- source: ../papers/arXiv-2305.19118v4/
- authors: Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, Zhaopeng Tu (Tsinghua University / Shanghai Jiao Tong University / Tencent AI Lab)
- venue / year: TeX 中に明示なし（`emnlp_2024.tex` は `\usepackage[final]{acl}` を使用）
- tags: [multi-agent, LLM, debate, reasoning, machine-translation, self-reflection]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: self-reflection 系の手法（Self-Refine, Reflexion など）は LLM 単体で内省→修正を繰り返すが、著者らは **Degeneration-of-Thought (DoT)** という新規問題を提起する。すなわち「LLM が一度自分の答えに自信を持つと、たとえ初手が誤っていても自己反省では新しい思考を生成できなくなる」。Figure 1 で 5 round 強制した debate と self-reflection の隣接 iteration 間 disagreement を比較し、self-reflection は disagreement が低くずっと誤答に張り付くことを示す。DoT の原因として (1) Bias/Distorted Perception、(2) Rigidity/Resistance to Change、(3) Limited External Feedback の3点を挙げる。
- **手法**: **Multi-Agent Debate (MAD)** フレームワーク。役割は3つ:
  - **Meta prompt**: 「objective is to find the correct answer なので必ずしも合意する必要はない」と "tit for tat" の場を作る。
  - **Debaters**: $N$ 人（デフォルト $N=2$、affirmative=devil と negative=angel）が固定順で発話。negative には「affirmative に同意するな、反論せよ」と明示。
  - **Judge**: 2モード。Discriminative Mode $J_d(H)\in\{\text{True},\text{False}\}$ で正解到達と判断したら早期終了（**adaptive break**）。上限 iteration（デフォルト3）に達したら Extractive Mode $J_e(H)=a$ で最終解を抽出。
  - "tit for tat" の強度は meta prompt の文言で4段階に変調（level 0 は全合意、level 3 は全反対、level 2 がデフォルト）。zero-shot, temperature=0。
- **結果**: 2つの challenging benchmark で評価。
  - **Common MT** (Chinese→English, He+2020 由来; Lexical 200 / Contextless 450 / Contextual 350 の計 1000 例): Table tab:common-mt。GPT-3.5-Turbo+MAD は GPT-4 を上回る。Lexical では COMET 82.0/82.0、BLEURT 70.1→70.9、HUMAN 3.41→3.78。Contextless では COMET 84.7→84.8、HUMAN 3.63→3.67。Contextual も同様。Self-Reflect / Rerank / MAPS は Turbo ベースラインに対して限定的改善のみ。Vicuna-7b/13b+MAD も全部 +MAD で改善。
  - **Counter-Intuitive AR (CIAR)**: 著者らが新規構築した 200問の反直感的算数推論。GPT-4 51.0、Turbo 26.0、+CoT 28.0、+Self-Consistency 29.5、+Self-Reflect 27.5、**+MAD 37.0**。GPT-4 には届かないが Turbo ベースで CoT/SC/Reflect を大きく上回る。
  - **Math/Symbolic 追加** (Appendix `tab:math_symbolic_results`): GSM 70.2→73.8, AddSub 87.3→92.1, Penguin 58.9→63.7, Date 56.4→65.2, Colored Objects 57.2→58.8 と CoT/Self-Reflect より一貫して良い。
  - **DoT 緩和** (Table `tab:mitigate-dot`, Common MT): Bias 29.0→24.8、Diversity (=100 − Self-BLEU) 19.3→49.7 と MAD が両方を改善。
  - **計算コスト** (Appendix `tab:computational_cost`, CIAR 上で計測): CoT を 1.0× とすると Self-Reflect 1.83×、MAD 2.46×（生成トークン数で計測）。
- **貢献**: (1) self-reflection の構造的欠陥 DoT を初めて定式化、(2) judge + adaptive break 付き MAD フレームワークを提案し Turbo+MAD で GPT-4 超え（Common MT）を実現、(3) 新規ベンチマーク Counter-Intuitive AR (200問) を構築、(4) 議論強度・debater 数・judge 設定の網羅的分析、特に "LLM judge は自分と同じバックボーンの debater を贔屓する" という不公平性を実証。

## Takeaway（自分にとっての要点）

- "Multi-agent debate" 系の中で本論文の差別化点は **judge + adaptive break**。Related Work では Du+2023 / Xiong+2023 との差分として、judge と adaptive break、DoT 対応、同一 backbone LLM の有効性を挙げる。Figure (COMET-iteration) では、強制 iteration の最高 COMET も adaptive break より低く、議論を強制継続すると翻訳結果を損なうと述べている。
- "tit for tat" 強度は **中庸が最良**。level 3（完全不同意を強制）は disagreement 0.988 まで上がるが最良性能ではない。著者は「共通基盤を見つけない継続的不同意は polarization につながり、真実や理解より勝敗が中心になる」と考察している。
- **debater を増やすと悪化**（2→3→4 で COMET 84.4→83.1→82.9）。原因は LLM の long-context 弱さ＝他者発話を忘れる＋judge が要約に失敗する、と著者は説明し、Lost in the Middle を引用している。
- **Judge bias の発見が一番怖い**: Table `tab:behavior-agent` の rows ③④ では **judge は両方とも GPT-4** で、debater のバックボーンだけを入れ替えている。row③ Aff=Turbo / Neg=GPT-4 のとき judge は GPT-4 (negative) を 136対52 で選び、row④ Aff=GPT-4 / Neg=Turbo のとき judge は GPT-4 (affirmative) を 120対77 で選ぶ。すなわち affirmative/negative の位置に関係なく、自分と同じ backbone 側を選ぶ傾向が示されている。（Turbo を judge に据えた異種 debaters の対照実験は本論文中には無い）
- 同一バックボーン設定 (rows ①②, judge も debaters も同一 LLM) では judge は **negative 側** を 87対104 (Turbo) / 67対124 (GPT-4) で多く選ぶ。著者は、複雑なタスクでは affirmative 側が誤りやすく、それを opposing side が訂正することが改善に寄与すると述べている。
- MAD は **divergent thinking を出すためのフレーム** であり、著者は Common MT と CIAR の共通点を「表面的表現だけに基づく直感が多くの場合誤りで、深い熟考が必要」と説明している。Appendix の math/symbolic reasoning でも MAD は CoT / Self-Reflect を上回るが、主実験は Common MT と CIAR。
- 実装的に: backbone は GPT-3.5-Turbo-0301 / GPT-4-0314 / vicuna-7b-v1.5-16k / vicuna-13b-v1.5-16k、temperature 0、debate 上限 3 round、N=2、コード公開あり (github.com/Skytliang/Multi-Agents-Debate)。

## Critical Thoughts（評価・疑問）

- **強み**:
  - DoT という現象を Figure 1 の disagreement curve で可視化してから手法に繋げる構成は綺麗。「self-reflection が効かない理由」を実験で示してから debate を導入する流れに説得力がある。
  - GPT-3.5-Turbo + MAD が GPT-4 を上回るという結果は、Common MT のように "common sense を要する翻訳" の領域で特に強い対比になっており、ベンチマークの選択が主張に噛んでいる。
  - 計算コストを Appendix で 2.46× と開示している。Self-Reflect が 1.83× なので、著者は MAD の追加コストを +0.63× と説明している。
  - LLM judge の自己バックボーン贔屓を Table behavior-agent で定量的に示した点は重要（評者補足）。
- **弱み / 疑問**:
  - **CIAR が 200問だけ**、しかも著者らが自作（elicitation.info と geeksforgeeks puzzle と "additional manual derivatives"）。Turbo+MAD が GPT-4 比 -14pt（37.0 vs 51.0）であることも踏まえると、ベンチマーク堅牢性の検証が薄い（評者補足）。
  - **Common MT の "Turbo+MAD > GPT-4" 主張は紙一重**。Contextless で COMET 84.7→84.8、Contextual で 85.0→85.3 など差は 0.1〜0.3pt の桁で、HUMAN（5段階）は Krippendorff α=0.76、3 評価者だが各設定の有意差検定は TeX 中に無い。ベンチマーク自動指標的には GPT-4 と互角どまりと読むのが妥当（評者補足）。
  - **fair comparison の問題**: MAD は 2.46×、Self-Consistency も「multiple responses」をサンプリングするとだけ説明されており、同じ token 予算で並べた pareto curve は TeX 中に無い（評者補足）。
  - **judge も同じ LLM** という設定は「debater のミスを judge が検知できる」ことを暗黙に仮定する。が、LLM が自分の誤りを見抜けないからこそ DoT が起きていたのではないか？ judge が affirmative の誤答を採用するケースの分析が無い（Table `tab:behavior-agent` で affirmative が選ばれた回数だけは示されている）。
  - debater は2人だけが基本設定で、3人以上は劣化と報告。これは MAD の「multi-」が実質「2-agent」止まりであることを意味する。タイトルに反する弱さ。
  - DoT 図 (Figure 1) の disagreement の人手アノテーション基準が「manually determine the disagreement as 1 and agreement as 0」とあるだけで、評価者数・サンプル数・任意のタスクなのか開示が弱い。
  - 著者自身が limitations で認めている: (a) 推論時間が増える、(b) long context での coherence 維持が苦手、(c) **LLM judge が自分の出力を贔屓する** ので judge と debater のバックボーンを揃えるか完全分離するよう推奨。(c) は本手法のフレームワーク的脆弱性で、運用ガイドラインとして無視できない。
- **次に試したいこと**:
  - 同一 token 予算で CoT / Self-Consistency / Reflexion / Du+2023 multi-agent debate / MAD を並べた pareto frontier を引く（評者補足）。
  - CIAR を増やし、さらに通常問題を mix した時の MAD の挙動を見る。Resistance to Intuition が本当に MAD の効く軸なのか分離する（評者補足）。
  - Judge bias の "calibration": judge と debaters を全部別 LLM に分けた時の精度 vs 全部同一にした時の精度のトレードオフを Common MT 以外でも測る（評者補足）。
  - "negative-side 優位" を逆手に取り、affirmative に意図的に弱いモデルを当てるなど非対称配置の効果を調べる（評者補足）。
  - adaptive break の判定そのものを学習させ、judge を small LM や rule-based に置き換えてコストを削れるか調べる（評者補足）。
  - MAD のログを SFT/DPO の教師信号として distill し、シングルエージェントが MAD 同等性能に届くか調べる（評者補足）。

## Notes / Quotes

- "Once the LLM-based agent has established confidence in its answers, it is unable to generate novel thoughts later through self-reflection even if the initial stance is incorrect." (introduction, DoT 定義)
- "It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer." (method, default level 2 の meta prompt の核)
- DoT の3要因: Bias and Distorted Perception / Rigidity and Resistance to Change / Limited External Feedback (introduction)
- "We speculate that continuous disagreement without finding common ground can contribute to polarization, where the debate becomes more about winning the argument than seeking truth or understanding." (analysis §debate level)
- "the key challenge of MAD with more debaters lies in the limitations of the LLMs to handle long texts" (analysis §debater number)
- "the judge shows a preference to the side with the same LLM as the backbone. This bias indicates that LLMs might not be a fair judge when different LLMs are used for the agents." (analysis §judge)
- Adaptive break の根拠: 強制 iteration を増やすと COMET は単調に上がらず、adaptive break が最良 (Figure COMET-iteration)
- コスト: CoT 1.0×, Self-Reflect 1.83×, MAD 2.46× (Appendix `tab:computational_cost`, on CIAR, 生成トークン数で計測)
- 議論 level prompt 4種 (Appendix `tab:tit-for-tat-prompt`): level 0 全合意 / level 1 反対が中心で軽微な合意あり / level 2 default (合意必須ではない) / level 3 全反対
- データ規模: Common MT = Lexical 200 + Contextless 450 + Contextual 350、CIAR = 200

- (verified 2026-05-20) Judge bias の記述を修正: 旧記述「Turbo を judge にして…」は誤り。`tab:behavior-agent` rows ③④ は judge が両方とも GPT-4 で debater backbone のみを入れ替えた設定であり、GPT-4 judge が自バックボーン側を Aff/Neg 位置に関わらず選ぶ、というのが原文の主張 (4-analysis.tex, tab:behavior-agent)。
- (verified 2026-05-20) Common MT データセット原典の引用タイトルを bbl に従い "The Box is in the Pen: Evaluating Commonsense Reasoning in Neural Machine Translation" に訂正、Kong+2022 も "Eliciting Thinking Hierarchy without a Prior" に補完 (emnlp_2024.bbl)。
- (verified 2026-05-20) 表番号の誤り（"Table 3", "Appendix Table 4/5/6"）を TeX のラベル参照（`tab:common-mt` / `tab:mitigate-dot` / `tab:math_symbolic_results` / `tab:tit-for-tat-prompt` / `tab:computational_cost` / `tab:behavior-agent`）に置き換え。
- (verified 2026-05-27) venue/year の EMNLP 2024 推定を削除し、TeX 中に明示が無いことと `acl` final style 使用だけに限定 (emnlp_2024.tex)。
- (verified 2026-05-27) TeX/bbl で裏取りできない比較・比喩・年表記を削除または評者補足として明示し、Related Papers のタイトル・年を bbl に合わせて修正 (4-analysis.tex, 5-related_work.tex, 8-appendix.tex, emnlp_2024.bbl)。

## Related Papers

- Du+ 2023, "Improving factuality and reasoning in language models through multiagent debate" — 並行研究の MAD。本論文は judge + adaptive break + DoT 定式化が差分（related work で明示）。
- Xiong+ 2023, "Diving into the inter-consistency of large language models: An insightful analysis through debate" — 同じく concurrent な multi-agent debate。
- Madaan+ 2024, "Self-refine: Iterative refinement with self-feedback" / Shinn+ 2024, "Reflexion: Language agents with verbal reinforcement learning" — DoT を起こす self-reflection 系 baseline。
- Wang+ 2022, "Self-consistency improves chain of thought reasoning in language models" — majority vote 系 baseline (CIAR で比較)。
- He+ 2024, "Exploring human-like translation strategy with large language models" (MAPS) — Common MT の比較対象、chain-of-thought 型翻訳。
- He+ 2020, "The box is in the pen: Evaluating commonsense reasoning in neural machine translation" (Findings of EMNLP 2020) — Common MT データセットの原典。
- Kong+ 2022, "Eliciting thinking hierarchy without a prior" (NeurIPS 2022) — CIAR の elicitation question 元データ。
- Wang+ 2023, "Large language models are not fair evaluators" (FairEval) — judge bias 議論の参照。
- Liu+ 2024, "Lost in the middle: How language models use long contexts" — long-context 失敗の参照、debater 数増加で悪化する説明根拠。
- Kahneman Daniel 2017, "Thinking, fast and slow" — CIAR の動機（system 1 / system 2）。
- Park+ 2023, "Generative agents: Interactive simulacra of human behavior" / Zhu+ 2023, "Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory" — 思想的源流の multi-agent LLM。
