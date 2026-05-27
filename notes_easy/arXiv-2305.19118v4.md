# Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate（self-reflection の Degeneration-of-Thought に対する multi-agent debate）

- arXiv: https://arxiv.org/abs/2305.19118
- 一次ソース: ../papers/arXiv-2305.19118v4/
- 正規ノート: ../notes/arXiv-2305.19118v4.md

---

## 一言で言うと

この論文は、LLM が自分の出力を反省して直す self-reflection では、初期回答に自信を持つと新しい考えを出せなくなる **Degeneration-of-Thought (DoT)** が起きる、という問題を提起する。提案手法 **Multi-Agent Debate (MAD)** は、複数の debater が "tit for tat" の状態で argument を出し合い、judge が debate の終了と最終解を管理することで、Common MT と Counter-Intuitive AR で self-reflection 系 baseline を上回ると著者は主張している。

## 何を議論する論文か

- **問題設定**: ChatGPT のような LLM は複雑な reasoning tasks でまだ失敗する。self-reflection は、LLM が前回の回答と自分で生成した feedback に基づいて反復的に回答を refinement する方法だが、著者はこれが LLM 自身の self-evaluation 能力に強く依存し、形式的には保証されていないと述べる（`1-introduction.tex`）。
- **対象範囲 / 仮定**: 実験は zero-shot instructions、temperature 0 で行う。中心設定では 3 agents、すなわち affirmative debater、negative debater、judge を使う。backbone は `GPT-3.5-Turbo-0301`、`GPT-4-0314`、`vicuna-7b-v1.5-16k`、`vicuna-13b-v1.5-16k`（`3-experiments.tex`）。
- **既存研究との差分**: CoT、Self-Consistency、Self-Reflect、Rerank、MAPS と比較する。multi-agent debate の並行研究として Du et al. 2023 と Xiong et al. 2023 を挙げ、本論文の差分を、judge と adaptive break、DoT 問題の扱い、同一 backbone LLM の agent で性能改善できる経験的知見としている（`5-related_work.tex`）。
- **この論文で答えたい問い**: self-reflection が DoT に陥るなら、debate によって外部的な反論と多様な chain-of-thoughts を導入することで、LLM の直感的・表面的な誤りを修正できるか。さらに、debate をどのタイミングで止めるべきか、"tit for tat" の強さや debater 数、judge の構成は性能にどう影響するかを調べる。

## 背景と前提

- **LLM と reasoning**: 論文は、LLM が一般的な言語タスクでは高性能だが、complex reasoning tasks ではまだ苦戦する、という背景から始まる。ここでの reasoning は、単に流暢な文章を作ることではなく、問題の表面的な表現に引きずられず、複数ステップの判断や常識知識を使って答えを出すことを指す。
- **self-reflection**: 論文中では、LLM が前回までの回答と feedback を使って新しい回答を生成し、その新しい回答にも feedback を与える iterative refinement process として説明される。Self-Reflect / Reflexion / Self-Refine 系の方法がこの文脈に入る。
- **DoT (Degeneration-of-Thought)**: 著者が「初めて propose and define」した問題。原文の定義は、LLM-based agent がいったん自分の答えに confidence を持つと、初期 stance が間違っていても self-reflection では novel thoughts を生成できなくなる、というもの（`1-introduction.tex` の quote）。Figure `fig:intro` では、5 rounds に強制した debate/self-reflection の隣接 iteration 間 disagreement を人手で 1/0 判定し、self-reflection の disagreement が低いことを DoT の観察として使う。
- **DoT の要因**: Introduction は 3 つの要因を挙げる。`Bias and Distorted Perception`、`Rigidity and Resistance to Change`、`Limited External Feedback`。MAD は他 agent の視点を debate として入れることで、とくに external feedback の不足を補う設計になっている。
- **評価対象の性質**: 主実験は Common MT と Counter-Intuitive AR。著者は両者に共通して、表面的表現だけに基づく直感が多くの場合誤りで、deeper levels of contemplation が必要だと説明する（`1-introduction.tex`, `3-experiments.tex`）。

## 提案手法

### コアアイデア

MAD は、複数の debater が固定順に発話し、judge が debate の進行と最終解を管理する framework である。デフォルトの図では、devil を affirmative side、angel を negative side とし、angel が devil の mistakes を correct する構成になっている（Figure `fig:method`）。

meta prompt は debate topic、debater 数、iteration limit、その他の要求を入れる。重要な文言は "It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer." で、これにより単なる合意ではなく、正答発見を目的にした "tit for tat" の雰囲気を作る。

debater は $N$ 人で、各 iteration に固定順で発話する。基本実験では affirmative と negative の 2 人で、negative prompt には "You disagree with the affirmative side's points. Provide your reasons and answer." と明示される。

judge は 2 つの mode を持つ。`Discrinative Mode` では、現在 iteration の全 debater の発話後に、正しい solution が得られたかを True/False で判定する。True なら debate を終了する。False のまま iteration limit に達した場合、`Extractive Mode` で debate history 全体から final solution を抽出する。

### 重要な定義・数式

$$
D = \{D_i\}_{i=1}^N
$$

**式の意味**: MAD framework に参加する debater の集合を表す定義である。論文では $N$ debaters が関与すると書くが、中心実験では主に affirmative と negative の 2 debaters を使う。

**記号の定義**:
- $D$ ... MAD 内の debaters 全体
- $D_i$ ... $i$ 番目の debater
- $N$ ... debater の人数

**この論文での役割**: MAD が single-agent self-reflection ではなく multi-agent の外部 feedback を使うことを表す基本定義である。後の analysis では $N=2,3,4$ を比較し、2 debaters が default で最良だったことを Table `tab:more-debaters` で示す。

$$
D_i(H) = h
$$

**式の意味**: $i$ 番目の debater が、それまでの debate history $H$ を入力として、新しい argument $h$ を生成することを表す。LLM を、履歴から発話を返す関数として書いている。

**記号の定義**:
- $D_i$ ... $i$ 番目の debater
- $H$ ... previous debate history
- $h$ ... debater が今回生成する発話・主張

**この論文での役割**: debate が各 agent の独立した一発回答ではなく、過去の発話を踏まえた逐次的な相互作用であることを示す。DoT に対する介入として、他 debater の発話が次の入力に入る点が重要である。

$$
J_d(H)= \begin{cases}
    \texttt{True},~~ & \textrm{solution obtained} \\
    \texttt{False},~~ & \textrm{otherwise}
\end{cases}
$$

**式の意味**: judge の `Discrinative Mode` を表す。debate history $H$ を読んで、正しい solution が得られたと判断すれば `True`、そうでなければ `False` を返す。

**記号の定義**:
- $J_d$ ... judge $J$ の discriminative mode
- $H$ ... current iteration までの debate history
- $\texttt{True}$ ... solution obtained と judge が判断した状態
- $\texttt{False}$ ... まだ solution が得られていないと judge が判断した状態

**この論文での役割**: adaptive break の中核である。`True` なら debate はそこで終わり、`False` なら継続する。Analysis の Figure `fig:comet-iteration` では、強制的に長く議論させるより adaptive break の方が良いことが示され、MAD の性能主張を支える。

$$
J_e(H) = a
$$

**式の意味**: judge の `Extractive Mode` を表す。iteration limit 内に正しい solution が識別されなかった場合、judge が debate history 全体から final solution $a$ を抽出する。

**記号の定義**:
- $J_e$ ... judge $J$ の extractive mode
- $H$ ... whole debate history
- $a$ ... 抽出された final solution

**この論文での役割**: MAD は debate を無期限に続けるのではなく、上限 iteration に達したら judge が final answer を決める。実験設定では最大 debate iteration は 3 で、judge は必要に応じて early stop する（`4-analysis.tex`）。

$$
\mathrm{Diversity} = 100-\mathrm{Self\_BLEU}\left(Cand_1, Cand_2\right)
$$

**式の意味**: translation candidates の多様性を Self-BLEU から計算する式である。Self-BLEU が高いほど候補同士が似ているため、$100-\mathrm{Self\_BLEU}$ を diversity として使う。

**記号の定義**:
- $\mathrm{Diversity}$ ... 生成候補の多様性。高いほど候補が似ていない
- $\mathrm{Self\_BLEU}$ ... 候補同士の類似度を BLEU 的に測る値
- $Cand_1$ ... initial translation。Self-Reflection では base answer、MAD では affirmative side's response
- $Cand_2$ ... current translation。Self-Reflection では modified answer、MAD では negative side's response

**この論文での役割**: DoT の要因のうち `Rigidity and Resistance to Change` を測るための分析指標である。Table `tab:mitigate-dot` では Diversity が Self-Reflect 19.3、MAD 49.7 となり、MAD がより多様な候補を出す根拠として使われる。

### 実装 / アルゴリズム上の要点

- step1: meta prompt で debate topic、debater 数、iteration limit、"tit for tat" の指示を設定する。arithmetic reasoning の例では "You are a debater" から始まり、正答発見が目的なので完全合意は不要だと指示する。
- step2: affirmative debater が previous debate history $H$ に基づいて argument を出す。Figure `fig:method` では devil が affirmative side として描かれる。
- step3: negative debater が affirmative side に disagreement を示し、理由と answer を出す。論文の prompt 例では "You disagree with the affirmative side's points." と明記される。
- step4: 各 round の終わりに judge が $J_d(H)$ を実行し、solution obtained なら debate を終了する。これが adaptive break である。
- step5: judge が `False` を返し続けて iteration limit に達したら、$J_e(H)=a$ で whole debate history から final solution を抽出する。
- step6: 実験は zero-shot instructions、temperature 0。主な backbone は `GPT-3.5-Turbo-0301`、`GPT-4-0314`、`vicuna-7b-v1.5-16k`、`vicuna-13b-v1.5-16k`。

## 実験・結果

- **データセット / ベンチマーク**: Common MT は Chinese $\Rightarrow$ English translation examples で、lexical ambiguity 200、contextless syntactic ambiguity 450、contextual syntactic ambiguity 350 の計 1000 examples を使う（Appendix `app:testbeds`）。Counter-Intuitive AR (CIAR) は著者作成の 200 questions で、elicitation questions、web data、manual derivatives から集められ、`Resistance to Intuition` と `Multi-Step Reasoning` を特徴とする。
- **比較対象 / baseline**: Common MT では `GPT-4`、`GPT-3.5-Turbo`、`+ Rerank`、`+ MAPS`、`+ Self-Reflect`、`+ MAD`、さらに `Vicuna-7b/13b` と `+ MAD` を比較する。CIAR では `GPT-4`、`GPT-3.5-Turbo`、`+ CoT`、`+ Self-Consistency`、`+ Self-Reflect`、`+ MAD` を比較する。
- **指標**: CIAR は accuracy (ACC) を報告する。Common MT は COMET (`Unbabel/wmt22-comet-da`)、BLEURT (`BLEURT-20`) と、professional human translators による 1 から 5 の HUMAN score を使う。human evaluation は 3 人の professional human translators、Krippendorff's Alpha = 0.76 と書かれている（Appendix, Human Evaluation Details）。
- **主な結果**: Common MT の Table `tab:common-mt` では `GPT-3.5-Turbo + MAD` が Lexical で COMET 82.0、BLEURT 70.9、HUMAN 3.78、Contextless で 84.8、73.7、3.67、Contextual で 85.3、74.0、3.67。`GPT-4` はそれぞれ Lexical 82.0、70.1、3.41、Contextless 84.7、73.6、3.63、Contextual 85.0、73.7、3.65。著者は、GPT-3.5 backbone の MAD が Common MT で GPT-4 を surpass できると述べる。
- **主な結果**: CIAR の Table `tab:CIAR` では ACC が `GPT-4` 51.0、`GPT-3.5-Turbo` 26.0、`+ CoT` 28.0、`+ Self-Consistency` 29.5、`+ Self-Reflect` 27.5、`+ MAD` 37.0。MAD は GPT-4 には届かないが、GPT-3.5-Turbo 上の比較手法を上回る。
- **主な結果**: Appendix Table `tab:math_symbolic_results` では、MAD は CoT / Self-Reflect より高い accuracy を報告する。GSM は CoT 70.2、Self-Reflect 70.8、MAD 73.8。AddSub は 87.3、87.6、92.1。Penguin は 58.9、61.0、63.7。Date は 56.4、58.0、65.2。Colored Objects は 57.2、58.0、58.8。
- **主な結果**: DoT mitigation の Table `tab:mitigate-dot` では、Bias は Self-Reflect 29.0、MAD 24.8、Diversity は Self-Reflect 19.3、MAD 49.7。著者は、MAD が inherent biases を correct し、DoT を mitigate し、performance を改善すると述べる。
- **著者が主張する貢献**: Introduction の contribution は、(1) self-reflection における DoT の提案・定義、(2) MAD framework による divergent chain-of-thoughts の探索と 2 challenging tasks での有効性、(3) adaptive break と modest level of "tit for tat" の必要性、および LLM-based judge が同じ backbone の side を好むという観察である。

## 妥当性と限界

- **この主張を支える根拠**: DoT の存在は Figure `fig:intro` の disagreement 分析で支える。著者は debate/self-reflection を 5 rounds に強制し、隣接 iteration 間の disagreement を人手で 1/0 判定する。self-reflection の低い disagreement を、CoT の誤答に stick し meaningful self-reflection ができないことの根拠としている。
- **この主張を支える根拠**: MAD の性能は、Common MT の Table `tab:common-mt` と CIAR の Table `tab:CIAR` で支える。特に Common MT は、specific words が common sense に合うかという token-level の差が重要なので、著者は automatic metrics だけでなく HUMAN score を重視している。
- **この主張を支える根拠**: adaptive break は Figure `fig:comet-iteration` で支える。各 iteration で強制的に $a=J_e(H)$ を抽出した場合、MAD は self-reflection より良いが、最高 COMET は first iteration に現れ、それでも adaptive break の結果より低いと述べる。したがって、長く議論すればよいのではなく、適切な時点で止めることが必要だという主張になる。
- **この主張を支える根拠**: "tit for tat" は Figure `fig:tit-for-tat` と Appendix Table `tab:tit-for-tat-prompt` で分析する。level 3 の "Both sides must disagree with each other on every point" は disagreement 0.988 だが最良性能ではなく、著者は合意点を見つけない継続的不同意が polarization につながる可能性を述べる。
- **この主張を支える根拠**: judge bias は Table `tab:behavior-agent` で示す。全 agent が同一 LLM の場合、judge は negative side を多く選ぶ（Turbo 全員: Aff 87 / Neg 104 / Tie 9、GPT-4 全員: Aff 67 / Neg 124 / Tie 9）。debater が異なる LLM の場合、judge が GPT-4 で、Aff Turbo / Neg GPT-4 では Aff 52 / Neg 136 / Tie 12、Aff GPT-4 / Neg Turbo では Aff 120 / Neg 77 / Tie 3 となり、judge と同じ backbone 側を好むと著者は解釈する。
- **著者が認めている limitations / future work**: Limitations では、複数 round の interaction が必要なので time cost が増えること、current LLM-based agents が long context scenarios で coherence and relevance を維持しにくいこと、LLM-based judge が自分の出力を好む可能性があることを認める。対策として、全 roles が同じ LLM を使う、または judge と debaters が distinct LLMs を使うことを推奨している。
- **著者が認めている limitations / future work**: Conclusion は future work として、より多くの agents の適切な scheduling、board games の multi-agent intelligence、model alignment のための AI feedback を挙げる。
- **読者として注意すべき点**: Common MT での `GPT-3.5-Turbo + MAD` と `GPT-4` の差は、Contextless COMET 84.8 vs 84.7、Contextual COMET 85.3 vs 85.0 のように小さい項目もある。著者は "surpass" と述べるが、TeX 中に各差の有意差検定は明示されていない。
- **読者として注意すべき点**: CIAR は 200 questions の著者作成 dataset である。source は elicitation questions、web data、manual derivatives と書かれているが、ベンチマークの広さや外部再現性については、この TeX 内では追加の大規模検証は示されていない。
- **読者として注意すべき点**: 計算コストは Appendix Table `tab:computational_cost` で、CoT 1.0、Self-Reflect 1.83x、MAD 2.46x の generated tokens と報告される。MAD は人間との interaction という external signals は不要とする一方で、token cost は増える。
- **追加で確認したい実験 / 疑問**: 同じ generated token 予算で CoT、Self-Consistency、Self-Reflect、MAD を比較した場合にも MAD が優位かは、この TeX 中には明示されていない。
- **追加で確認したい実験 / 疑問**: judge bias について、異種 debater + 別 backbone judge、あるいは rule-based / learned judge を使った場合に精度と公平性がどう変わるかは、追加で見る価値がある。

## 用語メモ

- **Degeneration-of-Thought (DoT)**: LLM-based agent が自分の答えに confidence を持つと、初期 stance が間違っていても self-reflection で novel thoughts を生成できなくなる現象。本論文が propose and define したとする。
- **Multi-Agent Debate (MAD)**: debaters が "tit for tat" の状態で argument を出し、judge が debate process を manage / monitor して final solution を得る framework。
- **debater**: $D_i$ と書かれる agent。previous debate history $H$ を入力にして argument $h$ を生成する。デフォルト例では affirmative side と negative side の 2 人。
- **affirmative side / negative side**: 論文の Figure `fig:method` では devil が affirmative、angel が negative。negative には affirmative への disagreement を明示的に促す prompt が与えられる。
- **judge**: debate の各 round 後に、solution obtained かどうかを discriminative mode で判定し、必要なら extractive mode で final solution を取り出す agent。
- **adaptive break**: judge が最適解が得られたと考えた時点で debate を早期終了する仕組み。論文は最大 iteration を 3 にし、それ以外の追加 stopping strategies は実装していないと書く。
- **tit for tat**: ここでは単なる敵対ではなく、相手に完全同意しない debate の状態を作る prompt 指示を指す。Appendix Table `tab:tit-for-tat-prompt` では level 0 から 3 まで自然言語 instruction で調整している。
- **Common MT**: Chinese-to-English の commonsense machine translation dataset。lexical、contextless syntactic、contextual syntactic ambiguity を扱い、直訳では誤るケースを含む。
- **Counter-Intuitive AR / CIAR**: 著者が作成した 200 questions の counter-intuitive arithmetic reasoning dataset。hidden traps による Resistance to Intuition と、correct answer に必要な Multi-Step Reasoning が特徴。
- **Bias**: Common MT の analysis では、specific words の翻訳が commonsense に合わないかどうかを human evaluation で判定する ambiguity error rate 的な指標として使われる。
- **Diversity**: Self-BLEU から $100-\mathrm{Self\_BLEU}(Cand_1,Cand_2)$ で計算される。self-reflection や MAD がどれだけ異なる候補を出すかを見る。
- **Rerank**: LLM から 4 回 translation を sample し、external quality estimation tool `wmt21-comet-qe-da` で best candidate を選ぶ Common MT の比較手法。
- **MAPS**: LLM に human translation process、すなわち analyze before translate を模倣させる translation method。論文では translation に適用した chain-of-thought 的手法として扱う。
- **CoT / Self-Consistency / Self-Reflect**: CIAR などの baseline。CoT は "Let's think step by step" を付ける。Self-Consistency は multiple responses を sample し majority vote する。Self-Reflect は LLM が現出力に満足するまで refinement する。

## 読む順番の提案

- まず `0-abstract.tex` と `1-introduction.tex` を読む。ここで DoT の定義、Figure `fig:intro`、DoT の 3 要因、MAD を導入する動機を押さえる。正規ノートでは Summary の「問題」と Takeaway の DoT 関連メモに対応する。
- 次に `2-methodology.tex` を読む。$D=\{D_i\}_{i=1}^N$、$D_i(H)=h$、$J_d(H)$、$J_e(H)=a$ が出るので、MAD の構成要素をここで確認する。正規ノートでは Summary の「手法」に対応する。
- その後 `3-experiments.tex` の Table `tab:common-mt` と `tab:CIAR` を読む。Common MT と CIAR の数値は論文の中心主張を支えるので、正規ノートの Results 部分と照合するとよい。
- 続いて `4-analysis.tex` を読む。Table `tab:mitigate-dot`、`tab:weak-judge`、`tab:behavior-agent`、`tab:more-debaters`、Figure `fig:tit-for-tat`、`fig:distribution`、`fig:comet-iteration` が、なぜ MAD が効くのか、どこで壊れるのかを示す。
- 最後に `7-limitations.tex` と `8-appendix.tex` を読む。付録には Common MT / CIAR の dataset details、human evaluation、math and symbolic reasoning、tit-for-tat prompt、computational cost、debate process examples がまとまっている。正規ノートの Critical Thoughts は、主にこの analysis / limitations / appendix の読みと対応している。
- 引用関係を確認したい場合は `emnlp_2024.bbl` を見る。Common MT の出典 He et al. 2020、MAPS の He et al. 2024、Self-Refine / Reflexion、Du et al. 2023、Xiong et al. 2023 などの正式タイトルを確認できる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2305.19118v4/`
- 正規ノート: `notes/arXiv-2305.19118v4.md`
