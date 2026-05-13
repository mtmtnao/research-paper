# Language Models are Few-Shot Learners

- arXiv: https://arxiv.org/abs/2005.14165
- source: ../papers/arXiv-2005.14165v4/
- authors: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (OpenAI)
- venue / year: NeurIPS 2020
- tags: [LLM, scaling, few-shot, in-context-learning, GPT-3]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 「pre-train → 大量ラベル付きデータで fine-tune」というパラダイムは、タスクごとに数千〜数十万件の教師データが要る。これは (1) 実用上のコスト、(2) 大モデル × 狭い fine-tune 分布での spurious correlation / OOD 一般化の弱さ、(3) 人間は数例または自然言語の指示だけで新タスクをこなす、という3点で限界。先行の in-context learning (GPT-2) は Natural Questions 4% など fine-tune に遠く及ばなかった。
- **手法**: 同じ Transformer 言語モデルを **3桁スケール（125M → 175B）** で 8 サイズ学習し、勾配更新なしの zero-shot / one-shot / few-shot（K = 10〜100 のデモを context に詰める）で評価。アーキは GPT-2 と同じ（modified init / pre-norm / reversible tokenization）に Sparse Transformer 風の alternating dense + locally banded sparse attention を加えたもの。context window $n_{\mathrm{ctx}}=2048$。データは CommonCrawl をフィルタリング＋fuzzy 重複除去した 410B tokens（mix 60%）に、WebText2（19B / 22%）、Books1（12B / 8%）、Books2（55B / 8%）、Wikipedia（3B / 3%）を「質の高いものは重く」サンプリング。全モデル合計 300B tokens を学習。175B モデルは 96 層・$d_{\mathrm{model}}$=12288・96 heads・batch 3.2M tokens・lr $0.6\times 10^{-4}$。V100 クラスタ（Microsoft 提供）で model parallel。
- **結果**: 24+ 個の NLP ベンチマークで scale と共に few-shot 性能が滑らかに向上し、log-loss は 100K〜175B の **10桁** にわたって power-law にほぼ従う。代表値: PTB ppl **20.50（SOTA を 15pt 更新, zero-shot）**, LAMBADA few-shot **86.4%**（先行 SOTA +18%）, TriviaQA few-shot **71.2%**（open-domain SOTA RAG 68.0 を上回る、closed-book で fine-tune 不要）, NaturalQS few-shot 29.9（T5-11B+SSM 36.6 に未達）, WebQS few-shot 41.5（T5-11B+SSM 44.7 に肉薄）, SuperGLUE few-shot 71.8（BERT-Large 69.0 超え, fine-tune SOTA 89.0 には未達）, COPA 92.0, WSC 80.1, **WiC 49.4（=chance）**, ANLI R3 は GPT-3 でやっと chance 超え。算術: 2D加算100%, 2D減算98.9%, 3D加算80.2%, 3D減算94.2%, 4D 25–26%, 5D 9–10%, 2D乗算29.2%, 1桁複合(1DC)21.3%。175B 生成のニュース記事を人間が機械生成と見抜く正答率は **約52%（≒chance, control model は約88%）**。翻訳は into-English で先行 unsupervised NMT 超え、En→Ro は -10 BLEU と非対称（GPT-2 由来の BPE が英語寄り）。
- **貢献**: (1) 175B autoregressive LM の訓練と公開的な実証、(2) zero/one/few-shot を「タスク特定データへの依存度の連続軸」として体系化、(3) 8 モデルサイズで in-context learning が scale と共に強化されること（zero/one/few-shot の gap が拡大）を示した、(4) ニュース記事生成の human evaluation で「人が見分けられない」水準を示した、(5) 13-gram overlap による系統的な test-set contamination 解析と影響評価、(6) bias / 誤用 / エネルギーを含む Broader Impacts の枠組み提示。

## Takeaway（自分にとっての要点）

- **スケーリング則がタスクの "学習方式" まで変える**: 同じモデル・同じ重みでも、サイズが上がると few-shot と zero-shot の差自体が広がる。「大モデルほど meta-learner として強い」という現象こそが本論文の核で、ベンチごとの数値より重要。
- **fine-tuning 一辺倒からの脱却**: TriviaQA で few-shot 71.2% が open-domain fine-tune RAG 68.0 を上回ったのは、知識を「外部 retrieval ＋ FT」ではなく「パラメータに焼き込む」方針が 175B 規模では成立しうることを示している。
- **得意/不得意がきれいに分かれる**: 「2文比較系」（WiC, ANLI, RTE, QuAC, RACE）は few-shot でも弱い。著者自身が「autoregressive・unidirectional」を疑い、bidirectional スケールアップを future work に挙げている → BERT/T5 系を完全に置換できる訳ではない、と読むべき。
- **算術タスクの digit-wise の急減カーブ**は memorization 仮説への反証になっている: 2D で 100% でも 5D で 10% に落ちる、carry ミスが残る、3D 加減算の test 問題の training data 一致率は 0.8% / 0.1% にすぎない、という3点セット。
- **contamination の扱い方が誠実**: 訓練後にバグで一部 overlap が残ったことを正直に書き、PIQA / Winograd には asterisk を付け、4 Wikipedia LM / 1BW / CBT は「測れない」として **そもそも報告しない**。PTB が今や貴重なクリーンベンチである、という指摘も実務的。
- **人間 vs 175B のニュース記事判別 52%** は、後の disinformation・detection 研究の出発点。同 paper 自体が Section 6 で危険性を述べている。
- 「demonstrations が "新タスクを学んで" いるのか "既学習タスクを認識して" いるのかは未解明」という limitations は、後の in-context learning 機構研究（induction heads など）の論点を先取り。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 同一プロトコル・同一モデルファミリーで 8 サイズ × 24+ タスク × 3 評価設定を流し切った **規模そのものが contribution**。これにより scaling law の議論が一気に応用 NLP まで広がった。
  - 自分たちの弱点（WiC chance / ANLI / QuAC / RACE, bidirectional 欠如, En→Ro 翻訳, 5桁算術, common sense physics, sample efficiency, calibration, 推論コスト, bias）を Section 5 で長文かつ具体的に列挙していて、後年から見ても古びにくい。
  - 「13-gram overlap で広めに contamination 候補を出して clean subset の差を比較」という枠組みは、現代の data-contamination 議論の事実上の出発点。LAMBADA は contamination が高いのに clean subset との差は 0.5% 以内、という反例も含めて報告しているのが誠実。
  - 「human eval で control model 88% / 175B 52%」のような対照群を取った設計は、生成品質評価のテンプレとして他分野でも参照しやすい。
- **弱み / 疑問**:
  - **同 token / 同 FLOPs 予算での比較が無い**。「scale が効いた」は正しいが、175B × 300B tokens を fine-tune や retrieval にどう振り直すと優位か、本文の議論からは引けない（後年の Chinchilla 等で別途検証された通り、本論文の token 量 300B は 175B には不足とされた）。
  - SuperGLUE 全体 71.8 は BERT-Large 超えだが fine-tune SOTA 89.0 には依然 17pt の差。図1の "aggregate performance" が 42 ベンチを単純平均しているので、見栄えは良いが、課題別の弱点を平均が隠している。
  - WiC が完全に chance（49.4）であることへの原因究明が「2文比較が苦手っぽい」までで止まっており、prompt 工夫の網羅性も限定的。
  - few-shot が「本当に学習しているのか、ただ既知タスクを再認識しているのか」が判別できない、と limitations 自身が認めている。サイズが大きいほど「学習データへの該当」が増える可能性も込みで未解決。
  - 訓練 corpus と test の overlap を防ぐためのフィルタにバグがあり、retrain は計算コスト的に不可能だった、と明記。PIQA・Winograd は asterisk のみで残してしまっている。
  - bias / misuse のセクションは「preliminary」を強調しているが、175B モデル自体は外部公開せず API のみという形に進んでいったので、本論文の broader impacts 議論と後年のデプロイ実態のギャップは大きい（読み手側の課題）。
  - エネルギーコストの議論はあるが、175B の総 FLOPs / kWh の具体的数字は本文中では明示されない（Appendix D 参照ベース）。
  - 翻訳は into-English ばかりで out-of-English の弱さが残っていて、BPE 再設計が必要との示唆だけで実験はしていない。
- **次に試したいこと**:
  - 同等 FLOPs 予算下で「175B × 300B tokens」 vs 「より小さいモデル × より多い tokens」（Chinchilla 設計）を WiC / ANLI / QuAC のような苦手タスクに限って引き直し、「比較タスクが苦手なのは scale 不足なのか、autoregressive bias なのか」を切り分ける。
  - few-shot で K を変えながら「タスク認識（recognition）」と「タスク学習（acquisition）」の比率を、in-context にゼロ情報 prompt や反事実 demo を混ぜて切り分ける（後年の研究の延長線）。
  - 算術の digit-wise 失敗例（"carry the 1" の取りこぼし）を categorize して、tokenizer を桁区切りにしたときに 5D が伸びるかを評価。
  - news 記事の 52% human-eval を、stylometric / 自動 detector（GROVER, GLTR）と組み合わせた現代的 detection と並べる。
  - PIQA / Winograd の asterisk を、現代の overlap detector で再評価し直す。

## Notes / Quotes

- 「For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.」（abstract）
- 「we use the term ``meta-learning'' to capture the inner-loop / outer-loop structure of the general method, and the term ``in context-learning" to refer to the inner loop」（introduction の脚注で in-context learning の語を定義）
- 「Unfortunately, a bug in the filtering caused us to ignore some overlaps, and due to the cost of training it was not feasible to retrain the model.」（training dataset, contamination の正直な告白）
- 「After extending this trend by two more orders of magnitude, we observe only a slight (if any) departure from the power-law.」（results 冒頭, scaling law の確認）
- 8 モデル: GPT-3 Small 125M / Medium 350M / Large 760M / XL 1.3B / 2.7B / 6.7B / 13B / 175B（96 層, $d_{\mathrm{model}}$=12288, 96 heads）, 全モデル 300B tokens 訓練 (table:param)。
- データ比率: CC 60% (410B, 0.44 epoch) / WebText2 22% (19B, 2.9) / Books1 8% (12B, 1.9) / Books2 8% (55B, 0.43) / Wikipedia 3% (3B, 3.4) — 「質の高いコーパスを多めに epoch する」設計 (table:dataset)。
- SuperGLUE: GPT-3 few-shot **71.8** vs Fine-tuned BERT-Large **69.0** vs Fine-tuned SOTA **89.0**。WiC は 49.4 で chance（table:superglue）。
- LAMBADA は contamination 大だが clean subset との差 0.5% 以内、PIQA は 29% 候補・3pt 落・* 付き、Winograd は 45% 候補・2.6pt 落・* 付き、Wiki LM 4本＋CBT＋1BW は報告自体を取り下げ（contamination section）。
- 算術が memorization でない根拠: 3D 加算 2,000 問中 training 中の overlap は 17件 (0.8%)、減算は 2件 (0.1%)。
- News human-eval: 175B → 約 52%（=chance）, control model → 約 86–88%。長記事（500 words 級）でも 175B はやはり 52%。
- limitations の自認リスト: 長文 coherence 喪失 / 自己矛盾 / common-sense physics（「冷蔵庫に入れたチーズは溶けるか？」）/ WiC・ANLI・RACE・QuAC が弱い / autoregressive で bidirectional 由来の改善を取りこぼす / 事前学習目的が「全 token 等価重み」で重要度を考慮しない / grounding 欠如 / pre-train サンプル効率が人間より大幅に悪い / few-shot が "新規学習" か "認識" か未解明 / 推論コスト / 解釈性 / calibration / training data の bias 継承（→ Broader Impacts）。

## Related Papers

- Radford+ 2019, GPT-2 — 同じアーキテクチャの直接前身。in-context learning の語と zero-shot transfer の起点。
- Kaplan+ 2020, "Scaling Laws for Neural Language Models" — log-loss が power-law に従うという主張の基盤。本論文はそれを下流タスクまで延長。
- Child+ 2019, Sparse Transformer — alternating dense / banded sparse attention の元ネタ。
- Vaswani+ 2017, Transformer / Devlin+ 2018, BERT / Raffel+ 2019, T5 / Liu+ 2019, RoBERTa — fine-tune パラダイムの比較対象。T5-11B(+SSM) は closed-book QA の主要 baseline。
- Lewis+ 2020, RAG — open-domain QA の fine-tune SOTA baseline（TriviaQA 68.0）。
- Roberts+ 2020 "How Much Knowledge Can You Pack…" — closed-book QA の枠組みを与えた直前研究。
- Nie+ 2019, ANLI / Wang+ 2019, SuperGLUE / Paperno+ 2016, LAMBADA / Joshi+ 2017, TriviaQA / Kwiatkowski+ 2019, Natural Questions / Berant+ 2013, WebQuestions / Bisk+ 2019, PIQA — 主要評価データセット。
- Hinton+ 2015, distillation — limitations が future work として挙げる軽量化方向。
- Ziegler+ 2019, RL from human feedback / Chen+ 2019, UNITER — limitations が示唆する次の方向（後の InstructGPT / マルチモーダル路線）。
- Zellers+ 2019 GROVER, Gehrmann+ 2019 GLTR, Ippolito+ 2019 — 機械生成テキストの自動検出 baseline。
