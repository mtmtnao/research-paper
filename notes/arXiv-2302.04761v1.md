# Toolformer: Language Models Can Teach Themselves to Use Tools

- arXiv: https://arxiv.org/abs/2302.04761
- source: ../papers/arXiv-2302.04761v1/
- authors: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (Meta AI Research; † UPF)
- venue / year: arXiv preprint, 2023（後に NeurIPS 2023 採択）
- tags: [tool-use, LLM, self-supervised, finetuning, API-calls]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: LLM は few-shot / zero-shot で多様なタスクを解けるが、算術・最新事実の参照・低リソース言語・時間進行の把握など、もっと小さい専用ツールなら簡単に解ける処理が苦手で hallucinate しがち。既存の tool 統合手法は (a) 大量の人手アノテーションに依存（WebGPT, LaMDA）、または (b) タスク固有 few-shot prompt で「どのツールを使うか」を人間が指定（PAL, TALM）、のいずれかで、自律的かつ汎用的な tool use になっていない。
- **手法**: \textbf{Toolformer}。pretrained LM 自身に in-context learning で API 呼び出し候補を生成させ、self-supervised に「将来トークン予測の loss を下げるか」で取捨選択し、その拡張データで finetune する。
  1. Sampling: 各位置 $i$ で $p_M(\texttt{<API>} \mid P(x), x_{1:i-1}) > \tau_s$（既定 0.05）の位置を top $k=5$、各位置で $m=5$ 個の API 呼び出しを生成（calculator/MT は $\tau_s=0$, $k=20$, $m=10$, $\tau_f=0.5$）。
  2. Execution: 実 API を叩いて結果 $r_i$ を得る。
  3. Filtering: 重み付き cross-entropy で $L_i^+ = L_i(\text{e}(c_i, r_i))$ と $L_i^- = \min(L_i(\varepsilon), L_i(\text{e}(c_i, \varepsilon)))$ を比較し、$L_i^- - L_i^+ \geq \tau_f$（既定 1.0）のものだけ残す。重みは $\tilde{w}_t = \max(0, 1-0.2t)$。
  4. Finetuning: API 呼び出しを本文中に interleave した $\mathcal{C}^*$ で GPT-J を通常 LM 目的関数で finetune。
  5. Inference: 生成中に `→` が出たら decoding を中断 → 実 API を呼ぶ → 結果と `</API>` を挿入して続行。`<API>` トークンを top-$k$（$k=10$）にあれば発火させる decoding 修正を入れる。
  - データ: CCNet サブセット、base LM: GPT-J 6.7B。ツールは QA (Atlas, NQ で finetuned)、Calculator (Python, 四則のみ, 小数2桁丸め)、Wikipedia Search (BM25 over KILT dump)、MT (NLLB 600M, source 言語は fastText で自動検出 → 英語へ)、Calendar (URL から日付抽出)。
- **結果**: 全部 zero-shot prompted setting。
  - **LAMA** (SQuAD/Google-RE/T-REx): GPT-J 17.8/4.9/31.9, GPT-3 175B 26.8/7.0/39.8, Toolformer \textbf{33.8/11.5/53.5}（QA ツールを 98.1\% 使用、Wikipedia Search は fair でないので禁止）。
  - **Math** (ASDiv/SVAMP/MAWPS): GPT-J 7.5/5.2/9.9, GPT-3 175B 14.0/10.0/19.8, Toolformer \textbf{40.4/29.4/44.0}（calculator 97.9\%）。
  - **QA** (WebQS/NQ/TriviaQA): GPT-3 175B 29.0/22.6/65.9 にはまだ届かない（Toolformer 26.3/17.7/48.8、Wikipedia Search 99.3\%）。BM25 の弱さと「クエリ書き換え不可」が原因と著者は分析。
  - **MLQA** (Es/De/Hi/Vi/Zh/Ar): MT ツールは Es/De/Vi で使用率 63.8〜94.9\% だが Hindi だけ 7.3\%。CCNet finetune による分布シフトで GPT-J vanilla を一貫しては上回らない。OPT 66B / GPT-3 175B は英語回答指示に従えず壊滅（Es 3.4, Zh 17.7 など）。
  - **Temporal** (TempLAMA/Dateset): Toolformer \textbf{16.3/27.3}。Dateset では Calendar を 54.8\% 使用して効くが、TempLAMA では Calendar は 0.2\% しか使われず、改善は QA/WikiSearch 由来。
  - **Perplexity** (WikiText/CCNet): GPT-J 9.9/10.6, Toolformer (disabled) 10.3/10.5。API 呼び出しを混ぜた finetune でも素の言語モデル性能は壊れない。
  - **Scaling laws**: GPT-2 124M/355M/775M/1.6B + GPT-J 6.7B で比較し、ツール使用が効くのは概ね 775M 以降。サイズが上がっても「API 有り vs 無し」のギャップは縮まらない。
- **貢献**: (1) self-supervised に API 呼び出しの「位置・引数・タイミング」を学習する一般枠組み、(2) perplexity 削減量でフィルタするシンプルかつ強力な信号、(3) QA / 計算 / Wiki / MT / Calendar の 5 ツール全部で zero-shot 改善を実証、(4) 6.7B モデルが GPT-3 175B を多くのタスクで上回るというサンプル効率の良さの提示、(5) ツール使用能力の emergence が ~775M パラ近辺で起こる scaling 観察。

## Takeaway（自分にとっての要点）

- 「API 呼び出しを sampling → 実行 → \emph{次トークン予測 loss を下げたかどうか} で残す」という filter が肝。報酬モデルも human label も要らず、pretrained LM 自身が信号源になっている。RLHF より遥かに安く tool use を仕込める。
- decoding 時の「`<API>` が top-$k$ にあれば発火」は地味だが効く。$k=1$（純 greedy）では T-REx で 47.8、$k=10$ で 53.5。逆に $k=1$ のとき、モデルは「自分が外しそうな例だけ」API を呼ぶような calibration を見せる（NC 44.3 < AC 53.0）— ただし $k$ を上げるとこの calibration は崩れる。「呼ぶ確率」を信頼度として使えるかもしれない可能性は面白い。
- API ごとに **どこを filter heuristic で絞り込むか** が実装上の急所。Calculator は「3 個以上の数字 / `=`/`equals` 直後の数字」で絞り、Calendar は URL から日付が抽出できる文書だけ残す（CCNet の 18\%）。100 万文書から calculator 用は数千例しか残らない、というサンプル効率の悪さは率直に書かれている。
- 「ツール使用は ~775M パラから emergence」は ToT / CoT 系の emergence 議論と整合する観察で、小モデルで tool use を試すのは筋が悪いという証拠。逆にそのサイズ以上であれば、6.7B でも 175B 級の zero-shot を超えうる。
- limitation の指摘がそのまま次研究のテーマリスト: \textbf{chain 不可・interactive 不可・wording sensitive・sample inefficient・API コスト無考慮}。とくに chain と interactive は ReAct / Toolformer 後継（Gorilla, ToolLLM など）が直接埋めにいく所。

## Critical Thoughts（評価・疑問）

- **強み**:
  - filtering 信号が「実行結果を入れたら loss が下がるか」というモデル内在的シグナルだけで完結している点が美しい。ground truth answer もタスク label もいらない。
  - 各ツールに専用 prompt は要るが、それは「数例の人手 demonstration」だけで済む。WebGPT / LaMDA のように人間が逐次教師信号を与える方式に比べてスケーラビリティが圧倒的に上。
  - 「API call を入れた finetune が WikiText perplexity を悪化させない」をきちんと検証している（Table 5）。tool use と言語モデリングのトレードオフを定量化する誠実さ。
  - decoding bias ($k=10$) のような実装ノブを ablation 込みで開示している（Table 6）。再現に必要な情報がそろっている。
- **弱み / 疑問**:
  - 著者自身が認める通り **chain of tool calls ができない**。TempLAMA で「Calendar → QA」が最適だと自ら指摘しつつ、訓練データ生成時に各 API を独立サンプリングしている設計上、根本的に学習できない。これが本手法最大の boundary。
  - **interactive な検索（クエリ書き換え・複数結果の閲覧）が不可能**。QA で GPT-3 175B に届かない主因がここだと著者も書いており、BM25 + 単発クエリだけで Wikipedia Search を評価していることは方法論的制約。
  - **wording sensitivity**: 「ほぼ同じ意味の入力でも API 呼ぶ／呼ばないが変わる」と limitation 節で告白。prompt engineering 依存度が高く、production 用途では不安定要因。
  - **sample inefficiency**: 100 万文書からカリキュレータ用に残るのは数千例（$\tau_f=1.0$ で 994 件、Table 2）。MT も 1,034 件。CCNet 全体に占めるツール有用例の比率が極小で、希少 API ほど不利。iterative bootstrapping で対処できる、と書いてあるが本論文ではやっていない。
  - **コスト考慮なし**: API 呼ぶ／呼ばないを loss 差だけで決めているので、安価な calculator と高価な MT/QA を区別せずに「呼んだ方が perplexity 下がる」だけで採択している。実運用では効用＝精度向上 - レイテンシ - 課金、を最適化する必要がある。
  - **MLQA で vanilla GPT-J に勝てない言語がある**（De/Zh/Ar）。CCNet finetune 由来の分布シフトと書かれているが、ツールが効くはずの多言語タスクで一貫した改善が出ないのは弱い。
  - **ベースが GPT-J 一機種**。手法の generality は GPT-2 124M〜1.6B との scaling では示されているが、Llama 系・OPT 系で同じ filtering 信号が機能するかは未検証。
  - **TempLAMA で Calendar が 0.2\% しか呼ばれない**事実は、「filtering 信号が将来トークン予測 loss だけ」ゆえに「日付があっても珍しい固有名詞が予測できない」ケースで API 呼び出しがそもそも生成 → 採択されにくいという構造的弱点を示している。タスク信号ではないので「正解できる API」が選ばれる保証がない。
- **次に試したいこと**:
  - filtering 段階で **chain（API₁ の結果を context に入れて API₂ をサンプリング）** を許容する反復生成 → 多段 tool use の Toolformer。著者が提示するが実装していない自然な拡張。
  - decoding 時の API 発火確率 $p_M(\texttt{<API>})$ を **信頼度スコアとして較正**し、ECE で confidence prompt と比較。$k=1$ のときの「外しそうな例だけ呼ぶ」現象（Table 6）の定量化。
  - **コスト付き objective**: $L_i^- - L_i^+ - \lambda \cdot \text{cost}(c_i)$ で tool ごとの計算コスト・課金を考慮した filtering に拡張。
  - iterative self-training: filter で生き残った例で finetune → 新モデルでもう一度 sampling → さらに filter、を 2〜3 周回した時の sample efficiency 改善幅。
  - 手法を Llama-3 8B / Qwen-2.5 7B のような近年のオープン LM に移植し、Toolformer-style filtering 信号が今でも反復通用するか。RLHF 済みモデルでは `<API>` 生成自体が抑制されている可能性を確認。
  - **interactive Wikipedia Search**: 1 文書中に複数の API call を許容し、最初の検索結果を context にした 2 段目クエリを生成・filtering する。

## Notes / Quotes

- "we let a LM annotate a huge language modeling dataset with potential API calls. We then use a self-supervised loss to determine which of these API calls actually help the model in predicting future tokens." (introduction)
- linearize: `e(c) = <API> a_c(i_c) </API>`, `e(c,r) = <API> a_c(i_c) → r </API>` で、実装上は `[`, `]`, `->` を予約トークンとして流用（vocab 拡張なし）。
- filter 基準: $L_i^- - L_i^+ \geq \tau_f$。$L_i^+$ は API+結果を prefix、$L_i^- = \min(L_i(\varepsilon), L_i(\text{e}(c_i, \varepsilon)))$ は「呼ばない」と「呼ぶが結果なし」の min。後者を含めるのは、API 呼び出し自身が prompt として情報を含んでしまうのを差し引くため。
- 重み $w_t \propto \max(0, 1-0.2t)$ により、API 呼び出し近傍の 5 トークン程度に loss を集中。
- decoding: greedy 中、`<API>` が top-$k=10$ にあれば発火させる。生成中 1 リクエスト 1 API call の制約あり（無限ループ防止）。
- LAMA で API 用途内訳: QA 98.1\%, 他ツール 0.7\%, ノーツール 1.2\%。Math では calculator 97.9\%。QA タスクでは Wikipedia Search 99.3\%。
- $\mathcal{C}^*$ サイズ ($\tau_f=1.0$): QA 18,526 / WikiSearch 60,974 / Calculator 994 / Calendar 20,587 / MT 1,034 例。calculator と MT が極端に少ない。
- 「emergence at ~775M parameters」(Section Scaling Laws, Figure 4)。
- limitations 抜粋（自認）:
  - no chain: 「API calls for each tool are generated independently」
  - no interaction: Webgpt 風のブラウジング/クエリ修正不可
  - prompt sensitivity: 「sensitive to the exact wording of their input when deciding whether or not to call an API」
  - sample inefficiency: 「processing more than a million documents results in only a few thousand examples of useful calls to the calculator API」
  - cost-blind: 「does not take into account the tool-dependent, computational cost incurred from making an API call」
- 興味深い decoding 観察 (Table 6, $k=1$): T-REx で API call した部分集合 (AC) 53.0 > しなかった部分集合 (NC) 44.3 > 全部禁止 34.9。モデルは「呼べば伸びる例」と「呼ばなくても解ける例」をある程度区別している。

## Related Papers

- Brown+ 2020, GPT-3 — zero-shot baseline、175B モデルの比較対象。
- Wang+ 2022 Self-Instruct, Honovich+ 2022 Unnatural Instructions — in-context generation で大規模データ自動生成という発想の系譜。
- Komeili+ 2022 (BlenderBot/Internet-Augmented), Thoppilan+ 2022 (LaMDA), Nakano+ 2021 (WebGPT) — 大量人手アノテーションで tool 使用を教える対極アプローチ。
- Gao+ 2022 PAL, Yao+ 2022 ReAct, Lazaridou+ 2022 — task-specific few-shot prompt で tool 使用を引き出すアプローチ。
- Parisi+ 2022 TALM — 最も近い研究。同様の self-supervised 学習で電卓と検索を扱うが、downstream 専用 finetune に限定。Toolformer は zero-shot 汎用。
- Izacard+ 2022 Atlas — 内部で使う QA システム本体。retrieval-augmented LM。
- Costa-jussà+ 2022 NLLB — MT ツールに使用した 600M 多言語翻訳モデル。
- Petroni+ 2019 LAMA, Patel+ 2021 SVAMP, Miao+ 2020 ASDiv, Koncel-Kedziorski+ 2016 MAWPS, Kwiatkowski+ 2019 Natural Questions, Joshi+ 2017 TriviaQA, Berant+ 2013 WebQS, Lewis+ MLQA, Dhingra+ 2022 TempLAMA — 評価ベンチマーク。
- Zelikman+ 2022 STaR, Schick & Schütze 2021 PET, Izacard+ 2021 — self-training/bootstrap の系譜（filtering で残った自分の生成で再学習する思想の源流）。
- Wei+ 2022 (Emergent abilities) — 「~775M で tool use が emerge」観察の参照点。
