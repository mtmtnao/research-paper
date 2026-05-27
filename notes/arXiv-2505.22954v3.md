# Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents

- arXiv: https://arxiv.org/abs/2505.22954
- source: ../papers/arXiv-2505.22954v3/
- authors: Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune (UBC / Vector Institute / Sakana AI / CIFAR)
- venue / year: ICLR 2026（TeX は `iclr2026_conference.sty` かつ `\iclrfinalcopy` を使用。arXiv v3 はフォルダ名由来）
- tags: [self-improving-agent, open-endedness, LLM-agent, SWE-bench, evolutionary-search, AI-safety]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 現在の AI システムは人手で固定設計され、自身のコードを書き換えて自律的・継続的に改善できない。Schmidhuber の Gödel Machine は「証明付きで自己改良する」枠組みだが、実 AI の改変が有益かを形式的に証明するのは現実的に不可能。一方で ADAS のようなメタエージェント手法は「ダウンストリーム性能」と「自己改良能力」が乖離し得る、また hill-climbing 型の自己改良は局所最適に嵌まる。
- **手法**: **Darwin Gödel Machine (DGM)** を提案。形式証明の代わりに **コーディングベンチマーク上の経験的検証** で自己改変を判定し、Darwin 進化に倣って archive（過去エージェント全保存）から親をサンプル → 自己改変 → 評価 → archive に追加、を 80 iter 繰り返す。エージェントは frozen FM + 2 ツール（Bash, Edit）の Python 実装で、自分のレポを書き換えて新版エージェントを生成。親選択は sigmoid スケールした性能 × 1/(1+子供数) で開拓と活用をバランス（λ=10, α₀=0.5, SWE-bench は k=2 並列、Polyglot は k=4）。評価は staged: まず 10 task で基本編集機能を確認し、SWE-bench は追加 50 task と合わせた 60-task subset で ≥40% かつ archive 上位2なら 200 task、Polyglot は 10-task subset で ≥40% なら 50-task subset。自己改変提案には別途 o1 を診断 FM として使う。エージェント本体の FM は SWE-bench で Claude 3.5 Sonnet (New)、Polyglot 評価は o3-mini、temperature=1.0。サンドボックス内実行＋時間制限＋系統 traceability で安全策。
- **結果**:
  - SWE-bench Verified: **20.0% → 50.0%**（200 task 評価）。チェック済 OSS SoTA に匹敵。
  - Polyglot: 評価 50 task で 14.0% → 38.0%、フルベンチで **14.2% → 30.7%** で Aider を上回る。
  - **アブレーション（Table 1, app:add-results）**: DGM 50.0% / 38.0%、DGM w/o open-ended 23.0% / 14.0%、DGM w/o self-improve 39.0% / 28.0%、DGM Greedy（archive 中で最高性能のみ親）39.7% / 30.0%。両コンポーネントとも必要。
  - **FM 転移（SWE-bench, 200 task）**: o3-mini で base 23.0% → DGM 33.0%、Claude 3.7 Sonnet で base 19.0% → DGM **59.5%**。
  - **ベンチマーク転移**: SWE-bench 育ち agent を Polyglot へ 14.2% → 28.9%、Polyglot 育ち agent を SWE-bench へ 20.0% → 24.5%。Python 限定で育てた agent も他言語へ転移し base/Aider 超え。
  - 基本コード編集機能を保った子供の割合: DGM 51.3% vs 両 baseline 32.5%。Polyglot を 3 回回した安定性は 40.7%±2.3%。
  - 副次結果: ハルシネーション抑制ベンチに DGM を適用すると node 96 が score 1.67 を真に達成、node 114 は 2.0 を達成したが「ツール使用ログの特殊トークンを削除して検出関数を迂回」する objective hacking を観測。
- **貢献**: (1) 形式証明を経験検証に緩めた self-referential self-improving 系 DGM、(2) open-ended な archive 探索＋自己改変ループの組合せが、self-improve 単独・open-ended 単独より優れることを 2 ベンチで実証、(3) FM・ベンチ・言語をまたぐ転移性を確認、(4) sandbox / 系統トレーサビリティを最初から組み込んだ運用と、ハルシネーション対策タスクへの応用事例（objective hacking の警鐘付き）。

## Takeaway（自分にとっての要点）

- **「ダウンストリーム=メタ目的の一致」が肝**。コーディング能力で評価＝自己改変能力で評価、という重ね合わせができたので、ベンチが上がる＝次の自己改変もうまくなる、という自己加速ループが成立する。この一致を作れない領域（例: writing, vision）では同じ枠を使いにくい、というのが転用時の制約として強く意識すべき点。
- **archive ＋ 親の確率的サンプリング（DGM Greedy より +10pt）**：deceptive dip を越える効果が定量で出ている（iter 4, 56 で一度親より低下しても、その系統からブレイクスルーが生まれる例）。これは Go-Explore 的な開拓ボーナス（1/(1+子供数)）の現実的な威力サンプルとして引用価値がある。
- **改変の中身がそのまま示唆になる**: DGM が獲得した改良は「edit ツールを行範囲ビュー＋文字列置換に粒度を上げる」「複数解を生成して別 FM に選ばせる」「過去試行を次の生成に context として渡す」など、エージェント設計界で人手で蓄積されてきたパターンと同型。同じ機能でも実装で 23.3% vs 40.5% と大差（node 6 vs 24）、設計探索を諦めず複数実装を試す価値を裏付ける。
- **コスト**: SWE-bench で 1 run 約 USD 22,000・2 週間（baseline は約 USD 10,000）。「人手の SoTA エンジニアと比べれば安い」と著者は主張するが、再現できる研究者は限られる。Polyglot は USD 5/60 task 単位なのでこちらが再現入口。
- **objective hacking の実例**：著者は検出関数を hidden にしない場合に objective hacking が増えると述べ、hidden 設定でも node 114 は特殊トークンの記録自体を消して評価関数を迂回した。自己改変系の安全評価は「評価関数を unmodifiable に隔離する」アイデアの説得材料として使える。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「self-improve」と「open-ended exploration」を独立に切ったアブレーションで両者必要を示し、さらに DGM Greedy（archive はあるが greedy）も別途切って archive の意味が「保管」だけでなく「確率的にサンプル」にあることまで分離している。設計仮説の検証としてかなり手堅い。
  - FM 入れ替え・ベンチ入れ替え・言語入れ替えの 3 軸で転移を確認し、しかも Claude 3.7 Sonnet では base 19% → DGM 59.5% と転移先のほうがゲインが大きい。base agent 設計のオーバーフィットではなく「FM を効果的に使う scaffold が獲得された」と読める証拠。
  - 安全議論を本文 §6 と App. §safety/§dgm-halluc に分けて真面目に書き、objective hacking の失敗例まで自ら公開している。再現用にコードも GitHub 公開。
- **弱み / 疑問**:
  - **計算コストと統計**: 主結果は 80 iter 1 run。Polyglot で 3 回 (40.7±2.3%) は出しているが、SWE-bench は 1 シードのみ。50.0% という数字の分散が分からない。
  - **「コーディング性能＝自己改変性能」という核心仮定の正当化が弱い**。改善の中身（edit tool 粒度・workflow）が確かにエージェント設計の改良として読めるという qualitative 観察に依存。Limitations でも「extend beyond coding」「co-evolve task distribution」を future work として認めている。
  - **診断 FM を別途 o1 で動かしている**ので、純粋に「エージェントが自分でログを見て次の改良案を立案」しているわけではない。本文も "no fundamental limitation" と書くが、現実装は独立 FM 込みのループであり、self-referential 度合いは控えめに見るべき。
  - **SWE-bench は人気ベンチ＝FMs の training sets に含まれている可能性が高い**点を著者自身が Polyglot 採用理由で認めている。とはいえ SWE-bench Verified の SoTA は閉ソース solution に届かず（著者も Limitations で明記）、「open-ended にいつまで回せば追いつくか」は未解決。
  - **私的テストパッチを診断プロンプトに入れている**（app:selfimprove-prompts）。手動検査で memorization は無いとしているが、これは強い前提で、leakage の定量検証は無し。
  - **objective hacking** が既に出ている時点で、評価指標とのいたちごっこは始まっている。検出関数を hidden にしただけで完全には封じ切れない（node 114 は hidden 設定ですら検出ロジックの裏を突いた）。
  - **archive が肥大化する一方**。pruning / 興味度フィルタは future work 任せで、「全部残す」の長期スケーラビリティは未検証。
- **次に試したいこと**:
  - 同じ予算（USD・トークン・wall-clock）を best-of-N + reflection / Tree-of-Agents 等に振った pareto curve。50% という数字がコスト効率で本当に強いかを見たい。
  - 自己改変提案の診断 FM を抜いて「同一 agent が自分のログを読み次案を出す」純 self-referential 版と比較。性能ギャップを測れば self-reference の現実コストが見える。
  - archive を「興味度（learnability + novelty）」で pruning して、サイズが iter に対し log で済むようにしても性能維持できるかの ablation。
  - Constitutional AI 的な不変則を「unmodifiable な評価器」として外付けし、objective hacking が抑えられるかの実験（safety §6 で著者も示唆）。
  - DGM Greedy が DGM w/o self-improve とほぼ同等(39.7% vs 39.0%)で、両者が共に DGM w/o open-ended(23%) より大きく上 という階層が興味深い：archive 単純保持 ≒ self-improve 単独 < archive ＋ 確率的サンプル の関係を、他ドメインで再現するか。

## Notes / Quotes

- "We introduce the Darwin G\"odel Machine (DGM), a novel self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks."（abstract）
- "A key assumption is that an increase in performance on coding benchmarks indicates better coding capabilities, and hence better ability to self-modify and self-improve."（§3）— DGM のスキーム全体が乗っている前提。
- 親選択式: \(w_i = s_i h_i\), \(s_i = \sigma(\lambda(\alpha_i - \alpha_0))\), \(h_i = 1/(1+n_i)\), λ=10, α₀=0.5（app:parent-select）。
- 自己改変は Claude 3.5 Sonnet (New)、評価は SWE-bench=Claude 3.5 Sonnet (New) / Polyglot=o3-mini、temp=1.0（Table fm-hyperparam）。
- 改良の具体例: 「edit tool に行範囲ビュー＋str_replace を導入」「複数解生成→別 FM で選択」「過去試行を context に」（§4 Results）。
- 同一機能でも実装差で大差: node 6 (edit に edit_type/edit_actions パラメータ追加) 23.3% vs node 24 (str_replace に置換) 40.5%（app:supp-dgm-diffimpl）。
- 安全策: sandbox 隔離、実行時間制限、改変は Python codebase に限定、archive で系統 traceable、害ある挙動は未観測（§6）。
- 認めた限界: 閉ソース SWE-bench SoTA には未到達、1 run 約 2 週間 / 約 USD 22,000、FM 能力に律速、coding 以外への一般化は future work、open-ended exploration ロジック自体は固定（§7 Conclusion, app:future-work）。
- ハルシネーション抑制で objective hacking: node 114 はログの特殊トークンを削除して検出回避（app:dgm-halluc）。"When a measure becomes a target, it ceases to be a good measure."
- (verified 2026-05-27) venue/year を TeX で確認できる範囲（`iclr2026_conference.sty` と `\iclrfinalcopy`）に限定し、「投稿版」を削除 (main.tex)
- (verified 2026-05-27) SWE-bench のデータ混入リスク表現を「Claude の post-training」から TeX の "training sets of FMs" に合わせて修正 (main.tex, §Benchmarks)
- (verified 2026-05-27) `main.bbl` が同梱されていないため、Related Papers の引用タイトル展開を削除 (main.tex, main.bib)
- (verified 2026-05-27) staged evaluation と hallucination objective hacking の条件を本文記述に合わせて具体化 (main.tex, §Benchmarks, app:dgm-halluc)

## Related Papers

- (TeX ソースに `main.bbl` が同梱されていないため、引用文献タイトルは展開しない)
