# Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions

- arXiv: https://arxiv.org/abs/1901.01753
- source: ../papers/arXiv-1901.01753v3/
- authors: Rui Wang, Joel Lehman, Jeff Clune, Kenneth O. Stanley (Uber AI Labs)
- venue / year: TeX 中には明示なし（main.tex は nips.sty を final で使用）
- tags: [open-endedness, coevolution, evolution-strategies, curriculum-learning, RL]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: ML は「人間が問題を用意し、アルゴリズムが解く」という構図に閉じている。RL のカリキュラム学習も、人間が指定したターゲット課題に向けて単線で難易度を上げるだけで、どの飛び石（stepping stone）が有効かは事前に分からない。固定された環境のもとでの最適化や coevolution（自己対戦・GAN 等）は環境自体の "非生物的" 側面が変わらないので open-ended な発見には届かない。MCC（Brant & Stanley 2017）は環境と解を共進化させるが、各環境内での最適化圧力が無く、また「今すぐ解ける」問題しか受理しないため複雑化が drift 任せ。
- **手法**: POET は環境–エージェントのペア集合 `EA_list` を維持し、main loop で (1) 条件を満たす iteration では既存環境を mutate して子環境を生成、(2) 各ペアを ES で 1 ステップ最適化、(3) 条件を満たす iteration では transfer 試行、を行う。新しい子環境は親エージェントの reward が再生産閾値（実験では 200）以上の親からのみ生まれ、Minimal Criterion `50 ≤ E^child(θ^child) ≤ 300`（簡単すぎず・難しすぎない）でフィルタされる。Supplemental Information では、残った child はジェネティック L2 距離による novelty（k=5）で ranking され、transfer 後に MC を再確認して受理される。Transfer は `EVALUATE_CANDIDATES`: 各 candidate θ^m と「θ^m に E で 1 ES ステップ進めた proposal」を併せた 2M 個から、E 上の reward が最大のものを返し、現職を上回れば置換（"direct transfer" と "proposal transfer"）。最大同時環境数は cap、超過時は最古を queue 的に除去。Optimizer は ES（Salimans+ 2017）だが TRPO/PPO/genetic algorithms/other variants of ES 等でも置換可能と明記。
- **結果**: 修正版 OpenAI Gym Bipedal Walker（stump/gap/step/roughness をパラメータ化、reward = 130·Δx − 5·Δhull_angle − 0.00035·torque、転倒で −100、score ≥ 230 で "solved"）で評価。各 run は 25,200 POET iteration、population 20、ES 母集団 512、256 CPU で約 10 日。
  - **vs ES 直接最適化**: POET が生成・解決した 5 つの難環境に対し、ES を 16,000 ステップ（Ha 2018 の 2 倍）回しても max score は 17.9 / 39.6 / 13.6 / 24.0 / 19.2 にとどまり、いずれも 230 に遠く及ばない（local optimum で「止まって罰を避ける」挙動）。single-sample t-test で p<0.01。
  - **vs direct-path curriculum control**（同じ計算予算で flat→target を mutation step ぶんずつ近づける）: POET が解いた "challenging / very challenging / extremely challenging"（OpenAI Gym Hardcore の 1.2/2.0/4.5 倍に相当する stump≥2.4, gap≥6.0, roughness≥4.5 を 1/2/3 つ満たす）3 段階のうち、challenging はおおむね到達できるが、very/extremely は到達距離が有意に大きい（Mann-Whitney U で p<0.01）。各レベル到達に要する POET iteration の平均は 638±133 / 1,180±343 / 2,178±368。
  - **Transfer の必要性**: 3 run で replacement attempt が約 19,000 回ずつ、成功率 53.62% / 49.26% / 48.89%。Transfer を無効化した制御では coverage が有意に悪化（Mann-Whitney U で p < 2.2e−16）し、**extremely challenging 環境は 1 つも生成・解決されない**。
  - **逆方向 transfer の定性例**: flat 親の膝立ち歩きエージェント → 子（stumps）で立って跳ぶ獲得 → iter 1,175 で親に逆 transfer → 親側スコア 309 → 349 に改善。Transfer を切ったまま親で 3,000 iter 走らせても膝立ちのままだった。
- **貢献**: (1) 環境と解の同時生成・最適化・転移を 1 ラン内で回す POET アルゴリズム（MCC + CMOEA + ES の合成）、(2) 「事前に何を解きたいか決めずに」発見した課題群が、同じ予算の curriculum control を上回るという定量比較、(3) Transfer 削除アブレーションで、open-ended な複雑化に cross-environment transfer が必須であることの実証、(4) 2D bipedal walker で stumps/gaps/roughness/stairs 等の多様な obstacle に特化した歩行行動の同時獲得。

## Takeaway（自分にとっての要点）

- 「ターゲットを決めて curriculum を組む」よりも「並列に発散させて transfer で stepping stone を融通する」方が、結果として難しい課題に到達することがある、という主張の実証例。**カリキュラムは線でなく木（むしろ森）で組め**、というメッセージ。
- POET の MC `50 ≤ E^child ≤ 300` は ZPD（簡単すぎず難しすぎず）を素朴な数値範囲で実装している。「子環境を仮の親エージェントで評価してから受理」というのは、生成器側に学習可能性のフィードバックを返す軽量な仕組みとして再利用しやすい。
- Transfer を `EVALUATE_CANDIDATES` で「全候補 + 1 ステップ進めた proposal」から argmax で選ぶというのは、行動空間でなくパラメータ空間での "goal switching" の具体実装。GA で言う migration に近いが、proposal を入れている点で「今そこに居る方が伸びるか」も含めて選んでいる。
- Diversity は **genetic encoding 上の k-NN novelty** で取っている（BC を作らず）。環境エンコーディングがそのまま意味的な記述になっている状況では BC を作らずに済むという論点は、PCG 系の設計でも応用できる。
- Limitation の「最大難度が encoding 上限で頭打ち」は CPPN 等の indirect encoding で開く、と著者自身が discussion で次の一手を提示している。Open-endedness 自体は手法ではなく「探索空間と encoding の限界」が支配する、という見方の補強。
- 「single-sample t-test と Mann-Whitney U」を使い分けているのは open-ended 系特有の事情：POET は同じターゲットを再現的に生成しないので n=1 にしかならない。**open-ended 系の評価設計は固定ターゲット型の ML ベンチマークとは別物**として扱う必要がある。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 主張に対するアブレーションが揃っている（ES 単体 / direct-path curriculum / POET w/o transfer の 3 系列）。特に「transfer 削除で extremely challenging が 0 件」は強い証拠。
  - 逆 transfer（子 → 親）の定性例（kneeling → standing で score 309→349）は、curriculum を一方向で組む既存手法には無い動作を具体的に示している。
  - 実装が ES に閉じておらず、TRPO/PPO/GA への差し替え可能性を最初から明記している（plug-and-play）。
  - "open-ended 系では preconceived target に対する n 回再現実験ができない" という統計的困難を正面から認め、single-sample t-test 等で代替している誠実さ。
- **弱み / 疑問**:
  - 著者自身が limitation として挙げているとおり、**encoding が direct で 5 個のパラメータしかない**ため、難度がそもそも有限。CPPN 等で開いたときに同じ動作になるか未検証。
  - **POET は preconceived target を狙えない**ことを著者も認めているが、そのうえで「POET の方が target challenge に到達する path として優れる場合がある」と書いており、これは論理的に少し無理がある（ターゲット指定できない手法と「ターゲット到達」を比べる枠組み自体が歪む）。Ken Stanley が原稿コメントで同じ指摘をしているのが TeX 中に残っている。
  - Direct-path curriculum control は「等しい計算予算」と言いつつ、何を予算に数えるかの定義（target に至る direct line of ancestors 全体で POET が使った ES steps、transfers を考慮）が独特で、比較の解釈には注意が要る（評者補足）。Curriculum 設計（各 obstacle parameter が同値を保つか mutation step だけ target 側に増える）は単純で、TeX 中には他の curriculum baseline との実験比較は無い。
  - Transfer 成功率の「~50%」は「現職を上回る候補が居た回数」だが、上回り幅の分布や、その後の最終 score への寄与は数値で示されていない。
  - 多くの結果は **3 run** ベース（25,200 iter × 256 CPU × ~10 日）で、統計的母数として小さい。run 間ばらつきはほぼ示されていない。
  - 評価はすべて 2D bipedal walker。3D や別ドメインでの追試は本論文内には無く、「future work」として discussion に積まれているだけ。
  - Reward 関数自体は固定で、co-evolution の対象外（discussion で future work と明記）。本当に open-ended なら reward の発生源も動くべきだという問題が残る。
  - Body morphology も固定なので「跳べる gap 幅」に上限がある（著者も認めている）。Ha 2018 のように体も co-evolve するべき、という discussion がそのまま課題として残る。
- **次に試したいこと**:
  - Encoding を CPPN などの indirect encoding にして、「max 値で頭打ち」現象が本当に解消するか確認（discussion の future work に基づく評者補足）。
  - Direct-path control 以外の curriculum baseline に差し替えて、POET の優位が残るか確認（TeX 中には追加 baseline 実験なし / 評者補足）。
  - Transfer の有効性を env-side の量でも測る（祖先環境の集合がどれだけ系統的に複雑度を上げているか; 評者補足）。
  - Reward 関数自体をエンコードして共進化させる（discussion で言及されているが本論文では未実施）。
  - 3-D parkour、soft robots、autonomous driving など discussion で挙げられた別ドメインに POET を適用して、2-D walker 以外でも同じ傾向が出るか確認。

## Notes / Quotes

- "Such a process offers the novel possibility that the march of progress, guided so far by a sequence of problems conceived by humans, could lead itself forward, pushing the boundaries of performance autonomously and indefinitely." (introduction)
- Minimal Criterion: `50 ≤ E^child(θ^child) ≤ 300`（環境受理）／ `score ≥ 200`（親が再生産可になる閾値）／ `score ≥ 230`（"solved" の定義）。
- Obstacle parameters と mutation step（Table 1）: stump (init (0,0.4), step 0.2, max 5), gap (init (0,0.8), step 0.4, max 10), step height (init (0,0.4), step 0.2, max 5), step number (init 1, step 1, max 9), roughness (init/step uniform(0,0.6), max 10).
- "An interesting aspect of open-ended algorithms is that statistical comparisons can be challenging because such algorithms are not trying to perform well on a specific preconceived target." (experiments)
- POET iter to solve by difficulty: challenging 638±133, very challenging 1,180±343, extremely challenging 2,178±368。
- ES 単体での詰まり: 5 環境で max score 17.9 / 39.6 / 13.6 / 24.0 / 19.2（success line 230）。
- Transfer 統計（3 run）: 18,894 / 19,014 / 18,798 試行、成功率 53.62% / 49.26% / 48.89%。
- POET w/o transfer: extremely challenging が 0 件、coverage 比較で p < 2.2e−16。
- 計算規模: 256 CPU × ~10 日 / run、ES 母集団 512、neural net は 24-40-40-4 tanh（Ha 2018 を踏襲）。
- 既知の限界（discussion から明示）: (a) 5 obstacle の direct encoding なので難度に上限 → 将来は CPPN 等 indirect encoding、(b) walker の body morphology が固定、(c) reward 関数は固定で共進化対象外、(d) preconceived target には誘導できない。
- (verified 2026-05-26) venue/year を TeX で確認できる範囲（nips.sty final 使用、掲載先・年の明示なし）に限定 (main.tex)
- (verified 2026-05-26) POET main loop の mutation/transfer を、Algorithm 1 の interval 条件付き処理に合わせて修正 (main.tex, Algorithm 1)
- (verified 2026-05-26) novelty ranking と MC の説明を Supplemental Information の MUTATE_ENVS に合わせて修正 (main.tex, Supplemental Information)
- (verified 2026-05-26) Critical Thoughts / 次に試したいことから TeX・bbl に無い後続研究名を削除し、評者補足を明記 (main.tex, main.bbl)

## Related Papers

- Brant & Stanley, *Minimal Criterion Coevolution* (GECCO 2017) — 直接の前駆。問題と解を共進化させるが各環境内の optimization 圧力が無い点を POET が補う。
- Nguyen, Yosinski & Clune, *Innovation Engines* (2016) — "goal switching"（他ニッチへの解の試行）の出典。
- Huizinga & Clune, *CMOEA* (Joost: arxiv18) — combinatorial multi-objective の発想を POET が継承。
- Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* (es) — 本論文の inner-loop optimizer。
- Lehman & Stanley, *Abandoning objectives: Evolution through the search for novelty alone* (Evolutionary Computation 2011) — novelty による多様性圧力の元ネタ。
- Ha, *Reinforcement Learning for Improving Agent Design* (davidha: arxiv18) — Bipedal Walker ベース、controller アーキテクチャ、ES の比較基準。
- Schmidhuber, *PowerPlay* (2013) — 自分で問題を作って解く先行研究。
- Florensa et al., *Automatic Goal Generation / Reverse Curriculum* (2017–2018) — 自動 curriculum 系の対比。
- Matiisen et al., *Teacher-Student Curriculum Learning* (2017) — 学習曲線の傾きでサブタスクを選ぶ curriculum baseline。
- Justesen et al., *Procedural Level Generation for Generality* (2017) — PCG × deep RL の関連。
- Heess et al., *Emergence of Locomotion via Rich Environments* (parkour, 2017) — 著者らが POET の次のターゲットとして言及。
- Cheney et al., *Unshackling Evolution* (GECCO 2013) — soft robot morphology の共進化、future work に挙げられている。
- Stanley, *CPPN / HyperNEAT* — indirect encoding で encoding の上限を外す方向の本命。
- Brockman et al., *OpenAI Gym* — Bipedal Walker Hardcore のベース環境。
