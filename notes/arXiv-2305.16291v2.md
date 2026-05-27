# Voyager: An Open-Ended Embodied Agent with Large Language Models

- arXiv: https://arxiv.org/abs/2305.16291
- source: ../papers/arXiv-2305.16291v2/
- authors: Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi "Jim" Fan, Anima Anandkumar (NVIDIA / Caltech / UT Austin / Stanford / UW Madison)
- venue / year: TeX 中には明示なし（ms.tex は neurips_2023.sty を preprint で使用、project page: https://voyager.minedojo.org）
- tags: [LLM-agent, embodied, lifelong-learning, code-generation, Minecraft, GPT-4]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: Minecraft のような open-ended な 3D 環境で「タスクを自分で提案 → スキルを獲得 → 記憶して再利用」を人間介入なしに無期限に続ける lifelong learning agent が存在しない。既存の LLM agent（ReAct / Reflexion / AutoGPT 等）は単発タスク向けで、知識の累積・転用が出来ない。低レベル RL/IL 系（VPT, DreamerV3 等）は系統的探索・解釈性・汎化に弱い。
- **手法**: GPT-4 を黒箱 API として駆動する 3 モジュール構成の Voyager:
  1. **Automatic Curriculum** — GPT-4 に「discover as many diverse things as possible」を上位目標として与え、現在の inventory / equipment / nearby blocks / biome / health / 既達成・失敗タスクを context に次のタスクを bottom-up に生成。budget 上の理由で self-ask/self-answer は GPT-3.5 で行う。temperature=0.1（他は全て 0）。
  2. **Skill Library** — 各スキルを実行可能 JavaScript コード（Mineflayer API）として保存。記述文の embedding (text-embedding-ada-002) を index とし、現タスクと環境フィードバックの embedding で類似スキルを retrieval して in-context に注入。複雑スキルは既存スキルの合成として組み立てる → catastrophic forgetting の回避。
  3. **Iterative Prompting Mechanism** — 3 種類のフィードバックでコードを反復改良: (a) Environment feedback（`bot.chat()` 経由の "I cannot make an iron chestplate because I need: 7 more iron ingots" 等）、(b) Execution errors（インタプリタの例外）、(c) Self-verification（別の GPT-4 を critic として現在状態とタスクから成功判定＋失敗時の助言）。self-verification が OK を出したらスキル登録、4 round 行き詰まったらカリキュラムに別タスクを要求。
  - 制御は Mineflayer JS API（高水準）に乗せており、低レベル知覚・運動制御は対象外。シミュレータは MineDojo ベース。
- **結果**:
  - **探索**: 160 prompting iterations で 63 unique items を発見、baselines の **3.3×**（Fig. 1 / Fig. main_experiment）。移動距離は **2.3×**。
  - **Tech tree mastery**（Table 1, 3 trials, max 160 iter; 数値は到達 iteration）: Wooden tool で Voyager **6±2 (3/3)** vs AutoGPT 92±72、Stone **11±2 (3/3)** vs 94±72、Iron **21±7 (3/3)** vs 135±103、Diamond は Voyager のみ **102 (1/3)** 到達。ReAct / Reflexion は全段階 0/3。木 15.3×、石 8.5×、鉄 6.4× の高速化。
  - **Zero-shot generalization**（Table 2, max 50 iter, 新規 world, inventory 空）: Diamond Pickaxe / Golden Sword / Lava Bucket / Compass の 4 タスク全てで Voyager は 3/3 達成（18–21 iter）。baselines は全て 0/3。AutoGPT に Voyager のスキルライブラリを"挿す"だけで Diamond Pickaxe / Golden Sword / Compass で 1–2/3 まで上がり（Lava Bucket は 0/3）、ライブラリが plug-and-play に効くことを示す。
  - **Ablation**（Fig. ablation）: random curriculum で item 数 **-93%**、self-verification 削除で **-73%**、code 生成を GPT-3.5 にすると Voyager は **5.7×** 少ない item 数しか得られない。
  - **人間フィードバック**（Fig. human, 視覚補助）: Nether Portal や家など 3D 構造物を、人間が critic / curriculum を肩代わりすれば構築可能。
- **貢献**: (1) LLM のみで動く初の embodied lifelong learning agent、(2) コードを action space に据えた合成可能・解釈可能スキルライブラリ、(3) 3 種フィードバックを統合する iterative prompting、(4) MineDojo 上で従来 LLM agent / 既存 Minecraft agent を大幅に上回るベンチマーク結果と、新ワールドへの zero-shot 転移の実証。

## Takeaway（自分にとっての要点）

- **「コードをスキルの単位にする」設計の強さ**: 自然言語プランや低レベル軌道ではなく実行可能関数（`craftStoneShovel()` 等）にすることで、composability・retrieval・debugging・継続学習が同じインフラに乗る。エージェントの memory 設計の有力な雛形。
- **Curriculum を LLM に外注する**: 報酬関数や手書きカリキュラム不要で、agent 状態・既達成/失敗タスク・GPT-3.5 による追加 context を入れて、GPT-4 が bottom-up に次タスクを出す。著者はこれを in-context novelty search と位置づけており、random curriculum では item 数が -93% 落ちる。
- **Self-verification > 他の feedback**: ablation で最も効くのが「成功判定する別 LLM」。Reflexion 的 self-reflection と違って "成功したか" を別 critic が判断するので、curriculum の next task に進むかどうかの gating 信号として機能している（method §2.3 / experiments §4.4）。
- **Skill library はモデル間で plug-and-play**: AutoGPT に挿しただけで成績が上がる。蓄積したコードが他 agent でも再利用できるので「LLM agent 用のオープンソース・スキルカタログ」を考える余地がある。
- **GPT-3.5 では成立しない**: code generation を GPT-3.5 に置換すると GPT-4 版 Voyager は 5.7× 多く unique items を得る。著者は GPT-4 の code generation quality が必要だと述べている。
- **temperature の使い分け**: 探索性は curriculum だけ 0.1、コード生成や critic は 0。多様性が必要な場所だけ温度を上げる、というのは多 agent 設計の実用パターン。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **3 モジュールが clean に分離**しており、各々を ablation で個別に倒している（curriculum -93%、self-verification -73%、GPT-4→3.5 で 1/5.7）。設計判断が「効くから残した」と説明可能。
  - **zero-shot 転移**: 新 world でも inventory ゼロから Diamond Pickaxe を 19±3 iter で達成 (3/3) は強い結果。スキルが world specific でないことを示しており、coding ベースのスキル表現の prediction を裏付ける。
  - **完全 black-box**: 勾配不要・パラメータ非公開で、GPT-4 への prompting / in-context learning によって動く。
  - **限界を著者自身が明示**（コスト・hallucination・自己検証ミス）しており、誇張がない。

- **弱み / 疑問**:
  - **コストの議論が定性的**: limitations で「GPT-4 は GPT-3.5 の 15×」と書くだけで、1 trial あたりの USD・トークン量は TeX 中に見当たらない。Tech tree 比較で「prompting iteration 数」を指標にしているが、AutoGPT との iteration 1 回あたりの API コストを揃えた比較は TeX 中には示されていない。
  - **試行回数が少ない**: tech tree も downstream も 3 trials。Diamond は 1/3 で「unlock した」と書いているが、再現率 33% を SOTA と呼ぶのは弱い。標準偏差が大きい指標もあり（AutoGPT iron 135±103 など）、優位性の有意性検定はない。
  - **hallucination が curriculum 側にも残る**: "copper sword" 等の存在しない item を要求する事例を著者が認めており、curriculum LLM がドメイン知識を完全に持っていない。これを「後で再試行」で吸収する設計は spurious task に対して頑健か疑問。
  - **self-verification が誤検出する例**（spider string を spider 撃破の成功シグナルと認識しない）も明記。self-verification の信頼性が崩れると skill library に「失敗例」が紛れ込む経路があるはずで、long horizon でどれくらい蓄積するかは未分析。
  - **Mineflayer 高レベル API に依存**: 「3D 知覚も低レベル制御も対象外」と明示。論文の貢献は LLM のプランニング部分に閉じており、本物のロボットや視覚必須環境への拡張は別問題（broader impacts でも safety 制約が必要と認めている）。
  - **比較ベースラインが NLP 由来**: ReAct / Reflexion / AutoGPT を Minecraft に「再解釈」して持ち込んでいる。著者は、low-level pixel ベースの Minecraft agent（VPT, DreamerV3 等）とは「apple-to-apple でない」と自ら断っている。Voyager の主張は、Mineflayer の高水準 API を使う GPT-4 lifelong embodied agent として読むべき。
  - **skill library のサイズ管理がブラックボックス**: スキルが無制限に増えた場合の retrieval 精度・冗長性除去・「悪いスキル」の上書きについての言及が薄い。lifelong learning と呼ぶには寿命の長い実験が必要だが 160 iter 上限。

- **次に試したいこと**:
  - **同 token / 同 USD 予算条件**で AutoGPT・ReAct と pareto curve を引き直す。Voyager の iteration 1 回が AutoGPT の何倍のコストか可視化する。
  - **Skill library を distillation の教師信号**として SFT した小モデルで、Voyager と同程度の lifelong 性能が出るか（GPT-3.5 や OSS LLM で coding 能力が足りるか）。
  - **self-verification を確率値付きにする**: 現在は OK/NG の二値。critic LLM の logprob や複数 sampling で confidence を出し、誤検出（spider string 例）を threshold で防ぐ。
  - **負例スキルの保持**: 失敗したコードも「これは効かない」という反例として indexing したらどうなるか。今は成功コードしか積んでいない。
  - **本当に長時間動かす**: 160 iter 上限ではなく 10k iter スケールで skill library のサイズ vs 探索効率を測り、catastrophic forgetting / interference が本当に起きないか確認する。
  - **multimodal 化**: GPT-4V 以降を critic に使い、人間 critic が肩代わりしていた 3D 構造物タスクを agent だけで閉じられるか検証。

## Notes / Quotes

- "the first LLM-powered embodied lifelong learning agent in Minecraft" (abstract)
- "3.3× more unique items, travels 2.3× longer distances, and unlocks key tech tree milestones up to 15.3× faster" (abstract)
- temperature: all 0 except curriculum = 0.1 (experiments §4.1)
- API: `gpt-4-0314`, `gpt-3.5-turbo-0301`, `text-embedding-ada-002`（experiments §4.1）
- iterative prompting は最大 4 round で打ち切り、別タスクへ（method §2.3）
- self-verification は Reflexion の self-reflection より「成功判定」を兼ねる点で comprehensive と主張（method §2.3）
- ablation 数値: random curriculum -93% / no self-verification -73% / GPT-3.5 code = Voyager の 1/5.7（experiments §4.4）
- 著者明示の limitations: コスト（GPT-4 が GPT-3.5 の 15×）、行き詰まり、self-verification 誤検出（spider string）、hallucination（copper sword、cobblestone を燃料扱い、存在しない API 呼び出し）(discussion §)
- Broader Impacts: 物理ロボに移す場合は safety 制約を別途人間が入れる必要があると明記
- 視覚は未対応。Nether Portal や家は人間 critic / human curriculum で代替（experiments §4.5, Fig. human）
- skill library を AutoGPT に挿すと downstream で成績が上がる → ライブラリが plug-and-play（experiments §4.3, Table 2）
- (verified 2026-05-20) Table 2 の AutoGPT w/ Skill Library 結果を 0–2/3 と訂正 (Lava Bucket は 0/3、Diamond Pickaxe / Golden Sword は 1/3、Compass は 2/3) (tables/downstream_table.tex)
- (verified 2026-05-20) Related Papers の Plan4MC を Wang+ 2023 → Yuan+ 2023 に訂正、DEPS = Wang+ 2023 と分離 (ms.bbl, wang2023describe / yuan2023plan4mc)
- (verified 2026-05-27) venue/year を TeX で確認できる範囲（neurips_2023.sty preprint 使用、project page 明記）に限定 (ms.tex)
- (verified 2026-05-27) automatic curriculum の説明から「Minecraft 知識を prompt に書き込まずに済む」を削除し、GPT-3.5 self-ask/self-answer と optional wiki knowledge base の記述に合わせた (2-method.tex, appendix/1-method-appendix.tex)
- (verified 2026-05-27) 再現性・コスト比較・baseline 評価で TeX 根拠より強い評者表現を削り、TeX 中に示された範囲へ限定 (3-experiment.tex, 4-discussion.tex)

## Related Papers

- MineDojo (Fan+ 2022) — シミュレータ・ベンチマーク基盤。
- VPT (OpenAI 2022) — YouTube 動画から学ぶ低レベル Minecraft agent。比較対象外だが対比軸。
- DreamerV3 (Hafner+ 2023) — world model 系で diamond を取った先行例。
- ReAct (Yao+ 2022) / Reflexion (Shinn+ 2023) / AutoGPT — Voyager の直接的 baseline、いずれも NLP 由来。
- Code as Policies (Liang+ 2022) / ProgPrompt (Singh+ 2022) — LLM でロボット実行コードを生成する系譜。
- DreamCoder (Ellis+ 2020) — プログラム単位での skill 蓄積思想の源流。
- Inner Monologue (Huang+ 2022) — 環境フィードバックを取り込むロボット計画の先行例。
- LEVER (Ni+ 2023) / CLAIRIFY (Skreta+ 2023) — 実行結果を使ってコード生成を改善する系統。
- DEPS (Wang+ 2023) / Plan4MC (Yuan+ 2023) / Nottingham+ 2023 — Minecraft で LLM を high-level planner に使う近接研究（exploration 自由度の点で Voyager と差別化）。
- SPRING (Wu+ 2023) — concurrent work、ゲームマニュアルから機構を抽出する LLM agent。
