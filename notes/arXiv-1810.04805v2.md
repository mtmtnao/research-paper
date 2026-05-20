# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- arXiv: https://arxiv.org/abs/1810.04805
- source: ../papers/arXiv-1810.04805v2/
- authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)
- venue / year: NAACL-HLT 2019（TeX は `naaclhlt2019.sty` を使用、arXiv 初出は 2018-10）
- tags: [pre-training, transformer, language-model, transfer-learning, NLP]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 既存の事前学習言語表現は単方向（OpenAI GPT は左→右の Transformer）または浅い双方向結合（ELMo の独立 LTR+RTL の concat）に限られ、Transformer の自己注意層内で「全層が左右両方向の文脈を同時に見る」という意味で真の双方向化はできていなかった。とくに SQuAD のようなトークン単位タスクでは右側文脈の欠落が致命的。
- **手法**: 多層双方向 Transformer encoder を、(1) **Masked LM (MLM)**：入力 WordPiece トークンの 15% をランダムに選び、そのうち 80% を `[MASK]` に置換・10% をランダム語に置換・10% は元のまま残して原語を当てさせる Cloze 風目的、(2) **Next Sentence Prediction (NSP)**：50/50 で A の次に来る本物 / ランダム文を `[CLS]` 表現から二値分類、の 2 タスクで同時事前学習する。fine-tuning では `[CLS]` ベクトル `C` または各トークン表現 `T_i` の上に最小限の output 層 1 枚を載せて end-to-end で全パラメータを再学習。事前学習・fine-tuning でアーキテクチャは出力層以外完全に共通。入力は token + segment + position の 3 種類の embedding の和、特殊トークン `[CLS]` / `[SEP]`、WordPiece 30,000 語彙、最大 512 トークン。モデルは `BERT_BASE` (L=12, H=768, A=12, 110M) と `BERT_LARGE` (L=24, H=1024, A=16, 340M)。事前学習データは BooksCorpus 800M words + English Wikipedia 2,500M words の合計 3.3B words、batch 256 系列 × 512 トークン = 128k tokens/batch を 1M step（約 40 epoch）、Adam lr=1e-4・β1=0.9・β2=0.999・weight decay 0.01・10k step warmup + linear decay、dropout 0.1、GELU。`BERT_BASE` は 4 Cloud TPU (16 chips)、`BERT_LARGE` は 16 Cloud TPU (64 chips) で各 4 日（TeX に TPU 世代は明示なし）。
- **結果**: 11 NLP タスクで SOTA を一斉に更新。GLUE leaderboard 公式 score は BERT_LARGE 80.5（OpenAI GPT 72.8 から +7.7、abstract）。Table 1（WNLI を除外した Average）では BERT_BASE 79.6 / BERT_LARGE 82.1、BERT_LARGE は全 8 タスク列で最高値（MNLI-m/mm 86.7/85.9、QQP 72.1、QNLI 92.7、SST-2 94.9、CoLA 60.5、STS-B 86.5、MRPC 89.3、RTE 70.1）。SQuAD v1.1 Test F1 は BERT_LARGE Sgl.+TriviaQA 91.8 / Ens.+TriviaQA 93.2（Human 91.2、従来トップ #1 ensemble nlnet 91.7、+1.5 F1）。SQuAD v2.0 Test F1 は BERT_LARGE single で 83.1（前 SOTA unet ensemble 74.9 から +5.1）。SWAG Test 86.3（ESIM+ELMo 59.2、OpenAI GPT 78.0、Human (5 ann.) 88.0、ESIM+ELMo 比 +27.1）。CoNLL-2003 NER Test F1 は BERT_LARGE fine-tune 92.8（feature-based では top 4 hidden の concat で Dev 96.1、fine-tune Dev 96.6 比 -0.3 F1）。アブレーション (Table 5) では BERT_BASE から NSP を外すと QNLI 88.4→84.9, MNLI 84.4→83.9, SQuAD F1 88.5→87.9 と劣化、さらに MLM を LTR に変える（No NSP → LTR&No NSP）と MRPC 86.5→77.5, SQuAD F1 87.9→77.8 と大幅悪化（BiLSTM を乗せても SQuAD 84.9 までしか戻らず、GLUE は逆に悪化）。モデルサイズ ablation (Table 6) では (L=3,H=768) から (L=24,H=1024) まで MNLI-m / MRPC / SST-2 の 3 タスクで単調改善し、訓練データ 3,600 件の MRPC でも例外なく伸びる。
- **貢献**: (1) 「全層で左右両方向の文脈に同時条件付け」する深層双方向事前学習の重要性を示し、それを可能にする MLM 目的を提示。(2) BERT 単体で出力層 1 枚を足すだけで sentence-level（GLUE, SWAG, NLI）と token-level（SQuAD, NER）の双方を最小タスク固有設計で SOTA 化、タスクごとの heavily-engineered アーキ設計の必要性を大幅削減。(3) 大規模事前学習モデルを「極端な」サイズまでスケールすると、MRPC（3.6k 件）のような小規模 downstream タスクでも単調に効くことを feature-based 系の先行研究（ELMo, context2vec）と対比して実証。(4) コードと pre-trained checkpoint を公開（https://github.com/google-research/bert ）。

## Takeaway（自分にとっての要点）

- 「**深層双方向**」と「**浅い concat 双方向**」と「**単方向**」は別物。ELMo 流の独立 LTR + RTL を最後に結合する方式は (a) 2 倍コスト、(b) QA で RTL 側が question に条件付けできない、(c) 各層で同時に左右を使えない、の 3 点で deep bidirectional より劣ると著者は明言（ablation.tex）。MLM はこの 3 つを一気に解く工学的トリック。
- MLM の 80/10/10 比率は単に regularizer ではなく「`[MASK]` が fine-tune に出ない」pre-train/fine-tune mismatch を埋めるためのもの。10% の random replace は全トークンの 1.5% にしか効かないので言語理解は壊れない、という量的議論がある（bert_details.tex）。
- `[CLS]` の最終隠れベクトル `C` は **fine-tune 前は意味のある文表現ではない**（NSP だけで訓練されているため）と脚注に明記。embedding として持ち出して classification する設計は事前学習だけでは正当化されていない。
- NSP 単体精度は 97–98% に到達するほど簡単なタスクだが、それでも外すと QNLI で 3.5pt 落ちる。後続の RoBERTa 等が NSP 不要と主張するための重要な比較点。
- モデルサイズの効果：feature-based 系（ELMo, context2vec）では hidden を 200→600 で頭打ちだったが、fine-tune 系では極小タスクでも一貫してスケールする。「fine-tune では新規パラメータがごく少数だから巨大表現の恩恵を吸い切れる」という仮説は、いまの巨大基盤モデル時代を正しく予言している部分。
- 実装規約：fine-tune 推奨 batch ∈ {16, 32}, lr ∈ {5e-5, 3e-5, 2e-5}, epochs ∈ {2, 3, 4}。BERT_LARGE は小データだと不安定でランダム再起動が必要、と素直に書いてある。後続研究の再現性議論の出発点。
- Feature-based でも fine-tune と 0.3 F1 差まで詰められる（NER, top-4 concat）→ ストリーミング推論や事前計算が必要な現場では fine-tune を諦めても十分使える、という工学的な逃げ道。

## Critical Thoughts（評価・疑問）

- **強み**:
  - GPT との比較で「アーキはほぼ同じ、違いは attention mask の方向と pre-train task と data 量」と統制を効かせた ablation を組んでおり、bidirectionality の貢献を切り分けようとする態度が誠実。BERT_BASE をわざわざ GPT と同じ L=12,H=768,A=12 にしている。
  - 11 タスク・GLUE 9 サブセット・SQuAD 1.1/2.0・SWAG・NER と評価対象が広く、sentence-level と token-level を同じ事前学習で押し切ったという主張に説得力がある。
  - fine-tune が単一 Cloud TPU で 1 時間以下、SQuAD で 30 分というコストの低さを明記（GPU では「数時間」、bert.tex 脚注）→ 公開重みの実用ハードルを意識的に下げにいっており、その後の普及（HuggingFace 等）の前提を作った。
  - アブレーションが「方向性」「サイズ」「ステップ数」「マスク戦略」「feature-based vs fine-tune」と必要な軸を網羅しており、reviewer 視点で穴が少ない。
- **弱み / 疑問**:
  - **著者が明示している limitation**:
    - `[MASK]` トークンが fine-tune には現れない pre-train/fine-tune mismatch（bert.tex §3.1）。80/10/10 で緩和すると言うが、本質的なバイアスは残る。
    - MLM は 1 batch あたり 15% のトークンしか教師信号にしないので LTR より収束が遅い（bert_details.tex, more_ablation.tex の Q&A）。「絶対精度ではすぐ抜く」と主張するが、計算効率としては不利。
    - BERT_LARGE が小データ GLUE タスクで fine-tune 不安定、random restart が必要（experiment.tex §4.1）。
  - 「11 タスクで SOTA」のうち SQuAD v1.1 の +1.5 F1 はアンサンブル + TriviaQA 追加データ込みの比較で、TriviaQA 無しでは「0.1-0.4 F1 落ちる」と paper 自身が認める（experiment.tex §4.2）。abstract の見出し数字はやや盛り気味。
  - GLUE で abstract は leaderboard 公式 score 80.5（+7.7%）、experiment.tex は「BERT_BASE 4.5% / BERT_LARGE 7.0% の平均改善」、Table 1 は WNLI 除外平均で BERT_LARGE 82.1 と、複数の差分定義が混在していて読みにくい。
  - 事前学習コーパスが BooksCorpus + English Wikipedia のみ。多言語や口語・コード等の generalization は本論文では一切評価されていない。
  - NSP の効果検証が「外したら下がる」レベルで、なぜ効くのかの機序解析は無い。後年 RoBERTa / ALBERT で「NSP は不要、SOP の方が良い」と覆されたのは、この ablation が too shallow だった証左。
  - MLM のマスク戦略 ablation は MNLI と NER だけで、SQuAD・GLUE 全体での感度は示されていない。
  - 計算資源（Cloud TPU 64 チップ × 4 日、TeX に世代記載なし）が当時の academic 再現を事実上不可能にしており、その後「巨大事前学習 + 公開重み」の研究形態を固定化した功罪両方を負う論文。
- **次に試したいこと**:
  - 80/10/10 比率を 0/0/100 や 100/0/0 などに振った時の downstream 性能を、本論文の MNLI+NER だけでなく SQuAD / SWAG にも広げて再測定し、後続の dynamic masking との比較軸を引く。
  - NSP を coherence-based な目的（SOP, contiguity）に置き換えた条件で Table 5 と同じセットアップを走らせて、QNLI の 3.5pt 劣化が NSP 固有か「文ペア事前学習があること」由来かを切り分ける。
  - Feature-based 経路で全 11 タスクを回し、Table 7 の NER で言う「fine-tune との 0.3 F1 差」が他タスクでも保たれるか確認。エッジ推論向けの distillation 目標設定に効く。
  - BERT_LARGE の fine-tune 不安定性を「random restart して best を選ぶ」ではなく、warmup / re-init top-k layers / LLRD などで安定化した時のメディアン性能を測り、報告値の bias を可視化する。

## Notes / Quotes

- "We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional…" (intro.tex)
- "the masked LM only make predictions on 15% of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge." (bert_details.tex)
- "The vector C is not a meaningful sentence representation without fine-tuning, since it was trained with NSP." (bert.tex 脚注)
- "All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU…" (bert.tex)
- "we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained." (ablation.tex §5.2)
- Masking 80/10/10：80% `[MASK]`、10% random token、10% unchanged（bert.tex §3.1 + bert_details.tex 例 `my dog is hairy` → `my dog is [MASK]` / `my dog is apple` / 不変）。
- Pre-training データ：BooksCorpus 800M words + English Wikipedia 2,500M words、document-level corpus（Billion Word Benchmark のような shuffled sentence-level は不可）。
- 推奨 fine-tune 探索範囲：batch ∈ {16, 32}, Adam lr ∈ {5e-5, 3e-5, 2e-5}, epochs ∈ {2, 3, 4}、dropout は 0.1 固定。
- GLUE Test (Table 1)：BERT_LARGE Average 82.1（Pre-OpenAI SOTA 74.0、OpenAI GPT 75.1、BiLSTM+ELMo+Attn 71.0）。
- SQuAD v1.1 Test (Table 2)：BERT_LARGE Sgl.+TriviaQA EM 85.1 / F1 91.8、Ens.+TriviaQA EM 87.4 / F1 93.2、Human 82.3/91.2、トップ ensemble nlnet 86.0/91.7。
- SQuAD v2.0 Test (Table 3)：BERT_LARGE Single EM 80.0 / F1 83.1、Human 86.9/89.5。
- SWAG Test (Table 4)：BERT_LARGE 86.3、ESIM+ELMo 59.2、OpenAI GPT 78.0、Human (expert) 85.0 / (5 ann.) 88.0。
- CoNLL-2003 NER (Table 7)：BERT_LARGE fine-tune Test F1 92.8、feature-based の最良は "Concat Last Four Hidden" の Dev 96.1。
- Direction ablation (Table 5)：BERT_BASE 84.4/88.4/86.7/92.7/88.5 → No NSP 83.9/84.9/86.5/92.6/87.9 → LTR&No NSP 82.1/84.3/77.5/92.1/77.8（MNLI / QNLI / MRPC / SST-2 / SQuAD F1）。
- (verified 2026-05-20) Summary 手法の計算資源を「Cloud TPU 64 チップで 4 日」から `BERT_BASE` 4 TPU(16 chips)/`BERT_LARGE` 16 TPU(64 chips) で各 4 日に補正（bert_details.tex §A.2）。
- (verified 2026-05-20) Summary 結果の Table 5 推移を `MRPC 86.7→77.5, SQuAD F1 87.9→77.8` から `No NSP→LTR&No NSP: MRPC 86.5→77.5, SQuAD F1 87.9→77.8` に整合 (direction_ablation_tab.tex)。
- (verified 2026-05-20) Summary 結果の Table 6 size ablation「全 4 タスクで単調改善」を実際のカラム数に合わせ「MNLI-m / MRPC / SST-2 の 3 タスク」に修正 (size_ablation_tab.tex)。
- (verified 2026-05-20) Summary 結果の GLUE 記述を、abstract の leaderboard score 80.5 と Table 1 (WNLI 除外平均) 82.1 を分けて記述 (abstract.tex, glue_official_tab.tex 脚注)。
- (verified 2026-05-20) Summary 結果の SQuAD v1.1「単体 91.8」を「Sgl.+TriviaQA 91.8」に明確化 (squad_tab.tex)。
- (verified 2026-05-20) Critical Thoughts のコスト記述を「シングル GPU で 1 時間以下」から「単一 Cloud TPU で 1 時間以下（GPU では数時間）」に修正 (bert.tex 脚注)。
- (verified 2026-05-20) 「TPU v3」表記を削除。TeX には Cloud TPU としか書かれていない (bert_details.tex §A.2)。

## Related Papers

- Vaswani+ 2017, "Attention Is All You Need" — Transformer encoder の母体。
- Radford+ 2018, OpenAI GPT — 単方向 fine-tuning の直接比較対象。BERT_BASE は同サイズに合わせて設計。
- Peters+ 2018, ELMo — 浅い concat 双方向 feature-based のベースライン。
- Howard & Ruder 2018, ULMFiT — 単方向 LM fine-tune の先行研究。
- Dai & Le 2015 — semi-supervised sequence learning、fine-tune 系の元祖。
- Taylor 1953, Cloze task — MLM 目的の概念的源流。
- Rajpurkar+ 2016 (SQuAD v1.1) / SQuAD 2.0、Wang+ 2018 (GLUE)、Zellers+ 2018 (SWAG)、Tjong Kim Sang & De Meulder 2003 (CoNLL-2003 NER) — 評価ベンチマーク。
- Vincent+ 2008, Denoising Auto-Encoder — マスク再構成と対比される目的関数。
- Melamud+ 2016, context2vec — 双方向 LSTM での文脈表現、feature-based の比較対象。
