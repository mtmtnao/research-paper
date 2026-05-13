# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

- arXiv: https://arxiv.org/abs/2402.14905
- source: ../papers/arXiv-2402.14905v2/
- authors: Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra (Meta)
- venue / year: ICML 2024
- tags: [on-device, small-LLM, architecture, weight-sharing, GQA]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: クラウド LLM のコスト・遅延・エネルギーが膨張する一方、モバイル端末側は DRAM 容量（iPhone 15 で 6GB、Pixel 8 Pro で 12GB）と電力（iPhone のフル充電 ≒ 50kJ、7B モデルは 0.7 J/token で 2 時間未満で枯れる）の制約により <1B パラメータの LLM が必要。だが既存の sub-billion 系（OPT-125M、GPT-Neo-125M、Pythia-160M、Galactica-125M、RWKV-169M、BLOOM-560M 等）は sub-billion 専用に最適化されておらず性能が頭打ち。一方 Kaplan の scaling law は「アーキテクチャは精度にほぼ影響しない」と主張してきた。
- **手法**: sub-billion 領域で重要なのは「データ量・パラメータ数」よりも「アーキテクチャ」だと主張し、次の4つの設計を組み合わせた **MobileLLM** を baseline として作る。(1) FFN を SwiGLU 化（125M で avg 42.6→43.9）、(2) **deep-and-thin**（layer 数 30〜32 を採用、scaling law の常識である 12 layer 系を否定）、(3) 入出力 **embedding sharing**（vocab 32k×embed 512 で 16M パラ削減、約11.8%）、(4) **grouped query attention** （head 16、kv-head 4、head_dim=64）。さらに immediate **block-wise layer sharing**（隣接 2 ブロックで重みを共有、計算は 2 回行うが SRAM 上に置けるので DRAM↔SRAM の重み転送を回避）を追加した版を **MobileLLM-LS** と呼ぶ。学習は 32×A100、各 GPU bs=32、Adam wd=0.1、lr=2e-3 cosine、120k iter / 0.25T tok で探索 → 480k iter / **1T tokens** で最終モデル。学習データの種類は本文では明示されていない。
- **結果**: zero-shot common sense reasoning 平均（Table 1）で **MobileLLM-125M 46.3 / -LS-125M 47.0**（先行 best RWKV-169M 43.6、Pythia-160M 42.5 を 2.7〜3.8pt 上回り、しかも 22〜26% 小さい）、**MobileLLM-350M 51.3 / -LS-350M 52.1**（先行 best RWKV-430M 47.0 を +4 以上）。layer sharing が +0.7 / +0.8。TQA F1 (Table 2) は -125M 1-shot 13.9（OPT-125M 8.7）、-350M 1-shot 22.0（OPT-350M 11.0、+10 以上）。RACE middle/high も同様に大幅勝ち。Chat（Table 6）: AlpacaEval 勝率 **MobileLLM-LS-350M 48.20%** が text-davinci-001（自己勝率 50%）に肉薄、Falcon-1.3B 30.38%・OPT-1.3B 38.84% を上回り 1B 超のモデルすら抜く。MT-Bench も -LS-125M 2.52、-LS-350M 3.16（OPT-125M 1.21 比）。API calling（Table 8）: **MobileLLM-350M の EM_intent 65.3 / EM_structure 48.8** は LLaMA-v2 7B（62.8 / 50.9）に同等以上（Rouge は 7B が上）。iPhone 13 ExecuTorch + MPS 実測（Table 9）: MobileLLM-LS の load 43.6ms / init 1388.2ms / execute 16.0ms は MobileLLM 比 +2.2% / +2.6% に対し、純粋に 60 層に倍化した non-shared モデルは +75% / +146% / +86%。W8A8 PTQ も accuracy drop <0.5pt（§4.5）。Appendix では 600M / 1B / 1.5B にスケールしても MobileLLM-1.5B avg 59.4 が Qwen1.5-1.8B を 2.9pt 上回る。
- **貢献**: (1) sub-billion での「深さ > 幅」の経験則を 19 モデル grid で実証、(2) embedding sharing + GQA を sub-billion 文脈で再評価して採用、(3) immediate block-wise layer sharing という単純で no-extra-param・低遅延な精度向上法、(4) 125M/350M で SOTA、加えて downstream（Chat、API calling）で 1B〜7B クラスに迫る結果。(5) 事前学習コードを `github.com/facebookresearch/MobileLLM` で公開。

## Takeaway（自分にとっての要点）

- sub-billion 帯では「層の数を倍にして幅を絞る」「embedding を共有して浮いたパラを層に回す」だけで scaling law が示唆する以上に伸びる。125M で 30 層、350M で 32 層という具体的な数値が出ているのは即使える指針。
- block-wise layer sharing は「実装が memcpy 削減の意味で SRAM-friendly」と「精度が +0.7〜0.8 上がる」が両立する稀な手法。on-device 推論を意識した時の design space で覚えておく価値が高い。
- GQA は 7B+ で KV-cache 圧縮のために使うイメージだったが、sub-billion でも「kv-head を共有して浮いたパラを embed dim に振る」weight 再利用テクとして効く（125M で +0.4）。
- 4.3 章（KD）が negative result として記録されている点が嬉しい。LLaMA-v2 7B を teacher にしても 125M/350M では速度 2.6–3.2× 遅くなるだけで精度はほぼ同等 or 劣る。sub-billion での KD は割に合わないという根拠データ。
- API calling で 350M が 7B と EM 同等という結果は「on-device で agent 化する」ロードマップを現実的に見せる。Rouge が低くても intent/structure が当たれば良い、というタスク特性も含めて応用先選定の参考になる。
- AlpacaEval 48.20% は text-davinci-001 の自己勝率 50% にほぼ並ぶ、と著者は強調するが、GPT-4 採点バイアスの余地は残るので「self-win rate ≒ 50% との相対比較」を見るのが妥当。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 19 個の 125M/350M モデルを実際に学習して depth vs width を grid 評価しているので、scaling law へのカウンター主張が経験則として強い。Section 2.2.2 / Appendix C のテーブルが裏付けになっている。
  - 「アーキ → embedding 共有 → GQA → layer sharing」の寄与が Table 11（appendix の roadmap、本文中で +1.3/+0.9/-0.2/+0.4/+1.1 と分解）で逐次的に検証されており、貢献の切り分けが明確。
  - on-device 実測（iPhone 13 ExecuTorch）まで踏み込んでいる点が、一般的な「精度だけ報告」のサブ B 系論文より一段強い。layer sharing は理屈上 SRAM フィットを根拠にしているので、それを実機の execute 時間（16.0 vs 29.0 ms）で示しているのは綺麗。
  - 1.5B まで同じレシピがスケールすることを appendix で示し「sub-billion 専用テク」ではないことに踏み込んでいる。
- **弱み / 疑問**:
  - 学習データの種類・出典が main text に明記されていない（TeX 中には明示されていない）。比較対象の OPT/Pythia/GPT-Neo はそれぞれ違うコーパスで学習されているので、「アーキ差で勝った」のか「データで勝った」のか厳密には分離できない。1T tokens で揃えたのは比較相手より多い可能性が高い。
  - 「深さ > 幅」の主張は 125M/350M の範囲では強いが、1B 超の領域では同じ手法（grid）での比較は appendix にも見当たらず、結論は extrapolation にとどまる。
  - Layer sharing は accuracy も上がっているが、Table 3 では immediate block-wise (45.0) は repeat-all-over (45.2) にわずかに劣る。本文では「SRAM に収まるから immediate を採る」と言うが、その精度差自体の有意性（seed variance）の議論がない。
  - 著者自身が認める limitation: **KD が効かない**（§4.4, appendix F）。「pre-training に teacher を入れても 2.6–3.2× の slowdown でほぼ same/worse」という結果は、small LLM の蒸留に対する素朴な期待を否定する重要な negative result。
  - AlpacaEval / MT-Bench は GPT-4 judge による評価で、verbosity bias など既知のバイアスがある。Chat 結果を「1B モデル超え」と読むには注意。
  - API calling のデータセットは synthetic 5000/2500 を著者が生成しており、tool 呼び出しの分布が真の on-device 利用と一致する保証はない。
- **次に試したいこと**:
  - 同じ学習トークン量・同じデータでアーキだけ変えた fair comparison（特に Pythia と完全同条件）を再現し、「アーキ寄与」を切り出す。
  - immediate block-wise sharing の repeat 回数を 3〜4 にしたとき性能が落ちる現象（Appendix E）を、loss landscape / 表現の冗長性の観点から分析。
  - GQA の kv-head 数を mixture-of-experts や per-layer で可変にしたら更に精度／メモリの pareto が動くか。
  - MobileLLM-350M を function-calling 専用に SFT した時、Llama-3-8B-Instruct とどこまで差が縮まるか。on-device agent の現実的下限を測る。
  - 蒸留が効かない原因（teacher logits の温度、batch 内 cross-entropy 形式）を変えて再検証。

## Notes / Quotes

- "Contradictory to the scaling law, we demonstrate that depth is more important than width for small LLMs." (introduction)
- 125M/350M baseline 構成: head=16, kv-head=4, head_dim=64, 30〜32 layers, embedding shared。
- Table 1: MobileLLM-LS-125M (30 distinct layers) avg **47.0** > 既存 350M 級モデル多数（Pythia-410M 46.6、BLOOM-560M 44.2）。
- Table 9 実測: MobileLLM-LS は 60-layer 非共有版より load 36.5%、init 58.5%、execute 44.8% 速い（ExecuTorch + MPS / iPhone 13）。
- Layer sharing ablation（Table 3）で immediate block-wise は repeat-all-over に僅差で負けるが、SRAM ローカリティ理由で immediate を選択。
- 学習: Adam wd=0.1、lr=2e-3 cosine、32×A100、bs=32/GPU、最終 480k iter × 1T tokens。
- Appendix F: KD with LLaMA-v2 7B teacher → 2.6–3.2× slower, comparable or inferior accuracy → 採用しない。
- §4.3 quantization: W8A8 PTQ で accuracy drop <0.5 pt、layer sharing と互換。

## Related Papers

- Kaplan+ 2020 "Scaling Laws for Neural Language Models" — 本論文が主に挑戦する scaling law。
- Zhang+ 2022 OPT — embedding sharing の先行例、本論文の主要比較対象 (125M/350M)。
- Ainslie+ 2023 / Chowdhery+ PaLM — grouped-query attention の出典。
- Dauphin+ / Shazeer SwiGLU — FFN 活性化選択の根拠。
- Touvron+ 2023 LLaMA / LLaMA-v2 — API calling の比較対象 (7B) と teacher。
- Biderman+ 2023 Pythia, Black+ GPT-Neo, Scao+ BLOOM, Dey+ Cerebras-GPT, Peng+ RWKV, Taylor+ Galactica, Zhang+ 2024 TinyLlama, Bai+ Qwen, Thawakar+ MobiLlama — sub-billion 比較対象群。
- Shen+ 2022 Sliced / Reid+ 2021 Subformer — 中間層の weight sharing 先行研究。
- Hinton+ 2015 KD — appendix で negative result として再検証。
- AlpacaEval, MT-Bench (Zheng+ 2023), TriviaQA, RACE, BoolQ, PIQA, SIQA, HellaSwag, ARC, OBQA, WinoGrande — 評価ベンチマーク群。
