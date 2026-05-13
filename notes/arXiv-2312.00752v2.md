# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- arXiv: https://arxiv.org/abs/2312.00752
- source: ../papers/arXiv-2312.00752v2/
- authors: Albert Gu (CMU), Tri Dao (Princeton)
- venue / year: arXiv preprint 2023 (v2)。著者注では COLM 2024 用フォーマットも同梱されている
- tags: [SSM, sequence-model, long-context, language-model, attention-free]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: Transformer は self-attention が文脈長 L に対して二次で、KV cache のため推論も線形時間。一方 S4 系の structured SSM は L に線形だが LTI（時間不変）に縛られているため、テキストや DNA のような離散・情報密度の高いモダリティで弱い。著者は SSM が「context-based reasoning（必要な情報だけを選んで状態に詰める）」を出来ないことを根本原因と特定する。
- **手法**: (1) **Selection（S6）**: SSM のパラメータ $\B, \C, \Delta$ を入力 $x$ の線形射影で生成し時間変化させる（$\A$ は $\Delta$ 経由で間接的に選択的になるので非選択のまま）。これにより RNN gate の一般化として $h_t = (1-g_t)h_{t-1} + g_t x_t$（Theorem 3.1: $N=1,\A=-1,\B=1$ の極限）が回収できる。(2) **Hardware-aware parallel scan**: 時間可変化で convolution が使えなくなる代わりに、kernel fusion + parallel scan + recomputation で状態 $h$ (B,L,D,N) を HBM に materialize せず SRAM 内で完結させる。FlashAttention 流の memory IO 最適化を scan に持ち込んだ形。(3) **Mamba block**: H3 ブロック（linear-attention 風）と Gated MLP を 1 ブロックに統合し、attention も MLP も無い同一ブロックを homogeneous に積む。$E=2$ で SwiGLU 互換、$12D^2$ パラ数を Transformer のブロックと揃える。
- **結果**:
  - Selective Copying: S4 単体 18.3% → S6 97.0%、H3 アーキ + S6 99.7%、Mamba+S6 **99.8%**（Table 1）。
  - Induction Heads: 長さ 256 で学習し、$2^{20}=1{,}048{,}576$（4000×）まで完全に外挿。他の方法はいずれも 2× 止まり。
  - Pile での scaling laws (125M〜1.3B): **attention-free モデルとして史上初めて Transformer++（LLaMA レシピ）に匹敵**。文脈長を 2k→8k に伸ばすほど差が広がる。
  - Zero-shot 8 タスク平均: Mamba-1.4B **59.7** vs Pythia-1.4B 55.2 / RWKV-1.5B 54.3。Mamba-2.8B **63.3** vs Pythia-2.8B 59.1 / RWKV-3B 59.6（≈ Pythia-6.9B 61.7、GPT-J-6B 63.0 とほぼ同水準で、約 2× サイズの Transformer に並ぶ）。Pile ppl も Mamba-2.8B 6.22 < Pythia-2.8B 6.73。
  - DNA (HG38): 同精度を Transformer++ / HyenaDNA より **3–4× 少ないパラメータ**で達成。系列長を $2^{10}$→$2^{20}$ まで伸ばすと perplexity が単調改善（HyenaDNA は逆に悪化）。
  - Audio (SC09 speech): Mamba 6.1M で **FID 0.94**（SaShiMi 5.8M の 1.99 から半減以上）、24.3M で FID 0.67、IS 7.33 と SOTA。
  - 効率: SSM scan は 2K 超で FlashAttention-2 より速く、PyTorch 標準 scan より 20–40× 高速。推論スループットは同サイズ Transformer 比 **4–5×**（KV cache が無く batch を大きく取れるため。Mamba-6.9B が Transformer-1.3B より高スループット）。
- **貢献**: (1) 入力依存パラメータ化（selection）による SSM の表現力強化、(2) 状態を materialize しない hardware-aware selective scan、(3) attention も MLP も無いシンプルな単一ブロック Mamba、(4) 言語・DNA・音声の 3 モダリティで Transformer と同等以上を線形時間で達成。

## Takeaway（自分にとっての要点）

- **「LTI vs データ依存」の本質**: アーキ gating（multiplicative interaction）はチャンネル軸の作用に過ぎず、系列軸の spacing を変えられないので Selective Copying を解けない。系列軸方向の動力学そのものを入力依存にする必要がある——これは LSTM/GRU の gate が本質、という古い直観を SSM の言葉で再定式化したもの。
- **Δ（discretization step）が gate の正体**: Theorem 3.1 で $g_t = \sigma(\mathrm{Linear}(x_t))$ が出てくる。Ablation でも selective Δ だけで 10.93→9.81 ppl 改善し、$\B,\C$ も足すと 8.71。**Δ こそ最重要パラメータで、$\A$ を selective にする必要は無い**（$\dA=\exp(\Delta \A)$ で Δ 経由になるため）。
- **状態次元 $N$ を上げるのは selective B,C が前提**: $N=1\to16$ で 9.88→9.81（非選択）vs 9.73→8.71（選択）。+1% のパラメータで 1.0+ ppl 改善。LTI SSM が $N$ を上げても効かないのは「全部覚える」だけで「選んで覚える」になっていないから。
- **Real-valued がデフォで良い**: 言語/DNA など離散モダリティでは S4D-Real が S4D-Lin（複素）より良い（8.85 vs 9.16）。複素は audio waveform のような連続モダリティで効く（discussion で著者自身が "no free lunch" と認める）。
- **inference 5× は KV cache フリーの帰結**: 同パラ Transformer よりむしろ少し大きい Mamba（6.9B vs 1.3B）でスループットが上回るというのが効いてくる場面 = 長文・大バッチ・edge inference。
- **「context が長いほど精度が単調改善する」のは selective モデルに限る**: LTI 系の global convolution は不要な情報を捨てる手段が無いので長文で崩れる、という説明と DNA 実験 ($2^{10}$→$2^{20}$) の単調改善は説得力あり。Filtering Context property を実証している点が重要。
- **アーキはむしろ簡素化方向**: H3 vs Mamba の差は小さく（8.95 vs 8.69 ppl）、勝因は S6 layer。アーキ凝るより selection を入れる方が効く、という結論は実装労力配分の指針になる。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「selection が無いから SSM は言語で弱い」という診断と、「Δ を入力依存にすればよい」という処方箋が一直線で繋がっており、ablation 表 (Table 6, 8, 9) で個別に検証されている。
  - 純粋な系列長線形・KV cache フリー・GPU 上で実用速度、という三拍子を同時に成立させた初の Transformer 同等モデル。FlashAttention の知見を scan に転用するという工学が肝で、論文に閉じず実装（state-spaces/mamba）と一体で評価できる。
  - 言語・DNA・音声の 3 モダリティで一貫して効く汎用 backbone であることを示しており、しかも DNA は great apes 分類で 99% 同一の DNA を分けるという挑戦的タスクを用意している。
  - Theorem 3.1 で gate 機構と SSM 離散化を理論的に接続している点は綺麗。「heuristic gating の原理的基礎が SSM 離散化」という主張は強い。
- **弱み / 疑問**:
  - 著者自身が "Scaling" 節で認める通り、評価は最大 1.3B（zero-shot は 2.8B）止まり。Llama-7B/13B/70B、RWKV-7B、RetNet-7B クラスで同等性が保たれるかは未検証。実際 2024 年以降の追試で 3B〜7B では Transformer に対して in-context learning / multi-task で差が出るとの報告もあり、本論文の範囲では断定不能。
  - "Downstream Affordances" を著者自身が open question として残している: fine-tuning, LoRA, in-context learning, RLHF, quantization 等が Transformer と同じ品質で乗るかは未知。Mamba ベースのチャットモデルが実用足り得るかは別研究待ち。
  - "No Free Lunch" の節で、selection を入れると LTI が得意な連続信号モダリティでむしろ劣化し得ると認めている。実際 audio waveform 実験では「ここだけ complex parameterization に切り替えた」と注記しており、デフォルトレシピが普遍ではない。
  - 推論 5× / 学習 scan 20–40× は印象的だが、ベースラインの中身（attention は FlashAttention-2 と比較、scan は PyTorch 標準と比較）に非対称があり、attention vs Mamba の総合 wall-clock 比較（同パープレに到達するまでの GPU 時間）は本文では示されていない。
  - 長文外挿（Induction Heads が 4000× 通る）は印象的だが、これは語彙 16 のおもちゃタスク。Pile の自然文で 1M token に外挿しても精度が保たれるかの直接実験は無い。
  - Selective Copying の比較表で「H3 + Hyena: 30.1」のような低スコアの存在から、SSM アーキを混ぜれば自動で良くなるわけではなく、結局 inner layer に S6 が乗るかどうかが全てになっている。アーキ研究側からはやや拍子抜けな結論。
  - Hybrid（attention + Mamba）の比較は付録に追いやられている。実応用では純 Mamba より hybrid の方が良いケースが現実に多いと示唆されており、その分純粋 Mamba 推しの結論は割引が必要。
- **次に試したいこと**:
  - 「Δ が gate である」という Theorem 3.1 を使って、学習済み Mamba 内部で「token ごとの Δ 値」を可視化し、attention map の代替として解釈性研究の材料にできるか試す。
  - Selective B/C/Δ の中で「Δ だけ」「B,C だけ」を切ったときの perplexity gap（10.93→9.81 と 10.93→9.98）を、文脈長を変えて再計測。短文では Δ、長文では B,C が効くという仮説を検証したい。
  - Audio で complex に戻す件を逆手に取り、「modality ごとに real / complex を mix した SSM block」をルーティングで切り替える hybrid を試す。
  - 推論 5× のうち何割が「KV cache 不在」由来で、何割が「scan kernel 最適化」由来かを decomposition する benchmark。
  - 1.3B 超の scaling（7B〜）を Llama-2 と同じデータで学習し、本論文の scaling 外挿がそのまま伸びるか、屈曲点があるかを見る（公開された Mamba-2 系列の知見と比較）。

## Notes / Quotes

- "we propose that a fundamental principle for building sequence models is **selectivity**: or the context-aware ability to focus on or filter out inputs into a sequential state." (method.tex)
- "discretization of SSMs is the principled foundation of heuristic gating mechanisms." (method.tex, Theorem 3.1 周辺)
- "selectivity in $\dt$ is enough to ensure selectivity in $\dAB$, and is the main source of improvement." (method.tex) — $\A$ を selective にしない設計判断の根拠。
- "Mamba is the first attention-free model to match the performance of a very strong Transformer recipe (Transformer++) that has now become standard." (experiments.tex)
- Selective Copying: S4 18.3 → S6 97.0、Mamba+S6 99.8（Table 1）。Induction Heads は 256 学習・$2^{20}$ 外挿で perfect。
- Pile ppl: Mamba-2.8B 6.22 / Pythia-2.8B 6.73 / RWKV-3B 7.00。Zero-shot avg: Mamba-2.8B 63.3 ≈ GPT-J-6B 63.0。
- 効率: SSM scan は seq>2K で FlashAttention-2 超え、推論スループット 4–5× over Transformer。
- DNA: Mamba は Transformer++/HyenaDNA を 3–4× 少パラで一致。長さ 1M まで単調改善。
- SC09: Mamba 6.1M で FID 0.94（SaShiMi 5.8M の 1.99 から半減超）、24.3M で 0.67。
- Ablation: selective Δ単独 9.81、全部 selective 8.71、非選択 10.93。state dim $N=16$ で +1% パラに対し >1.0 ppl 改善（B,C selective のとき）。
- 著者の認める limitation: scaling は ≤1.3B で未検証、downstream affordances（fine-tune/RLHF 等）は open、continuous モダリティでは LTI の方が良い場合あり（"no free lunch"）。
- Audio waveform 実験は本論文中で唯一 complex parameterization に切り替えた例（method.tex の Real vs Complex 節と合わせて読む）。

## Related Papers

- Gu+ 2021/2022 S4 / S4D (`gu2021combining`, `gu2022efficiently`, `gu2022parameterization`) — Mamba が直接置き換える LTI ベース。
- Dao+ 2023 H3 (`dao2023hungry`) — Mamba ブロックの基。
- Poli+ 2023 Hyena — convolution 系の比較対象。
- Peng+ 2023 RWKV / Sun+ 2023 RetNet — recurrent rival、scaling 比較で言及。
- Dao 2023 FlashAttention-2 — hardware-aware kernel の系譜、scan 実装の比較対象。
- Olsson+ 2022 Induction Heads / Elhage+ 2021 — mechanistic interpretability、合成タスク設計の根拠。
- Hoffmann+ 2022 Chinchilla — scaling law プロトコル。
- Touvron+ 2023 LLaMA — Transformer++ レシピの参照点。
- Nguyen+ 2023 HyenaDNA — DNA ベンチマークの直接の比較対象。
- Goel+ 2022 SaShiMi — 音声 U-Net baseline。
- Blelloch 1990 / Smith+ 2023 S5 — parallel scan アルゴリズムの源流。
