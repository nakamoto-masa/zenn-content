---
title: "NumPyの乱数APIはなぜ「面倒」になったのか"
emoji: "🎲"
type: "tech"
topics: ["numpy", "pytorch", "jax", "機械学習", "python"]
published: true
---

NumPyの乱数の推奨的な書き方が変わったのをご存知でしょうか。変わったのはずいぶん前（2019年、NumPy 1.17）ですが、まだ浸透していない印象があります。

```python
# NumPy: 以前の書き方
np.random.seed(42)
data = np.random.randn(100)

# NumPy: 現在の推奨
rng = np.random.default_rng(42)
data = rng.standard_normal(100)
```

PyTorchも同じ方向に動いています。`torch.Generator` が用意されていて、関数に渡せるようになっている。

```python
# PyTorch: おなじみの書き方
torch.manual_seed(42)
x = torch.randn(10)

# PyTorch: Generatorを使う書き方
gen = torch.Generator().manual_seed(42)
x = torch.randn(10, generator=gen)
```

Generatorオブジェクトを作って、そこからメソッドを呼ぶ（またはgenerator引数で渡す）。わざわざオブジェクトを挟む意味がわからない、という人は多いと思います。私も最初はそうでした。

ただ、背景を知ると印象が変わります。この変更は「面倒にした」のではなく、並列計算や関数型プログラミングの世界で長年議論されてきた問題への回答です。同じ問題意識はPyTorchやJAXにも波及していて、各フレームワークの対応はきれいに分かれています。歴史的な経緯を含めて、なぜこうなったのかを追ってみます。

**対象読者**: NumPy/PyTorch/JAXを日常的に使っているMLエンジニアで、乱数APIの変更に「なぜ？」と感じている方。

## グローバルstateの3つの罠

旧APIの問題は `np.random.seed()` そのものではなく、その裏にある**グローバルstate**です。

`np.random.seed(42)` を呼ぶと、モジュール内部にある1つの `RandomState` オブジェクトのstateが書き換わる。以降の `np.random.rand()` や `np.random.randn()` はすべてこの同じstateを参照し、呼ぶたびにstateが進む。

問題はこの「1つのstate」をプロセス内の全コードが共有していること。

### 罠1: 他のライブラリが勝手にstateを変える

```python
np.random.seed(42)
# ↓ この関数の中でscikit-learnがnp.random.randint()を呼んでいたら？
model = train_model(data)
result = np.random.rand(10)  # seedを42に設定した直後の値ではない
```

自分のコードでseedを固定しても、間に呼ばれるライブラリが内部で `np.random` を使えばstateが進む。再現性があるように見えて、ライブラリのバージョンを上げただけで結果が変わったりする。

実際に起きた例として、vLLMがモデルのロード時に内部で `np.random.seed()` を呼んでいて、利用者側の乱数が意図せずリセットされるという[問題が報告されています](https://zenn.dev/aratako_lm/articles/701832f1e346a1)。

### 罠2: コードを1行変えると全部壊れる

```python
np.random.seed(42)
a = np.random.rand(10)
b = np.random.rand(5)   # ← この行を消すと
c = np.random.rand(10)  # ← cの値が変わる
```

グローバルstateは呼び出し順にカウンターのように進んでいく。途中の呼び出しを1つ増やす・減らすだけで、以降のすべての乱数が変わる。

実験コードの開発中に「このサンプリングいらないな」と1行消したら、学習結果が全く変わった——という経験をしたことがある人もいるはず。seed固定しているのに。

### 罠3: NumPy自身が改善できなくなる

NumPyは長年「同じseedなら同じ乱数列を返す」ことを保証してきました（stream compatibility）。

この制約が足枷になった。たとえば正規分布の生成アルゴリズムをより高速なものに変えたくても、出力される乱数列が変わるので変えられない。ユーザーが再現性のためにseedに依存している以上、アルゴリズム改善を入れるたびに「結果が変わった」という報告が来る。

[NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html)（2018年提案）は、このstream compatibilityポリシーの緩和が主な目的の一つでした。

## 「面倒さ」の正体——stateを共有しないという選択

3つの罠に共通するのは「**1つのstateをみんなで共有している**」こと。

じゃあどうするか。2つの方向がある。

**方向A: 共有stateをスレッドセーフにする**
lockやmutexで保護して、同時アクセスを制御する。

**方向B: 共有stateをやめる**
各自が自分専用のstateを持ち、共有しない。

方向Aは一見まっとうに見えます。が、lockを取る時点でそこが逐次実行になる。並列処理の意味が薄れる上に、再現性の問題（どのスレッドが先にlockを取るかは非決定的）は解決しない。

NumPyが選んだのは方向B。これが `default_rng()` です。

```python
rng = np.random.default_rng(42)
```

この1行で「自分専用のGenerator」が作られる。他のどのコードとも共有されないstate。他のライブラリが何をしようが、このrngの出力は決定的に決まる。

```python
# 研究者Aの実験コード
rng_a = np.random.default_rng(42)
result_a = rng_a.standard_normal(100)

# 同じプロセス内で動く前処理ライブラリ
rng_lib = np.random.default_rng(0)
_ = rng_lib.random(1000)  # rng_aには一切影響しない
```

「面倒になった」のではなく、「**stateの所有権を明示する**」ようになった。暗黙の共有を、明示的な分離に変えた。

ここまでが「何が変わったか」と「なぜ変わったか」の話。では、この「共有stateをやめる」という発想はどこから来たのか。

## この設計思想はどこから来たのか

NumPyの設計者がある日思いついたわけではありません。2つの異なる領域で、同じ結論にたどり着いた流れがある。

### 並列計算の世界——Salmon et al. (2011)

D.E. Shaw Researchの研究者John Salmonらが2011年に発表した論文 "[Parallel Random Numbers: As Easy as 1, 2, 3](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)" が転換点でした。

従来のPRNG（Mersenne Twisterなど）は内部stateを逐次的に更新していく設計。1つのstateを複数のスレッドで共有すると、前述の通り破綻する。

Salmonらが提案したのは**counter-based PRNG**。Mersenne Twisterのような巨大な内部state（2.5KB）を逐次更新する代わりに、「key + counter」という最小限の入力から乱数を生成する。生成関数自体は純粋関数で、keyが違えば独立した乱数列になる。各スレッドに別々のkeyを配れば並列化できる。lockもバリアもいらない。

Threefry、Philoxといったアルゴリズムがここで生まれました。後にJAXが採用するのもThreefryです。

### 関数型プログラミングの世界——Haskell

Haskellは純粋関数型言語で、**グローバルな可変stateはそもそも存在できない**。すべての関数は入力だけから出力が決まる。

だからHaskellの乱数生成器 `StdGen` には最初から `split` があった。1つのgeneratorを2つの独立したgeneratorに分割する機能です。

ただし初期の `split` の実装は統計的品質が低く、長年問題を抱えていました。これを改善する研究が2013〜2014年に進みます。

- 2013年: Claessen & Pałka "[Splittable PRNGs using Cryptographic Hashing](https://dl.acm.org/doi/10.1145/2503778.2503784)"
- 2014年: Steele, Lea, Flood "[SplitMix](https://dl.acm.org/doi/10.1145/2660193.2660195)" — 高速で統計的に優れたsplittable PRNG

Haskellの `random` ライブラリは[v1.2（2020年）でSplitMixを正式採用](https://www.tweag.io/blog/2020-06-29-prng-test/)しました。

### 2018年、2つの流れが合流する

2018年は偶然の一致のように、複数の動きが同時に起きた年です。

- **2018年5月**: Robert Kernが[NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html)を提案。NumPyの乱数ポリシー変更の議論が始まる
- **2018年7月**: NEP 19が承認される
- **2018年12月**: GoogleがJAXをオープンソースとしてリリース

NumPyのNEP 19とJAXは独立に進行していましたが、解決しようとしていた問題は同じ。「グローバルな可変stateを使った乱数生成は、現代のコンピューティングに合わない」ということ。

Kevin Sheppardの[randomgen](https://github.com/bashtage/randomgen)プロジェクト（「NextGen NumPy RandomState」の後継）が実装を先行し、2019年7月のNumPy 1.17でGenerator/BitGeneratorとしてマージされました。

## NumPy・PyTorch・JAX——三者三様の選択

同じ問題に対して、3つのフレームワークは異なるアプローチを取りました。

### NumPy——明示的stateを**推奨**

NumPyは巨大な既存エコシステムを抱えています。旧APIを一夜にして廃止するわけにはいかない。

そこで取ったのは「旧APIは残すが、新APIを推奨する」という段階的な移行。旧APIはまだ動くし、deprecation warningも出ない。

新APIでは `Generator` と `BitGenerator` が分離されていて、乱数アルゴリズム（BitGenerator）を差し替えられる設計になっています。デフォルトはPCG64。Mersenne Twisterより高速で統計的性質も良い。

```python
from numpy.random import Generator, MT19937, PCG64, Philox

rng_pcg = Generator(PCG64(42))       # デフォルト
rng_mt = Generator(MT19937(42))      # 旧来のMersenne Twister
rng_philox = Generator(Philox(42))   # counter-based（並列向き）
```

NEP 19にはこう書かれています。

> The implicit global RandomState behind the np.random.* convenience functions can cause problems, especially when threads or other forms of concurrency are involved. Global state is always problematic. We categorically recommend avoiding using the convenience functions when reproducibility is involved.
>
> （np.random.*の便利関数の裏にある暗黙のグローバルRandomStateは、スレッドなどの並行処理が絡むと問題を起こしうる。グローバルstateは常に問題である。再現性が求められる場面では、便利関数の使用を避けることを強く推奨する。）

### PyTorch——グローバルstateが**まだ主流**

PyTorchの立ち位置は興味深い。`torch.Generator` オブジェクトは存在していて、一部の関数では `generator=` 引数で渡せる。しかし公式の推奨は依然として `torch.manual_seed()` です。

2020年に "[Do not modify global random state](https://github.com/pytorch/pytorch/issues/39716)" というIssueが立ちました。議論の方向は「ドキュメントでGeneratorの存在に触れる」程度に落ち着きましたが、そのドキュメント改善すら実施されないまま、2026年2月時点でもIssueはOpenのままです。

PyTorchがグローバルstateを維持できているのには理由がある。典型的な学習ループは「1プロセス・逐次実行」で、`manual_seed()` だけで再現性が確保できるケースが多い。JITやTPU並列を前提とするJAXとは使われ方が違うわけです。

ただ、分散学習やマルチワーカーの `DataLoader` が当たり前になりつつある現在、この設計がいつまで持つかは疑問が残ります。

### JAX——純粋関数型を**強制**

JAXは最も徹底しています。すべての乱数生成にkeyの明示的な受け渡しが必須。

```python
key = jax.random.key(42)
key, subkey = jax.random.split(key)
data = jax.random.normal(subkey, (100,))
```

`jax.random.normal(100)` のように「keyなしで呼ぶ」ことはできない。これは制約ではなく必然です。JAXの関数はJITコンパイルと自動微分の対象になるため、純粋関数（出力が入力だけで決まる）でなければならない。隠れたグローバルstateがあるとJITが成り立たない。

並列計算の面でも、TPU/GPU上での並列実行はcounter-based PRNG（Threefry）+ key-splittingが前提。[JEP 263](https://docs.jax.dev/en/latest/jep/263-prng.html)にこの設計判断が詳しくまとめられています。

JAXは「ゼロから設計できた」ので、妥協する必要がなかった。

## 「暗黙」から「明示」へ——Pythonの大きな流れ

乱数APIの変遷は、Python全体で起きている設計思想の転換と重なっています。

| 暗黙的 | 明示的 | 背景 |
|---|---|---|
| `np.random.seed()` | `default_rng()` | グローバルstate排除 |
| `def f(x):` | `def f(x: int) -> str:` | 型ヒント（PEP 484, 2015） |
| `from os import *` | `from os import path` | 名前空間の明示化 |

Pythonは「書きやすさ」を重視してきた言語。その設計は小規模なスクリプトでは自由で生産的ですが、規模が大きくなると暗黙の挙動が足を引っ張る。

乱数もまさにそう。1ファイルのJupyter Notebookなら `np.random.seed(42)` で何も困らない。でもチームで開発し、複数のライブラリが絡み、再現性を論文レベルで保証する必要がある場面では、暗黙のグローバルstateは爆弾になる。

The Zen of Pythonの "Explicit is better than implicit." が、ようやく乱数APIにも適用されたということです。

## 結局どうすればいいのか

ここまで読んで「なるほどわかった、でも自分のコードはどうすれば？」という方へ。

**新規コードは `default_rng()` を使う。** これだけです。

```python
rng = np.random.default_rng(42)
```

このrngを必要な関数に渡していく。グローバルstateに依存しないので、ライブラリの更新で結果が変わることもなければ、コードの順序を変えて壊れることもない。

既存コードの `np.random.seed()` を今すぐ全部書き換える必要はありません。動いているなら動いている。ただ、再現性が本当に必要な場面——論文の実験、本番のA/Bテスト、並列処理——では、Generatorへの移行を検討する価値があります。

PyTorchの場合は、`generator=` 引数を受け取る関数では `torch.Generator()` を渡すことで同じ考え方が適用できます。公式の推奨にはなっていませんが、再現性が重要な場面では意識しておいて損はありません。

面倒に見えた1行の変更が、実は「乱数のstateを自分で管理する」という、より堅牢な設計への入り口です。

## 参考文献

- [NEP 19 — Random number generator policy](https://numpy.org/neps/nep-0019-rng-policy.html)（NumPy Enhancement Proposal, 2018）
- [Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)（SC'11, 2011）
- [JAX PRNG Design (JEP 263)](https://docs.jax.dev/en/latest/jep/263-prng.html)
- [Best Practices for Using NumPy's Random Number Generators](https://blog.scientific-python.org/numpy/numpy-rng/)（Scientific Python Blog）
- [PyTorch Issue #39716: Do not modify global random state](https://github.com/pytorch/pytorch/issues/39716)
- [Steele, Lea, Flood "Fast Splittable Pseudorandom Number Generators"](https://dl.acm.org/doi/10.1145/2660193.2660195)（OOPSLA 2014）
