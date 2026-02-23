---
title: "Genesisは弾性体シミュレーションに使えるのか"
emoji: "🔬"
type: "tech"
topics: ["python", "3d", "robotics", "simulation", "数値計算"]
published: true
---

弾性体のリアルタイムシミュレーションができるツールを探していて、Genesisというプロジェクトを見つけました。MPMソルバーを搭載していて、弾性体を数行のコードで扱える。GitHubのスター推移を見たらMuJoCoの2倍以上、公開から1年ちょっとで28,000超。何がそんなに刺さったのか気になって調べ始めたら、速度の宣伝に問題があり、弾性体の速度は未知数で、でも統合ソルバーは本物の強みでした。**弾性体＋剛体を1つのライブラリで扱いたいなら、今のところGenesisに行き着くんじゃないかと思います。**

ソルバーの仕様、速度主張の独立検証、MuJoCoとの選定比較の3点から評価しています。

## この記事の対象読者

- 弾性体・流体など剛体以外の物理シミュレーションツールを探している人
- ロボティクス・強化学習のために物理シミュレータを選定中の人
- 「Genesisって実際どうなの？」と気になっているエンジニア・研究者

## 統合物理ソルバー — Genesisの独自性

Genesisを調べてまず目を引いたのは、1つのフレームワークに複数のソルバーを統合している点です。MuJoCoやIsaac Gymは剛体シミュレーション特化。

| ソルバー | 対象 |
|---|---|
| 剛体 | ロボットアーム等 |
| MPM | 変形体・粒状体 |
| SPH | 流体 |
| FEM | 弾性体 |
| PBD | 布・薄殻 |
| Stable Fluid | 気体・煙 |

1つのPythonスクリプトで剛体＋流体＋布を混ぜたシミュレーションが書ける。「ロボットが液体を扱う」「布を掴んで畳む」のようなマルチフィジクスなタスクをRL環境として組みたいなら、Genesisの統合ソルバーは検討に値します。

ただし成熟度にはばらつきがある。

| 特徴 | 独自性 | 成熟度 |
|---|---|---|
| 統合物理ソルバー | 高い | 中（剛体の接触が未成熟） |
| 微分可能シミュレーション | 中（他にもある） | 低（MPM/Toolソルバーのみ） |
| 生成AI連携（4Dワールド生成） | 非常に高い | **ほぼ未公開** |

最も派手に宣伝されている生成AI連携はデモ動画止まり。期待して選ぶには早すぎます。

## 弾性体シミュレーションの実力

私がGenesisに惹かれた理由はMPMソルバーです。弾性体を扱いたいなら、ここが評価の本丸。

### MPMで何ができるか

弾性体は剛体と違い、全粒子の変形を毎ステップ計算します。メッシュが細かいほど計算量が爆発する。従来のCPUベースFEMでは数千要素でもリアルタイムは困難でした。

GenesisのMPMは粒子＋グリッドのハイブリッドで、GPU並列化と相性が良い。APIも簡潔です。

```python
import genesis as gs

scene = gs.Scene()
scene.add_entity(
    morph=gs.morphs.Mesh(file='shape.obj', scale=0.1),
    material=gs.materials.MPM.Elastic(),
)
```

OBJ、STL、PLY等の任意メッシュを弾性体として読み込めます。プリミティブだけでなく実際の3Dモデルが使える。微分可能なので、勾配ベースの最適化にも対応。

### 粒子密度と速度のトレードオフ

粒子の密度は `grid_density`（1mあたりのグリッドセル数、デフォルト64）で制御します。

| grid_density | セルサイズ | 0.2m立方体の粒子数 |
|---|---|---|
| 64（デフォルト） | 約1.6cm | 約8,000 |
| 128 | 約0.8cm | 約64,000 |
| 256 | 約0.4cm | 約512,000 |

2倍にすると粒子数は約8倍。数千〜数万粒子ならGPU上でリアルタイム（30FPS以上）は現実的です。デフォルトの64は「RLの報酬計算に足る程度に変形すればいい」という割り切りの値で、工学的な精度が必要なら128以上に上げることになり、計算コストとのトレードオフになります。

### ベンチマーク未公開という問題

ここが最大の懸念です。**弾性体ソルバーの速度ベンチマークは公開されていません。** あの「430,000倍リアルタイム」はすべて剛体ソルバーの数字。MPMやFEMで実際にどれだけ速いかは、自分で計測するしかない。

GenesisのバックエンドはTaichiベースで、Taichi自体がGPU並列MPMに強い実績があります。ただしGenesisが特別速いかは別の話。Genesisの貢献は速さよりも、**弾性体＋剛体を統合的に扱えるAPIの利便性**にあると見るのが妥当でしょう。

GPU MPMで弾性体を扱えるツールは他にもあります（DiffTaichi, warp-mpm等）。ただし「弾性体＋剛体を統合的に扱えるPythonライブラリ」という条件では、現時点でGenesisが最も現実的な選択肢です。

## 「430,000倍速い」の実態

弾性体の速度が不明な一方で、剛体の速度主張も調べてみたら問題がありました。

Genesis公表の速度は、実用シナリオとかけ離れていました。

| 指標 | Genesis公表値 | 独立検証 |
|---|---|---|
| Franka腕（物理のみ） | 43M FPS | **0.29M FPS**（約150倍の下方修正） |
| キューブ掴みタスク | — | ManiSkill/SAPIENより **3〜10倍遅い** |
| レンダリング込み | — | **約10倍リアルタイム** まで低下 |

ManiSkillやIsaac Labはレンダリング込みで約1,000倍リアルタイムを達成しています。430,000倍どころか、既存ツールより遅い場面すらある。

独立検証（Stone Tao氏のブログ、MuJoCo GitHub Discussion）を読むと、ベンチマーク条件に問題がありました。

1. **比較対象のすり替え** — 「10-80倍速い」はIsaac（NVIDIA）との比較であり、MuJoCo MJXとは直接比較していません
2. **substeps=1** — 物理計算の精度を最小限にした状態で測定
3. **自己衝突を無効化** — 実用シミュレーションでは通常オンにします
4. **90%以上がアイドル状態** — 1ステップ動作後に999ステップ静止させています。ソルバーが早期終了するので、**物理エンジンがほぼ何もしていない時間を測っていた**ことになります
5. **レンダリングなし** — カメラを入れると速度が数桁落ちます

「最も楽な条件で測った最大瞬間風速」を代表値として掲げていたわけです。

:::message
Genesisチームは独立検証を受けて修正版ベンチマークを公開しました。特定条件（自己衝突あり・ロボットアームのみ）での43M+ FPS達成自体は確認されています。ただしこれは実用的なRLや操作タスクの速度とは別物です。
:::

この速度の宣伝がGitHubスター28,000超のバズを生んだ主因です。MuJoCoの2倍以上、10年以上の歴史があるPyBulletすら超えるスター数ですが、「使われているから」ではなく「話題になったから」の数字。20研究機関の連名による権威、「汎用物理エンジン＋生成AI」という壮大なビジョン、AI・ロボティクスバブルのタイミングが重なった結果です。

![Star History Chart](https://api.star-history.com/svg?repos=Genesis-Embodied-AI/Genesis,google-deepmind/mujoco,bulletphysics/bullet3,isaac-sim/IsaacLab&type=Date)

## 選定の現実

### 採用状況

| 指標 | Genesis | MuJoCo |
|---|---|---|
| 査読付き論文での使用 | ほぼなし | 3,500+ |
| 産業界での採用 | 未知数 | Google DeepMind, Tesla等 |
| コミュニティの成熟度 | 初期段階 | チュートリアル・教材が豊富 |
| 公開からの年数 | 約1年 | 10年以上 |

MuJoCoはロボティクスRLのデファクト標準です。論文3,500本超の実績があり、コードの再現性、バグ報告の蓄積、学習リソースの量で新参には太刀打ちできません。MuJoCo Playground等、エコシステムも広がっています。

### 導入の手軽さ

| シミュレータ | インストール | GPU要件 |
|---|---|---|
| MuJoCo | `pip install mujoco` | **不要**（CPUで動く） |
| Genesis | `pip install genesis-world` | NVIDIA GPU推奨 |
| Isaac Lab | pip可能だが依存が重い | **RTX必須** (VRAM 16GB+) |

Isaac Lab/Simと比べるとGenesisは手軽です。MuJoCoの「CPUだけで動く」気軽さには及びませんが、GPUがあればすぐ始められます。

:::details Isaac Lab/Simの要求スペック（参考）
| 項目 | 最小 | 推奨 |
|---|---|---|
| GPU | RTX 4080 (VRAM 16GB) | RTX PRO 6000 (VRAM 48GB) |
| RAM | 32GB | 64GB |
| ストレージ | 50GB SSD | 1TB NVMe SSD |

RTコア必須（A100/H100は非対応）、Python 3.11限定。以前はOmniverse Launcherからの巨大バイナリダウンロードが必要でしたが、現在は`pip install isaacsim`で入ります。
:::

### どちらを選ぶか

**MuJoCoを選ぶべき場合:**

- RL研究で論文を書く（再現性・先行研究との比較で圧倒的に有利）
- CPUのみの環境で動かしたい
- 剛体＋接触の精度と安定性が重要

**Genesisを検討してもよい場合:**

- 流体・布・弾性体を含むマルチフィジクスなシミュレーションが必要
- MuJoCoでカバーできないソルバー（MPM, SPH等）が欲しい
- 「まず触ってみる」段階で、本番採用の判断はまだ先

**今は待った方がいい場合:**

- 生成AI連携（4Dワールド生成）に期待している → まだ使えません
- 「速いから」が選定理由 → 宣伝ほどは速くないです

2026年2月時点での私の判断は、**弾性体を含むマルチフィジクスが必要ならGenesisを試す価値はある。それ以外ならMuJoCo一択**です。ただしGenesisの弾性体ソルバーはベンチマークすら公開されていない段階なので、自分で計測して判断する覚悟が要ります。

将来のポテンシャルを先食いしてバズったプロジェクトという印象は、調べれば調べるほど強くなりました。統合物理エンジンというコンセプト自体には価値があります。1〜2年後に剛体の接触処理が成熟し、弾性体のベンチマークが公開され、生成AI連携が実際に使えるようになれば評価は変わるかもしれません。そのときはまた検証して共有したいと思います。

## 参考文献

**Genesis**
- [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [Genesis Documentation（日本語版）](https://genesis-world.readthedocs.io/ja/latest/)
- [Genesis Speed Benchmark（公式）](https://github.com/zhouxian/genesis-speed-benchmark)

**速度の独立検証**
- [How fast is the new hyped Genesis simulator? - Stone Tao](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)
- [Genesis vs MuJoCo MJX Discussion](https://github.com/google-deepmind/mujoco/discussions/2303)

**シミュレータ比較**
- [ManiSkill Performance Benchmarking](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html)
- [A Review of Nine Physics Engines for RL Research](https://arxiv.org/html/2407.08590v1)
- [Simulately - Overall Comparison](https://simulately.wiki/docs/comparison/)

**MuJoCo**
- [MuJoCo GitHub](https://github.com/google-deepmind/mujoco)
- [MuJoCo 日本語ドキュメント](https://mujoco-docs-ja.readthedocs.io)（筆者がメンテしています）
