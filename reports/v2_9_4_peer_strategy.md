# Kaggle Auto Scientist V2.9.4｜榜单前三与公开思路复盘

核验：2026-10-19 08:09:07 UTC（经仓库 GitHub Actions 上 Kaggle CLI 认证获取）
比赛：https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard
机器可核对数据：[`state/v2_9_4_top3.json`](../state/v2_9_4_top3.json)

## 1. 当时的官方 Public Top 3

| Public 排位 | 团队 | 官方 Public AUC | 原始 API 来源 |
|---:|---|---:|---|
| 1 | Masaya Kawamata | 0.96321 | Kaggle CLI leaderboard 返回的第 1 条 |
| 2 | Ahmet Bugrahan ALTUNOK | 0.96319 | Kaggle CLI leaderboard 返回的第 2 条 |
| 3 | Prior | 0.96241 | Kaggle CLI leaderboard 返回的第 3 条 |

**事实边界：**以上经认证可确认队伍、分数与官方返回顺序，但**无法根据排名推断代码、训练方式或是否使用外部数据**。不能把别人的公开方案冒充榜一/二/三的私有方案。比赛尚未结束，此表不代表 Private 榜和领奖名单。

## 2. 已公开、可复现的方法（并非声称来自榜一/二/三）

- [alvis1111 公共复盘](https://github.com/alvis1111/s6e10-airline-solution)：强 TargetEncoder（全部 21 字段，inner 5 折）、精确 Flight Distance 画像、外部 129k 原始数据训练 teacher、context-conditioned features、RealMLP 产生模型互补；作者自述后续自有 Public LB 0.96126。该仓库自称其他复现已到 0.96169，但该数字不是本站验收或前 3 身份依据。
- [公开路线 / 原始数据辅助 Notebook](https://www.kaggle.com/code/goodpjw2008/s6e10-route-id-fe-og-model-stack-lb-0-96108)：公开版标题与记录有 0.96108 Public LB。
- [sachith7 消融实验 Notebook](https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth)：公开说明中的最佳分数 0.96153；应检查输入与是否属于同等独立验证，不能直接混合外部测试预测文件。
- [Masaya Kawamata 过往其他比赛冠军解法](https://www.kaggle.com/masayakawamata)：模型多样性、多个 OOF 与正则化融合是他**历史比赛**公开的策略，不能认定就是本场 S6E10 第一名的具体方法。

## 3. 优先级与止损

1. P0 **V2.9.4 独立原始数据 teacher + fold-safe 航线类别频率**：独立复现、去竞赛 train/test 特征重复、只用原始数据标签训练 teacher，完全不使用竞赛验证折标签构造特征。
2. P1 **深层 XGB/LightGBM 与 RealMLP 模型多样化**：必须在 P0 的真实 OOF 和 GPU 资源审计后决定；不凭公共榜反复试融合权重。
3. P2 **重复实验止损**：V2.9.3 仅变 CatBoost 深度/低阶评分差值，没有超越既有最高 CV，避免再次尝试基本同构特征/参数。
4. 验证协议：外层 3-fold 继续沿用 20261018 固定划分；目标编码在每折训练内交叉拟合；外层验证不参与早停；同折配对统计、新特征消融和真实硬件来源均入档。
5. 提交门槛：只有 CV 超过最佳已验证值并满足独立验证、防重放锁、官方配额时才考虑新 Kaggle 提交。不能假设 CV 提分等于 Public/Private 提分。

## 4. V2.9.4 仓库代码

- `kaggle_agent/v294_original_teacher_train.py`：新算法和外部数据 teacher、GPU XGB，输出 OOF、测试预测、特征重要性、原始数据去重证据。
- `kaggle_agent/v294_pipeline.py`：使用**独立 Kaggle Notebook** 跑三折，核实 GPU 与 CV，不覆盖 V2.5 线上冠军。
- `kaggle_agent/v294_ingest.py`：真实 GPU 模型完成后写入实验账本与资源报告。
- `tests/test_v294_original_teacher.py`：外部重叠排除、fold-only 路线和频率生成、不触碰外层验证早停及禁止无保护提交。
- `.github/workflows/kaggle_v294_original_teacher.yml`：可恢复式独立 Notebook 流程，原有 `main` 不变。

## 5. 结果说明

本报告是 **研究计划和出处说明**，而非 V2.9.4 模型完成报告。CV / GPU 完成时间 / 正式 LB 必须读取 `state/v2_9_4_status.json` 中 Kaggle 验证后的数据；仅 Github Actions 成功、kernel queued 不代表训练完成。
