# V2.9.7 条件目标编码独立 CPU 实验

- UTC：2026-10-19T16:20:57.050539+00:00
- 阶段：kernel_queued
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/kas-v297-conditional-cpu-s6e10
- 新模型：LightGBM，服务评分 × 旅客分组的条件 TE，配对评分 × 旅客类型，数值 × 类别 TE。
- 频率特征：严格训练折内 5-fold cross-fit；不从验证集或测试集估计目标。
- 资源：仅 Kaggle CPU（不占 T4），不使用外部标签或伪标签。
- 独立模型真实三折 OOF：尚未完成
- 历史三折最佳：0.9607535043719558
- 固定 85% 旧基线 + 15% 独立模型 OOF：尚未完成
- CV 改进：未计算
- 各折配对增量：未计算
- Kaggle 新 Public LB：无（此阶段绝不提交）

