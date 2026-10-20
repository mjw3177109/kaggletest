# V2.9.5｜RealMLP CUDA 单折探索性实验

- UTC：2026-10-19T13:42:04.246658+00:00
- 状态：pilot_evaluated
- Kaggle Notebook：https://www.kaggle.com/code/mjw3177109/v2-9-5-realmlp-strict-pilot
- 独立模型：RealMLP-TD，4 模型内集成、3 固定轮数、T4 GPU。
- **这是原三折协议的第 0 折，不是完整三折 CV，也不是线上分数。**
- 同折 V2.9.3 参照 AUC：0.9603317085463999
- 同折 RealMLP AUC：0.9599579808498494
- 固定 80/20 融合该折 AUC：0.9605299179748474
- 相对同折增量：0.00019820942844750444
- 实际训练显卡：Tesla T4
- GPU 实测时长：166.2 秒
- Kaggle 新提交：无；该试验不产生 submission.csv。
- 第一折单次收益不足以证明模型稳定提分；后续必须另跑剩余两折。
- 不允许把单折 AUC 写入完整实验的 champion CV。

