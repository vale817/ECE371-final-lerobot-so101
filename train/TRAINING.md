# ACT 训练准备 — task1_grab

数据集概况（来自 `task1_grab/meta/info.json`）：

| 项目 | 值 |
|---|---|
| 格式 | LeRobotDataset v3.0 |
| episodes | 60 |
| frames | 35,989 |
| fps | 30 |
| robot_type | so_follower |
| 相机 | `observation.images.handeye`, `observation.images.fixed` (480×640) |
| action / state | 6-DOF (`shoulder_pan/lift, elbow_flex, wrist_flex/roll, gripper`) |
| 视频编码 | AV1 (pyav backend) |

策略：**ACT**（Action Chunking Transformer）。设备：**CUDA GPU 服务器**。

---

## 流程总览

1. 本地（Mac）把数据集上传到 HuggingFace —— `train/upload_to_hf.sh`
2. GPU 服务器装好 lerobot 并拉取数据集
3. 服务器跑训练 —— `train/train_act.sh`
4. 监控 → 取 checkpoint → 回机械臂上评估

---

## 1. 上传数据集到 HuggingFace（本地 Mac，跑一次）

```bash
pip install -U "huggingface_hub[cli]"
hf auth login            # 粘贴 WRITE token: https://huggingface.co/settings/tokens
bash train/upload_to_hf.sh
```

成功后数据集在 `https://huggingface.co/datasets/Valaris1028/task1_grab`，
顺便也完成了之前"上传 task1_grab"的待办。

> 不想过 HF？用 `train_act.sh` 里的 **Option B**：直接 `rsync` 数据集文件夹到服务器的
> `~/.cache/huggingface/lerobot/Valaris1028/task1_grab/`，训练时加 `--dataset.root` 指过去即可。

## 2. GPU 服务器环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip "lerobot[act]" # 或从 lerobot checkout: pip install -e ".[act]"
hf auth login                     # 数据集私有时才需要；用来拉数据集 / 推策略
nvidia-smi                        # 确认 GPU 可见
```

## 3. 训练

```bash
# 先跑 100 steps，确认下载、视频解码、显存和模型 forward 正常
JOB=act_task1_grab_smoke STEPS=100 BATCH_SIZE=2 SAVE_FREQ=100 bash train/train_act.sh

# smoke test 通过后正式训练；两个任务使用不同输出目录
bash train/train_act.sh
```

核心命令（脚本里已写好）：

```bash
lerobot-train \
  --dataset.repo_id=Valaris1028/task1_grab \
  --policy.type=act \
  --output_dir=outputs/train/act_task1_grab \
  --job_name=act_task1_grab \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --wandb.enable=false
```

要点：

- ACT 会自动识别两个相机。原始分辨率分别为 480×640 和 360×640，脚本会把两路都
  resize 到 360×640；ACT 多相机编码要求两路特征 token 数一致，不能直接省略这个步骤。
- 默认 `chunk_size`/`n_action_steps=100`，SO-101 抓取任务够用；想缩短预测窗口加
  `--policy.chunk_size=50 --policy.n_action_steps=50`。
- 想看训练曲线：`--wandb.enable=true`（先 `wandb login`）。
- 中断后续训：
  `lerobot-train --config_path=outputs/train/act_task1_grab/checkpoints/last/pretrained_model/train_config.json --resume=true`。
- 显存吃紧就调小 `--batch_size`（如 4），双相机 + AV1 解码占用会高一些。

## 4. 评估（回到机械臂）

训练完 checkpoint 在 `outputs/train/act_task1_grab/checkpoints/last/`。用 `lerobot-record`
加载策略跑实物评估：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --policy.path=outputs/train/act_task1_grab/checkpoints/last/pretrained_model \
  --dataset.repo_id=Valaris1028/eval_task1_grab \
  --dataset.single_task="grab the cube"
```

> 评估前确认夹爪过载问题已处理（`gripper range_min` 标定），否则抓取动作可能触发过载保护。

---

## 训练时长粗估

60 episodes / ~36k frames 属于小数据集。单张现代 GPU（如 RTX 4090 / A100）跑
ACT 100k steps 大约几小时；先跑到 ~50k steps 看 loss 是否收敛即可中途评估。
