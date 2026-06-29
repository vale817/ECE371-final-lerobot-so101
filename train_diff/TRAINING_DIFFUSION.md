# Diffusion Policy 训练准备 — task1_grab

目标：在同一个 `Valaris1028/task1_grab` 数据集上训练 **Diffusion Policy**，后续和组长的 ACT baseline 对比实物抓取效果。

数据集概况（来自 Hugging Face 数据集 `meta/info.json`）：

| 项目 | 值 |
|---|---|
| 格式 | LeRobotDataset v3.0 |
| episodes | 60 |
| frames | 35,989 |
| fps | 30 |
| robot_type | so_follower |
| 相机 | `observation.images.handeye` (480x640), `observation.images.fixed` (360x640) |
| action / state | 6-DOF (`shoulder_pan/lift, elbow_flex, wrist_flex/roll, gripper`) |
| 视频编码 | AV1 (pyav backend) |

---

## 流程总览

1. GPU 服务器装好 LeRobot Diffusion 依赖。
2. 先跑 100 steps smoke test，确认数据拉取、视频解码、显存和模型 forward/backward 正常。
3. smoke test 通过后正式训练。
4. 取 checkpoint 回机械臂上评估，并和 ACT 结果比较成功率和失败原因。

---

## 1. GPU 服务器环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip "lerobot[diffusion]"
hf auth login   # 数据集私有、或后续要上传模型时才需要
nvidia-smi      # 确认 GPU 可见
```

当前数据集是公开的，默认训练会通过 `--dataset.repo_id=Valaris1028/task1_grab` 从 Hugging Face Hub 拉取数据，不需要 Google Drive 原始数据。

如果服务器不能访问 Hugging Face，可以先把本地数据复制到服务器：

```bash
rsync -avhP ~/.cache/huggingface/lerobot/Valaris1028/task1_grab/ \
    user@gpu-server:~/.cache/huggingface/lerobot/Valaris1028/task1_grab/
```

然后在训练命令里手动加：

```bash
--dataset.root="${HOME}/.cache/huggingface/lerobot/Valaris1028/task1_grab"
```

---

## 2. 训练

先跑 smoke test：

```bash
JOB=diffusion_task1_grab_smoke STEPS=100 BATCH_SIZE=2 SAVE_FREQ=100 bash train_diff/train_diffusion.sh
```

通过后正式训练：

```bash
bash train_diff/train_diffusion.sh
```

核心命令（脚本里已写好）：

```bash
lerobot-train \
  --dataset.repo_id=Valaris1028/task1_grab \
  --policy.type=diffusion \
  --output_dir=outputs/train_diff/diffusion_task1_grab \
  --job_name=diffusion_task1_grab \
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

- Diffusion Policy 使用和 ACT baseline 相同的数据集，方便后续对比。
- 数据集的两路相机分辨率不同：`handeye` 是 480x640，`fixed` 是 360x640。脚本默认把两路都 resize 到 360x640，避免多相机输入形状不一致。
- 默认不启用 wandb，避免服务器上因为未登录 wandb 卡住。想看训练曲线可以设置 `WANDB_ENABLE=true` 并先运行 `wandb login`。
- 显存吃紧就调小 `BATCH_SIZE`，例如：
  ```bash
  BATCH_SIZE=4 bash train_diff/train_diffusion.sh
  ```
- 中断后续训默认 run：
  ```bash
  lerobot-train \
    --config_path=outputs/train_diff/diffusion_task1_grab/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
  ```

---

## 3. Checkpoint 和评估

正式训练的 checkpoint 默认在：

```text
outputs/train_diff/diffusion_task1_grab/checkpoints/last/
```

回到机械臂上评估时，用 `lerobot-record` 加载最后的 policy：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --policy.path=outputs/train_diff/diffusion_task1_grab/checkpoints/last/pretrained_model \
  --dataset.repo_id=Valaris1028/eval_task1_grab_diffusion \
  --dataset.single_task="grab the cube"
```

建议记录：

- 成功次数 / 总次数，例如 10 次里成功几次。
- 失败类型：没对准、碰倒方块、夹爪没闭合、运动太慢或太抖等。
- 训练 steps、训练时长、batch size、是否启用 wandb。
- 和 ACT baseline 的同条件对比结果。

> 评估前确认夹爪标定和过载问题已处理，尤其是 `gripper range_min`。否则模型动作本身可能没问题，但实物执行会被硬件保护打断。
