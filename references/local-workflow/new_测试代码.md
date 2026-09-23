### 1、校准

（从臂）

```
python -m lerobot.scripts.lerobot_calibrate --robot.type=so101_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm
```

（主臂）

```
python -m lerobot.scripts.lerobot_calibrate --teleop.type=so101_leader --teleop.port=COM22 --teleop.id=my_awesome_leader_arm
```



### 2、遥操作

 无相机

```
python -m lerobot.scripts.lerobot_teleoperate --robot.type=so_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm --teleop.type=so_leader --teleop.port=COM22 --teleop.id=my_awesome_leader_arm
```

 带相机

```
python -m lerobot.scripts.lerobot_teleoperate --robot.type=so101_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm --robot.cameras="{ fixed: {type: opencv,index_or_path: 0, width: 640, height: 480, fps: 30}, handeye: {type: opencv,index_or_path: 1, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM22 --teleop.id=my_awesome_leader_arm --display_data=true
```



### 3、数据采集

```
python -m lerobot.scripts.lerobot_record --robot.type=so101_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm --robot.cameras="{ handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM22 --teleop.id=my_awesome_leader_arm --display_data=true --dataset.root="D:/TanZiyan/lerobot-main/lerobot-main/tzy_data/data_02" --dataset.repo_id=admin/demo --dataset.push_to_hub=false --dataset.num_episodes=20 --dataset.single_task="Grab the screwdriver"
```


### 4、训练

```
python -m lerobot.scripts.lerobot_train --dataset.repo_id=admin/demo --dataset.root="D:/TanZiyan/lerobot-main/tzy_data/data_01" --policy.type=act --output_dir=outputs/train/act_so101_test --job_name=act_so101_test --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false
```



### 5、推理

```
python -m lerobot.scripts.lerobot_rollout --robot.type=so101_follower --robot.port=COM24 --robot.id=my_awesome_follower_arm --robot.cameras="{ handeye: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --display_data=true --policy.path="outputs/train/act_so101_test/checkpoints/last/pretrained_model" --policy.device=cuda --strategy.type=base
```
