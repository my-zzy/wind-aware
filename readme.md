# Wind-aware Structured End-to-End Navigation Framework

## Logs

### 2025.12.31

+ neural-fly-airsim folder deleted. to find it, clone https://github.com/my-zzy/neural-fly-airsim
+ 暂时抛弃windows-wsl通信方案，转为纯windows+python方案



## TODO:

1. Windows(Airsim) - WSL(YOPO) 数据传输
2. Airsim中的仿真数据发布 ros topic, 适配YOPO
3. YOPO生成的轨迹通过话题给controller, 适配meta-pinn

![structure](new_structure.png)

```
YOPO/
├── yopo_airsim.py          ← Main entry (run this)
├── requirements.txt        ← Updated (no ROS deps)
├── train_yopo.py           ← Keep for training
├── yopo_trt_transfer.py    ← Keep for TensorRT (optional)
├── config/
│   ├── __init__.py
│   ├── config.py
│   └── traj_opt.yaml
├── policy/
│   ├── __init__.py
│   ├── poly_solver.py
│   ├── primitive.py
│   ├── state_transform.py
│   ├── yopo_network.py
│   ├── yopo_dataset.py     ← Only for training
│   ├── yopo_trainer.py     ← Only for training
│   └── models/
│       ├── backbone.py
│       ├── head.py
│       └── resnet.py
├── loss/                   ← Only for training
└── saved/
    └── YOPO_1/
        └── epoch50.pth
```

## Windows(Airsim) - WSL(YOPO) 数据传输搭建步骤如下：

1. wsl中使用python3的ros环境（不要使用conda 下的python3）：  
终端 1（WSL，系统 ROS）  
`source /opt/ros/noetic/setup.bash`  
`roscore`  

2. 终端 2（WSL，系统 ROS，先“清理环境”）  
`unset PYTHONPATH`  
`unset PYTHONHOME`  
`export PATH=/usr/bin:/bin:/usr/sbin:/sbin`  
`source /opt/ros/noetic/setup.bash`  
`roslaunch rosbridge_server rosbridge_websocket.launch`  

3. 终端 3（WSL，YOPO）  
`source /opt/ros/noetic/setup.bash`  
`source ~/anaconda3/etc/profile.d/conda.sh`  
`conda activate yopo`  
`python test_yopo_ros.py`  

4. 在windows启动Airsim+UE，然后运行python airsim_yopo_bridge.py  （windows下的python环境），输出：  
`[INFO] Connected to rosbridge`  
`Connected!`    
`Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)`  
`[INFO] Connected to AirSim`  
`[INFO] Listening to /yopo/cmd_vel`  
说明YOPO+AirSim已经调通  

