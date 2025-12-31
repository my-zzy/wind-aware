### Understanding the Test Workflow

Each command in the testing process starts specific components that work together for autonomous navigation:

#### 1. Controller and Dynamics Simulator
```bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

**What it does:**
- **Quadrotor Dynamics Simulator**: Simulates the physics of the drone (motors, IMU, dynamics)
  - Location: `Controller/src/so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp`
  - Subscribes to: `/so3_cmd` (attitude commands)
  - Publishes: `/odom` (position, velocity, orientation)
  
- **SO(3) Attitude Controller**: Geometric controller for tracking
  - Location: `Controller/src/so3_control/src/SO3Control.cpp`
  - Converts position commands → attitude commands using SO(3) control
  - Subscribes to: `/position_cmd`
  - Publishes: `/so3_cmd`

**Result**: A simulated quadrotor ready to receive commands

#### 2. Environment and Sensors Simulator
```bash
rosrun sensor_simulator sensor_simulator_cuda
```

**What it does:**
- **CUDA-Accelerated Environment Generation**: Creates synthetic 3D environments
  - Random forests, 3D Perlin noise, or maze structures
  - Configurable via `Simulator/src/config/config.yaml`
  
- **Sensor Simulation**: Renders realistic sensor data using GPU ray-tracing
  - Location: `Simulator/src/src/sensor_simulator.cu`
  - Depth camera (RGB-D images)
  - LiDAR point clouds
  
- **ROS Communication**:
  - Subscribes to: `/odom` (from quadrotor simulator)
  - Publishes: `/depth_image`, `/pointcloud`

**Result**: Real-time sensor data based on drone's current position

#### 3. YOPO Planner
```bash
python test_yopo_ros.py --trial=1 --epoch=50
```

**What it does:**
- **Neural Network Inference**: 
  - Loads trained model: `YOPO/saved/YOPO_1/epoch50.pth`
  - Network architecture: ResNet backbone + trajectory prediction heads
  
- **Planning Pipeline**:
  1. Receives depth images from simulator
  2. Predicts trajectory primitives with scores
  3. Selects best trajectory avoiding obstacles
  4. Generates smooth position commands
  
- **Key Components** (`test_yopo_ros.py`):
  - `YopoNetwork`: Neural network for trajectory prediction
  - `PolySolver`: Polynomial trajectory generation
  - State transformation and primitive generation
  
- **ROS Communication**:
  - Subscribes to: `/depth_image`, `/odom`
  - Publishes: `/position_cmd` → SO(3) controller

**Result**: Autonomous trajectory commands for obstacle avoidance

#### 4. Visualization
```bash
rviz -d yopo.rviz
```

**What it does:**
- Opens RViz with pre-configured visualization settings
- **Displays**:
  - 3D environment point cloud
  - Camera images (RGB + Depth)
  - Predicted trajectory path
  - Drone pose and orientation
  - Goal marker
  
- **Interactive Control**:
  - Click "2D Nav Goal" tool to set target position
  - YOPO planner receives goal and generates collision-free trajectory

**Result**: Real-time visualization of planning and navigation

#### Complete System Data Flow

```
┌─────────────────────────────────────────────────────┐
│  Quadrotor Simulator (C++)                          │
│  • Simulates drone physics and dynamics             │
│  → Publishes: /odom (position, velocity)            │
│  → Subscribes: /so3_cmd (attitude commands)         │
└──────────────┬──────────────────────────────────────┘
               │ /odom
               ↓
┌─────────────────────────────────────────────────────┐
│  Environment Simulator (CUDA)                       │
│  • Generates 3D obstacles and environment           │
│  • Renders depth images via GPU ray-tracing         │
│  → Reads: /odom                                     │
│  → Publishes: /depth_image, /pointcloud             │
└──────────────┬──────────────────────────────────────┘
               │ /depth_image, /odom
               ↓
┌─────────────────────────────────────────────────────┐
│  YOPO Planner (Python/PyTorch)                      │
│  • Neural network inference                         │
│  • Predicts trajectories + scores                   │
│  • Selects best collision-free trajectory           │
│  → Reads: /depth_image, /odom                       │
│  → Publishes: /position_cmd                         │
└──────────────┬──────────────────────────────────────┘
               │ /position_cmd
               ↓
┌─────────────────────────────────────────────────────┐
│  SO(3) Controller (C++)                             │
│  • Converts position → attitude commands            │
│  → Reads: /position_cmd                             │
│  → Publishes: /so3_cmd ─────────┐                  │
└─────────────────────────────────┼───────────────────┘
                                   │ (loops back to quadrotor)
                                   └────────┐
```

All components run as independent ROS nodes, communicating through topics for modular, real-time autonomous navigation.