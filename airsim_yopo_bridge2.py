import airsim
import roslibpy
import time
import numpy as np
import base64

# ========== 1. 连接 rosbridge ==========
ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()
print('[INFO] Connected to rosbridge')

# ========== 2. 连接 AirSim (双通道) ==========
# 通道 A: 用于传感器读取 (运行在主线程)
client_sensor = airsim.MultirotorClient(ip="172.25.48.1", port=41451)
client_sensor.confirmConnection()
client_sensor.enableApiControl(True)
client_sensor.armDisarm(True)

# 通道 B: 用于控制指令 (运行在 roslibpy 回调线程)
# 注意：我们需要第二个独立的连接对象来避免线程冲突
client_cmd = airsim.MultirotorClient(ip="172.25.48.1", port=41451)
client_cmd.confirmConnection()
# 注意：不需要在第二个客户端上再次 enableApiControl，只要有一个开启即可，或者都开启也没关系

print('[INFO] Connected to AirSim (Dual Channels)')

# 起飞 (使用控制通道)
client_cmd.takeoffAsync().join()
client_cmd.moveToZAsync(-20, 1).join()  # AirSim中负数是向上
time.sleep(2)

# ========== 3. ROS 发布器 ==========
depth_pub = roslibpy.Topic(ros, '/depth_image', 'sensor_msgs/Image')
odom_pub = roslibpy.Topic(ros, '/sim/odom', 'nav_msgs/Odometry')


# ========== 4. 回调函数 (使用 client_cmd) ==========
def cmd_callback(msg):
    """接收 YOPO 的速度指令"""
    # 这里的 msg 是字典类型
    try:
        vx = msg['linear']['x']
        vy = msg['linear']['y']
        vz = msg['linear']['z']
        yaw_rate = msg['angular']['z']

        # 【关键修改】使用 client_cmd 而不是 client_sensor
        client_cmd.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.05,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
    except Exception as e:
        print(f"[ERROR] In cmd_callback: {e}")


# ========== 5. 订阅 YOPO 输出 ==========
listener = roslibpy.Topic(ros, '/yopo/cmd_vel', 'geometry_msgs/Twist')
listener.subscribe(cmd_callback)
print('[INFO] Listening to /yopo/cmd_vel')


# ========== 6. 主循环：发布传感器数据 (使用 client_sensor) ==========
def publish_sensor_data():
    """发布深度图和里程计"""
    # 【关键修改】相机名称改为 "0" (对应 settings.json)
    responses = client_sensor.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
    ])

    if responses:
        depth_data = responses[0]
        # 检查数据是否为空
        if len(depth_data.image_data_float) == 0:
            return

        depth_array = np.array(depth_data.image_data_float, dtype=np.float32)

        # Base64 编码
        encoded_data = base64.b64encode(depth_array.tobytes()).decode('utf-8')

        # 发布深度图
        depth_msg = {
            'header': {
                'stamp': {'secs': int(time.time()), 'nsecs': int((time.time() % 1) * 1e9)},
                'frame_id': 'camera'
            },
            'height': depth_data.height,
            'width': depth_data.width,
            'encoding': '32FC1',
            'is_bigendian': 0,
            'step': depth_data.width * 4,
            'data': encoded_data
        }
        depth_pub.publish(roslibpy.Message(depth_msg))

    # 获取里程计
    state = client_sensor.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation

    odom_msg = {
        'header': {
            'stamp': {'secs': int(time.time()), 'nsecs': int((time.time() % 1) * 1e9)},
            'frame_id': 'world'
        },
        'pose': {
            'pose': {
                'position': {'x': pos.x_val, 'y': pos.y_val, 'z': pos.z_val},
                'orientation': {'x': ori.x_val, 'y': ori.y_val, 'z': ori.z_val, 'w': ori.w_val}
            }
        },
        'twist': {
            'twist': {
                'linear': {'x': vel.x_val, 'y': vel.y_val, 'z': vel.z_val}
            }
        }
    }
    odom_pub.publish(roslibpy.Message(odom_msg))


# ========== 7. 保持运行 ==========
try:
    rate = 30  # 30 Hz
    while ros.is_connected:
        publish_sensor_data()
        time.sleep(1.0 / rate)
except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    pass

listener.unsubscribe()
client_cmd.landAsync().join()  # 使用 cmd 客户端降落
client_cmd.armDisarm(False)
client_cmd.enableApiControl(False)
ros.terminate()