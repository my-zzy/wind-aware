import airsim
import roslibpy
import time
import numpy as np
import base64
import threading

ROSBRIDGE_HOST = 'localhost'
ROSBRIDGE_PORT = 9090

AIRSIM_IP   = "172.16.26.6"
AIRSIM_PORT = 41451

CTRL_DT = 0.05     # 控制周期（s）
SENSOR_HZ = 40     # 传感器发布频率

# =========================
# 1. 连接 rosbridge
# =========================
ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
ros.run()
print('[INFO] Connected to rosbridge')

# =========================
# 2. 连接 AirSim（双 Client）
# =========================
client_sensor = airsim.MultirotorClient(ip=AIRSIM_IP, port=AIRSIM_PORT)
client_cmd    = airsim.MultirotorClient(ip=AIRSIM_IP, port=AIRSIM_PORT)

client_sensor.confirmConnection()
client_cmd.confirmConnection()

# 控制权：两个 client 都打开
client_sensor.enableApiControl(True)
client_sensor.armDisarm(True)

client_cmd.enableApiControl(True)
client_cmd.armDisarm(True)

print('[INFO] Connected to AirSim (dual clients)')

# 起飞并上升
client_cmd.takeoffAsync().join()
client_cmd.moveToZAsync(-3, 1).join()
time.sleep(2.0)

# =========================
# 3. ROS Publisher
# =========================
depth_pub = roslibpy.Topic(ros, '/depth_image', 'sensor_msgs/Image')
odom_pub  = roslibpy.Topic(ros, '/sim/odom', 'nav_msgs/Odometry')

# =========================
# 4. YOPO 控制回调（ENU → NED）
# =========================
def cmd_callback(msg):
    try:
        vx = msg['linear']['x']
        vy = msg['linear']['y']
        vz = msg['linear']['z']
        yaw_rate = msg['angular']['z']

        client_cmd.moveByVelocityAsync(
            vx,
            vy,
            -vz,   # ENU -> NED
            duration=CTRL_DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
    except Exception as e:
        print(f"[ERROR] cmd_callback: {e}")

listener = roslibpy.Topic(ros, '/yopo/cmd_vel', 'geometry_msgs/Twist')
listener.subscribe(cmd_callback)
print('[INFO] Listening to /yopo/cmd_vel')

# =========================
# 5. 传感器发布函数
# =========================
def publish_sensor_data():
    # 统一时间戳
    t = time.time()
    secs = int(t)
    nsecs = int((t - secs) * 1e9)

    # ---------- Depth ----------
    responses = client_sensor.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
    ])

    if responses and len(responses[0].image_data_float) > 0:
        depth = responses[0]
        depth_array = np.array(depth.image_data_float, dtype=np.float32)

        encoded = base64.b64encode(depth_array.tobytes()).decode('utf-8')

        depth_msg = {
            'header': {
                'stamp': {'secs': secs, 'nsecs': nsecs},
                'frame_id': 'camera'
            },
            'height': depth.height,
            'width': depth.width,
            'encoding': '32FC1',
            'is_bigendian': 0,
            'step': depth.width * 4,
            'data': encoded
        }
        depth_pub.publish(roslibpy.Message(depth_msg))

    # ---------- Odometry ----------
    state = client_sensor.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation

    odom_msg = {
        'header': {
            'stamp': {'secs': secs, 'nsecs': nsecs},
            'frame_id': 'world'
        },
        'pose': {
            'pose': {
                'position': {
                    'x': pos.x_val,
                    'y': pos.y_val,
                    'z': pos.z_val
                },
                'orientation': {
                    'x': ori.x_val,
                    'y': ori.y_val,
                    'z': ori.z_val,
                    'w': ori.w_val
                }
            }
        },
        'twist': {
            'twist': {
                'linear': {
                    'x': vel.x_val,
                    'y': vel.y_val,
                    'z': vel.z_val
                }
            }
        }
    }
    odom_pub.publish(roslibpy.Message(odom_msg))

try:
    dt = 1.0 / SENSOR_HZ
    while ros.is_connected:
        publish_sensor_data()
        time.sleep(dt)

except KeyboardInterrupt:
    print('[INFO] Shutting down...')

listener.unsubscribe()
client_cmd.landAsync().join()
client_cmd.armDisarm(False)
client_cmd.enableApiControl(False)
ros.terminate()
