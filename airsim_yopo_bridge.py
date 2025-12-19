import airsim
import roslibpy
import time

# ========== 1. 连接 rosbridge ==========
ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()
print('[INFO] Connected to rosbridge')

# ========== 2. 连接 AirSim ==========
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print('[INFO] Connected to AirSim')

# 可选：起飞
client.takeoffAsync().join()

# ========== 3. 回调函数 ==========
def cmd_callback(msg):
    vx = msg['linear']['x']
    vy = msg['linear']['y']
    vz = msg['linear']['z']
    yaw_rate = msg['angular']['z']

    client.moveByVelocityAsync(
        vx, vy, vz,
        duration=0.05,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(
            is_rate=True,
            yaw_or_rate=yaw_rate
        )
    )

# ========== 4. 订阅 YOPO 输出 ==========
listener = roslibpy.Topic(
    ros,
    '/yopo/cmd_vel',
    'geometry_msgs/Twist'
)

listener.subscribe(cmd_callback)
print('[INFO] Listening to /yopo/cmd_vel')

# ========== 5. 保持运行 ==========
try:
    while ros.is_connected:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

listener.unsubscribe()
ros.terminate()
