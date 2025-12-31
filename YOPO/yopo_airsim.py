import airsim
import cv2
import time
import torch
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.poly_solver import Poly5Solver, calculate_yaw
from policy.state_transform import StateTransform
from policy.primitive import LatticePrimitive


class YopoAirSim:
    def __init__(self, config, weight):
        # ---------- Config ----------
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(config['goal'])
        self.plan_from_reference = config['plan_from_reference']
        self.Rotation_bc = R.from_euler('ZYX', [0, config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ---------- State Variables ----------
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = threading.Lock()
        self.arrive = False
        self.running = True
        
        # ---------- Helpers ----------
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time
        
        # ---------- Load Network ----------
        state_dict = torch.load(weight, weights_only=True)
        self.policy = YopoNetwork()
        self.policy.load_state_dict(state_dict)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        self.warm_up()
        
        # ---------- Connect AirSim ----------
        self.client = airsim.MultirotorClient(ip=config['airsim_ip'])
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[INFO] Connected to AirSim")
        
        # Takeoff
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-3, 1).join()
        time.sleep(1.0)
        
        print("YOPO AirSim Ready!")
    
    # ==================== Main Loop ====================
    def run(self):
        """Main loop: get sensor data -> inference -> control"""
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()
        
        while self.running:
            try:
                # 1. Get Odometry
                state = self.client.getMultirotorState()
                self._update_odometry(state)
                
                if not self.odom_init:
                    time.sleep(0.01)
                    continue
                
                # 2. Get Depth Image
                depth = self._get_depth_image()
                if depth is None:
                    continue
                
                # 3. YOPO Inference
                self._yopo_inference(depth)
                
                time.sleep(0.02)  # ~50Hz
                
            except KeyboardInterrupt:
                self.running = False
                break
    
    # ==================== Sensor Functions ====================
    def _get_depth_image(self):
        """Get depth image from AirSim"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        
        if not responses or len(responses[0].image_data_float) == 0:
            return None
        
        depth = responses[0]
        depth_array = np.array(depth.image_data_float, dtype=np.float32)
        depth_array = depth_array.reshape(depth.height, depth.width)
        
        # Resize if needed
        if depth_array.shape[0] != self.height or depth_array.shape[1] != self.width:
            depth_array = cv2.resize(depth_array, (self.width, self.height), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        depth_array = np.minimum(depth_array, self.max_dis) / self.max_dis
        
        # Handle NaN
        nan_mask = np.isnan(depth_array) | (depth_array < self.min_dis / self.max_dis)
        interpolated = cv2.inpaint(np.uint8(depth_array * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        depth_array = interpolated.astype(np.float32) / 255.0
        
        return depth_array.reshape(1, 1, self.height, self.width)
    
    def _update_odometry(self, state):
        """Update odometry from AirSim state"""
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        ori = state.kinematics_estimated.orientation
        
        self.position = np.array([pos.x_val, pos.y_val, pos.z_val])
        self.velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        self.orientation = np.array([ori.x_val, ori.y_val, ori.z_val, ori.w_val])
        
        if not self.odom_init:
            self.desire_pos = self.position.copy()
            self.desire_vel = self.velocity.copy()
            self.desire_acc = np.zeros(3)
            ypr = R.from_quat(self.orientation).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
            self.odom_init = True
        
        # Check arrival
        if np.linalg.norm(self.position - self.goal) < 5 and not self.arrive:
            print("Arrive!")
            self.arrive = True
    
    # ==================== Inference ====================
    @torch.inference_mode()
    def _yopo_inference(self, depth):
        """YOPO network inference"""
        # Prepare inputs
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_norm = self._process_odom().to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)
        
        # Forward
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        score_pred = score_pred.cpu().numpy()
        
        # Post-process
        endstate, score = self._process_output(endstate_pred, score_pred)
        
        # Transform to world frame
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        Rotation_wb = R.from_quat(self.orientation).as_matrix()
        Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        endstate_w = np.matmul(Rotation_wc, endstate_c)
        
        action_id = 0
        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else self.position
            start_vel = self.desire_vel if self.plan_from_reference else self.velocity
            
            self.optimal_poly_x = Poly5Solver(
                start_pos[0], start_vel[0], self.desire_acc[0],
                endstate_w[action_id, 0, 0] + start_pos[0],
                endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(
                start_pos[1], start_vel[1], self.desire_acc[1],
                endstate_w[action_id, 1, 0] + start_pos[1],
                endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], self.desire_acc[2],
                endstate_w[action_id, 2, 0] + start_pos[2],
                endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
    
    def _process_odom(self):
        """Process odometry for network input"""
        Rotation_wb = R.from_quat(self.orientation).as_matrix()
        Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = Rotation_wc.T
        
        vel_w = self.desire_vel if self.plan_from_reference else self.velocity
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_c = np.dot(Rotation_cw, self.desire_acc)
        
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)
        
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm
    
    def _process_output(self, endstate_pred, score_pred):
        """Process network output"""
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)
        
        action_id = np.argmin(score_pred)
        lattice_id = self.lattice_primitive.traj_num - 1 - action_id
        endstate = self.state_transform.pred_to_endstate_cpu(
            endstate_pred[action_id, :][np.newaxis, :], lattice_id)
        return endstate, score_pred[action_id]
    
    # ==================== Control Loop ====================
    def _control_loop(self):
        """Control loop running in separate thread"""
        while self.running:
            if self.ctrl_time is None or self.ctrl_time > self.traj_time:
                time.sleep(self.ctrl_dt)
                continue
            
            if self.arrive:
                self.odom_init = False  # Ready for next goal
                time.sleep(self.ctrl_dt)
                continue
            
            with self.lock:
                self.ctrl_time += self.ctrl_dt
                
                # Get trajectory point
                vx = self.optimal_poly_x.get_velocity(self.ctrl_time)
                vy = self.optimal_poly_y.get_velocity(self.ctrl_time)
                vz = self.optimal_poly_z.get_velocity(self.ctrl_time)
                
                # Update desired state
                self.desire_pos = np.array([
                    self.optimal_poly_x.get_position(self.ctrl_time),
                    self.optimal_poly_y.get_position(self.ctrl_time),
                    self.optimal_poly_z.get_position(self.ctrl_time)
                ])
                self.desire_vel = np.array([vx, vy, vz])
                self.desire_acc = np.array([
                    self.optimal_poly_x.get_acceleration(self.ctrl_time),
                    self.optimal_poly_y.get_acceleration(self.ctrl_time),
                    self.optimal_poly_z.get_acceleration(self.ctrl_time)
                ])
                
                # Calculate yaw
                goal_dir = self.goal - self.desire_pos
                yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
                self.last_yaw = yaw
                
                # Send command to AirSim (NED: negate z)
                self.client.moveByVelocityAsync(
                    vx, vy, -vz,
                    duration=self.ctrl_dt,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=np.degrees(yaw_dot))
                )
            
            time.sleep(self.ctrl_dt)
    
    def warm_up(self):
        """Warm up the network"""
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        self.policy(depth, obs)


if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = base_dir + "/saved/YOPO_1/epoch50.pth"
    
    settings = {
        'airsim_ip': '172.16.26.6',  # Your AirSim IP
        'goal': [50, 0, -3],          # Goal position (NED)
        'pitch_angle_deg': -10,       # Camera pitch angle
        'plan_from_reference': False,
    }
    
    yopo = YopoAirSim(settings, weight)
    yopo.run()