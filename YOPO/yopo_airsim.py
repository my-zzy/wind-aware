import airsim
import cv2
import time
import torch
import numpy as np
import threading
import matplotlib.pyplot as plt
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
        self.last_yaw_error = 0.0
        self.last_yaw_rate = 0.0
        self.ctrl_dt = 0.01  # 100Hz control loop for better yaw tracking
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
        
        # ---------- Command Buffer (for thread-safe AirSim calls) ----------
        self.cmd_lock = threading.Lock()
        self.pending_cmd = None  # (vx, vy, vz, yaw_dot)
        self.print_counter = 0  # For periodic printing
        
        # ---------- Data Logging ----------
        self.log_time = []
        self.log_x = []
        self.log_y = []
        self.log_z = []
        self.log_yaw = []
        self.log_pitch = []
        self.log_roll = []
        self.log_x_des = []
        self.log_y_des = []
        self.log_z_des = []
        self.log_yaw_des = []
        self.log_pitch_des = []
        self.log_roll_des = []
        self.start_time = None
        
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
        print("Connecting to airsim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[INFO] Connected to AirSim")
        
        # Takeoff
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-10, 3).join()
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
                
                # 4. Execute pending command (thread-safe AirSim call)
                self._execute_pending_command()
                
                # 5. Display depth image at ~10Hz (every 5 iterations at 50Hz)
                # Display depth image
                depth_display = (depth[0, 0] * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colored)
                cv2.waitKey(1)
                
                # Print status at ~2Hz (every 50 iterations at 100Hz)
                self.print_counter += 1
                if self.print_counter >= 50:
                    self.print_counter = 0
                    yaw_deg = np.degrees(self.last_yaw)
                    goal_dir = self.goal - self.position
                    goal_yaw_deg = np.degrees(np.arctan2(goal_dir[1], goal_dir[0]))
                    
                    # Check depth statistics
                    depth_min = np.min(depth[0, 0])
                    depth_max = np.max(depth[0, 0])
                    depth_mean = np.mean(depth[0, 0])
                    print(f"[POS] x:{self.position[0]:6.2f} y:{self.position[1]:6.2f} z:{self.position[2]:6.2f} | yaw:{yaw_deg:6.1f}° | goal_yaw:{goal_yaw_deg:6.1f}°")
                    # print(f"[DEPTH] min:{depth_min:.3f} max:{depth_max:.3f} mean:{depth_mean:.3f} | range: 0.0=near, 1.0=far(20m)")
                    # cv2.waitKey(1)
                
                time.sleep(0.05)  # ~50Hz
                
            except KeyboardInterrupt:
                self.running = False
                self.plot_data()
                break
    
    def _execute_pending_command(self):
        """Execute pending velocity command in main thread (AirSim is not thread-safe)"""
        with self.cmd_lock:
            if self.pending_cmd is not None:
                vx, vy, vz, yaw_dot = self.pending_cmd
                self.client.moveByVelocityAsync(
                    vx, vy, vz,
                    duration=self.ctrl_dt * 1.5,  # 1.5x for smooth transition without delay
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=np.degrees(yaw_dot))
                )
                self.pending_cmd = None
    
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
        
        # Normalize (same as training data: depth / max_depth_dist -> clip [0,1])
        depth_array = np.minimum(depth_array, self.max_dis) / self.max_dis
        depth_array = np.clip(depth_array, 0.0, 1.0)
        
        # Mask bottom part to ignore drone body (adjust crop_bottom if needed)
        crop_bottom = int(self.height * 0.05)  # Mask bottom 15% of image
        if crop_bottom > 0:
            depth_array[-crop_bottom:, :] = 1.0  # Set to far distance
        
        # Handle NaN (use uint16 like training data for better precision)
        nan_mask = np.isnan(depth_array) | (depth_array < self.min_dis / self.max_dis)
        interpolated = cv2.inpaint(np.uint16(depth_array * 65535), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        depth_array = interpolated.astype(np.float32) / 65535.0
        
        return depth_array.reshape(1, 1, self.height, self.width)
    
    def _update_odometry(self, state):
        """Update odometry from AirSim state"""
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        ori = state.kinematics_estimated.orientation
        
        self.position = np.array([pos.x_val, pos.y_val, pos.z_val])
        self.velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        self.orientation = np.array([ori.x_val, ori.y_val, ori.z_val, ori.w_val])
        
        # Always update last_yaw from actual drone state to prevent drift
        ypr = R.from_quat(self.orientation).as_euler('ZYX', degrees=False)
        self.last_yaw = ypr[0]
        
        if not self.odom_init:
            self.desire_pos = self.position.copy()
            self.desire_vel = self.velocity.copy()
            self.desire_acc = np.zeros(3)
            self.odom_init = True
        
        # Check arrival
        if np.linalg.norm(self.position - self.goal) < 5 and not self.arrive:
            print("Arrive!")
            self.arrive = True
    
    # ==================== Inference ====================
    @torch.inference_mode()
    def _yopo_inference(self, depth):
        """YOPO network inference"""
        # Debug: check depth distribution
        if self.print_counter % 50 == 0:
            print(f"[DEPTH] min:{np.min(depth):.3f} max:{np.max(depth):.3f} mean:{np.mean(depth):.3f}")
        
        # Prepare inputs
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_norm = self._process_odom().to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)
        
        # Forward
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        score_pred = score_pred.cpu().numpy()
        
        # Debug: check network output
        if self.print_counter % 50 == 0:
            print(f"[SCORE] min:{np.min(score_pred):.3f} max:{np.max(score_pred):.3f} best_action:{np.argmin(score_pred)}")
        
        # Post-process
        endstate, score, action_id = self._process_output(endstate_pred, score_pred)
        
        # Transform to world frame
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        Rotation_wb = R.from_quat(self.orientation).as_matrix()
        Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        endstate_w = np.matmul(Rotation_wc, endstate_c)
        
        # Use index 0 because _process_output already selected the best action
        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else self.position
            start_vel = self.desire_vel if self.plan_from_reference else self.velocity
            
            self.optimal_poly_x = Poly5Solver(
                start_pos[0], start_vel[0], self.desire_acc[0],
                endstate_w[0, 0, 0] + start_pos[0],
                endstate_w[0, 0, 1], endstate_w[0, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(
                start_pos[1], start_vel[1], self.desire_acc[1],
                endstate_w[0, 1, 0] + start_pos[1],
                endstate_w[0, 1, 1], endstate_w[0, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], self.desire_acc[2],
                endstate_w[0, 2, 0] + start_pos[2],
                endstate_w[0, 2, 1], endstate_w[0, 2, 2], self.traj_time)
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
        return endstate, score_pred[action_id], action_id
    
    # ==================== Control Loop ====================
    def _control_loop(self):
        """Control loop running in separate thread - computes commands only"""
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
                
                # Calculate yaw (PD controller outputs angular acceleration)
                goal_dir = self.goal - self.desire_pos
                yaw, yaw_rate, yaw_error = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, 
                                                          self.ctrl_dt, self.last_yaw_error, self.last_yaw_rate)
                self.last_yaw_error = yaw_error
                self.last_yaw_rate = yaw_rate
                # Note: last_yaw is now updated from actual drone state in _update_odometry
                
                # Queue command for main thread (NED: negate z)
                with self.cmd_lock:
                    self.pending_cmd = (vx, vy, -vz, yaw_rate)
                
                # Log data
                if self.start_time is None:
                    self.start_time = time.time()
                self.log_time.append(time.time() - self.start_time)
                self.log_x.append(self.position[0])
                self.log_y.append(self.position[1])
                self.log_z.append(self.position[2])
                
                # Get actual pitch and roll
                ypr = R.from_quat(self.orientation).as_euler('ZYX', degrees=True)
                self.log_yaw.append(ypr[0])
                self.log_pitch.append(ypr[1])
                self.log_roll.append(ypr[2])
                
                self.log_x_des.append(self.desire_pos[0])
                self.log_y_des.append(self.desire_pos[1])
                self.log_z_des.append(self.desire_pos[2])
                # Desired yaw based on actual trajectory direction (from velocity)
                yaw_des = np.degrees(np.arctan2(self.desire_vel[1], self.desire_vel[0]))
                self.log_yaw_des.append(yaw_des)
                # Desired pitch/roll are 0 for level flight
                self.log_pitch_des.append(0.0)
                self.log_roll_des.append(0.0)
            
            time.sleep(self.ctrl_dt)
    
    def warm_up(self):
        """Warm up the network"""
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        self.policy(depth, obs)
    
    def plot_data(self):
        """Plot logged data in 6 subplots"""
        if len(self.log_time) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle('Flight Data', fontsize=14)
        
        # X position
        axes[0, 0].plot(self.log_time, self.log_x, 'b-', label='Actual')
        axes[0, 0].plot(self.log_time, self.log_x_des, 'r--', label='Desired')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('X (m)')
        axes[0, 0].set_title('X Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Y position
        axes[0, 1].plot(self.log_time, self.log_y, 'b-', label='Actual')
        axes[0, 1].plot(self.log_time, self.log_y_des, 'r--', label='Desired')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Y (m)')
        axes[0, 1].set_title('Y Position')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Z position
        axes[1, 0].plot(self.log_time, self.log_z, 'b-', label='Actual')
        axes[1, 0].plot(self.log_time, self.log_z_des, 'r--', label='Desired')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Z (m)')
        axes[1, 0].set_title('Z Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Yaw angle
        axes[1, 1].plot(self.log_time, self.log_yaw, 'b-', label='Actual')
        axes[1, 1].plot(self.log_time, self.log_yaw_des, 'r--', label='Desired')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Yaw (deg)')
        axes[1, 1].set_title('Yaw Angle')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Pitch angle
        axes[2, 0].plot(self.log_time, self.log_pitch, 'b-', label='Actual')
        axes[2, 0].plot(self.log_time, self.log_pitch_des, 'r--', label='Desired')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Pitch (deg)')
        axes[2, 0].set_title('Pitch Angle')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Roll angle
        axes[2, 1].plot(self.log_time, self.log_roll, 'b-', label='Actual')
        axes[2, 1].plot(self.log_time, self.log_roll_des, 'r--', label='Desired')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Roll (deg)')
        axes[2, 1].set_title('Roll Angle')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('flight_data.png', dpi=150)
        print("Plot saved to flight_data.png")
        # plt.show()


if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = base_dir + "/saved/YOPO_1/epoch50.pth"
    
    settings = {
        'airsim_ip': 'localhost',  # Your AirSim IP
        'goal': [250, 250, -10],          # Goal position (NED)
        'pitch_angle_deg': -10,       # Camera pitch angle
        'plan_from_reference': False,
    }
    
    yopo = YopoAirSim(settings, weight)
    yopo.run()