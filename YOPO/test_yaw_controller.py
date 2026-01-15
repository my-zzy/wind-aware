import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from policy.poly_solver import calculate_yaw

class YawControllerTest:
    def __init__(self):
        # Connect to AirSim
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("Connected!")
        
        # Control parameters
        self.ctrl_dt = 0.01  #100Hz
        self.last_yaw = 0.0
        self.last_yaw_error = 0.0
        self.last_yaw_rate = 0.0
        
        # Data logging
        self.log_time = []
        self.log_yaw_actual = []
        self.log_yaw_desired = []
        
        # Takeoff
        print("Taking off...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-3, 1).join()
        time.sleep(2.0)
        print("Ready!")
    
    def get_current_yaw(self):
        """Get current yaw angle from AirSim"""
        state = self.client.getMultirotorState()
        ori = state.kinematics_estimated.orientation
        orientation = np.array([ori.x_val, ori.y_val, ori.z_val, ori.w_val])
        ypr = R.from_quat(orientation).as_euler('ZYX', degrees=False)
        return ypr[0]
    
    def test_step_response(self, target_yaw_deg, duration=10.0):
        """Test step response to a target yaw angle"""
        print(f"\nTesting step response to {target_yaw_deg}°...")
        target_yaw = np.radians(target_yaw_deg)
        
        # Reset state
        self.last_yaw = self.get_current_yaw()
        self.last_yaw_error = 0.0
        self.last_yaw_rate = 0.0
        
        self.log_time = []
        self.log_yaw_actual = []
        self.log_yaw_desired = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get current state
            current_yaw = self.get_current_yaw()
            
            # Calculate control using the existing controller
            # Simulate velocity direction aligned with current yaw
            vel_dir = np.array([np.cos(current_yaw), np.sin(current_yaw), 0])
            goal_dir = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
            
            yaw, yaw_rate, yaw_error = calculate_yaw(
                vel_dir, goal_dir, self.last_yaw,
                self.ctrl_dt, self.last_yaw_error, self.last_yaw_rate
            )
            
            self.last_yaw = current_yaw  # Use actual yaw for feedback
            self.last_yaw_error = yaw_error
            self.last_yaw_rate = yaw_rate
            
            # Send command (hover in place, only control yaw)
            self.client.moveByVelocityAsync(
                0, 0, 0,
                duration=self.ctrl_dt * 1.5,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=np.degrees(yaw_rate))
            )
            
            # Log data
            self.log_time.append(time.time() - start_time)
            self.log_yaw_actual.append(np.degrees(current_yaw))
            self.log_yaw_desired.append(target_yaw_deg)
            
            time.sleep(self.ctrl_dt)
        
        print("Test complete!")
    
    def test_sinusoidal_tracking(self, amplitude_deg=30, period=5.0, duration=15.0):
        """Test sinusoidal yaw tracking"""
        print(f"\nTesting sinusoidal tracking (amplitude={amplitude_deg}°, period={period}s)...")
        
        # Reset state
        self.last_yaw = self.get_current_yaw()
        self.last_yaw_error = 0.0
        self.last_yaw_rate = 0.0
        
        self.log_time = []
        self.log_yaw_actual = []
        self.log_yaw_desired = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            
            # Sinusoidal desired yaw
            target_yaw_deg = amplitude_deg * np.sin(2 * np.pi * t / period)
            target_yaw = np.radians(target_yaw_deg)
            
            # Get current state
            current_yaw = self.get_current_yaw()
            
            # Calculate control
            vel_dir = np.array([np.cos(current_yaw), np.sin(current_yaw), 0])
            goal_dir = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
            
            yaw, yaw_rate, yaw_error = calculate_yaw(
                vel_dir, goal_dir, self.last_yaw,
                self.ctrl_dt, self.last_yaw_error, self.last_yaw_rate
            )
            
            self.last_yaw = current_yaw
            self.last_yaw_error = yaw_error
            self.last_yaw_rate = yaw_rate
            
            # Send command
            self.client.moveByVelocityAsync(
                0, 0, 0,
                duration=self.ctrl_dt * 1.5,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=np.degrees(yaw_rate))
            )
            
            # Log data
            self.log_time.append(t)
            self.log_yaw_actual.append(np.degrees(current_yaw))
            self.log_yaw_desired.append(target_yaw_deg)
            
            time.sleep(self.ctrl_dt)
        
        print("Test complete!")
    
    def plot_results(self, title="Yaw Tracking Performance", task=""):
        """Plot the results"""
        if len(self.log_time) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title, fontsize=14)
        
        # Yaw tracking
        axes[0].plot(self.log_time, self.log_yaw_actual, 'b-', label='Actual', linewidth=2)
        axes[0].plot(self.log_time, self.log_yaw_desired, 'r--', label='Desired', linewidth=2)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Yaw (deg)')
        axes[0].set_title('Yaw Angle Tracking')
        axes[0].legend()
        axes[0].grid(True)
        
        # Tracking error
        error = np.array(self.log_yaw_actual) - np.array(self.log_yaw_desired)
        axes[1].plot(self.log_time, error, 'g-', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Error (deg)')
        axes[1].set_title('Tracking Error')
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[1].grid(True)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        axes[1].text(0.02, 0.98, f'RMSE: {rmse:.2f}°\nMax Error: {max_error:.2f}°',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'yaw_controller_test_{task}.png', dpi=150)
        print("\nResults saved to yaw_controller_test.png")
        print(f"RMSE: {rmse:.2f}°")
        print(f"Max Error: {max_error:.2f}°")
        plt.show()
    
    def cleanup(self):
        """Land and disconnect"""
        print("\nLanding...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Done!")


if __name__ == "__main__":
    tester = YawControllerTest()
    
    try:
        # Test 1: Step response to 45 degrees
        tester.test_step_response(target_yaw_deg=45, duration=8.0)
        tester.plot_results(title="Step Response to 45°", task="step")
        
        time.sleep(1)
        
        # Test 2: Sinusoidal tracking
        tester.test_sinusoidal_tracking(amplitude_deg=30, period=5.0, duration=15.0)
        tester.plot_results(title="Sinusoidal Yaw Tracking (±30°, 5s period)", task="sinusoidal")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        tester.cleanup()
