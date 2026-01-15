import numpy as np


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ 5-th order polynomial at each Axis """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t


class Polys5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ multiple 5-th order polynomials at each Axis (only used for visualization of multiple trajectories) """
        N = len(pos1)
        State_Mat = np.array([[pos0] * N, [vel0] * N, [acc0] * N, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_position(self, t):
        """Return the position array at time t."""
        t = np.atleast_1d(t)
        result = (self.A[0][:, np.newaxis] + self.A[1][:, np.newaxis] * t + self.A[2][:, np.newaxis] * t ** 2 +
                  self.A[3][:, np.newaxis] * t ** 3 + self.A[4][:, np.newaxis] * t ** 4 + self.A[5][:, np.newaxis] * t ** 5)
        return result.flatten()

def wrap_to_pi(angle):
    """将角度限制在 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, last_yaw_error=0.0, last_yaw_rate=0.0, max_yaw_rate=0.7, kp=4.0, kd=6.0):
    # Normalize velocity and goal directions
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)

    # Goal yaw and weighting
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
    weight = 6 * abs(delta_yaw) / np.pi  # weight ∈ [0,6]; equal weight at 30°, goal weight increases as delta_yaw grows

    # Desired direction and yaw
    # weight = min(weight, 0.1)
    # print(f"weight: {weight:.3f}")
    dir_des = vel_dir + weight * goal_dir
    yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw

    # PD controller outputs angular acceleration
    yaw_error = wrap_to_pi(yaw_desired - last_yaw)
    yaw_error_dot = (yaw_error - last_yaw_error) / dt
    yaw_ddot = kp * yaw_error + kd * yaw_error_dot  # angular acceleration
    
    # Integrate to get angular velocity
    yaw_rate = last_yaw_rate + yaw_ddot * dt
    
    # Limit yaw rate
    max_yaw_change = max_yaw_rate * np.pi
    # print(f"yaw rate, max yaw rate: {yaw_rate:.3f}, {max_yaw_change:.3f}")
    # if abs(yaw_rate) > abs(max_yaw_change):
    #     print("[Warning] yaw rate exceeds max yaw rate.")
    yaw_rate = np.clip(yaw_rate, -max_yaw_change, max_yaw_change)

    # Integrate to get yaw angle
    yaw = wrap_to_pi(last_yaw + yaw_rate * dt)

    return yaw, yaw_rate, yaw_error

