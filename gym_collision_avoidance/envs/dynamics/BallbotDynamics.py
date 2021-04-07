import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math


def integrator(S, U, dt):
    M = 4
    dt_ = float(dt) / M
    S_next = np.array(S)
    for i in range(M):
        k1 = dt_ * state_dot(S, U)
        k2 = dt_ * state_dot(S + (0.5 * k1), U)
        k3 = dt_ * state_dot(S + (0.5 * k2), U)
        k4 = dt_ * state_dot(S + k3, U)
        S_next += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return S_next


def state_dot(S0, U):
    S_dot = np.array(S0)
    S_dot[0] = S0[1]
    S_dot[1] = ((-38.73 * S0[0]) + (-11.84 * S0[1]) + (-6.28 * S0[2]) +
                 (51.61 * U[0]) + (11.84 * U[1]) + (6.28 * U[2]))

    S_dot[2] = ((13.92 * S0[0]) + (2.0 * S0[1]) + (1.06 * S0[2]) +
                 (-8.72 * U[0]) + (-2.0 * U[1]) + (-1.06 * U[2]))

    S_dot[3] = S0[4]
    S_dot[4] = ((-38.54 * S0[3]) + (-11.82 * S0[4]) + (-6.24 * S0[5]) +
                 (51.36 * U[3]) + (11.82 * U[4]) + (6.24 * U[5]))

    S_dot[5] = ((14.00 * S0[3]) + (2.03 * S0[4]) + (1.07 * S0[5]) +
                 (-8.81 * U[3]) + (-2.03 * U[4]) + (-1.07 * U[5]))
    return S_dot


def velocity_adjuster(v_body_x_ref, v_body_y_ref, v_body_x, v_body_y, thresh):
    adjusted_vx = v_body_x_ref
    adjusted_vy = v_body_y_ref
    if np.linalg.norm((v_body_x_ref - v_body_x, v_body_y_ref - v_body_y)) > thresh:
        theta_diff = np.arctan2((v_body_y_ref - v_body_y), (v_body_x_ref - v_body_x))
        adjusted_vx = v_body_x + thresh * np.cos(theta_diff)
        adjusted_vy = v_body_y + thresh * np.sin(theta_diff)
    return adjusted_vx, adjusted_vy


class BallbotDynamics(Dynamics):
    """ Convert a speed & heading to a new state according to Unicycle Kinematics model.

    """

    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        """ 

        In the global frame, assume the agent instantaneously turns by :code:`heading`
        and moves forward at :code:`speed` for :code:`dt` seconds.  
        Add that offset to the current position. Update the velocity in the
        same way. Also update the agent's turning direction (only used by CADRL).

        Args:
            action (list): [delta heading angle, speed] command for this agent
            dt (float): time in seconds to execute :code:`action`
    
        """
        selected_speed = action[0]
        selected_heading_global = wrap(action[1] + self.agent.heading_global_frame)

        # States: [theta_x, theta_dot_x, vx, theta_y, theta_dot_y, vy, x, y]
        theta_body_x = self.agent.theta_ego_frame[0]
        theta_body_y = self.agent.theta_ego_frame[1]
        theta_body_dot_x = self.agent.theta_dot_ego_frame[0]
        theta_body_dot_y = self.agent.theta_dot_ego_frame[1]
        v_body_x = self.agent.vel_global_frame[0]
        v_body_y = self.agent.vel_global_frame[1]

        # Current State
        S_curr = np.asarray([theta_body_x, theta_body_dot_x, v_body_x,
                             theta_body_y, theta_body_dot_y, v_body_y])

        # Inputs: [theta_x_ref, theta_dot_x_ref, v_body_x_ref, theta_y_ref, theta_dot_y_ref, v_y_ref]
        theta_body_x_ref = 0
        theta_body_dot_x_ref = 0
        v_body_x_ref = selected_speed * np.cos(selected_heading_global)
        theta_body_ref_y = 0
        theta_body_dot_ref_y = 0
        v_body_y_ref = selected_speed * np.sin(selected_heading_global)

        adjuster_thresh = 0.5
        v_body_x_ref, v_body_y_ref = velocity_adjuster(v_body_x_ref, v_body_y_ref, v_body_x, v_body_y, adjuster_thresh)

        # print("BALLBOT vref_x: ", v_body_x_ref)
        # print("BALLBOT vref_y: ", v_body_y_ref)

        # Reference input
        U = np.asarray([theta_body_x_ref, theta_body_dot_x_ref, v_body_x_ref,
                        theta_body_ref_y, theta_body_dot_ref_y, v_body_y_ref])

        # Simulate dynamics
        S_next = integrator(S_curr, U, dt)

        # theta_x, theta_dot_x, vx
        self.agent.theta_ego_frame[0] = S_next[0]
        self.agent.theta_dot_ego_frame[0] = S_next[1]
        self.agent.vel_global_frame[0] = S_next[2]

        # theta_y, theta_dot_y, vy
        self.agent.theta_ego_frame[1] = S_next[3]
        self.agent.theta_dot_ego_frame[1] = S_next[4]
        self.agent.vel_global_frame[1] = S_next[5]

        # dx, dy (global frame)
        dx = self.agent.vel_global_frame[0] * dt
        dy = self.agent.vel_global_frame[1] * dt

        # x, y (global frame)
        self.agent.pos_global_frame += np.array([dx, dy])
        #print("BALLBOT px: ", self.agent.pos_global_frame[0])
        #print("BALLBOT py: ", self.agent.pos_global_frame[1])
        #print("DT: ", dt)

        # v (global frame)
        self.agent.speed_global_frame = np.linalg.norm(self.agent.vel_global_frame)

        # yaw (global frame)
        new_heading = np.arctan2(self.agent.vel_global_frame[1], self.agent.vel_global_frame[0])
        self.agent.delta_heading_global_frame = wrap(new_heading - self.agent.heading_global_frame)
        self.agent.heading_global_frame = new_heading

        # turning dir: needed for cadrl value fn
        if abs(self.agent.turning_dir) < 1e-5:
            self.agent.turning_dir = 0.11 * np.sign(new_heading)
        elif self.agent.turning_dir * new_heading < 0:
            self.agent.turning_dir = max(-np.pi, min(np.pi, -self.agent.turning_dir + new_heading))
        else:
            self.agent.turning_dir = np.sign(self.agent.turning_dir) * max(0.0, abs(self.agent.turning_dir) - 0.1)
