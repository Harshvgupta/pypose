import os
import time
import torch
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from torch.linalg import cross
# from pypose.module.pid import PID
from pypose.lietensor.basics import vec2skew
import torch
from torch import nn


class PID(nn.Module):

    def __init__(self, kp, ki, kd):
        super().__init__()
        self.integrity_initialized = False
        self.integity = None
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def forward(self, error, error_dot, ff=None):

        if not self.integrity_initialized:
            self.integity = torch.zeros_like(error)
            self.integrity_initialized = True

        self.integity += error

        if ff == None:
            ff = torch.zeros_like(error)

        return self.kp * error + self.ki * self.integity + self.kd * error_dot + ff

    def reset(self):
        r"""
        This method is used to reset the internal error integrity.
        """
        if self.integrity_initialized:
            self.integity = None
            self.last_error = None
            self.integrity_initialized = False


def angular_vel_2_quaternion_dot(quaternion, w):
    device = quaternion.device
    p, q, r = w

    zero_t = torch.zeros(1, device=device)

    omega_1 = torch.cat((zero_t, -r, q, -p))
    omega_2 = torch.cat((r, zero_t, -p, -q))
    omega_3 = torch.cat((-q, p, zero_t, -r))
    omega_4 = torch.cat((p, q, r, zero_t))

    omega_matrix = torch.stack((omega_1, omega_2, omega_3, omega_4))

    return -0.5 * omega_matrix @ quaternion.T


def skew2vec(input):
    # Convert batched skew matrices to vectors.
    return torch.vstack([-input[1, 2], input[0, 2], -input[0, 1]])


class MultiCopter(pp.module.NLS):
    def __init__(self, mass, g, J, dt):
        super(MultiCopter, self).__init__()
        self.device = J.device
        self.m = mass
        self.J = J
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.tau = dt
        self.e3 = torch.tensor([[0., 0., 1.]], device=self.device).reshape(3, 1)

    def state_transition(self, state, input, t=None):
        new_state = self.rk4(state, input, self.tau)
        self.pose_normalize(new_state)
        return new_state

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = state + k1 * t / 2
        self.pose_normalize(k1_state)

        k2 = self.xdot(k1_state, input)
        k2_state = state + k2 * t / 2
        self.pose_normalize(k2_state)

        k3 = self.xdot(k2_state, input)
        k3_state = state + k3 * t
        self.pose_normalize(k3_state)

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

    def pose_normalize(self, state):
        state[3:7] = state[3:7] / torch.norm(state[3:7])

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        pose, vel, angular_speed = state[3:7], state[7:10], state[10:13]
        thrust, M = input[0], input[1:4]

        # convert the 1d row vector to 2d column vector
        M = torch.unsqueeze(M, 0)
        pose = torch.unsqueeze(pose, 0)
        pose_SO3 = pp.LieTensor(pose, ltype=pp.SO3_type)
        Rwb = pose_SO3.matrix()[0]

        acceleration = (Rwb @ (-thrust * self.e3) + self.m * self.g * self.e3) / self.m

        angular_speed = torch.unsqueeze(angular_speed, 1)
        w_dot = self.J_inverse \
            @ (M.T - cross(angular_speed, self.J @ angular_speed, dim=0))

        # transfer angular_speed from body frame to world frame
        return torch.concat([
                vel,
                torch.squeeze(angular_vel_2_quaternion_dot(pose, angular_speed)),
                torch.squeeze(acceleration),
                torch.squeeze(w_dot)
            ]
        )


class GeometricController(torch.nn.Module):
    def __init__(self, parameters, mass, J, g):
        self.device = J.device
        self.parameters = parameters
        self.g = g
        self.m = mass
        self.J = J
        self.e3 = torch.tensor([0., 0., 1.], device=self.device).reshape(3, 1)

    def compute_pose_error(self, pose, ref_pose):
        err_pose =  ref_pose.T @ pose - pose.T @ ref_pose
        return 0.5 * torch.squeeze(skew2vec(err_pose), dim=0)

    def forward(self, state, ref_state):
        device = state.device
        des_pos = torch.unsqueeze(ref_state[0:3], 1)
        des_vel = torch.unsqueeze(ref_state[3:6], 1)
        des_acc = torch.unsqueeze(ref_state[6:9], 1)
        des_acc_dot = torch.unsqueeze(ref_state[9:12], 1)
        des_acc_ddot = torch.unsqueeze(ref_state[12:15], 1)
        des_b1 = torch.unsqueeze(ref_state[15:18], 1)
        des_b1_dot = torch.unsqueeze(ref_state[18:21], 1)
        des_b1_ddot = torch.unsqueeze(ref_state[21:24], 1)

        # extract specific state from state tensor
        position = torch.unsqueeze(state[0:3], 1)
        pose = state[3:7]
        vel = torch.unsqueeze(state[7:10], 1)
        angular_vel = torch.unsqueeze(state[10:13], 1)
        pose_Rwb = pp.LieTensor(pose, ltype=pp.SO3_type).matrix()

        # extract parameters
        kp, kv, kori, kw = self.parameters
        position_pid = PID(kp, 0, kv)
        pose_pid = PID(kori, 0, kw)

        # position controller
        des_b3 = - position_pid.forward(position - des_pos, vel - des_vel) \
            - self.m * self.g * self.e3 \
            + self.m * des_acc

        b3 = pose_Rwb @ self.e3
        thrust_des = torch.squeeze(-des_b3.T @ b3)

        # attitude controller
        err_vel_dot = self.g * self.e3 - thrust_des / self.m * b3 - des_acc
        des_b3_dot = - kp * (vel - des_vel) - kv * err_vel_dot + self.m * des_acc_dot

        # calculate des_b3, des_b3_dot, des_b3_ddot
        b3_dot = pose_Rwb @ vec2skew(torch.squeeze(angular_vel)) @ self.e3
        thrust_dot = torch.squeeze(- des_b3_dot.T @ b3 - des_b3.T @ b3_dot)
        err_vel_ddot = (-thrust_dot * b3 - thrust_des * b3_dot) / self.m - des_acc_dot
        des_b3_ddot = -kp * err_vel_dot - kv * err_vel_ddot + self.m * des_acc_ddot

        des_b3 = -des_b3 / torch.norm(des_b3)
        des_b3_dot = -des_b3_dot / torch.norm(des_b3_dot)
        des_b3_ddot = -des_b3_ddot / torch.norm(des_b3_ddot)

        # calculate des_b2, des_b2_dot, des_b3_ddot
        des_b2 = cross(des_b3, des_b1, dim=0)
        des_b2_dot = cross(des_b3_dot, des_b1, dim=0) + cross(des_b3, des_b1_dot, dim=0)
        des_b2_ddot = cross(des_b3_ddot, des_b1, dim=0) \
            + 2*cross(des_b3_dot, des_b1_dot, dim=0) \
            + cross(des_b3, des_b1_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b2_dot = des_b2 / torch.norm(des_b2_dot)
        des_b2_ddot = des_b2 / torch.norm(des_b2_ddot)

        # calculate des_b1, des_b1_dot, des_b1_ddot
        des_b1 = cross(des_b2, des_b3, dim=0)
        des_b1_dot = cross(des_b2_dot, des_b3, dim=0) + cross(des_b2, des_b3_dot, dim=0)
        des_b1_ddot = cross(des_b2_ddot, des_b3, dim=0) \
            + 2 * cross(des_b2_dot, des_b3_dot, dim=0) \
            + cross(des_b2, des_b3_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b1_dot = des_b2 / torch.norm(des_b1_dot)
        des_b1_ddot = des_b2 / torch.norm(des_b1_ddot)

        des_pose_Rwb = torch.concat([des_b1, des_b2, des_b3], dim=1)
        des_pose_Rwb_dot = torch.concat([des_b1_dot, des_b2_dot, des_b3_dot], dim=1)
        des_pose_Rwb_ddot = torch.concat([des_b1_ddot, des_b2_ddot, des_b3_ddot], dim=1)

        des_augular_vel = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_dot)
        wedge_des_augular_vel = vec2skew(des_augular_vel.T)[0]
        des_augular_acc = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_ddot
                                   - wedge_des_augular_vel @ wedge_des_augular_vel)

        M = - pose_pid.forward(self.compute_pose_error(pose_Rwb, des_pose_Rwb),
                               angular_vel - pose_Rwb.T @ (des_pose_Rwb @ des_augular_vel)) \
          + cross(angular_vel, self.J @ angular_vel, dim=0)
        temp_M = torch.squeeze(vec2skew(angular_vel.T)) \
          @ (pose_Rwb.T @ des_pose_Rwb @ des_augular_vel \
          - pose_Rwb.T @ des_pose_Rwb @ des_augular_acc)
        M = (M - self.J @ temp_M).reshape(-1)

        zero_force_tensor = torch.tensor([0.], device=device)
        return torch.concat([torch.max(zero_force_tensor, thrust_des), M])

def get_ref_states4(dt, N, coeff_x, coeff_y, coeff_z, device):
    ref_state = torch.zeros(N, 24, device=device)
    time = torch.arange(0, N, device=device) * dt

    # Determine the number of waypoints based on the number of columns in coeff_x
    num_waypoints = coeff_x.shape[1] + 1

    # Calculate the number of time steps for each trajectory segment
    segment_steps = N // (num_waypoints - 1)

    for i in range(num_waypoints - 1):
        segment_start = i * segment_steps
        segment_end = (i + 1) * segment_steps

        segment_time = time[segment_start:segment_end] - time[segment_start]

        # Position set points
        ref_state[segment_start:segment_end, 0] = coeff_x[0, i] * segment_time**5 + coeff_x[1, i] * segment_time**4 + coeff_x[2, i] * segment_time**3 + coeff_x[3, i] * segment_time**2 + coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 1] = coeff_y[0, i] * segment_time**5 + coeff_y[1, i] * segment_time**4 + coeff_y[2, i] * segment_time**3 + coeff_y[3, i] * segment_time**2 + coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 2] = coeff_z[0, i] * segment_time**5 + coeff_z[1, i] * segment_time**4 + coeff_z[2, i] * segment_time**3 + coeff_z[3, i] * segment_time**2 + coeff_z[4, i] * segment_time + coeff_z[5, i]

        # Velocity set points
        ref_state[segment_start:segment_end, 3] = coeff_x[1, i] * segment_time**4 + coeff_x[2, i] * segment_time**3 + coeff_x[3, i] * segment_time**2 + coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 4] = coeff_y[1, i] * segment_time**4 + coeff_y[2, i] * segment_time**3 + coeff_y[3, i] * segment_time**2 + coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 5] = coeff_z[1, i] * segment_time**4 + coeff_z[2, i] * segment_time**3 + coeff_z[3, i] * segment_time**2 + coeff_z[4, i] * segment_time + coeff_z[5, i]

        # Acceleration set points
        ref_state[segment_start:segment_end, 6] = coeff_x[2, i] * segment_time**3 + coeff_x[3, i] * segment_time**2 + coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 7] = coeff_y[2, i] * segment_time**3 + coeff_y[3, i] * segment_time**2 + coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 8] = coeff_z[2, i] * segment_time**3 + coeff_z[3, i] * segment_time**2 + coeff_z[4, i] * segment_time + coeff_z[5, i]

        # Jerk set points
        ref_state[segment_start:segment_end, 9] = coeff_x[3, i] * segment_time**2 + coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 10] = coeff_y[3, i] * segment_time**2 + coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 11] = coeff_z[3, i] * segment_time**2 + coeff_z[4, i] * segment_time + coeff_z[5, i]

        # Snap set points
        ref_state[segment_start:segment_end, 12] = coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 13] = coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 14] = coeff_z[4, i] * segment_time + coeff_z[5, i]

    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state
def get_ref_states(dt, N, coeff_x, coeff_y, coeff_z, device):
    ref_state = torch.zeros(N, 24, device=device)
    time = torch.arange(0, N, device=device) * dt

    # Determine the number of waypoints based on the number of columns in coeff_x
    num_waypoints = coeff_x.shape[1] + 1

    # Calculate the number of time steps for each trajectory segment
    segment_steps = N // (num_waypoints - 1)

    for i in range(num_waypoints - 1):
        segment_start = i * segment_steps
        segment_end = (i + 1) * segment_steps

        segment_time = time[segment_start:segment_end] - time[segment_start]

        # Position set points
        ref_state[segment_start:segment_end, 0] = coeff_x[0, i] * segment_time**5 + coeff_x[1, i] * segment_time**4 + coeff_x[2, i] * segment_time**3 + coeff_x[3, i] * segment_time**2 + coeff_x[4, i] * segment_time + coeff_x[5, i]
        ref_state[segment_start:segment_end, 1] = coeff_y[0, i] * segment_time**5 + coeff_y[1, i] * segment_time**4 + coeff_y[2, i] * segment_time**3 + coeff_y[3, i] * segment_time**2 + coeff_y[4, i] * segment_time + coeff_y[5, i]
        ref_state[segment_start:segment_end, 2] = coeff_z[0, i] * segment_time**5 + coeff_z[1, i] * segment_time**4 + coeff_z[2, i] * segment_time**3 + coeff_z[3, i] * segment_time**2 + coeff_z[4, i] * segment_time + coeff_z[5, i]

        # Velocity set points
        ref_state[segment_start:segment_end, 3] = 5 * coeff_x[0, i] * segment_time**4 + 4 * coeff_x[1, i] * segment_time**3 + 3 * coeff_x[2, i] * segment_time**2 + 2 * coeff_x[3, i] * segment_time + coeff_x[4, i]
        ref_state[segment_start:segment_end, 4] = 5 * coeff_y[0, i] * segment_time**4 + 4 * coeff_y[1, i] * segment_time**3 + 3 * coeff_y[2, i] * segment_time**2 + 2 * coeff_y[3, i] * segment_time + coeff_y[4, i]
        ref_state[segment_start:segment_end, 5] = 5 * coeff_z[0, i] * segment_time**4 + 4 * coeff_z[1, i] * segment_time**3 + 3 * coeff_z[2, i] * segment_time**2 + 2 * coeff_z[3, i] * segment_time + coeff_z[4, i]

        # Acceleration set points
        ref_state[segment_start:segment_end, 6] = 20 * coeff_x[0, i] * segment_time**3 + 12 * coeff_x[1, i] * segment_time**2 + 6 * coeff_x[2, i] * segment_time + 2 * coeff_x[3, i]
        ref_state[segment_start:segment_end, 7] = 20 * coeff_y[0, i] * segment_time**3 + 12 * coeff_y[1, i] * segment_time**2 + 6 * coeff_y[2, i] * segment_time + 2 * coeff_y[3, i]
        ref_state[segment_start:segment_end, 8] = 20 * coeff_z[0, i] * segment_time**3 + 12 * coeff_z[1, i] * segment_time**2 + 6 * coeff_z[2, i] * segment_time + 2 * coeff_z[3, i]

        # Jerk set points
        ref_state[segment_start:segment_end, 9] = 60 * coeff_x[0, i] * segment_time**2 + 24 * coeff_x[1, i] * segment_time + 6 * coeff_x[2, i]
        ref_state[segment_start:segment_end, 10] = 60 * coeff_y[0, i] * segment_time**2 + 24 * coeff_y[1, i] * segment_time + 6 * coeff_y[2, i]
        ref_state[segment_start:segment_end, 11] = 60 * coeff_z[0, i] * segment_time**2 + 24 * coeff_z[1, i] * segment_time + 6 * coeff_z[2, i]

        # Snap set points
        ref_state[segment_start:segment_end, 12] = 120 * coeff_x[0, i] * segment_time + 24 * coeff_x[1, i]
        ref_state[segment_start:segment_end, 13] = 120 * coeff_y[0, i] * segment_time + 24 * coeff_y[1, i]
        ref_state[segment_start:segment_end, 14] = 120 * coeff_z[0, i] * segment_time + 24 * coeff_z[1, i]

    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state
def get_ref_states2(dt, N, coeff_x, coeff_y, coeff_z, device):
    ref_state = torch.zeros(N, 24, device=device)
    time = torch.arange(0, N, device=device) * dt

    # Position set points
    ref_state[..., 0] = coeff_x[5, 0] * time**5 + coeff_x[4, 0] * time**4 + coeff_x[3, 0] * time**3 + coeff_x[2, 0] * time**2 + coeff_x[1, 0] * time + coeff_x[0, 0]
    ref_state[..., 1] = coeff_y[5, 0] * time**5 + coeff_y[4, 0] * time**4 + coeff_y[3, 0] * time**3 + coeff_y[2, 0] * time**2 + coeff_y[1, 0] * time + coeff_y[0, 0]
    ref_state[..., 2] = coeff_z[5, 0] * time**5 + coeff_z[4, 0] * time**4 + coeff_z[3, 0] * time**3 + coeff_z[2, 0] * time**2 + coeff_z[1, 0] * time + coeff_z[0, 0]

    # Velocity set points
    ref_state[..., 3] = coeff_x[5, 1] * time**5 + coeff_x[4, 1] * time**4 + coeff_x[3, 1] * time**3 + coeff_x[2, 1] * time**2 + coeff_x[1, 1] * time + coeff_x[0, 1]
    ref_state[..., 4] = coeff_y[5, 1] * time**5 + coeff_y[4, 1] * time**4 + coeff_y[3, 1] * time**3 + coeff_y[2, 1] * time**2 + coeff_y[1, 1] * time + coeff_y[0, 1]
    ref_state[..., 5] = coeff_z[5, 1] * time**5 + coeff_z[4, 1] * time**4 + coeff_z[3, 1] * time**3 + coeff_z[2, 1] * time**2 + coeff_z[1, 1] * time + coeff_z[0, 1]

    # Acceleration set points
    ref_state[..., 6] = coeff_x[5, 2] * time**5 + coeff_x[4, 2] * time**4 + coeff_x[3, 2] * time**3 + coeff_x[2, 2] * time**2 + coeff_x[1, 2] * time + coeff_x[0, 2]
    ref_state[..., 7] = coeff_y[5, 2] * time**5 + coeff_y[4, 2] * time**4 + coeff_y[3, 2] * time**3 + coeff_y[2, 2] * time**2 + coeff_y[1, 2] * time + coeff_y[0, 2]
    ref_state[..., 8] = coeff_z[5, 2] * time**5 + coeff_z[4, 2] * time**4 + coeff_z[3, 2] * time**3 + coeff_z[2, 2] * time**2 + coeff_z[1, 2] * time + coeff_z[0, 2]

    # Jerk set points
    ref_state[..., 9] = coeff_x[5, 3] * time**5 + coeff_x[4, 3] * time**4 + coeff_x[3, 3] * time**3 + coeff_x[2, 3] * time**2 + coeff_x[1, 3] * time + coeff_x[0, 3]
    ref_state[..., 10] = coeff_y[5, 3] * time**5 + coeff_y[4, 3] * time**4 + coeff_y[3, 3] * time**3 + coeff_y[2, 3] * time**2 + coeff_y[1, 3] * time + coeff_y[0, 3]
    ref_state[..., 11] = coeff_z[5, 3] * time**5 + coeff_z[4, 3] * time**4 + coeff_z[3, 3] * time**3 + coeff_z[2, 3] * time**2 + coeff_z[1, 3] * time + coeff_z[0, 3]

    # Snap set points
    ref_state[..., 12] = coeff_x[5, 4] * time**5 + coeff_x[4, 4] * time**4 + coeff_x[3, 4] * time**3 + coeff_x[2, 4] * time**2 + coeff_x[1, 4] * time + coeff_x[0, 4]
    ref_state[..., 13] = coeff_y[5, 4] * time**5 + coeff_y[4, 4] * time**4 + coeff_y[3, 4] * time**3 + coeff_y[2, 4] * time**2 + coeff_y[1, 4] * time + coeff_y[0, 4]
    ref_state[..., 14] = coeff_z[5, 4] * time**5 + coeff_z[4, 4] * time**4 + coeff_z[3, 4] * time**3 + coeff_z[2, 4] * time**2 + coeff_z[1, 4] * time + coeff_z[0, 4]

    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state
def get_ref_states1(dt, N, coeff_x, coeff_y, coeff_z,device):
    ref_state = torch.zeros(N, 24, device=device)
    time = torch.arange(0, N, device=args.device) * dt
    # print(coeff_x[1,0])
    # Position set points
    ref_state[..., 0] = coeff_x[0, 0] * time**5 + coeff_x[1, 0] * time**4 + coeff_x[2, 0] * time**3 + coeff_x[3, 0] * time**2 + coeff_x[4, 0] * time + coeff_x[5, 0]
    ref_state[..., 1] = coeff_y[0, 0] * time**5 + coeff_y[1, 0] * time**4 + coeff_y[2, 0] * time**3 + coeff_y[3, 0] * time**2 + coeff_y[4, 0] * time + coeff_y[5, 0]
    ref_state[..., 2] = coeff_z[0, 0] * time**5 + coeff_z[1, 0] * time**4 + coeff_z[2, 0] * time**3 + coeff_z[3, 0] * time**2 + coeff_z[4, 0] * time + coeff_z[5, 0]
    print(ref_state[...,0:3])
    # Velocity set points
    ref_state[..., 3] = 5 * coeff_x[0, 0] * time**4 + 4 * coeff_x[1, 0] * time**3 + 3 * coeff_x[2, 0] * time**2 + 2 * coeff_x[3, 0] * time + coeff_x[4, 0]
    ref_state[..., 4] = 5 * coeff_y[0, 0] * time**4 + 4 * coeff_y[1, 0] * time**3 + 3 * coeff_y[2, 0] * time**2 + 2 * coeff_y[3, 0] * time + coeff_y[4, 0]
    ref_state[..., 5] = 5 * coeff_z[0, 0] * time**4 + 4 * coeff_z[1, 0] * time**3 + 3 * coeff_z[2, 0] * time**2 + 2 * coeff_z[3, 0] * time + coeff_z[4, 0]

    # Acceleration set points
    ref_state[..., 6] = 20 * coeff_x[0, 0] * time**3 + 12 * coeff_x[1, 0] * time**2 + 6 * coeff_x[2, 0] * time + 2 * coeff_x[3, 0]
    ref_state[..., 7] = 20 * coeff_y[0, 0] * time**3 + 12 * coeff_y[1, 0] * time**2 + 6 * coeff_y[2, 0] * time + 2 * coeff_y[3, 0]
    ref_state[..., 8] = 20 * coeff_z[0, 0] * time**3 + 12 * coeff_z[1, 0] * time**2 + 6 * coeff_z[2, 0] * time + 2 * coeff_z[3, 0]

    # Jerk set points
    ref_state[..., 9] = 60 * coeff_x[0, 0] * time**2 + 24 * coeff_x[1, 0] * time + 6 * coeff_x[2, 0]
    ref_state[..., 10] = 60 * coeff_y[0, 0] * time**2 + 24 * coeff_y[1, 0] * time + 6 * coeff_y[2, 0]
    ref_state[..., 11] = 60 * coeff_z[0, 0] * time**2 + 24 * coeff_z[1, 0] * time + 6 * coeff_z[2, 0]

    # Snap set points
    ref_state[..., 12] = 120 * coeff_x[0, 0] * time + 24 * coeff_x[1, 0]
    ref_state[..., 13] = 120 * coeff_y[0, 0] * time + 24 * coeff_y[1, 0]
    ref_state[..., 14] = 120 * coeff_z[0, 0] * time + 24 * coeff_z[1, 0]

    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state

def subPlot(ax, x, y, style, xlabel=None, ylabel=None, label=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Geometric controller Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/pid/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 200  # Number of time steps
    dt = 0.1
    # States: x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
    state = torch.zeros(N, 13, device=args.device)
    state[0][6] = 1
    # coeff_x = torch.tensor([
    #     [0.0000e+00, 5.8040e-03, 2.5949e+02],
    #     [7.7716e-09, -2.2906e-02, -4.2303e+02],
    #     [0.0000e+00, 3.6089e-02, 2.7567e+02],
    #     [1.1060e+00, 1.0776e+00, -8.8660e+01],
    #     [-5.9557e-01, -5.8444e-01, 1.4013e+01],
    #     [8.0603e-02, 7.8859e-02, -8.7038e-01]
    # ])
    # coeff_y = torch.tensor([
    #     [-3.5527e-09, -9.3668e-01, 1.1181e+02],
    #     [-2.1316e-08, 3.6545e+00, -1.8016e+02],
    #     [-1.3323e-09, -5.7032e+00, 1.1408e+02],
    #     [-6.0732e-01, 3.8429e+00, -3.5159e+01],
    #     [6.9566e-01, -1.0406e+00, 5.3045e+00],
    #     [-1.7304e-01, 9.7934e-02, -3.1470e-01]
    # ])
    # coeff_z = torch.tensor([
    #     [2.0000e+00, 7.4102e-01, 7.5327e+01],
    #     [2.1316e-08, 4.9120e+00, -1.1669e+02],
    #     [5.3291e-09, -7.6659e+00, 7.1579e+01],
    #     [-1.7183e+00, 4.2636e+00, -2.1539e+01],
    #     [1.3260e+00, -1.0079e+00, 3.1899e+00],
    #     [-2.7772e-01, 8.6522e-02, -1.8647e-01]
    # ])
    coeff_x=torch.tensor([[ 0.0000e+00,  6.5232e-01,  1.3870e+00],
        [-9.5843e-11, -4.1467e-01, -1.9037e+00],
        [ 3.4694e-11,  1.0512e-01,  4.3171e-01],
        [ 6.2791e-03, -7.0071e-03, -3.5137e-02],
        [-7.1485e-04,  1.2242e-04,  1.2095e-03],
        [ 2.1762e-05,  7.1450e-07, -1.5060e-05]])
    coeff_y=torch.tensor([[ 0.0000e+00, -1.9523e+00,  3.8582e+00],
        [ 1.3357e-10,  1.2411e+00, -3.7676e+00],
        [-1.1048e-10, -3.1462e-01,  6.6529e-01],
        [-4.4954e-03,  3.5269e-02, -4.5260e-02],
        [ 8.4389e-04, -1.6620e-03,  1.3688e-03],
        [-3.5246e-05,  2.7748e-05, -1.5459e-05]])
    coeff_z=torch.tensor([[ 1.0000e+00,  9.5042e-01,  1.2968e-01],
        [-1.3214e-13,  2.3292e-02,  2.2474e-01],
        [ 1.3553e-14, -4.2199e-03, -2.2875e-02],
        [ 6.2547e-07,  3.6783e-04,  1.1478e-03],
        [ 1.3665e-08, -1.5425e-05, -2.8427e-05],
        [-1.1481e-08,  2.4938e-07,  2.7832e-07]])

    ref_state = get_ref_states(dt, N, coeff_x, coeff_y, coeff_z, args.device)
    # ref_state = get_ref_states(dt, N, args.device)
    time  = torch.arange(0, N, device=args.device) * dt
    parameters = torch.ones(4, device=args.device) # kp, kv, kori, kw
    mass = torch.tensor(0.18, device=args.device)
    g = torch.tensor(9.81, device=args.device)
    inertia = torch.tensor([[0.0820, 0., 0.00000255],
                            [0., 0.0845, 0.],
                            [0.00000255, 0., 0.1377]], device=args.device)

    controller = GeometricController(parameters, mass, inertia, g)
    model = MultiCopter(mass, g, inertia, dt).to(args.device)

    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], controller.forward(state[i], ref_state[i]))


    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=3, sharex=True)
    subPlot(ax[0], time, state[:, 0], '-', ylabel='X position (m)', label='true')
    subPlot(ax[0], time, ref_state[:, 0], '--', ylabel='X position (m)', label='sp')
    subPlot(ax[1], time, state[:, 1], '-', ylabel='Y position (m)', label='true')
    subPlot(ax[1], time, ref_state[:, 1], '--', ylabel='Y position (m)', label='sp')
    subPlot(ax[2], time, state[:, 2], '-', ylabel='Z position (m)', label='true')
    subPlot(ax[2], time, ref_state[:, 2], '--', ylabel='Z position (m)', label='sp')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    figure = os.path.join(args.save + 'geometric_controller.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
