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

def get_ref_states3(dt, N, traj_x, traj_y, traj_z, device):
    """
    Generate the reference states based on the waypoint trajectory
    """
    ref_state = torch.zeros(N, 27, device=device)
    time = torch.arange(0, N, device=device) * dt

    num_waypoints = traj_x.shape[1] + 1
    num_segments = num_waypoints - 1
    segment_duration = N // num_segments

    for i in range(num_segments):
        start_index = i * segment_duration
        end_index = (i + 1) * segment_duration

        t = time[start_index:end_index] - i * segment_duration * dt

        x_pos, x_vel, x_acc, x_jerk, x_snap, x_crackle = traj_x[:, i]
        y_pos, y_vel, y_acc, y_jerk, y_snap, y_crackle = traj_y[:, i]
        z_pos, z_vel, z_acc, z_jerk, z_snap, z_crackle = traj_z[:, i]

        ref_state[start_index:end_index, 0] = x_pos + x_vel * t + 0.5 * x_acc * t**2 + (1/6) * x_jerk * t**3 + (1/24) * x_snap * t**4 + (1/120) * x_crackle * t**5
        ref_state[start_index:end_index, 1] = y_pos + y_vel * t + 0.5 * y_acc * t**2 + (1/6) * y_jerk * t**3 + (1/24) * y_snap * t**4 + (1/120) * y_crackle * t**5
        ref_state[start_index:end_index, 2] = z_pos + z_vel * t + 0.5 * z_acc * t**2 + (1/6) * z_jerk * t**3 + (1/24) * z_snap * t**4 + (1/120) * z_crackle * t**5

        ref_state[start_index:end_index, 3] = x_vel + x_acc * t + 0.5 * x_jerk * t**2 + (1/6) * x_snap * t**3 + (1/24) * x_crackle * t**4
        ref_state[start_index:end_index, 4] = y_vel + y_acc * t + 0.5 * y_jerk * t**2 + (1/6) * y_snap * t**3 + (1/24) * y_crackle * t**4
        ref_state[start_index:end_index, 5] = z_vel + z_acc * t + 0.5 * z_jerk * t**2 + (1/6) * z_snap * t**3 + (1/24) * z_crackle * t**4

        ref_state[start_index:end_index, 6] = x_acc + x_jerk * t + 0.5 * x_snap * t**2 + (1/6) * x_crackle * t**3
        ref_state[start_index:end_index, 7] = y_acc + y_jerk * t + 0.5 * y_snap * t**2 + (1/6) * y_crackle * t**3
        ref_state[start_index:end_index, 8] = z_acc + z_jerk * t + 0.5 * z_snap * t**2 + (1/6) * z_crackle * t**3

        ref_state[start_index:end_index, 9] = x_jerk + x_snap * t + 0.5 * x_crackle * t**2
        ref_state[start_index:end_index, 10] = y_jerk + y_snap * t + 0.5 * y_crackle * t**2
        ref_state[start_index:end_index, 11] = z_jerk + z_snap * t + 0.5 * z_crackle * t**2

        ref_state[start_index:end_index, 12] = x_snap + x_crackle * t
        ref_state[start_index:end_index, 13] = y_snap + y_crackle * t
        ref_state[start_index:end_index, 14] = z_snap + z_crackle * t

        ref_state[start_index:end_index, 15] = x_crackle
        ref_state[start_index:end_index, 16] = y_crackle
        ref_state[start_index:end_index, 17] = z_crackle

    # b1 axis orientation
    ref_state[..., 18:21] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 24:27] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state
def get_ref_states(dt, N, coeff_x, coeff_y, coeff_z, device):
    """
    Generate the reference states based on the piecewise trajectory coefficients
    """
    ref_state = torch.zeros(N, 24, device=device)
    time = torch.arange(0, N, device=device) * dt

    num_segments = coeff_x.shape[1] - 1
    segment_duration = N // num_segments

    for i in range(num_segments):
        start_index = i * segment_duration
        end_index = (i + 1) * segment_duration

        t = time[start_index:end_index] - i * segment_duration * dt
        t_pow = torch.stack([t ** j for j in range(6)], dim=-1)

        x = torch.sum(coeff_x[:, i].unsqueeze(0) * t_pow, dim=-1)
        y = torch.sum(coeff_y[:, i].unsqueeze(0) * t_pow, dim=-1)
        z = torch.sum(coeff_z[:, i].unsqueeze(0) * t_pow, dim=-1)

        ref_state[start_index:end_index, 0:3] = torch.stack([x, y, z], dim=-1)

        dx = torch.sum(coeff_x[1:, i].unsqueeze(0) * t_pow[:, :-1] * torch.arange(1, 6, device=device).float(), dim=-1)
        dy = torch.sum(coeff_y[1:, i].unsqueeze(0) * t_pow[:, :-1] * torch.arange(1, 6, device=device).float(), dim=-1)
        dz = torch.sum(coeff_z[1:, i].unsqueeze(0) * t_pow[:, :-1] * torch.arange(1, 6, device=device).float(), dim=-1)

        ref_state[start_index:end_index, 3:6] = torch.stack([dx, dy, dz], dim=-1)

        ddx = torch.sum(coeff_x[2:, i].unsqueeze(0) * t_pow[:, :-2] * torch.arange(2, 6, device=device).float() * (torch.arange(1, 5, device=device).float()), dim=-1)
        ddy = torch.sum(coeff_y[2:, i].unsqueeze(0) * t_pow[:, :-2] * torch.arange(2, 6, device=device).float() * (torch.arange(1, 5, device=device).float()), dim=-1)
        ddz = torch.sum(coeff_z[2:, i].unsqueeze(0) * t_pow[:, :-2] * torch.arange(2, 6, device=device).float() * (torch.arange(1, 5, device=device).float()), dim=-1)

        ref_state[start_index:end_index, 6:9] = torch.stack([ddx, ddy, ddz], dim=-1)

        dddx = torch.sum(coeff_x[3:, i].unsqueeze(0) * t_pow[:, :-3] * torch.arange(3, 6, device=device).float() * (torch.arange(2, 5, device=device).float()) * (torch.arange(1, 4, device=device).float()), dim=-1)
        dddy = torch.sum(coeff_y[3:, i].unsqueeze(0) * t_pow[:, :-3] * torch.arange(3, 6, device=device).float() * (torch.arange(2, 5, device=device).float()) * (torch.arange(1, 4, device=device).float()), dim=-1)
        dddz = torch.sum(coeff_z[3:, i].unsqueeze(0) * t_pow[:, :-3] * torch.arange(3, 6, device=device).float() * (torch.arange(2, 5, device=device).float()) * (torch.arange(1, 4, device=device).float()), dim=-1)

        ref_state[start_index:end_index, 9:12] = torch.stack([dddx, dddy, dddz], dim=-1)

        ddddx = torch.sum(coeff_x[4:, i].unsqueeze(0) * t_pow[:, :-4] * torch.arange(4, 6, device=device).float() * (torch.arange(3, 5, device=device).float()) * (torch.arange(2, 4, device=device).float()) * (torch.arange(1, 3, device=device).float()), dim=-1)
        ddddy = torch.sum(coeff_y[4:, i].unsqueeze(0) * t_pow[:, :-4] * torch.arange(4, 6, device=device).float() * (torch.arange(3, 5, device=device).float()) * (torch.arange(2, 4, device=device).float()) * (torch.arange(1, 3, device=device).float()), dim=-1)
        ddddz = torch.sum(coeff_z[4:, i].unsqueeze(0) * t_pow[:, :-4] * torch.arange(4, 6, device=device).float() * (torch.arange(3, 5, device=device).float()) * (torch.arange(2, 4, device=device).float()) * (torch.arange(1, 3, device=device).float()), dim=-1)

        ref_state[start_index:end_index, 12:15] = torch.stack([ddddx, ddddy, ddddz], dim=-1)

    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state
def get_ref_states1(dt, N, device):
    """
    Generate the circle trajectory with fixed heading orientation
    """
    ref_state = torch.zeros(N, 24, device=device)
    time  = torch.arange(0, N, device=args.device) * dt
    # position set points
    ref_state[..., 0:3] = torch.stack(
        [2*(1-torch.cos(time)), 2*torch.sin(time), 0.1*torch.sin(time)], dim=-1)
    print(ref_state[...,0:3])
    # velocity set points
    ref_state[..., 3:6] = torch.stack(
        [2*torch.sin(time), 2*torch.cos(time), 0.1*torch.cos(time)], dim=-1)
    # acceleration set points
    ref_state[..., 6:9] = torch.stack(
        [2*torch.cos(time), -2*torch.sin(time), -0.1*torch.sin(time)], dim=-1)
    # acceleration dot set points
    ref_state[..., 9:12] = torch.stack(
        [-2*torch.sin(time), -2*torch.cos(time), -0.1*torch.cos(time)], dim=-1)
    # acceleration ddot set points
    ref_state[..., 12:15] = torch.stack(
        [-2*torch.cos(time), 2*torch.sin(time), 0.1*torch.sin(time)], dim=-1)
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

    N = 200   # Number of time steps
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

    # ref_state = get_ref_states(dt, N, coeff_x, coeff_y, coeff_z, args.device)
    ref_state = get_ref_states(dt, N, args.device)
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
