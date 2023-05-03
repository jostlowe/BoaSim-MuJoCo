import csv
import math
import threading
import time
from copy import deepcopy

import mujoco
from mujoco import viewer
from itertools import pairwise
import numpy as np
from matplotlib import pyplot as plt

N_LINKS = 13
N_JOINTS = N_LINKS - 1

def get_joint_angles(data: mujoco.MjData):
    joint_names = [f"joint_{i}" for i in range(1, N_LINKS)]
    return np.array([data.joint(joint_name).qpos[0] for joint_name in joint_names])


def calculate_target_joint_angles(data: mujoco.MjData):
    SPEED = 0.5   # Link lengths per second

    path = [0.0] * N_JOINTS + [math.pi / 3] * 3 + [0] * 2 + [-math.pi / 3] * 3 + [0]*2 + [-math.pi/4] + [0] + [math.pi/3]*3 + [0] * N_JOINTS
    progress = data.time * SPEED
    step = math.floor(progress)
    target_angles = np.array(path[step:(step + N_JOINTS)])
    return target_angles


def controller(_model: mujoco.MjModel, data: mujoco.MjData):
    MAX_SPEED = 0.5
    GAIN = 4

    # Control
    joint_angles = get_joint_angles(data)
    target_joint_angles = calculate_target_joint_angles(data)
    error = target_joint_angles - joint_angles
    data.ctrl = np.clip(GAIN * error, -MAX_SPEED, MAX_SPEED)

    # Logging
    log_data.append([data.time] + list(data.body("head").xpos) + list(data.actuator_force))


# Setup
model = mujoco.MjModel.from_xml_path('mjcf/scene.xml')
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)
log_data = []

# Launch the simulation
try:
    with viewer.launch(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
finally:
    with open("data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_data)
    print("Data logged!")
