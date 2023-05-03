import csv
import math
import time

import mujoco
from mujoco import viewer
import numpy as np

from path import SnakePath


N_LINKS = 15
N_JOINTS = N_LINKS - 1
SPEED = 0.1

def get_joint_angles(data: mujoco.MjData):
    joint_names = [f"joint_{i}" for i in range(1, N_LINKS)]
    return np.array([data.joint(joint_name).qpos[0] for joint_name in joint_names])


def calculate_target_joint_angles(data: mujoco.MjData):
    path = SnakePath(
        control_points=[(0, 0), (2.6, 0), (2.6, 0.34), (1.85, 0.34), (1.85, 0.66), (3, 0.66), (3.4, 0), (5, 0)],
        min_radius=0.15,
        n_links=N_LINKS,
        link_length=0.2
    )
    target_angles = np.array(path.get_joint_angles(data.time*SPEED))
    return target_angles


def controller(_model: mujoco.MjModel, data: mujoco.MjData):
    MAX_SPEED = 2
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
