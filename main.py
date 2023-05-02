import math
import time

import mujoco
from mujoco import viewer
from itertools import pairwise

N_LINKS = 13
N_JOINTS = N_LINKS-1

angles = [0.0]*N_JOINTS + [math.pi/4] + [math.pi/4] + [0.0]*N_JOINTS


def controller(model: mujoco.MjModel, data: mujoco.MjData):
    progress = data.time/2
    step = math.floor(progress)
    print(step)
    ctrl = angles[step:(step+N_JOINTS)]
    data.ctrl = ctrl


# Setup
model = mujoco.MjModel.from_xml_path('mjcf/scene.xml')
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)

# Launch the simulation
with viewer.launch(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        print(data.qpos)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
