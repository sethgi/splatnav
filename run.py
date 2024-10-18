#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from tqdm import tqdm
import json

from splat.splat_utils import GSplatLoader
from splatplan.splatplan import SplatPlan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Methods for the simulation
n = 100         # number of different configurations
n_steps = 10   # number of time discretizations
n_t = 500       # number of time steps. NOTE:This is only used for myopic methods like single-step splatplan!!!

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

### ----------------- Possible Methods ----------------- ###
# method = 'splatplan'
# method = 'sfc'
# TODO: splatplan-single-step, A*
### ----------------- Possible Distance Types ----------------- ###

for scene_name in ['flight', 'stonehenge', 'old_union', 'statues']:
    for method in ['splatplan']:

        resolution = 100

        # NOTE: POPULATE THE UPPER AND LOWER BOUNDS FOR OTHER SCENES!!!
        if scene_name == 'old_union':
            radius_z = 0.01     # How far to undulate up and down
            radius_config = 1.35/2  # radius of xy circle
            mean_config = np.array([0.14, 0.23, -0.15]) # mean of the circle

            path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored

            radius = 0.01       # radius of robot
            amax = 0.1
            vmax = 0.1

        elif scene_name == 'stonehenge':
            radius_z = 0.01
            radius_config = 0.784/2
            mean_config = np.array([-0.08, -0.03, 0.05])

            path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

            radius = 0.015
            amax = 0.1
            vmax = 0.1

        elif scene_name == 'statues':
            radius_z = 0.03    
            radius_config = 0.475
            mean_config = np.array([-0.064, -0.0064, -0.025])

            path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

            radius = 0.03
            amax = 0.1
            vmax = 0.1

        elif scene_name == 'flight':
            radius_z = 0.06
            radius_config = 0.545/2
            mean_config = np.array([0.19, 0.01, -0.02])

            path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

            radius = 0.02
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
            upper_bound = torch.tensor([1, 0.5, 0.26], device=device)

        # elif scene_name == 'flight-low-res':
        #     radius_z = 0.06
        #     radius = 0.03
        #     radius_config = 0.545/2
        #     mean_config = np.array([0.19, 0.01, -0.02])
        #     path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
        #     path_to_gsplat_high_res = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

            # radius = 0.03
            # amax = 0.1
            # vmax = 0.1

        print(f"Running {scene_name} with {method}")

        # Robot configuration
        robot_config = {
            'radius': radius,
            'vmax': vmax,
            'amax': amax,
        }

        # Environment configuration (specifically voxel)
        voxel_config = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'resolution': resolution,
        }

        tnow = time.time()
        gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)
        
        # # Load high res gsplat for flight-low-res for comparison
        # if scene_name == 'flight-low-res':
        #     gsplat_high_res = GSplatLoader(path_to_gsplat_high_res, device)

        if method == 'splatplan' or method == 'splatplan-single-step':
            planner = SplatPlan(gsplat, robot_config, voxel_config, device)
        
        elif method == 'sfc':
            raise NotImplementedError
            # planner = SFC(gsplat, device)

        else:
            raise ValueError(f"Method {method} not recognized")
        

        ### Create configurations in a circle
        x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
        x0 = x0 + mean_config

        xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
        xf = xf + mean_config

        # Run simulation
        total_data = []

        for trial, (start, goal) in enumerate(zip(x0, xf)):

            # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
            x = torch.tensor(start).to(device).to(torch.float32)
            goal = torch.tensor(goal).to(device).to(torch.float32)

            traj = [x]
            times = [0]
            u_values = []
            safety = []
            sucess = []
            feasible = []

            # We only do this for the single-step SplatPlan
            if method == 'splatplan' or method == 'sfc':
                output = planner.generate_path(x, goal, savepath=scene_name)

                raise NotImplementedError

            elif method == 'splatplan-single-step':
                raise NotImplementedError
                for i in tqdm(range(n_steps), desc=f"Simulating trajectory {trial}"):
                    ### ----------------- Safety Filtering ----------------- ###
                    u = cbf.solve_QP(x, u_des)
                    ### ----------------- End of Safety Filtering ----------------- ###

                    # We end the trajectory if the solver fails (because we short-circuit the control input if it fails)
                    if cbf.solver_success == False:
                        print("Solver failed")
                        sucess.append(False)
                        feasible.append(False)
                        break
                    else:
                        feasible.append(True)

                    # Propagate dynamics
                    x_ = x.clone()

                    traj.append(x)
                    times.append((i+1) * dt)
                    u_values.append(u.cpu().numpy())
                    u_des_values.append(u_des.cpu().numpy())

                    # It's not moving
                    if torch.norm(x - x_) < 0.001:
                        # If it's at the goal
                        if torch.norm(x_ - goal) < 0.001:
                            print("Reached Goal")
                            sucess.append(True)
                        else:
                            sucess.append(False)
                        break
                traj = torch.stack(traj)
                u_values = np.array(u_values)

                data = {
                'traj': traj.cpu().numpy().tolist(),
                'u_out': u_values.tolist(),
                'time_step': times,
                'safety': safety,
                'sucess': sucess,
                'feasible': feasible,
                'cbf_solve_time': cbf.times_cbf,
                'qp_solve_time': cbf.times_qp,
                'prune_time': cbf.times_prune,
                }

            total_data.append(data)

        # Save trajectory
        data = {
            'scene': scene_name,
            'method': method,
            'radius': radius,
            'amax': amax,
            'vmax': vmax,
            'radius_z': radius_z,
            'radius_config': radius_config,
            'mean_config': mean_config.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'resolution': resolution.tolist(),
            'n_steps': n_steps,
            'n_time': n_t,
            'total_data': total_data,
        }

        # create directory if it doesn't exist
        os.makedirs('trajs', exist_ok=True)

        # write to the file
        with open(f'trajs/{scene_name}_{method}.json', 'w') as f:
            json.dump(data, f, indent=4)

# %%
