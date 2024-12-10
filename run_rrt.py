#%%
import os
from os.path import abspath, dirname, join, expanduser
from functools import partial
from scipy.spatial import KDTree

import sys
import torch
from pathlib import Path    
import time
import pickle 
import numpy as np
import json
from splat.splat_utils import GSplatLoader
from splatplan.spline_utils import SplinePlanner
from ellipsoids.intersection_utils import gs_sphere_intersection_test, compute_intersection_linear_motion
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix

ompl_app_root = dirname(dirname(dirname(abspath(__file__))))

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import app as oa
except ImportError:
    sys.path.insert(0, join(ompl_app_root, 'ompl/py-bindings'))
    from ompl import base as ob
    from ompl import geometric as og

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Methods for the simulation
n = 5         # number of different configurations
n_steps = 10   # number of time discretizations

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)


### ----------------- Possible Methods ----------------- ###
# method = 'splatplan'
# method = 'sfc'
# method = 'ompl'
# TODO: splatplan-single-step, A*
### ----------------- Possible Distance Types ----------------- ###


class EMV(ob.MotionValidator):
    '''
    Custom Ellipsoid Motion Validator to implement line search between sampled points in graph
    '''
    def __init__(self, si, evc):
        super().__init__(si)
        self.space_info = si
        self.evc = evc

    def checkMotion(self, s1, s2) -> bool:
        self.evc.line_count += 1
        current_point_np = np.array([s1[0], s1[1], s1[2]])

        # find closest ellipsoids
        neigh_indexs = self.evc.kdtree.query_ball_point(current_point_np, self.evc.search_radius)
        if len(neigh_indexs) == 0:
            # This isn't very good. Think of something that makes more sense. Maybe find nearest n?
            neigh_indexs = [0, 1]
            # print('Nothing nearby')

        # Grab necessary parameters
        x0 = torch.tensor(current_point_np).to(self.evc.device).to(torch.float32)
        next_point_np = np.array([s2[0], s2[1], s2[2]])
        delta_x = torch.tensor(next_point_np).to(self.evc.device).to(torch.float32)#torch.tensor(next_point_np - current_point_np).to(self.evc.device).to(torch.float32)
        # Perform line search
        S_A = self.evc.scales[neigh_indexs]
        R_A = self.evc.rots[neigh_indexs]
        mu_A = self.evc.means[neigh_indexs]

        tnow = time.time()
        search_data = compute_intersection_linear_motion(x0, delta_x, R_A, S_A, mu_A, S_B=self.evc.robot_radius, collision_type='sphere', mode='bisection', N=5)
        results = search_data['is_not_intersect']
        result = torch.all(results)
        self.evc.times_line.append(time.time() - tnow)
        return result.item()



class EVC(ob.StateValidityChecker):
    '''
    Ellipsoidal Validity Checker. Takes in all ellipsoids and creates KD tree
    '''
    def __init__(self, gsplat, robot_radius, device):
        self.device = device
        # robot parameters
        self.robot_radius = robot_radius

        # grab means, covs, Rs (as Rmat), and Ss from gsplat 
        self.means, self.covs, self.rots, self.scales  = gsplat.means, gsplat.covs, quaternion_to_rotation_matrix(gsplat.rots), gsplat.scales

        # build KD tree
        self.kdtree = KDTree(self.means.cpu().numpy())
        largest_dim = np.sqrt(np.max(np.linalg.eigvalsh(self.covs.cpu().numpy())))
        self.search_radius = largest_dim + robot_radius
    
        # count calls to the checker
        self.line_count = 0
        self.point_count = 0

        # validity timer
        self.times_point = []
        # motion timer
        self.times_line = []

    def isValid(self, si, state) -> bool:
        '''
        Applies collision checker for a point and ellipsoid

        Inputs:
            state: robot state values

        Returns:
            valid_flag: [bool] Collision condition; boolean (true if state is valid (no collision), false if not valid (collision))
        '''
        # grab current position from robot state
        current_point_np = np.array([state[0], state[1], state[2]])

        # find closest ellipsoids
        neigh_indexs = self.kdtree.query_ball_point(current_point_np, self.search_radius)
        if len(neigh_indexs) == 0:
            # This isn't very good. Think of something that makes more sense. Maybe find nearest n?
            neigh_indexs = [0, 1]
            # print('Nothing nearby')
        
        self.point_count += 1

        mu_A = self.means[neigh_indexs]
        R_A = self.rots[neigh_indexs]
        S_A = self.scales[neigh_indexs] 
        mu_B = torch.tensor(current_point_np).to(self.device).to(torch.float32)
        
        ss = torch.linspace(0., 1., 100, device=self.device)[1:-1].reshape(1, -1, 1)

        tnow = time.time()
        results, _ = gs_sphere_intersection_test(R_A, S_A, self.robot_radius, mu_A, mu_B, ss)
        result = torch.all(results)
        self.times_point.append(time.time() - tnow)
        return result.item()


def plan(evc, space, start_point, goal_point):
    # define setup state object
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    # define motion checking function
    motion_validator = EMV(si, evc)
    si.setMotionValidator(motion_validator)
    

    # define validity checking function
    ss.setStateValidityChecker(ob.StateValidityCheckerFn( \
        partial(evc.isValid, si)))

    # create a start state
    start = ob.State(space)
    start()[0] = start_point[0].item()
    start()[1] = start_point[1].item()
    start()[2] = start_point[2].item()

    # create a goal state
    goal = ob.State(space)
    goal()[0] = goal_point[0].item()
    goal()[1] = goal_point[1].item()
    goal()[2] = goal_point[2].item()
    ss.setStartAndGoalStates(start, goal, 0.001) # set acceptable end tolerance

    # set planner
    path_resolution = 0.001  # sample step resolution
    print(f'Using Planner RRT')
    planner = og.RRTstar(ss.getSpaceInformation())
    planner.setRange(path_resolution)
    planner.setGoalBias(0.1)
    ss.setPlanner(planner)
    ss.setup()

    # termination condition
    termination_condition = 30.0 # set time to to solve and optimize 

    # solve
    if ss.solve(termination_condition):
        # simplifly
        #ss.simplifySolution()

        # print solutcion
        solution = ss.getSolutionPath().printAsMatrix()
        print(solution)
        print('Collision count is {}'.format(evc.point_count))
        print('Line Search count is {}'.format(evc.line_count))
        return solution
    else: 
        print('No solution found')
        return None

# Perform sim

for scene_name in ['stonehenge']: #['stonehenge', 'statues', 'flight', 'old_union']:
    for method in ['ompl']:

        # NOTE: POPULATE THE UPPER AND LOWER BOUNDS FOR OTHER SCENES!!!
        if scene_name == 'old_union':
            radius_z = 0.01     # How far to undulate up and down
            radius_config = 1.35/2  # radius of xy circle
            mean_config = np.array([0.14, 0.23, -0.15]) # mean of the circle

            #path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml') # points to where the gsplat params are stored
            path_to_gsplat = Path('outputs/old_union2/sparse-splat/2024-10-25_113753/config.yml') # points to sparse splat version
            radius = 0.01       # radius of robot
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-.8, -.7, -0.2], device=device)
            upper_bound = torch.tensor([1., 1., -0.1], device=device)

            resolution = 75

        elif scene_name == 'stonehenge':
            radius_z = 0.01
            radius_config = 0.784/2
            mean_config = np.array([-0.08, -0.03, 0.05])

            #path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')
            path_to_gsplat = Path('outputs/stonehenge/sparse-splat/2024-10-25_120323/config.yml') # sparse-splat


            radius = 0.015
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-5., -.5, -0.], device=device)
            upper_bound = torch.tensor([5., .5, 0.1], device=device)

            resolution = 40

        elif scene_name == 'statues':
            radius_z = 0.03    
            radius_config = 0.475
            mean_config = np.array([-0.064, -0.0064, -0.025])

            #path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')
            path_to_gsplat = Path('outputs/statues/sparse-splat/2024-10-25_114702/config.yml') # sparse-splat
            radius = 0.03
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-.5, -.5, -0.1], device=device)
            upper_bound = torch.tensor([.5, .5, 0.2], device=device)

            resolution = 60

        elif scene_name == 'flight':
            radius_z = 0.06
            radius_config = 0.545/2
            mean_config = np.array([0.19, 0.01, -0.02])

            #path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')
            path_to_gsplat = Path('outputs/flight/sparse-splat/2024-10-25_115216/config.yml') # sparse-splat
            radius = 0.02
            amax = 0.1
            vmax = 0.1

            lower_bound = torch.tensor([-1.33, -0.5, -0.17], device=device)
            upper_bound = torch.tensor([1, 0.5, 0.26], device=device)

            resolution = 100

        print(f"Running {scene_name} with {method}")

        # Robot configuration
        robot_config = {
            'radius': radius,
            'vmax': vmax,
            'amax': amax,
        }


        tnow = time.time()
        gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        spline_planner = SplinePlanner(device=device)

        # setup validity chcker
        evc = EVC(gsplat, robot_config['radius'], device)

        # setup space   
        space = ob.RealVectorStateSpace(3)
        bounds = ob.RealVectorBounds(3)
        bound_inflation_scale = 3 # give slack to search space bounds
        lower_bound_np = (lower_bound.cpu().numpy().tolist()) 
        upper_bound_np = (upper_bound.cpu().numpy().tolist())
        bounds.setLow(0, lower_bound_np[0]* bound_inflation_scale) 
        bounds.setLow(1, lower_bound_np[1]* bound_inflation_scale) 
        bounds.setLow(2, lower_bound_np[2]* bound_inflation_scale) 
        bounds.setHigh(0, upper_bound_np[0]* bound_inflation_scale)
        bounds.setHigh(1, upper_bound_np[1]* bound_inflation_scale)
        bounds.setHigh(2, upper_bound_np[2]* bound_inflation_scale)
        space.setBounds(bounds)

        x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
        x0 = x0 + mean_config
        xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
        xf = xf + mean_config

        # Run simulation
        total_data = []
        # solve
        for trial, (start, goal) in enumerate(zip(x0, xf)):
            # define start and goal
            x = torch.tensor(start).to(device).to(torch.float32)
            goal = torch.tensor(goal).to(device).to(torch.float32)

            # generate paths
            tnow = time.time()
            output = plan(evc, space, x, goal) # ompl saves paths as a string
            times_rrt = time.time() - tnow
            if output is None:
                path = []
            else: 
                lines = output.strip().split('\n')
                path = [tuple(map(float, line.split())) for line in lines]
                                
            traj_data = {
                'traj': path,
                'times_rrt': times_rrt,
                #'start': start.tolist(),
                #'goal': goal.tolist(),
            }

            total_data.append(traj_data)
            print(f"Trial {trial} completed")

        # save data
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
            'resolution': resolution,
            'n_steps': n_steps,
            'n_time': 0,
            'total_data': total_data,
        }

        # create directory if it doesn't exist
        os.makedirs('trajs', exist_ok=True)

        # write to the file
        with open(f'trajs/{scene_name}_{method}.json', 'w') as f:
            json.dump(data, f, indent=4)

            


# %%
