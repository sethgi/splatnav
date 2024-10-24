#%%
import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

scene_names = ['statues', 'flight', 'old_union']
methods = ['splatplan', 'sfc']

n = 100

t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

fig, ax = plt.subplots(3, 2, figsize=(10, 10), dpi=200)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
for k, scene_name in enumerate(scene_names):

    for j, method in enumerate(methods):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}_processed.json'

        with open(save_fp, 'r') as f:
            meta = json.load(f)

        datas = meta['total_data']

        if method == 'sfc':
            col = '#34A853'
            linewidth= 3
 
            success = []
            safety = []
            times = []
            polytope_vols = []
            polytope_radii = []
            path_length = []

            # Per trajectory
            for i, data in enumerate(datas):

                # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
                if not data['feasible']:
                    success.append(False)
                    continue

                else:
                    success.append(True)

                num_polytopes = data['num_polytopes']

                # record the times
                traj_time = np.array([data['times_astar'], data['times_collision_set'], data['times_ellipsoid'],
                                    data['times_polytope'], data['times_opt']])

                times.append(traj_time)

                # record the min safety margin
                safety.append(np.array(data['safety_margin']).min())
                path_length.append(data['path_length'])

                # record the polytope stats (min/max/mean/std)
                polytope_vols_entry = np.array(data['polytope_vols'])
                polytope_radii_entry = np.array(data['polytope_radii'])

                polytope_vols.append([polytope_vols_entry.min(), polytope_vols_entry.max(), polytope_vols_entry.mean(), polytope_vols_entry.std()])
                polytope_radii.append([polytope_radii_entry.min(), polytope_radii_entry.max(), polytope_radii_entry.mean(), polytope_radii_entry.std()])

            success = np.array(success)
            safety = np.array(safety)
            times = np.array(times)
            polytope_vols = np.array(polytope_vols)
            polytope_radii = np.array(polytope_radii)
            path_length = np.array(path_length)

        elif method == 'splatplan':
            col = '#4285F4'
            linewidth=3

            success = []
            safety = []
            times = []
            polytope_vols = []
            polytope_radii = []
            path_length = []

            # Per trajectory
            for i, data in enumerate(datas):

                # If the trajectory wasn't feasible to solve, we don't store any data on it besides the success.
                if not data['feasible']:
                    success.append(False)
                    continue

                else:
                    success.append(True)

                num_polytopes = data['num_polytopes']

                # record the times
                traj_time = np.array([data['times_astar'], data['times_collision_set'],
                                    data['times_polytope'], data['times_opt']])
                times.append(traj_time)
                
                # record the min safety margin
                safety.append(np.array(data['safety_margin']).min())
                path_length.append(data['path_length'])

                # record the polytope stats (min/max/mean/std)
                polytope_vols_entry = np.array(data['polytope_vols'])
                polytope_radii_entry = np.array(data['polytope_radii'])

                polytope_vols.append([polytope_vols_entry.min(), polytope_vols_entry.max(), polytope_vols_entry.mean(), polytope_vols_entry.std()])
                polytope_radii.append([polytope_radii_entry.min(), polytope_radii_entry.max(), polytope_radii_entry.mean(), polytope_radii_entry.std()])

            success = np.array(success)
            safety = np.array(safety)
            times = np.array(times)
            polytope_vols = np.array(polytope_vols)
            polytope_radii = np.array(polytope_radii)
            path_length = np.array(path_length)

        # elif method == 'rrt':
        #     col = '#FBBc05'
        #     linewidth=3

        # elif method == 'nerf-nav':
        #     col = '#EA4335'
        #     linewidth=3

        print(f'{scene_name}_{method}')

        # Computation Time

        # TODO: This plots the individual times of each component of the algorithm
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, qp_solve_time.mean(), bottom = 0, width=0.15, color= adjust_lightness(col, 0.5), linewidth=3, ec='k', label='qp')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, cbf_solve_time.mean(), bottom=qp_solve_time.mean(), width=0.15, color=adjust_lightness(col, 1.0), linewidth=3, ec='k', label='cbf')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean(), bottom=cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3, hatch='x', ec='k', label='prune')
        # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean() + cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3,  ec='k', label='prune')

        # TODO: This plots just the total time
        ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, times.sum(axis=1).mean(), width=0.15, color=col, capsize=10, edgecolor='black', linewidth=linewidth, 
                    linestyle='-', joinstyle='round', rasterized=True)

        # Safety Margin
        errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

        ax[0, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        ax[0, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.04)
        ax[0, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, safety.mean(), s=200, color=col, alpha=1, marker='>')
            
        # Polytope Volume
        errors = np.abs(polytope_vols[:, 2].mean().reshape(-1, 1) - np.array([polytope_vols[:, 0].min(), polytope_vols[:, 1].max()]).reshape(-1, 1))

        ax[1, 0].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_vols[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        ax[1, 0].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_vols[:, 2])), polytope_vols[:, 2], s=250, color=col, alpha=0.04)
        ax[1, 0].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_vols[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

        # Polytope Radii
        errors = np.abs(polytope_radii[:, 2].mean().reshape(-1, 1) - np.array([polytope_radii[:, 0].min(), polytope_radii[:, 1].max()]).reshape(-1, 1))

        ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, polytope_radii[:, 2].mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(polytope_radii[:, 2])), polytope_radii[:, 2], s=250, color=col, alpha=0.04)
        ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, polytope_radii[:, 2].mean(), s=200, color=col, alpha=1, marker='>')

        # Path Length
        errors = np.abs(path_length.mean().reshape(-1, 1) - np.array([path_length.min(), path_length.max()]).reshape(-1, 1))

        ax[2, 0].errorbar(k + 0.75*j/len(methods) + 0.25/2, path_length.mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
        ax[2, 0].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(path_length)), path_length, s=250, color=col, alpha=0.04)
        ax[2, 0].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, path_length.mean(), s=200, color=col, alpha=1, marker='>')

        # Success Rate
        ax[2, 1].bar(k + 0.75*j/len(methods) + 0.25/2, success.sum()/len(success), width=0.15, color=col, capsize=10, edgecolor='black', linewidth=linewidth, 
                    linestyle='-', joinstyle='round', rasterized=True)

# COMPUTATION TIME
ax[0, 0].set_title(r'Computation Time (s) $\downarrow$' , fontsize=25, fontweight='bold')
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[0, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 0].spines[location].set_linewidth(4)
# ax[0,0].set_yscale('log')

# SAFETY MARGIN
ax[0, 1].set_title('Minimum Distance', fontsize=25, fontweight='bold')
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].axhline(y = 0., color = 'k', linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 
ax[0, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[0, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 1].spines[location].set_linewidth(4)
ax[0, 1].set_ylim(-0.005, 0.005)

# POLYTOPE VOLUME
ax[1, 0].set_title(r'Polytope Volume $\uparrow$', fontsize=25, fontweight='bold')
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[1, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 0].spines[location].set_linewidth(4)

# POLYTOPE RADII
ax[1, 1].set_title(r'Polytope Radius $\uparrow$', fontsize=25, fontweight='bold')
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[1, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 1].spines[location].set_linewidth(4)

# PATH LENGTH
ax[2, 0].set_title(r'Path Length $\downarrow$', fontsize=25, fontweight='bold')
ax[2, 0].get_xaxis().set_visible(False)
ax[2, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[2, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[2, 0].spines[location].set_linewidth(4)

# SUCCESS RATE
ax[2, 1].set_title(r'Success Rate $\uparrow$', fontsize=25, fontweight='bold')
ax[2, 1].get_xaxis().set_visible(False)
ax[2, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[2, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[2, 1].spines[location].set_linewidth(4)



plt.savefig(f'simulation_stats.png', dpi=500)

#%%