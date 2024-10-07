import numpy as np
import os, sys
import open3d as o3d
import copy
import cv2
import matplotlib.pyplot as plt
from enum import Enum, auto
import torch
import open3d as o3d

from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union
from open3d.pipelines.registration import Feature

from lightglue import LightGlue, SuperPoint, DISK, ALIKED, DoGHardNet

from torch.autograd.gradcheck import get_numerical_jacobian

import copy
from ns_utils.nerfstudio_utils import *

# # # # #
# # # # # Utils
# # # # #

# scaling factor for FPFH
FPFH_MAX = 200.0

class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class PrintOptions:
    '''
    Print Options for printing to the CONSOLE.
    '''
    def __init__(self, width=100):
        # separators
        self.sep_0 = '*' * width
        self.sep_1 = '-' * width
        self.sep_space = '\x0c' * 3

class POI_Detector(Enum):
    SIFT = auto()
    ORB = auto()
    SURF = auto()
    LIGHTGLUE = auto()

class LIGHTGLUE_Extractor(Enum):
    SUPERPOINT = auto()
    DISK = auto()
    ALIKED = auto()
    DOGHARDNET = auto()
    
    
class Global_Registration(Enum):
    # RANSAC
    RANSAC = auto()
    # Fast Global Registration
    FGR = auto()
    
    
class Local_Registration(Enum):
    # Iterative Closest Point (ICP)
    ICP = auto()
    # Colored Iterative Closest Point
    COLORED_ICP = auto()
    
    
class Local_Refinement(Enum):
    # Iterative Closest Point (ICP)and its variant
    ICP = auto()
    # PnP-RANSAC
    PnP_RANSAC = auto()
    # iNeRF
    iNeRF = auto()


class Open3dCustomFeature(Feature):
    def __init__(self, feat):
        super().__init__(feat)
        
        @Feature.data.setter
        def data(self, value):
            self.data = value
            
            
def SE3error(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(np.arccos(min(max(((Terr[0:3,0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3,3])
    return (rerr*180/np.pi, terr)

def downsample_point_cloud(pcd, voxel_size=0.01,
                           print_stats: bool = False):
    # downsample point cloud
    if print_stats:
        print(f":: Downsample with a voxel size {voxel_size:3f}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    return pcd_down

def visualize_point_cloud(pcds: List[object]= [], 
                          enable_downsampled_visualization: bool = True,
                          downsampled_voxel_size: float = 0.01):
    # visualize point cloud
    if enable_downsampled_visualization:
        pcds_down = [downsample_point_cloud(pcd, voxel_size=downsampled_voxel_size)
                     for pcd in pcds]
        
        o3d.visualization.draw_plotly(pcds_down)
    else:
        o3d.visualization.draw_plotly(pcds)
        
def visualize_registration_result(source, target, 
                                  transformation, 
                                  enable_downsampled_visualization: bool = True,
                                  downsampled_voxel_size: float = 0.01):
    source_disp = copy.deepcopy(source)
    source_disp.transform(transformation)

    visualize_point_cloud([target, source_disp],
                          enable_downsampled_visualization=enable_downsampled_visualization,
                          downsampled_voxel_size=downsampled_voxel_size)   
    
def extract_geometric_feature(pcd, voxel_size,
                              downsample_pcd: bool = True,
                              print_stats: bool = False):
    # downsample point cloud
    if downsample_pcd:
        if print_stats:
            print(f":: Downsample with a voxel size {voxel_size:3f}")
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = copy.deepcopy(pcd)
    
    # estimate the normals
    radius_normal = voxel_size * 2
    if print_stats:
        print(f":: Estimate normal with search radius {radius_normal:3f}")
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # FPFH feature
    radius_feature = voxel_size * 5
    if print_stats:
        print(f":: Compute FPFH feature with search radius {radius_feature:3f}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))


    fpfh_data = np.asarray(pcd_fpfh.data)
    fpfh_data = fpfh_data / np.linalg.norm(fpfh_data)
    fpfh_rgb = np.vstack((fpfh_data, np.asarray(pcd_down.colors).T))
    pcd_fpfh =  Open3dCustomFeature(pcd_fpfh)
    
    pcd_fpfh.data = fpfh_rgb
    
    return pcd_down, pcd_fpfh

def preprocess_point_clouds(source, target, voxel_size,
                            downsample_pcd: bool = True,
                            preprocess_target: Optional[bool] = True):
    # extract the geometric features
    source_down, source_fpfh = extract_geometric_feature(pcd=source, voxel_size=voxel_size, downsample_pcd=downsample_pcd)
    
    if preprocess_target:
        target_down, target_fpfh = extract_geometric_feature(pcd=target, voxel_size=voxel_size, downsample_pcd=downsample_pcd)
    else:
        target_down, target_fpfh = None, None
        
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh,
                                voxel_size,
                                print_stats: bool = False):
    # distance threshold for convergence criterion
    distance_threshold = voxel_size * 0.8 #1.5
    
    if print_stats:
        print(f":: RANSAC Registration on downsampled point clouds.")
        print(f"   Using a distance threshold of {distance_threshold}, given a downsampling voxel size of {voxel_size}.")
    
    # transformation
    transformation = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    return transformation
        
def execute_fast_global_registration(source_down, target_down,
                                     source_fpfh, target_fpfh,
                                     voxel_size,
                                     print_stats: bool = False):
    # distance threshold for convergence criterion
    distance_threshold = voxel_size * 0.95
    
    if print_stats:
        print(f":: Fast Global Registration on downsampled point clouds.")
        print(f"   Using a distance threshold of {distance_threshold}, given a downsampling voxel size of {voxel_size}.")
    
    # transformation
    transformation = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, 
        source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    
    return transformation

# local refinement
def ICP_refinement_registration(source, target,
                                source_fpfh, target_fpfh,
                                voxel_size, init_transformation,
                                distance_threshold: float = None,
                                point_to_plane_registration: bool = True,
                                print_stats: bool = False):
    # distance threshold
    if distance_threshold is None:
        distance_threshold = voxel_size * 0.4#  * 1e1
    
    # Point-to-Plane ICP
    if print_stats:
        if point_to_plane_registration:
            method_descrip = 'Point-to-Plane ICP registration'
        else:
            method_descrip = 'Point-to-Point ICP registration'
            
        print(f"{method_descrip} is applied on the original point clouds, with a distance threshold of {distance_threshold:.3f}.")
           
    if point_to_plane_registration:
        icp_registration_option = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        icp_registration_option = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    # transformation
    transformation = o3d.pipelines.registration.registration_icp(source, target,
                                                                 distance_threshold,
                                                                 init_transformation,
                                                                 icp_registration_option
                                                                 )
    
    return transformation

def Colored_ICP_refinement_registration(source, target,
                                        source_fpfh, target_fpfh,
                                        voxel_size,
                                        init_transformation,
                                        print_stats: bool = False):
    # radius
    radius = voxel_size
    
    # Colored ICP
    if print_stats:
        print(f"Colored ICP registration is applied on the original point clouds, with a radius of {radius:.3f}.")
    
    transformation = o3d.pipelines.registration.registration_colored_icp(source, target,
                                                                 radius,
                                                                 init_transformation,
                                                                 o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                                                                 o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                                                                   relative_rmse=1e-6,
                                                                                                                   max_iteration=50)
                                                                 )
    
    return transformation

def setup_lightglue(feature_extractor: LIGHTGLUE_Extractor,
                    max_num_keypoints: Optional[int] = 2048):
    if feature_extractor == LIGHTGLUE_Extractor.SUPERPOINT:
        # SuperPoint+LightGlue
        # load the extractor
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()

        # load the matcher
        matcher = LightGlue(features='superpoint').eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.DISK:
        # DISK+LightGlue
        # load the extractor
        extractor = DISK(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features='disk').eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.ALIKED:
        # ALIKED+LightGlue
        # load the extractor
        extractor = ALIKED(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features='aliked').eval().cuda()
    elif feature_extractor == LIGHTGLUE_Extractor.DOGHARDNET:
        # DoGHardNet+LightGlue
        # load the extractor
        extractor = DoGHardNet(max_num_keypoints=2048).eval().cuda()

        # load the matcher
        matcher = LightGlue(features='doghardnet').eval().cuda()
    else:
        raise NotImplementedError('The specified extractor is not implemented in LightGlue!')

    # speed optimization (Spikes in the computation time become more likely.)
    # matcher.compile(mode='reduce-overhead')
    
    # further speed optimization
    torch.set_float32_matmul_precision('medium')

    return extractor, matcher
    
def find_POI(img_rgb, npts, gray=True, viz=False, mask=None, detector=POI_Detector.SIFT): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_rgb2 = np.copy(img_rgb)
    if gray is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img_rgb[..., 0]

    if detector == POI_Detector.SIFT:
        #detector = cv2.SIFT_create(nfeatures=npts, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=15, sigma=1.)
        detector = cv2.SIFT_create(nfeatures=npts)
    elif detector == POI_Detector.ORB:
        detector = cv2.ORB_create(npts)
    elif detector == POI_Detector.SURF:
        detector = cv2.xfeatures2d.SURF_create(npts)
    else:
        raise ValueError(f'Detector: {detector} does not exist!')

        # # FAST Detector
        # detector = cv2.FastFeatureDetector_create()
        
        # # SURF Detector
        # detector = cv2.xfeatures2d.SURF_create(400)
        # keypoints = detector.detect(img, None)
        # descriptors = detector.compute(img, keypoints)

    keypoints, descriptors = detector.detectAndCompute(img, mask)

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    descriptors = np.array(descriptors)

    if gray is False:
        for i in range(2):
            keypoints_rgb, descriptors_rgb = detector.detectAndCompute(img_rgb[..., i+1], mask)

            xy_rgb = [keypoint.pt for keypoint in keypoints_rgb]
            xy_rgb = np.array(xy_rgb).astype(int)
            descriptors_rgb = np.array(descriptors_rgb)

            xy = np.concatenate([xy, xy_rgb], axis=0)
            descriptors = np.concatenate([descriptors, descriptors_rgb], axis=0)
            keypoints = list(keypoints)
            keypoints.extend(list(keypoints_rgb))
            keypoints = tuple(keypoints)

    if viz is True:
        kp_img = cv2.drawKeypoints(img_rgb2, keypoints, img_rgb2, color=(0,255,0))

    # Perform lex sort and get sorted data
    if len(xy) > 0:
        sorted_idx = np.lexsort(xy.T)
        xy =  xy[sorted_idx,:]
        descriptors = descriptors[sorted_idx,:]
        keypoints = [keypoints[i] for i in sorted_idx]
    else:
        print(f'xy: {xy}')
        print(f'keypoints:{keypoints}')
        raise RuntimeError('No keypoint was detected!')

    # Get unique row mask
    row_mask = np.append([True],np.any(np.diff(xy,axis=0),1))

    # Get unique rows
    xy = xy[row_mask]
    descriptors = descriptors[row_mask]
    keypoints = [keypoints[i] for i in np.where(row_mask)[0]]

    return xy, descriptors, keypoints # pixel coordinatess


def ratio_test(matches, thresh=0.95):
    # store all the good matches as per Lowe's ratio test.
    match_passed = []

    for match in matches:
        if len(match) == 2:
            m,n = match
        else:
            continue
        
        if m.distance < thresh*n.distance:
            match_passed.append(match)

    return  match_passed

def sym_test(matches12, matches21):
    good = []
    for m1,n1 in matches12:
        for m2, n2 in matches21:
            if (m1.queryIdx == m2.trainIdx) and (m1.trainIdx == m2.queryIdx):
                good.append(m1)

    return good

def feature_matching(des1, des2, detector=POI_Detector.SIFT):
    # FLANN parameters
    if detector == POI_Detector.SIFT or detector == POI_Detector.SURF:
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
    elif detector == POI_Detector.ORB:
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 0) #2
        search_params = dict(checks=10)   # or pass empty dictionary
    else:
        raise RuntimeError(f'Detector: {detector} does not exist!')

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    Matches = flann.knnMatch(des1,des2,k=2)

    # matches21 = flann.knnMatch(des2, des1, k=2)

    matches12 = ratio_test(Matches, thresh=0.9)

    #matches21 = ratio_test(matches21, thresh=0.8)

    #matches = sym_test(matches12, matches21)

    matches = [m1 for (m1, n1) in matches12]

    return matches, Matches

def skew_matrix(vec):
    batch_dims = vec.shape[:-1]
    S = torch.zeros(*batch_dims, 3, 3).to(vec.device)
    S[..., 0, 1] = -vec[..., 2]
    S[..., 0, 2] =  vec[..., 1]
    S[..., 1, 0] =  vec[..., 2]
    S[..., 1, 2] = -vec[..., 0]
    S[..., 2, 0] = -vec[..., 1]
    S[..., 2, 1] =  vec[..., 0]
    return S

def rot_matrix_to_vec(R):
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    def acos_safe(x, eps=1e-7):
        """https://github.com/pytorch/pytorch/issues/8069"""
        slope = np.arccos(1-eps) / eps
        # TODO: stop doing this allocation once sparse gradients with NaNs (like in
        # th.where) are handled differently.
        buf = torch.empty_like(x)
        good = abs(x) <= 1-eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
        return buf

    # angle = torch.acos((trace - 1) / 2)[..., None]
    angle = acos_safe((trace - 1) / 2)[..., None]
    # print(trace, angle)

    vec = (
        1
        / (2 * torch.sin(angle + 1e-10))
        * torch.stack(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ],
            dim=-1,
        )
    )

    # needed to overwrite nanes from dividing by zero
    vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

    # eg TensorType["batch_size", "views", "max_objects", 3, 1]
    rot_vec = (angle * vec)[...]

    return rot_vec

def vec_to_rot_matrix(rot_vec):
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)

    axis = rot_vec / (1e-10 + angle)
    S = skew_matrix(axis)
    # print(S.shape)
    # print(angle.shape)
    angle = angle[...,None]
    rot_matrix = (
            torch.eye(3).to(rot_vec.device)
            + torch.sin(angle) * S
            + (1 - torch.cos(angle)) * S @ S
            )
    return rot_matrix

def SE3_to_se3(pose):
    # pose in se(3)
    pose_se3 = torch.zeros(6).to(pose.device)
    
    # rotation component
    pose_se3[:3] = rot_matrix_to_vec(pose[:3, :3])
    
    # translation components
    pose_se3[3:] = pose[:3, 3]
    
    return pose_se3
 
def se3_to_SE3(pose):
    # pose in SE(3)
    pose_SE3 = torch.eye(4).to(pose.device)
    
    # rotation component
    pose_SE3[:3, :3] = vec_to_rot_matrix(pose[:3])
    
    # translation components
    pose_SE3[:3, 3] = pose[3:]
    
    return pose_SE3
    
def photometric_loss(nerf, pose, source_img, batch):
    # pose in SE(3)
    pose = se3_to_SE3(pose)
        
    # Batch of rays to compute loss on:
    # (x, y) pixel coordinates corresponding to rays we want to sample (N x 2).
    batch = torch.from_numpy(batch)
    
    # Coordinates for rendering from the NeRF:
    # in the form (y, x)
    img_coords = torch.zeros_like(batch)
    img_coords[:, 0] = batch[:, 1]
    img_coords[:, 1] = batch[:, 0]

    # Renders the subset image. 
    # Output also contains a depth channel for use with depth data if one chooses
    rendered_img = nerf.render(pose, img_coords=img_coords)["rgb"]
    
    # sample a subset of pixels from source image
    source_img = source_img[batch[:, 1], batch[:, 0]]

    # Computes MSE Loss
    loss_rgb = torch.mean((rendered_img - source_img) ** 2.0)

    return loss_rgb

def iNeRF_loss_fn(inputs):
    # inputs
    return photometric_loss(*inputs)

def execute_iNeRF(nerf, init_guess,
                  rgb_input,
                  feature_detector: POI_Detector,
                  dil_iteration: int = 3,
                  dil_kernel_size: Literal[5, 7, 9] = 5,
                  learning_rate: float = 1e-2,
                  convergence_threshold: float = 1e-3,
                  max_num_iterations: int = 100,
                  batch_size: int = 512,
                  save_rendered_image: bool = False,
                  rendered_img_filepath: Optional[str] = './inerf_guess',
                  visualize_inerf_keypoints: bool = False,
                  keypoint_img_filepath: Optional[str] = './keypoints.png',
                  print_stats: bool = False,
                  detector_params: Optional[Dict[str, Any]] = {}):
        # image dimensions
        H, W = rgb_input.shape[:2]
        
        # source image in the range [0, 255] (Image obtained from the camera.)
        source_img = (rgb_input.cpu().numpy() * 255).astype(np.uint8)
        
        # fing points of interest (keypoints) in the input
        # keypoints: (x, y) coordinates in the image (N, 2)
        source_xy, *_ = find_POI(source_img, 1000, viz=visualize_inerf_keypoints, detector=feature_detector)

        if print_stats:
            print(f'Found {source_xy.shape[0]} keypoints!')
            
        # raise an error if no keypoints are found
        if source_xy.shape[0] < 1:
            raise RuntimeError("No keypoints were found in the input image!")
        
        # Clone the RGB image.
        source_img = rgb_input.detach().clone()
        
        # identify the interest regions
        
        # image coordinate grid (H, W,)
        img_coords = np.stack(np.meshgrid(np.linspace(0, W - 1, W),
                                          np.linspace(0, H - 1, H)),
                                          axis=-1).astype(int)
        
        # coordinates mask for region sampling
        interest_regions = np.zeros((H, W), dtype=np.uint8)
        
        # create the mask for the interest regions
        interest_regions[source_xy[:, 1], source_xy[:, 0]] = 1
        
        # dilate the interest regions to expand the interest region
        interest_regions = cv2.dilate(interest_regions, 
                                      kernel=np.ones((dil_kernel_size, dil_kernel_size), dtype=np.uint8),
                                      iterations=dil_iteration)
        
        # Cast to bool.
        interest_regions = np.array(interest_regions, dtype=bool)
        
        # identify the (x, y) coordinates associated with the interest regions
        interest_regions = img_coords[interest_regions]
        
        # optimization variable for the pose
        optim_pose = SE3_to_se3(init_guess).detach().clone().requires_grad_(True)
        
        # optimizer
        optimizer = torch.optim.Adam(params=[optim_pose], lr=learning_rate, betas=(0.9, 0.999))
                
        for iter_idx in range(max_num_iterations):
            # cache the value at the previous iteration
            optim_pose_prev = optim_pose.detach().clone()
            
            optimizer.zero_grad()
            # randomly select a batch of rays from the interest regions for optimization
            rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
            batch = interest_regions[rand_inds]
            
            # compute the loss
            # AutoDiff Not Supported!
            # inputs = (nerf, optim_pose, source_img, batch)
            # loss = iNeRF_loss_fn(inputs)
            
            # compute the gradients
            # loss.backward()
            
            # compute the gradient of the loss function via finite differences
            grad_n = get_numerical_jacobian(iNeRF_loss_fn, (nerf,
                                                            optim_pose,
                                                            source_img,
                                                            batch
                                                            ),
                                            target=optim_pose, eps=1e-3,
                                            grad_out=1.0)
            
            # update the gradients
            optim_pose.grad = grad_n[0].view(-1,)
            
            # update the pose
            optimizer.step()
            
            # compute the error
            error = torch.linalg.norm(optim_pose - optim_pose_prev, ord=2)
            
            if error < convergence_threshold:
                # exit
                if print_stats:
                    print(f'iNeRF converged in {iter_idx + 1} iterations!')
                break
            
            if save_rendered_image:
                if iter_idx % 50 == 0:
                    # file path
                    rendered_img_filepath = Path(rendered_img_filepath)

                    # create directory, if needed
                    Path(rendered_img_filepath).parent.mkdir(parents=True, exist_ok=True)
        
                    # image file path
                    img_filename = f'{os.path.splitext(rendered_img_filepath)[0]}_{iter_idx}.png'
                    
                    r_pose = se3_to_SE3(optim_pose).detach().to(nerf.device)
                    
                    # render
                    outputs = nerf.render(r_pose,
                                        compute_semantics=False)
                    # figure
                    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
                    plt.tight_layout()

                    # plot rendering
                    axs[0].imshow(outputs['rgb'].cpu().numpy())
                    axs[1].imshow(outputs['depth'].cpu().numpy())

                    for ax in axs:
                        ax.set_axis_off()
                    plt.show()
                    
                    # save figure
                    fig.savefig(img_filename)
            
        return se3_to_SE3(optim_pose).detach().cpu().numpy()