import torch

def compute_bounding_box(path, rs, up):
    # path: N+1 x 3

    # Find the bounding box hyperplanes of the path

    # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
    # We artibrarily choose y and z accordingly.
    midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
    lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N
    lengths_x = lengths/2 + rs
    local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)
    local_y = torch.cross(up / torch.linalg.norm(up), local_x)   # This is the direction perpendicular to the path and the z-axis
    local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

    rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)

    # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
    # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
    xyz_lengths = torch.stack([lengths_x, lengths, lengths], dim=-1)
    intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + torch.stack([lengths_x, lengths, lengths], dim=-1)    # N x 3
    intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

    # Represent as x.T A <= b
    A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1)    # N x 3 x 6
    b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

    return A, b

# def bounding_box_and_points(self, path, point_cloud):
#     # path: N+1 x 3
#     # point_cloud: M x 3

#     # Find the bounding box hyperplanes of the path

#     # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
#     # We artibrarily choose y and z accordingly.
#     # NOTE: Might have to keepdims to the norms
#     midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
#     lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N
#     lengths_x = lengths + self.rs
#     local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)
#     local_y = torch.cross(self.up, local_x)   # This is the direction perpendicular to the path and the z-axis
#     local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

#     rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)

#     # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
#     # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
#     xyz_lengths = torch.stack([lengths_x, lengths, lengths], dim=-1)
#     intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + torch.stack([lengths_x, lengths, lengths], dim=-1)    # N x 3
#     intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

#     # Represent as x.T A <= b
#     A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1)    # N x 3 x 6
#     b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

#     # Now return points that are only within the bounding box
#     xA = torch.einsum('mk, nkl -> nml', point_cloud, A)    # N x M x 6

#     # In order for a pt to be within the bounding box, it must satisfy xA <= b, i.e. xA-b <= 0
#     mask = (xA - b[:, None, :] <= 0.)    # N x M x 6
#     keep_mask = mask.all(dim=-1)    # N x M      # For every path segment N, keep_pts is a boolean of every point in M if it is in the bounding box or not

#     # Returning just the points is not parallelizable because there can be different number of points for each bounding box, so we need to loop, selecting the points
#     # where the mask is True.
#     keep_points = []
#     for keep_per_n in keep_mask:
#         keep_points.append(point_cloud[keep_per_n])

#     # saves bounding box constraints for polytope definition
#     self.box_As = torch.transpose(A[0], 0, 1)
#     self.box_bs = b[0]

#     # Return both the bounding box half-space representation and the relevant points
#     output = {'A': A, 'b': b, 'midpoints': midpoints, 'keep_points': keep_points}
#     return output

class CollisionSet():
    def __init__(self, gsplat, vmax, amax, device):
        self.gsplat = gsplat
        self.vmax = vmax
        self.amax = amax
        self.device = device

        self.rs = vmax**2 / (2*amax)    # Safety radius

    def compute_set(self, path):

