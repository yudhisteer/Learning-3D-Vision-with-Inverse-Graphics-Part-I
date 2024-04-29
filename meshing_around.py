import os
import numpy as np
import imageio
import math
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    TexturesUV,
)
import pytorch3d
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Any
import pickle


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_mesh_renderer(image_size: int = 512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """

    device = get_device()

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def load_obj_file(path):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, face_props, text_props = load_obj(path)
    faces = face_props.verts_idx
    return vertices, faces, text_props


def load_rgbd_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.

    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device),
        in_ndc=False,
        from_ndc=False,
        world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )

    return renderer


class MeshTextureRender:
    def __init__(
        self,
        vertices,
        mesh=None,
        face_props=None,
        text_props=None,
        texture_map=None,
        color1=[0.7, 0.7, 1.0],
        color2=None,
    ) -> None:    

        self.mesh = mesh
        self.vertices = vertices
        self.face_props = face_props
        self.text_props = text_props
        self.texture_map = texture_map
        self.color1 = color1
        self.color2 = color2
        self.device = get_device()
        



    

    @staticmethod
    def get_device():
        """
        Checks if GPU is available and returns device accordingly.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Device: ", device)
        return device

    def prepare_mesh(self):
        
        if getattr(self.face_props, 'verts_idx', None) is not None:
            verts_uvs = self.text_props.verts_uvs
            faces_uvs = self.face_props.textures_idx
            faces = self.face_props.verts_idx
        else:
            faces = self.face_props

        if self.texture_map is not None:
            textures = pytorch3d.renderer.TexturesUV(
                maps=torch.tensor([self.texture_map]),
                faces_uvs=faces_uvs.unsqueeze(0),
                verts_uvs=verts_uvs.unsqueeze(0),
            ).to(self.device)
        else:
            N = vertices.unsqueeze(0).shape[1]
            color1 = torch.tensor(self.color1).repeat(N, 1)
            texture_rgb = color1.unsqueeze(0)

            if self.color2 is not None:
                # Re-texturing with 2 colors
                color2 = torch.tensor(self.color2).repeat(N, 1)
                z_min = self.vertices.unsqueeze(0)[:, :, 2].min()
                z_max = vertices.unsqueeze(0)[:, :, 2].max()
                alpha = ((vertices.unsqueeze(0)[0, :, 2] - z_min) / (z_max - z_min)).unsqueeze(1)
                color = alpha * color2 + (1 - alpha) * color1
                texture_rgb = color.unsqueeze(0)
            textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

        if self.mesh is None:
            self.mesh = pytorch3d.structures.Meshes(
                verts=self.vertices.unsqueeze(0),
                faces=faces.unsqueeze(0),
                textures=textures,
            )
        self.mesh = self.mesh.to(self.device)
        return self.mesh

    def render(
        self,
        image_size: int = 512,
        fov: int = 60,
        R: torch.Tensor = torch.eye(3),  # [3, 3],
        T: torch.Tensor = torch.tensor([0, 0, 3]),  # [3]
    ) -> None:

        mesh = self.prepare_mesh()

        renderer = get_mesh_renderer(image_size=image_size, device=self.device)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.unsqueeze(0), T=T.unsqueeze(0), fov=fov, device=self.device
        )

        lights = pytorch3d.renderer.PointLights(
            location=[[0, 0, -3]], device=self.device
        )

        image = renderer(mesh, cameras=cameras, lights=lights)
        plt.imshow(image[0].cpu().numpy())
        plt.show()
        
        fig = plot_scene({"Mesh": {"Mesh": mesh, "Camera": cameras}})
        fig.show()

    def gif_render(
        self,
        elev: Any,
        azim: Any,
        image_size: int = 512,
        num_views: int = 30,
        dist: float = 2.7,
        fov: int = 60,
        FPS: int = 20,
        filename: str = "output.gif",
    ) -> None:

        mesh = self.prepare_mesh()

        renderer = get_mesh_renderer(image_size=image_size, device=self.device)

        Rs, Ts = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=dist, elev=elev, azim=azim, device=self.device
        )
        lights = pytorch3d.renderer.PointLights(
            location=[[0, 0, -3]], device=self.device
        )

        images = []
        for _, (R, T) in enumerate(
            tqdm(zip(Rs, Ts), total=num_views, desc="Rendering ...", colour="Green")
        ):
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R.unsqueeze(0), T=T.unsqueeze(0), fov=fov, device=self.device
            )
            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend = rend.cpu().numpy()[0, ..., :3]  # (1, 3, H, W) -> (H, W, 3)
            rend = (rend * 255).astype(np.uint8)
            images.append(rend)

        # Convert numpy array images to PIL images
        pil_images = [Image.fromarray(img) for img in images]

        # Save the GIF using PIL
        pil_images[0].save(
            f"output/{filename}",
            save_all=True,
            append_images=pil_images[1:],
            loop=0,
            duration=1000 / FPS,
        )
        print(f"GIF rendered successfully at {filename}!")





class PointCloudRender:
    def __init__(self, rgb_image=None, mask=None, depth=None, cameras=None, mesh=None, x=None, y=None, z=None) -> None:
        
        # Check input combinations
        if (rgb_image is None and mask is None and depth is None) and (mesh is None) and (x is None or y is None or z is None):
            raise ValueError("Insufficient data: Provide either 'rgb_image, mask, depth' or 'x, y, z'.")
        elif (rgb_image is not None and mask is not None and depth is not None) and (x is not None or y is not None or z is not None) and (mesh is not None):
            raise ValueError("Conflicting data: Provide only one set of inputs, not both.")
        
        self.rgb_image = rgb_image
        self.mask = mask
        self.depth = depth
        self.cameras = cameras
        self.mesh = mesh
        self.x = x
        self.y = y
        self.z = z
        self.device = get_device()
        


    @staticmethod
    def get_device():
        """
        Checks if GPU is available and returns device accordingly.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Device: ", device)
        return device

    def prepare_pointcloud(self):
        rgb, mask, depth = (
            torch.Tensor(self.rgb_image),
            torch.Tensor(self.mask),
            torch.Tensor(self.depth),
        )
        points, colors = unproject_depth_image(
            image=rgb, mask=mask, depth=depth, camera=self.cameras
        )
        point_cloud = pytorch3d.structures.Pointclouds(
            points=points.unsqueeze(0), features=colors.unsqueeze(0)
        ).to(self.device)
        
        return point_cloud

    def prepare_parametric(self):
        points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1) #[N*2, 3]
        color = (points - points.min()) / (points.max() - points.min()) #[N*2, 3]
        
        # Create point cloud
        point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(self.device)
        
        return point_cloud
    

    def mesh_to_pcd(self, N_pts=100):
        
        vertices = self.mesh.verts_list()[0]
        faces = self.mesh.faces_list()[0]
        
        # Compute areas of the faces and use them to create a sampling probability distribution
        areas = self.mesh.faces_areas_packed()
        prob = areas / areas.sum()
        
        # Sample face indices based on face area probabilities
        sampled_faces_ix = prob.multinomial(num_samples=N_pts, replacement=True)
        sampled_faces = faces[sampled_faces_ix]

        # Gather vertices of sampled faces
        sampled_verts = vertices[sampled_faces]  # N_f, 3 (v1, v2, v3), 3 (x, y, z)

        # Generate random barycentric coordinates
        alpha = torch.rand(N_pts)
        beta = (1 - alpha) * torch.rand(N_pts)
        gamma = 1 - alpha - beta
        abg = torch.stack([alpha, beta, gamma], dim=-1).unsqueeze(-1).to(self.device)

        # Calculating the new vertices based on random barycentric coordinates
        verts_new = (sampled_verts * abg).sum(1)
        
        # Normalize the vertices to a [0,1] scale for coloring
        normalized_verts = (verts_new - verts_new.min(0, keepdim=True)[0]) / \
                           (verts_new.max(0, keepdim=True)[0] - verts_new.min(0, keepdim=True)[0])
        
        point_cloud = pytorch3d.structures.Pointclouds(points=verts_new.unsqueeze(0), features=normalized_verts.unsqueeze(0)).to(self.device)
    
        return point_cloud

    def render_pointcloud(
        self,
        point_cloud: torch.Tensor,
        image_size: int = 512,
        fov: int = 60,
        R: torch.Tensor = torch.eye(3),  # [3, 3],
        T: torch.Tensor = torch.tensor([0, 0, 3]),
    ):

        # Create camera
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.unsqueeze(0), T=T.unsqueeze(0), fov=fov, device=self.device
        )

        # Create Renderer
        renderer = get_points_renderer(image_size=image_size, device=self.device)
        images = renderer(point_cloud, cameras=cameras)

        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.show()

        fig = plot_scene({"Pointcloud": {"person": point_cloud}})

        fig.show()

    def gif_render(
        self,
        point_cloud: torch.Tensor,
        elev = 0,
        azim = torch.linspace(-180, 180, 36),
        image_size: int = 512,
        num_views: int = 30,
        dist: float = 2.7,
        fov: int = 60,
        FPS: int = 20,
        filename: str = "pointcloud.gif",
    ) -> None:

        # Create Renderer
        renderer = get_points_renderer(image_size=image_size, device=self.device)

        Rs, Ts = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=dist, elev=elev, azim=azim, device=self.device
        )

        images = []
        for _, (R, T) in enumerate(
            tqdm(zip(Rs, Ts), total=num_views, desc="Rendering ...", colour="Green")
        ):
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R.unsqueeze(0), T=T.unsqueeze(0), fov=fov, device=self.device
            )
            rend = renderer(point_cloud, cameras=cameras)
            rend = rend.cpu().numpy()[0, ..., :3]  # (1, 3, H, W) -> (H, W, 3)
            rend = (rend * 255).astype(np.uint8)
            images.append(rend)

        # Convert numpy array images to PIL images
        pil_images = [Image.fromarray(img) for img in images]

        # Save the GIF using PIL
        pil_images[0].save(
            f"output/{filename}",
            save_all=True,
            append_images=pil_images[1:],
            loop=0,
            duration=1000 / FPS,
        )
        print(f"GIF rendered successfully at {filename}!")













if __name__ == "__main__":

    # Set the working directory
    os.chdir(os.path.join(os.getcwd(), "scripts"))
    print("New working directory:", os.getcwd())

    ### 1. Load .obj data
    filepath = os.path.join(os.getcwd(), "data", "cow.obj")
    mesh = pytorch3d.io.load_objs_as_meshes([filepath])
    vertices, face_props, text_props = load_obj(filepath)
    
    texture_map = plt.imread(
        os.path.join(os.getcwd(), "data", "cow_texture.png")
    )  # can be None
    
    # Parameters
    filename = "cow_1024.gif"
    num_views = 30
    color1 = [0.7, 0.7, 1.0]
    color2 = [1.0, 0.0, 0.0]  # can be None

    # 360 rotation
    azim = torch.linspace(0, 2 * np.pi, num_views) * 180 / np.pi - 180
    elev = 45 * torch.sin(torch.linspace(0, 2 * np.pi, num_views))

    # z-axis spin
    elev = 0
    azim = torch.linspace(-180, 180, 36)

    # Rotate mesh
    R_rel = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([0, np.pi / 2, 0]), "XYZ"
    )  # [3, 3]
    vertices_rotate = vertices @ R_rel  # [N_v, 3]

    # Rotate and translate camera
    R_rel = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([0.0, 0.0, 0.0]), "XYZ"
    )  # [3, 3]
    T_rel = torch.tensor([2 / 4, -2 / 4, 0])
    R = R_rel @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])  # [3, 3]
    T = R_rel @ torch.tensor([0.0, 0, 3]) + T_rel  # [3]
    
    ## Create mesh instance without texture_map
    # cow_mesh = MeshTextureRender(
    #     vertices=vertices,
    #     face_props=face_props,
    #     text_props=text_props,
    #     color1=color1,
    #     color2=color2,
    # )

    ## Create mesh instance with texture_map
    # cow_mesh = MeshTextureRender(
    #     vertices=vertices,
    #     face_props=face_props,
    #     text_props=text_props,
    #     texture_map=texture_map,
    # )

    ### Render 1 image
    #cow_mesh.render(R=R, T=T)

    ### Render GIF
    # cow_mesh.gif_render(elev=elev, azim=azim)







    ############ --------------------------------------------- ############

    ### 2. Load point cloud data
    filepath = os.path.join(os.getcwd(), "data", "rgbd_data.pkl")
    data = load_rgbd_data(filepath)
    keys = ["rgb1", "mask1", "depth1", "rgb2", "mask2", "depth2"]

    # Create a figure with 2 rows and 3 columns
    # fig, axes = plt.subplots(2, 3, figsize=(8, 15))

    # # Iterate over the keys and axes to plot the images
    # for ax, key in zip(axes.flatten(), keys):
    #     ax.imshow(data[key])
    #     ax.set_title(key)
    #     ax.axis('off')  # Turn off axis

    # plt.tight_layout()
    # plt.show()

    ## 2.1 Convert depth image to point cloud
    rgb1, mask1, depth1, cameras1 = (
        data["rgb1"],
        data["mask1"],
        data["depth1"],
        data["cameras1"],
    )
    rgb2, mask2, depth2, cameras2 = (
        data["rgb2"],
        data["mask2"],
        data["depth2"],
        data["cameras2"],
    )

    # Create instances of PointCloudRender
    plant_pcd_1 = PointCloudRender(
        rgb_image=rgb1, mask=mask1, depth=depth1, cameras=cameras1
    )
    plant_pcd_2 = PointCloudRender(
        rgb_image=rgb2, mask=mask2, depth=depth2, cameras=cameras2
    )

    # Parameters
    image_size = 1024
    dist = 9
    FPS = 20
    fov = 60
    num_views = 120
    R, T = pytorch3d.renderer.look_at_view_transform(8, 10, 0)  # [1, 3, 3]
    R_rel = pytorch3d.transforms.axis_angle_to_matrix(
        torch.tensor([[0.0, 0, 180 * math.pi / 180.0]])
    )  # [1, 3, 3]
    R = R_rel @ R  # [1, 3, 3]
    
    #Prepare point cloud
    point_cloud_1 = plant_pcd_1.prepare_pointcloud()
    point_cloud_2 = plant_pcd_2.prepare_pointcloud()

    # Render point cloud
    #plant_pcd_1.render_pointcloud(point_cloud=point_cloud_1, image_size=image_size, R=R, T=T)
    #plant_pcd_2.render_pointcloud(point_cloud=point_cloud_2, image_size=image_size, R=R, T=T)

    # z-axis spin
    elev = 0
    azim = torch.linspace(-180, 180, 36)
    # plant_pcd_2.gif_render(
    #     point_cloud=point_cloud,
    #     image_size=image_size,
    #     elev=elev,
    #     azim=azim,
    #     dist=dist,
    #     FPS=FPS,
    #     fov=fov,
    #     num_views=num_views,
    #     filename="pcd_2.gif",
    # )


    ############ --------------------------------------------- ############

    ### 3. Parametric function
    num_samples = 50 # N
    r = 1
    x_0 = 0
    y_0 = 0
    z_0 = 0

    phi = torch.linspace(0, 2 * np.pi, num_samples) # [N]
    theta = torch.linspace(0, 2 * np.pi, num_samples) # [N]

    Phi, Theta = torch.meshgrid(phi, theta, indexing="ij") # [N, N], [N, N]

    # Parametric function of sphere
    x = x_0 + r * torch.sin(Theta) * torch.cos(Phi) #[N, N]
    y = y_0 + r * torch.sin(Theta) * torch.sin(Phi) #[N, N]
    z = z_0 + r * torch.cos(Theta) #[N, N]
    
    # Parametric equation for torus
    R_torus = 10
    r_torus = 8
    x = (R_torus + r_torus * torch.cos(Theta)) * torch.cos(Phi)
    y = (R_torus + r_torus * torch.cos(Theta)) * torch.sin(Phi)
    z = r_torus * torch.sin(Theta)

    
    # Parameters
    elev = 0
    azim = torch.linspace(-180, 180, 36)
    dist = 60
    R, T = pytorch3d.renderer.look_at_view_transform(dist, 0, 0)  # [1, 3, 3]
    R_rel = pytorch3d.transforms.axis_angle_to_matrix(
        torch.tensor([[0.0, 0, 180 * math.pi / 180.0]])
    )  # [1, 3, 3]
    R = R_rel @ R  # [1, 3, 3]
    filename = "torus.gif"
    
    #Create instance of parametric point cloud
    parametric_pcd = PointCloudRender(x=x, y=y, z=z)
    # Prepare point cloud
    point_cloud = parametric_pcd.prepare_parametric()
    # Render point cloud
    #parametric_pcd.render_pointcloud(point_cloud=point_cloud, R=R, T=T)
    # Render gif
    #parametric_pcd.gif_render(point_cloud=point_cloud, elev=elev, azim=azim, dist=dist, filename=filename)
    



    ############ --------------------------------------------- ############

    ### 3. Implicit Surfaces
    import mcubes
    
    # Parameters
    color1 = [1., 0., 0.]
    color2 = [0.0, 0.0, 1.0]
    
    image_size=1024
    
    min_value = -1.6
    max_value = 1.6
    voxel_size = 90
    filename = f"torus_{voxel_size}.gif"
    
    R_torus = 0.6
    r_torus = R_torus / 3 * 2
    
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3) #[64, 64, 64]
    voxels = (torch.sqrt(X**2 + Y**2) - R_torus) ** 2 + Z**2 - r_torus**2 #[64, 64, 64]
    
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float() #[5352, 3]
    faces = torch.tensor(faces.astype(int)) # [10704, 3]
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value ##[5352, 3]
    

    # Create instance of implicit function
    implicit_fn = MeshTextureRender(
            vertices=vertices,
            face_props=faces,
            color1=color1,
            color2=color2,
                )
    
    ### Render 1 image
    #implicit_fn.render()

    ### Render GIF
    #implicit_fn.gif_render(image_size=image_size, elev=elev, azim=azim, filename=filename)
    





    ############ --------------------------------------------- ############

    ### 4. Sample points on mesh
    filepath = os.path.join(os.getcwd(), "data", "cow.obj")
    vertices, face_props, text_props = load_obj(filepath)
    color = [0.7, 0.7, 1]
    N_pts = 100000
    num_views = 36
    filename = f"cow_mesh_{N_pts}.gif"
    elev = 0
    azim = torch.linspace(-180, 180, num_views)
    
    # Create instance of mesh
    cow_mesh = MeshTextureRender(
        vertices=vertices,
        face_props=face_props,
        text_props=text_props,
        color1=color1
    )
    
    # Prepare mesh
    meshes = cow_mesh.prepare_mesh()
    
    # Create instance of pointcloud
    mesh_pcd = PointCloudRender(mesh=meshes)
    
    # Sample point on mesh
    point_cloud = mesh_pcd.mesh_to_pcd(N_pts=N_pts)
    mesh_pcd.render_pointcloud(point_cloud=point_cloud)
    mesh_pcd.gif_render(point_cloud=point_cloud, filename=filename, image_size=1024, num_views=120)


    
    print("Finish!")
    

 