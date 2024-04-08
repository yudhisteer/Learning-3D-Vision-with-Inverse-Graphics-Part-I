import torch
import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer
import numpy as np
from tqdm import tqdm
import pytorch3d.structures
from utils import get_mesh_renderer
from PIL import Image

"""
Credit: https://github.com/learning3d/assignment1
"""



class MeshGifRenderer:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        self.vertices = vertices
        self.faces = faces
        self.color = [0.7, 0.7, 1]
        self.image_size = (224, 224)
        self.device = self.get_device()
        
    def get_device(self):
        """
        Get the device (CPU or GPU)
        """
        if torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        return device
    
    def gif_renderer(self, filename: str, num_views: int) -> None:
        
        # Create renderer
        renderer = get_mesh_renderer(
            image_size=self.image_size,
            device=self.device)
        
        # Create mesh
        vertices = self.vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        vertices -= vertices.mean(1, keepdims=True)  # Place vertices at origin
        faces = self.faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(self.color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(self.device)
        
        # Place a point light in front of mesh
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=self.device)
        
        # Set elevation and azimuth of the views
        elev = torch.ones(num_views) * num_views
        azim = torch.linspace(-180, 180, num_views)

        # Create corresponding camera extrinsics
        Rs, Ts = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.7, elev=elev, azim=azim, device=self.device)

        images = []
        for _, (R, T) in enumerate(tqdm(zip(Rs, Ts), total=num_views, desc="Rendering ...", colour="Green")):
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T.unsqueeze(0), fov=60, device=self.device)
            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend = rend.cpu().numpy()[0, ..., :3]  # (1, 3, H, W) -> (H, W, 3)
            rend = (rend * 255).astype(np.uint8)
            images.append(rend)

        # Convert numpy array images to PIL images
        pil_images = [Image.fromarray(img) for img in images]
        # Save the GIF using PIL
        pil_images[0].save(f"../Output/{filename}",
            save_all=True,
            append_images=pil_images[1:],
            loop=0,
            duration=1000 / 15,
        )


if __name__ == "__main__":
    
    # Triangle Mesh
    vertices = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    filename = "triangle_mesh.gif"
    num_views = 30
    triangle_mesh = MeshGifRenderer(vertices=vertices, faces=faces)
    triangle_mesh.gif_renderer(filename=filename, num_views=num_views)
    
    # Square Mesh
    vertices = torch.tensor([[0, 1, 1], [0, 1, 0], [1, 1, 0],[1, 1, 1], 
                             [0, 0, 0], [1, 0,0 ], [0, 0, 1], [1, 0, 1]], dtype=torch.float32)
    
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5 ,6], [5, 6, 7], [0, 1, 6], [1, 4 ,6], [2, 3 , 5], [3, 5, 7],
                          [0, 3, 6], [0, 3 ,7], [1, 2 ,5], [1, 2, 4]], dtype=torch.int64)
    
    faces_list = []
    num_views = 30

    for i, face in enumerate(faces): 
        filename = f"square_mesh_{i}.gif"
        faces_list.append(face.unsqueeze(0))
        triangle_mesh = MeshGifRenderer(vertices=vertices, faces=torch.cat(faces_list, dim=0))
        triangle_mesh.gif_renderer(filename=filename, num_views=num_views)