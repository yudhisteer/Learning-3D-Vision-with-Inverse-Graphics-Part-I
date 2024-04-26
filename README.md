# Learning 3D Vision with Inverse Graphics

## Plan of Action

1. [Meshing Around](#ma)
2. [Photorealism Spectrum](#ps)
3. [Differential Rendering](#dr)




-------------------------
<a name="ma"></a>
## 1. Meshing Around
In  order to define a mesh, let's start with a ```point cloud``` which is an **unordered set of points** - ```{p_1, p_2, ..., p_N}```. When we represent a 3D model with a point cloud such as the sphere in red as shown below, we have no explicit connectivity information. Hence,  how do we answer the question: _How do we know if a point lies inside or outside the surface?_ Hence, the need for connectivity - **meshes**.

Meshes are ```piecewise linear approximations of the underlying surface```. Which means they are **discrete parametrizations** of a 3D scene. We start from our point cloud, now called **vertices**, joining them by **edges** to form **faces**. Thus, we establish **connectivity** by having ```3``` vertices to make a face. So now we need to answer again the question: _How do we know if a point lies inside or outside the surface?_ It turns out that now indeed we can answer this question due to the ```"watertight"``` property of meshes. That is, if we filled the mesh with water, we would have no leakage. Therefore, if our mesh is watertight, we can indeed define "inside" and "outside". 


<p align="center">
  <img src="https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/9a4a3334-cd07-4276-8a2d-b0b22574dddd" width="70%" />
</p>


Let's build our mesh with a base triangular polygon. We need to establish the vertices in ```x,y,z``` coordinates in a ```[3, 3]``` tensor and our faces in a ```[1, 3]```. Note that the elements in the face tensor are just the **indices** of the vertices tensor. However, PyTorch3D expects our tensor to be batched so we **unsqueeze** them later to become ```[1, 3, 3]``` and ```[1, 1, 3]``` respectively. We then use ```pytorch3d.structures.Meshes``` to create our mesh. The ```MeshGifRenderer``` class has a function to render our mesh from multiple viewpoints.

```python
# Triangle Mesh
vertices = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
filename = "triangle_mesh.gif"
num_views = 30
triangle_mesh = MeshGifRenderer(vertices=vertices, faces=faces)
triangle_mesh.gif_renderer(filename=filename, num_views=num_views)
```

<p align="center">
  <img src="https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/8aa00eb7-2e95-4a59-84b8-1502aec647aa" width="20%" />
</p>

### 1.1 Building mesh by mesh

Now that we have built a triangular mesh. We can use this as a base to create more complex 3D models such as a **cube**. Note that we need to use ```two``` sets of triangle faces to represent ```one``` face of the cube. Our cube will have ```8``` vertices and ```12``` triangular faces. Below is a step-by-step of joining all the 12 faces to form the final cube:



![square_mesh_0](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/5c2ffa90-5a6a-423e-8e49-6778bb92dbdf)
![square_mesh_1](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/4c93c08a-9af8-47b6-9bed-7f9b9c9de148)
![square_mesh_2](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/10b999ac-2477-42cc-9bfb-e4e4810fdd92)
![square_mesh_3](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/7826394f-9569-45dc-a3f8-299d8c7badef)
![square_mesh_4](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/c63cd921-fd99-4f10-96dd-4c5352bda481)
![square_mesh_5](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/8a155c59-9092-498e-a00b-800a8429db42)
![square_mesh_6](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/1605f495-7657-4042-b857-10646950fe00)
![square_mesh_7](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/154ee8e4-40dc-4988-9691-3c4d3c04b996)
![square_mesh_8](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/8240347a-96ed-4988-a5ce-63609862f752)
![square_mesh_9](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/0169c30c-ae4d-48b3-8fbd-352070a6741c)
![square_mesh_10](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/79857298-9029-4251-bce7-6ed8d13504d8)
![square_mesh_11](https://github.com/yudhisteer/Rendering-Basics-with-PyTorch3D/assets/59663734/64cc9fac-6f51-40a4-ab2e-092afc10844a)

### 1.1 Render Mesh with Texture
Although we showed how our 3D model are made up of triangular meshes, we kind of jump ahead in rendering a mesh. Now let's look at a step by step process of how we can import a ".obj" file, its texture from a ```.mtl``` file and render it.

#### 1.1.1 Load data
We first start by loading our data using the ```load_obj``` function from ```pytorch3d.io```. This returns the vertices of shape ```[N_v, 3]```, the ```face_props``` tuple which contains the **vertex indices** (**verts_idx**) of shape ```[N_f, 3]``` and **texture indices** (**textures_idx**) of similar shape ```[N_f, 3]```, and the ```aux``` tuple which contains the **uv coordinate per vertex** (**verts_uvs**) of shape ```[N_t, 2]```.

```python
vertices, face_props, aux = load_obj(data_file)
```

```python
print(vertices.shape) #[N_v, 3]

faces = face_props.verts_idx #[N_f, 3]
faces_uvs = face_props.textures_idx #[N_f, 3]

verts_uvs = text_props.verts_uvs #[N_t, 2]
```

Note that all Pytorch3D elements need to be batched.

```python
vertices = vertices.unsqueeze(0)  # [1 x N_v x 3]
faces = faces.unsqueeze(0)  # [1 x N_f x 3]
```

#### 1.1.2 Load Texture
Pytorch3d mainly supports 3 types of textures formats **TexturesUV**, **TexturesVertex** and **TexturesAtlas**. TexturesVertex has only one color per vertex. TexturesUV has rather one color per corner of a face. The 3D object file ```.obj``` directs to the material ```.mtl``` file and the material file directs to the texture ``.png``` file. So if we only have a ```.obj``` file we can still render our mesh using a texture of our choice as such:

```python
texture_rgb = torch.ones_like(vertices.unsqueeze(0)) # [1 x N_v X 3]
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
```

We use ```TexturesVertex``` to define a texture for the rendering:

```python
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)
```

However if we do have a texture map, we can load it as a normal image and visualize it:

```python
texture_map = plt.imread("cow_texture.png") #(1024, 1024, 3)
plt.imshow(texture_map)
plt.show()
```

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/d177293c-feab-46af-9eb1-ee5c5f63f4d7" width="40%" />
</p>


We then use ```TexturesUV``` which is an auxiliary datastructure for storing vertex uv and texture maps for meshes.

```python
textures = pytorch3d.renderer.TexturesUV(
                        maps=torch.tensor([texture_map]),
                        faces_uvs=faces_uvs.unsqueeze(0),
                        verts_uvs=verts_uvs.unsqueeze(0)).to(device)
```


#### 1.1.3 Create Mesh
Next, we create an instance of a mesh using ```pytorch3d.structures.Meshes```. Our arguments are the vertices and faces batched, and the textures.

```python
meshes = pytorch3d.structures.Meshes(
    verts=vertices.unsqueeze(0), # batched tensor or a list of tensors
    faces=faces.unsqueeze(0),
    textures=textures)
```

#### 1.1.4 Position a Camera
We want to be able to generate images of our 3D model so we set up a camera. Below are the 4 coordinate systems for 3D data:

1. **World Coordinate System**: The environment where the object or scene exists.
2. **Camera View Coordinate System**: Originates at the image plane with the Z-axis perpendicular to this plane, and orientations are such that +X points left, +Y points up, and +Z points outward. A rotation (R) and translation (T) transform this from the world system.
3. **NDC (Normalized Device Coordinate) System**: Normalizes the coordinates within a view volume, with specific mappings for the corners based on aspect ratios and the near and far planes. This transformation uses the camera projection matrix (P).
4. **Screen Coordinate System**: Maps the view volume to pixel space, where (0,0) and (W,H) represent the top left and bottom right corners of the viewable screen, respectively.


<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/38bc9210-6967-43cd-9854-c7b160a384d1" width="90%" />
</p>
<div align="center">
    <p>Image source: <a href="https://arxiv.org/abs/1612.00593">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></p>
</div>


We use the ```pytorch3d.renderer.FoVPerspectiveCameras``` function to generate a camera. Our 3D object lives in the world coordinates and we want to visualzie it in the image coordinates. We first need a **rotation** and **translation** matrix to build the **extrinsic matrix** of the camera, the **intrinsic matrix** will be supplied by PyTorch3D. 

```python
R = torch.eye(3).unsqueeze(0) # [1, 3, 3]
T = torch.tensor([[0, 0, 3]]) # [1, 3]

cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=R,
    T=T,
    fov=60,
    device=device)
```

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/246c18fe-64f7-4623-80ef-fe0e60e1552b" width="40%" />
</p>


Below we have the extrinsic matrix which consists of the translation and rotation matrix in **homogeneous** coordinates. 

```python
transform = cameras.get_world_to_view_transform()
print(transform.get_matrix()) # [1, 4, 4]
```

```python
tensor([[[ 1.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.],
         [ 0.,  0.,  1.,  0.],
         [ 0.,  0., 3.,  1.]]], device='cuda:0')
```
In the project [Pseudo-LiDARs with Stereo Vision](https://github.com/yudhisteer/Pseudo-LiDARs-with-Stereo-Vision), I explain more about the camera coordinate system:

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/63ce3160-35c1-4bda-94e7-1d1a8e58fa2c" width="50%" />
</p>

Now when rendering an image, we may experience that our rendered image is white because the camera is not face our mesh. We have 2 solutions for this: **move the mesh** or **move the camera**.

We rotate our mesh 90 degrees clockwise. Notice how the camera is always facing towards the x-axis.
```python
relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, np.pi/2, 0]), "XYZ") # [3, 3]
vertices_rotate = vertices @ relative_rotation # [N_v, 3]
```

<table>
  <tr>
    <th><b>Before rotation</b></th>
    <th><b>After rotation</b></th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/71b564b1-b3da-42bb-9c93-29c7f940fa91" alt="Image 1">
    </td>
    <td>
      <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/08e755f3-6cf9-4fff-a613-fc6ae9ab3439" alt="Image 2">
    </td>
  </tr>
</table>

Or we rotate the camera. Notice how the camera is now facing towards teh z-axis:

<table>
  <tr>
    <th><b>Before rotation</b></th>
    <th><b>After rotation</b></th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/71b564b1-b3da-42bb-9c93-29c7f940fa91" alt="Image 1">
    </td>
    <td>
      <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/9075d493-87a4-420b-bbf2-42a1b26d09be" alt="Image 2">
    </td>
  </tr>
</table>


#### 1.1.5 Create a renderer
To create a render we need a **rasterizer** which is given a pixel, which triangles correspond to it and a **shader**, that is, given triangle, texture, lighting, etc, how should the pixel be colored. 

```python
image_size = 512

# Rasterizer
raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
rasterizer = pytorch3d.renderer.MeshRasterizer(
    raster_settings=raster_settings)

# Shader
shader = pytorch3d.renderer.HardPhongShader(device=device)
```

```python
# Renderer
renderer = pytorch3d.renderer.MeshRenderer(
    rasterizer=rasterizer,
    shader=shader)
```


#### 1.1.6 Set up light
Our image will be pretty dark if we do not set up a light source in our world.

```python
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
```

#### 1.1.7 Render Mesh


```python
image = renderer(meshes, cameras=cameras, lights=lights)
plt.imshow(image[0].cpu().numpy())
plt.show()
```


<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/f554efe4-3a91-4faa-8f66-7ecdfbb7d405" width="40%" />
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/e228231f-4f51-4c53-bae2-c29bd23060db" width="40%" />
</p>


### 1.2 Rendering Generic 3D Representations

#### 1.2.1 Rendering Point Clouds from RGB-D Images



#### 1.2.2 Parametric Functions




#### 1.2.3 Implicit Surfaces



-------------------------
<a name="ps"></a>
## 2. Photorealism Spectrum




-------------------------
<a name="dr"></a>
## 3. Differential Rendering



-------------------------
## References
1. https://www.andrew.cmu.edu/course/16-889/projects/
2. https://www.educative.io/courses/3d-machine-learning-with-pytorch3d
3. https://towardsdatascience.com/how-to-render-3d-files-using-pytorch3d-ef9de72483f8
4. https://towardsdatascience.com/glimpse-into-pytorch3d-an-open-source-3d-deep-learning-library-291a4beba30f
5. https://www.youtube.com/watch?v=MOBAJb5nJRI
6. https://www.youtube.com/watch?v=v3hTD9m2tM8&t
7. https://www.youtube.com/watch?v=468Cxn1VuJk&list=PL3OV2Akk7XpDjlhJBDGav08bef_DvIdH2&index=4
8. https://github.com/learning3d
9. https://geometric3d.github.io/
10. https://learning3d.github.io/schedule.html
11. https://www.scenerepresentations.org/courses/inverse-graphics-23/
12. https://www-users.cse.umn.edu/~hspark/CSci5980/csci5980_3dvision.html
13. https://github.com/mint-lab/3dv_tutorial
14. https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/
15. https://www.youtube.com/watch?v=_M21DcHaMrg&list=PLZk0jtN0g8e_4gGYEpm1VYPh8xNka66Jt&index=6
16. https://learn.udacity.com/courses/cs291
17. https://madebyevan.com/webgl-path-tracing/
18. https://numfactory.upc.edu/web/Geometria/signedDistances.html
19. https://mobile.rodolphe-vaillant.fr/entry/86/implicit-surface-aka-signed-distance-field-definition
