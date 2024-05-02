# Learning 3D Vision with Inverse Graphics

## Plan of Action

1. [Meshing Around](#ma)
2. [Single View to 3D](#sv3d)
3. [Photorealism Spectrum](#ps)
4. [Differential Rendering](#dr)




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

Or we rotate the camera. Notice how the camera is now facing towards the z-axis:

```python
relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, np.pi/2, 0]), "XYZ") # [3, 3]
R_rotate = relative_rotation.unsqueeze(0) # [1, 3, 3]
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
Our dataset contains 3 images of the same plan. We have the RGB image, a depth map, a mask, and a Pytorch3D camera corresponding to the pose that the image was taken from. Frst, we want to convert the depth map int oa point cloud. For  that, we make use of the ```unproject_depth_image``` function which uses the camera intrinsics and extrinisics to cast a ray from every pixel in the image into world coordinates space. The ray's final distance is the depth value at that pixel, and the color of each point can be determined from the corresponding image pixel.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/27add765-3897-4b15-b847-146e0798a6bf" width="60%" />
</p>

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/a9b0ab66-b165-404f-87b0-c4747b76df6d" width="49%" />
  <img src="https://github.com/yudhisteer/Learning-for-3D-Vision-with-Inverse-Graphics/assets/59663734/447cfc5b-c8c6-4de1-a313-bc9ddfaa1e5e" width="49%" />
</p>


#### 1.2.2 Parametric Functions
We can define a 3D object as a **parameteric function** and sample points along its surface and render these points. If we were to define the equation of a sphere with center ```(x_0, y_0, z_0)``` and radius ```R```. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/b616a09b-9428-4323-82c8-d963b73244cd"/>
</p>

Now if we were to define the **parameteric function** of the sphere using the elevation angle (theta) and the azimuth angle (phi). Note that by sampling values of theta and phi, we can generate a sphere point cloud. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e9decac9-5f5b-4def-afd6-42c57686502e"/>
</p>

Below are the rendered point clouds where we sampled ```50```, ```300``` and ```1000``` points on the surface respectively.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/6a66b82b-e239-48ae-8bf1-1629f4fc40a7" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2c1d61fc-24be-463b-b14e-dab07c824b81" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/9dd55c63-539e-40de-a5a5-8f9b8cf74c36" width="30%" />
</p>




#### 1.2.3 Implicit Surfaces
An implicit function is a way to define a shape **without** explicitly listing its **coordinates**. The function ```F(x, y, z)``` describes the surface by its "**zero level-set**," which means all points ```(x, y, z)``` that satisfy ```F(x, y, z) = 0``` belong to the surface. 

To visualize a shape defined by an implicit function, we start by **discretizing** 3D space into a ```grid of voxels``` (**volumetric pixels**). We then evaluate the function ```F``` at each voxel's coordinates to determine whether each voxel should be part of the shape (i.e., does it satisfy the equation ```F = 0```?). The result of this process is stored in a voxel grid, a 3D array where each value indicates whether the corresponding voxel is inside or outside the shape.

To reconstruct the mesh, we use the **marching cubes algorithm**, which helps us **extract surfaces** at a specific threshold level (0-level set). In practice, we can create our voxel grid using ```torch.meshgrid```, which helps in setting up coordinates for each voxel in our space. We use these coordinates to evaluate our mathematical function. After setting up the voxel grid, we apply the ```mcubes``` library to transform this grid into a **triangle mesh**.

The implicit function for a torus:

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4092a638-5327-4f1d-8f0f-685ec2c6e7a6"/>
</p>

Below we have the torus with voxel size ```20```, ```30```, and ```80``` respectively.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/741c5bd9-2c44-4fcd-b346-5a4f85fa8ef6" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4e0465af-af8f-425c-815c-3f19069344cc" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4002533d-b898-45c1-88fa-2755e96e3ef6" width="30%" />
</p>

So how is these torus different from the point cloud ones? With implicit surfaces, we have **connectivity** between the vertices as compared to point clouds which has no connectivity.

#### 1.2.4 Sampling Points on Meshes 

One way to convert meshes into point clouds would be simply to use the **vertices**.But this can be problematic if the triangular mesh - **faces**- are of different sizes. A better method is **uniform sampling** of the surface through **stratified sampling**. Below is the process:

1. Choose a triangle to sample from based on its size; larger triangles (larger area) have a higher chance of being chosen.
2. Inside the chosen triangle, pick a random spot. This is done using something called **barycentric coordinates**, which help in defining a point in relation to the triangle’s corners.
3. Calculate the exact position of this random spot on the triangle to get a uniform spread of points across the entire mesh.

Below is an example whereby we take a triangle mesh and the number of samples and outputs a point cloud. We randomly sample ```1000```, ```10000```, and ```100000``` points respectively.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0ce2baa6-e279-4729-88cb-6652c793467d" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/bcf98e06-8a0f-4699-b19e-cae5d8ef4e5c" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/fcf169f1-fa94-4304-a687-1d59feafabf8" width="30%" />
</p>


-------------------------
<a name="sv3d"></a>
## 2. Single View to 3D

### 2.1 Learning to Predict Volumetric 3D

![image](https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/a54c54b4-2104-4101-af49-e8299255e49b)

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/56b1ecc8-c7e3-44bb-ab57-1cac9c4e0e49" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/b21dde0e-2d7b-48e8-af03-5222f1d08195" width="30%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/f63c2471-3cd4-49a7-97d5-ee08931fcdcd" width="30%" />
</p>

### 2.2 Image to voxel grid





### 2.2 Learning to Predict Point Clouds



### 2.3 Learning to Predict Meshes










-------------------------
<a name="ps"></a>
## 3. Photorealism Spectrum




-------------------------
<a name="dr"></a>
## 4. Differential Rendering



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
20. https://www.youtube.com/watch?v=KnUFccsAsfs&t=2512s
