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

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0f30d9b9-65d9-4156-8eae-e7b703f17172" width="80%" />
</p>

### 2.1 Fitting a Voxel Grid 
To fit a voxel, we wil first generate a **randomly initalized** voxel of size ```[b x h x w x d]``` and define a **binary cross entropy (BCE)** loss that can help us fit a **3D binary voxel grid** using the ```Adam``` optimizer. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e28d2d01-9c75-424d-b288-c9810ebac72c" width="50%" />
</p>

In a 3D voxel grid, a value of ```0``` indicates an **empty** cell, while ```1``` signifies an **occupied** cell. Thus, when fitting a voxel grid to a target, the process essentially involves solving a **binary classification** problem aimed at ```maximizing the log-likelihood``` of the ground-truth label in each voxel. That is, we will be predicting an **occupancy score** for every point in the voxel grid and we compare that with the binary occupancy in our ground truths. 

In summary, the BCE loss function is the mean value of the voxel-wise binary cross entropies between the reconstructed object and the ground truth. In the equation below, ```N``` is the number of voxels in the ground truth. ```y``` and ```y-hat``` is the predicted occupancy and the corresponding ground truth respectively. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/58283911-ff26-46ba-a18b-de972e9a2533"/>
</p>

We will define a Binary Cross Entropy loss with logits which combines a Sigmoid layer and the BCELoss in one single class. The ```pos_weight``` factor calculates a **weightage** for occupied voxels based on the average value of the target voxels. By dividing 0.5 the weight **inversely** adjusts according to the frequency of occupied voxels in the data. This method addresses **class imbalances** where we have more unoccupied cells than occupied ones.

```python
def voxel_loss(voxel_src: torch.Tensor, voxel_tgt: torch.Tensor) -> torch.Tensor:
    # voxel_src: b x h x w x d
    # voxel_tgt: b x h x w x d
    pos_weight = (0.5 / voxel_tgt.mean())
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    loss = criterion(voxel_src, voxel_tgt)
    return loss
```

Below is the code to fit a voxel:

```python
# Generate voxel source with randomly initialized values
voxels_src = torch.rand(feed_cuda["voxels"].shape, requires_grad=True, device=args.device)

# Initialize optimizer to optimize voxel source
optimizer = torch.optim.Adam([voxels_src], lr=args.lr)

for step in tqdm(range(start_iter, args.max_iter)):
    # Calculate loss
    loss = voxel_loss(voxels_src, voxels_tgt)
    # Zero the gradients before backpropagation.
    optimizer.zero_grad()
    # Backpropagate the loss to compute the gradients.
    loss.backward()
    # Update the model parameters based on the computed gradients.
    optimizer.step()
```

We train our data for ```10000``` iterations and observe the loss steadily decreases to about ```0.1```.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/a54c54b4-2104-4101-af49-e8299255e49b" width="50%" />
</p>

Below are the visualization for the ```ground truth```, the ```fitted voxels```, and the ```optimization progress``` results.

<table>
  <tr>
    <th style="width:50%; text-align:center">Ground Truth</th>
    <th style="width:50%; text-align:center">Fitted</th>
    <th style="width:50%; text-align:center">Progress</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/56b1ecc8-c7e3-44bb-ab57-1cac9c4e0e49" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/b21dde0e-2d7b-48e8-af03-5222f1d08195" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/f63c2471-3cd4-49a7-97d5-ee08931fcdcd" width="100%" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d5b45a63-fff1-4626-8fcd-df8574fdb789" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d108d801-916f-4df6-9758-3de685454cee" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/be6c7249-56dc-44c8-b8bc-14494418620a" width="100%" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/dae78f84-59a7-4af7-aefc-cf1c2bf99c93" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/032affb2-78e5-439a-a36d-a928c2e150ad" width="100%" /></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/6b7e0163-8273-4a0b-9d17-e77f38a2f155" width="100%" /></td>
  </tr>
</table>



### 2.2 Image to voxel grid
Fitting a voxel grid is easy but now we want to 3D reconstruct a vocel grid from a single image only. For that, we will make use of an ```auto-encoder``` which first ```encode``` the **image** into **latent code** using a ```2D encoder```. We use a **pre-trained** ```ResNet-18``` model from ```torchvision``` to extract **features** from the image. The final classification layer is to make it a ```feature encoder```. Our image will be transformed to a ```latent code```.

Our input image is of size ```[batch_size, 137, 137, 3]```. The encoder transforms it into a latent code of size ```[batch_size, 512]```.  Next, we need to **reconstruct** the latent code into a voxel grid. For that, we first build a decoder using multi-layer perceptron (MLP) only as shown below.

```python
self.decoder = torch.nn.Sequential(
    nn.Linear(512, 1024),
    nn.PReLU(),
    nn.Linear(1024, 32*32*32)
)
```

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/3ba408e7-f03d-494d-9fa4-5c16c904a0bb" width="60%" />
</p>

Secondly, we change our **decoder** to fit the architecture of the paper [Pix2Vox](https://arxiv.org/abs/1901.11153) which uses **3D de-convolutional network** (**transpose convolution**) to upsample ```1 x 1 x 1 ch``` to ```N x N x N x ch```. Note that the latent code is what is actually encoding the ```scene``` (the image) and decoding the latents will give us a ```scene representation``` (3D model). The input of the decoder is of size ```[batch_size, 512]``` and the output of it is ```[batch_size x 32 x 32 x 32]```.

```python
self.fc = nn.Linear(512, 128 * 4 * 4 * 4)
self.decoder = nn.Sequential(
    nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm3d(64),
    nn.ReLU(),
    nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm3d(32),
    nn.ReLU(),
    nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm3d(8),
    nn.ReLU(),
    nn.Conv3d(8, 1, kernel_size=1),
    nn.Sigmoid()
)
```


<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/742d27ed-5258-4002-9894-4e07f9485312" width="120%" />
</p>

```python
# Set model to training mode
model.train()
# Initialize the Adam optimizer with model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Loop through the training steps
for step in range(start_iter, args.max_iter):
    # Restart the iterator when a new epoch begins
    if step % len(train_loader) == 0:
        train_loader = iter(loader)

    # Fetch the next batch of data
    feed_dict = next(train_loader)
    # Preprocess the data into the required format
    images_gt, ground_truth_3d = preprocess(feed_dict, args)  # [32, 137, 137, 3], [32, 1, 32, 32, 32]
    # Generate predictions from the model
    prediction_3d = model(images_gt, args)  # [32, 1, 32, 32, 32])  # voxels_pred
    # Calculate the loss based on predict
    loss = calculate_loss(prediction_3d, ground_truth_3d, args)
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Backpropagate to compute gradients
    loss.backward()
    # Update model parameters
    optimizer.step()
```

After training for ```3000``` epochs with a batch size of ```32``` and a learning rate of ```4e-4```, we achive a loss of ```0.395```. For some reason, we got worst result with the deconvolutional network. In the paper, the authors describe their decoder as a coarse voxel generator before passing it into a refiner. We will continue with the MLP network for evaluation.

<table style="width:100%">
  <tr>
    <th style="width:50%; text-align:center">Decoder with MLP</th>
    <th style="width:50%; text-align:center">Decoder with 3D De-conv</th>
  </tr>
  <tr>
    <td style="text-align:center"><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/22060226-ffb1-4f5c-8539-a713d218082b" style="width:100%"/></td>
    <td style="text-align:center"><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2e9b5a92-f5e5-4242-8f96-73ebe112b502" style="width:100%"/></td>
  </tr>
</table>

In the first row are the **single view image**, **ground truths** of the mesh and the second row is the **predicted voxels**.

<table style="width:100%">
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/abb9c391-372d-4910-b9f8-2e76cd88fe4f" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/7f2547fb-1dc3-4a09-ac72-9ab2142ed1c8" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4f267071-b1e0-4452-b24c-8c5c3fc88991" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/342587bd-e886-41d3-b76e-d2258fd3c2ba" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/86d07580-d209-47a2-8fb3-facb3122e083" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/3a19d1bb-8b55-402b-9844-5b7b289c791a" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/bdf7004f-6f9a-412c-9a52-18a562901080" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/1a9c5530-acad-4866-a593-b3b66081a5b3" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d1951b89-cb72-4053-a9c0-811d3a469a71" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/7af9c989-94e9-493e-be0c-c7aa670b394a" style="width:100%"/></td>
    <td></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/5c3f8d59-895b-4ac7-b9d8-daae55736224" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0b63e219-867f-4b03-9dbd-87fed4a74fa3" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/aad59035-1631-4307-8c0f-7c4f97ecca61" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/7dff48f0-1fa2-4849-ac4a-e3a76aea132d" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2d65ea1b-140d-4c21-ab5b-747bcee30ae3" style="width:100%"/></td>
    <td></td>
  </tr>
</table>


<!---
<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/a715cb5f-5516-412a-aba7-aa541ea796d5"/>
</p>
<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/63b1b4ff-9db6-4f0c-9814-2f93301f7543" width="20%" />
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/cafd3c8d-73e2-477e-835e-c2aec9bb8656" width="20%" />
</p>
--->

### 2.3 Fitting a Point Cloud
Similarly, to fitting a voxel, we generate a point cloud with random ```xyz``` values. We define the ```chamfer loss``` function that will allow us to fit the random point cloud into our target point cloud again using the ```Adam``` optimizer.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/be835f55-6da3-4b3e-8aa3-71aeea9d11b3" width="50%" />
</p>

Note that a point cloud represents a set of P points in 3D space. It can represent fine structures a huge numebr of poitns as we see below in the visualizations which uses ```1000``` points only. However, it does not explicitly represent the surface of the of a shape hence, we need to extract a mesh from the point cloud. I explain more about point cloud in my other projects: [Point Clouds: 3D Perception with Open3D](https://github.com/yudhisteer/Point-Clouds-3D-Perception-with-Open3D) and [Robotic Grasping Detection with PointNet](https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet).

To fit a point cloud, we need a differentiable way to compare pointclouds as sets because **order does not matter**! Therefore, we need a **permutation invariant** learning objective. We use the ```Chamfer distance``` which is the sum of L2 distance to each point's nearest neighbor in the other set. 

Suppose we have a gorund truth point cloud and a predicted point cloud. For each point in the ground truth set, we get its **nearest neighbor** in the predicted set, and their **Euclidean distance** is calculated. These distances are summed to form the first term of the equation below. Similarly, for each predicted point, the nearest ground truth point is found, and the distance to this neighbor is similarly summed to create the second term of the loss. The Chamfer loss is the total of these two sums, indicating the average mismatch between the two sets. A ```zero Chamfer loss```, signifying **perfect alignment**, occurs only when each point in one set **exactly coincides** with a point in the other set. Ibn summary, the chamfer loss guides the learning process by comparing the predicted point cloud against a ground truth set, regardless of the order of points in either set.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/737c31c4-f7a7-410c-b62b-e82e2dd148af" />
</p>

```python
from pytorch3d.ops import knn_points
def chamfer_loss(point_cloud_src: torch.Tensor, point_cloud_tgt: torch.Tensor) -> torch.Tensor:
    # point_cloud_src: b x n_points x 3  
    # point_cloud_tgt: b x n_points x 3  
    dist1 = knn_points(point_cloud_src, point_cloud_tgt)
    dist2 = knn_points(point_cloud_tgt, point_cloud_src)
    loss_chamfer = torch.sum(dist1.dists) + torch.sum(dist2.dists)
    return loss_chamfer
```

We train our data for ```10000``` iterations and observe the loss steadily decreases to about ```0.014``` Note that we have a lower loss compared to fitting a voxel.
<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/92ceddbd-e8a9-4e43-9d11-a8de9a2d232a" width="50%" />
</p>


Below are the visualization for the ```ground truth```, the ```fitted point cloud```, and the ```optimization progress``` results.

<table style="width:100%">
  <tr>
    <th style="width:50%; text-align:center">Ground Truth</th>
    <th style="width:50%; text-align:center">Fitted</th>
    <th style="width:50%; text-align:center">Progress</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/3c4dd82f-4d93-4b31-b24f-c18855b42308" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0427949e-8758-4d28-a3c8-f987349a9ddf" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/fef4f274-fc17-4105-921d-c254f97b1a18" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/252ad3b5-2d53-483b-9adb-5b91b843558e" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/10975277-c046-4550-b12e-3900719eac7b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/605cfad4-76f3-43b5-b5bc-b07abd621fda" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/bcabc41e-8671-40c0-922b-099bc0801010" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/6057f7dc-ec2d-44a3-8da7-2a5efe9cd6aa" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2df8826e-33a2-4565-869a-44cffbfb3135" style="width:100%"/></td>
  </tr>
</table>


### 2.4 Image to Point Cloud
For single view image to 3D reconstruction, we will use a similar approach to that for the image-to-voxelgrid as shown above. We will have the ResNet18 encode the image into a latent code and build an MLP to decode the latter into ```N x 3``` output. Recall that the output of the encoder is of size ```[batch_size x 512]``` and the output of the decoder will be of size ```[batch_size x n_points x 3]```. Note that explicit prediction yields a fixed size point cloud denoted as ```n_points``` here. 

Our MLP has starts with an input feature vector of size ```512```, the model employs a series of fully connected layers with increasing size—```1024```, ```2048```, and ```4096```—each followed by a **LeakyReLU** activation with a negative slope of ```0.1```. The final layer expands the output to ```n_points * 3```, where n_point is the number of points each representing three coordinates ```(x, y, z)```. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/9aaf18df-dfb7-4cfc-afc9-299aff4550c7" />
</p>

```python
# Input: b x 512
# Output: b x args.n_points x 3 # b x N x 3
self.n_point = args.n_points
self.decoder = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(1024, 2048),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(2048, 4096),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(4096, self.n_point * 3),
)
```

We train our model for ```3000``` epochs with ```n_points = 5000```. The loss curve depicts a rapid initial decrease followed by fluctuating stability at a ```0.1```.

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2351598c-b455-477e-80a3-e8cd931c349e" width="50%" />
</p>

In the first row are the **single view image**, **ground truths** of the mesh and the second row is the **predicted pointcloud**.

<table style="width:100%">
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/da6c2904-a621-4dfc-b131-e0e11640659b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/c72c8db4-a1c3-44bf-8f09-a5d1ae9235a0" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/bb1d6a5e-b278-477b-860e-ef7a3a6234e5" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/795aee60-b077-4753-8a34-dc0bda5cd613" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/fa03a3ec-bba8-4833-8f07-2e894b72b1e5" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/96fe276f-3799-44b0-a9e5-e0b13161692a" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/5be9208e-ccd6-4fa4-bd43-c8206b79fd15" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e19ff2d2-d41f-4431-a354-ab41bddd0d36" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4fb9ef2b-9af5-40e1-aa3c-77864eeec91b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/5d42a8f7-1300-4f26-96ac-b3bc9ed4b9c8" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/6de873b8-79b7-46fa-bf40-f7e3164a8b24" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/073860f1-243a-481a-ad88-54dade190c42" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/431bd2c8-87a7-414d-815f-3c7ee8fe80f4" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e4b9e5af-44cd-40ab-95ef-35d021cbd28b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/bd2e9863-a522-4e15-8e8f-34ed957a56d0" style="width:100%"/></td>
  </tr>
</table>

### 2.5 Fitting a Mesh

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/00729a94-fff9-4ec0-817e-c560ab54aea3" width="50%" />
</p>


<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/900a6749-bae1-41fe-b9cc-e814da213846" width="50%" />
</p>





<table style="width:100%">
  <tr>
    <th style="width:10%; text-align:center">Icosphere Level</th>
    <th style="width:50%; text-align:center">Ground Truth</th>
    <th style="width:50%; text-align:center">Fitted</th>
    <th style="width:50%; text-align:center">Progress</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e8601a26-7e81-4474-8b92-0d04f532321e" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/94cd8ed1-b838-4392-a242-a22cfe38fbcb" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/477e5d27-d60b-46cd-91da-f61798841209" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/5b37e5fa-61c0-40af-84a4-c20365b406f4" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/f88886c6-ab90-4c6d-b904-32e80124b6cb" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/b3af6ebc-b4b7-42c0-88f3-388d895607df" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/f737c539-ac02-4e14-86c1-089242e1de16" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/6c1dc90a-c326-4d4c-860d-ab8b4dc8f643" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/b5f89281-28de-4a09-95df-b7104c394883" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/a603dc1c-eebf-4c3c-8102-838a2204117c" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/1492a522-4000-4bb9-ac78-c99272c07957" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/80e16760-3793-4dea-9155-2eb2ebd35203" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0a51dec9-e113-4c99-b04d-51ffece37ef3" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/08844638-5ce4-4f28-ae34-35baf5556aa2" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/3a420ec6-f083-4f4b-8a55-9a19245751e2" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/37447573-acec-4ed4-a6dd-d6ae96ca5bd9" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d5129df0-4daf-4636-8ffe-1f700c49d646" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/dc9de70a-747a-4a0a-9a34-aa22bc112e17" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/545bdae9-4465-4893-8aa8-1b450c9f7916" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d63abc43-b337-43aa-9b60-a4db86915374" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/93e95a3d-4c97-4f3f-a90f-bc7ecaf8dd3b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/50573e49-48d0-4069-b941-10cdd4c00043" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/c9e6fc04-2b8e-4296-9106-aad604a79aa6" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/1c6e029e-f899-44b2-b883-1f862090fe4f" style="width:100%"/></td>
  </tr>
</table>




<table style="width:100%">
  <tr>
    <th style="width:50%; text-align:center">Ground Truth</th>
    <th style="width:50%; text-align:center">Fitted</th>
    <th style="width:50%; text-align:center">Progress</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/ec8de90e-910e-486b-9b27-20408bfb928f" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/ce71f740-922c-4e36-9dda-4ad0ae340049" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/42e26370-6ff4-4416-a588-c2c65d307afe" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/7d132686-7fe5-48b2-b643-3aea490851fb" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/562c38d7-93aa-4e53-8d30-2e3cea7af438" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/131f3ea9-a9a1-4c24-8ae9-6e1580e39fe8" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/21204ca5-3796-4f7a-8ee7-7b154f0fa60f" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e9b79f7b-ec1a-48ad-bbf1-1e2a36c3273a" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/ae3ba8ff-f531-4028-a420-942952b2b968" style="width:100%"/></td>
  </tr>
</table>








### 2.6 Image to Mesh



<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/726ea258-08de-4336-84f1-ff564fa6ae41" width="80%" />
</p>


<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d71effb6-24f6-4b8e-88d1-a8a3a739dc6f" width="50%" />
</p>



<table style="width:100%">
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/314adc3e-3b66-4159-9146-43f8753f8334" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/c260e63f-e127-4a3c-9337-60f9a2758944" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/d0815bca-14ba-4521-ac46-8688a9f0fe58" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/7074e87b-7bb4-4ed7-a169-814ba6c0520f" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/baac17d9-7490-48c5-a373-c20685034239" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/a282cfe3-f955-4d92-bf41-785bb4ab1c29" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/fa6b73b6-c2b2-437d-b07f-6886cd95200c" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/ec211c06-becd-46f6-a0ec-c8a4dcb9acf7" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e67ce0ed-5547-48f9-be4f-50ad812fa02d" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4a6c8c20-bb90-4bed-8c78-9e98c92d0729" style="width:100%"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/ad00cbce-7689-4fdb-acea-4a8bc7d62b62" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/14726123-deaf-46a8-8928-99595fefcc9c" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/4c01a652-e7db-447a-baa8-e472a723086c" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/256dfb01-878b-40ac-891c-dec8b6e5216b" style="width:100%"/></td>
    <td><img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/2d16c1fb-8f13-4f4c-b888-2938bbb56e19" style="width:100%"/></td>
  </tr>
</table>

-------------------------
<a name="ps"></a>
## 3. Photorealism Spectrum





-------------------------
<a name="dr"></a>
## 4. Differential Rendering


-------------------------
## References
1. https://www.andrew.cmu.edu/course/16-889/projects/
2. https://www.andrew.cmu.edu/course/16-825/projects/
3. https://www.educative.io/courses/3d-machine-learning-with-pytorch3d
4. https://towardsdatascience.com/how-to-render-3d-files-using-pytorch3d-ef9de72483f8
5. https://towardsdatascience.com/glimpse-into-pytorch3d-an-open-source-3d-deep-learning-library-291a4beba30f
6. https://www.youtube.com/watch?v=MOBAJb5nJRI
7. https://www.youtube.com/watch?v=v3hTD9m2tM8&t
8. https://www.youtube.com/watch?v=468Cxn1VuJk&list=PL3OV2Akk7XpDjlhJBDGav08bef_DvIdH2&index=4
9. https://github.com/learning3d
10. https://geometric3d.github.io/
11. https://learning3d.github.io/schedule.html
12. https://www.scenerepresentations.org/courses/inverse-graphics-23/
13. https://www-users.cse.umn.edu/~hspark/CSci5980/csci5980_3dvision.html
14. https://github.com/mint-lab/3dv_tutorial
15. https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/
16. https://www.youtube.com/watch?v=_M21DcHaMrg&list=PLZk0jtN0g8e_4gGYEpm1VYPh8xNka66Jt&index=6
17. https://learn.udacity.com/courses/cs291
18. https://madebyevan.com/webgl-path-tracing/
19. https://numfactory.upc.edu/web/Geometria/signedDistances.html
20. https://mobile.rodolphe-vaillant.fr/entry/86/implicit-surface-aka-signed-distance-field-definition
21. https://www.youtube.com/watch?v=KnUFccsAsfs&t=2512s
22. https://towardsdatascience.com/understanding-pytorch-loss-functions-the-maths-and-algorithms-part-2-104f19346425
23. https://towardsdatascience.com/3d-object-classification-and-segmentation-with-meshcnn-and-pytorch-3bb7c6690302
24. https://towardsdatascience.com/generating-3d-models-with-polygen-and-pytorch-4895f3f61a2e
25. https://www.youtube.com/watch?v=S1_nCdLUQQ8&t
