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

### 2.1 Fitting a Voxel Grid 
Here, wil generate randomly initalized voxel of size ```[b x h x w x d]``` and define **binary cross entropy (BCE)** loss that can help us fit a **3D binary voxel grid** using the ```Adam``` optimizer. 

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/e28d2d01-9c75-424d-b288-c9810ebac72c" width="50%" />
</p>

In a 3D voxel grid, a value of ```0``` indicates an **empty** cell, while ```1``` signifies an **occupied** cell. Thus, when fitting a voxel grid to a target, the process essentially involves a **logistic regression** problem aimed at ```maximizing the log-likelihood``` of the ground-truth label in each voxel. In summary, the loss function is the mean value of the voxel-wise binary cross entropies between the reconstructed object and the ground truth. In the equation below, N is the number of voxels in thr ground truth. ```y``` and ```y-hat``` is the predicted occupancy and the corresponding ground truth respectively. 

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

We train our data for 10000 iterations and observe the loss steadily decreases to about ```0.1```. This reflects effective learning and model optimization.

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

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/0f30d9b9-65d9-4156-8eae-e7b703f17172" width="80%" />
</p>

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

Secondly, we change our **decoder** to fit the architecture of the paper [Pix2Vox](https://arxiv.org/abs/1901.11153) which uses **3D de-convolutional network** (**transpose convolution**) to upsample ```1 x 1 x 1 ch``` to ```N x N x N x ch```. Note that the latent code is what is actually encoding the ```scene``` (the image) and decoding the latents and decoding the latent will give us a ```scene representation``` (3D model). The input of the decoder is of size ```[batch_size, 512]``` and the output of it is ```[batch_size x 32 x 32 x 32]```.

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

After training for ```3000``` epochs with a batch size of ```32``` and a learning rate of ```4e-4```, we achive a loss of ```0.395```.


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


In the first row are the **ground truths** and the second row is the **predicted voxels**.



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


### 2.4 Image to voxel grid

### 2.5 Fitting a Mesh

<p align="center">
  <img src="https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/40e4c04b-9a7c-49b6-b619-59885da478c1" width="50%" />
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





<table style="width:100%">
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

### 2.6 Image to Mesh


![mesh_ onnx](https://github.com/yudhisteer/Learning-3D-Vision-with-Inverse-Graphics/assets/59663734/726ea258-08de-4336-84f1-ff564fa6ae41)


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
