# Learning for 3D Vision with Inverse Graphics

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

### 1.1 Implicit Surfaces


### 1.2 Explicit Surfaces


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
