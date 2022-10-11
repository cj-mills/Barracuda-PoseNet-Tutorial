# Barracuda PoseNet Tutorial 2nd Edition
## Tutorial Links
* [Part 1](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-1/): Install the Barracuda package in a Unity project and import the required video files and PoseNet models.
* [Part 2](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-2/): Set up a video player and webcam in Unity.
* [Part 3](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-3/): Implement the preprocessing steps for the MobileNet and ResNet PoseNet models.
* [Part 4](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-4/): Load, modify, and execute the PoseNet models.
* [Part 5](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-5/): Implement the post-processing steps for single pose estimation with PoseNet.
* [Part 6](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-6/): Implement the post-processing steps for multi-pose estimation with PoseNet.
* [Part 7](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-7/): Create pose skeletons and manipulate them using output from a PoseNet model.
* [WebGL](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/webgl/): Modify the Barracuda PoseNet project to run in a browser using WebGL.


### Single Pose Estimation

https://user-images.githubusercontent.com/9126128/191867974-7ba156d7-40a7-4f0d-a509-ed4521fe4d0d.mp4


### Multi-Pose Estimation

https://user-images.githubusercontent.com/9126128/191868000-b07938e3-7bcd-415b-a6cf-b29b11860bdd.mp4

---

## Update - 3/10/22
Added a new [branch](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/WebGL) and [tutorial](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/webgl/) for adapting the project to run in a browser using WebGL.

You can try out a live demo at the link below.
* [Barracuda PoseNet WebGL Demo](https://cj-mills.github.io/Barracuda-PoseNet-WebGL-Demo/)

---

## Update - 12/7/21
Added a new [branch](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/barracuda-2.3.1) that uses Barracuda version [2.3.1](https://docs.unity3d.com/Packages/com.unity.barracuda@2.3/changelog/CHANGELOG.html).
Version 2.3.1 contains some improvements over version 2.1.0 used in the main branch that may be especially relevant when building for non-Windows platforms.
<br><br>
It has support for creating a tensor from a RenderTexture even when compute shaders or GPUs are not available.
<br><br>
It also now has a Pixel Shader worker. This allows models to run on GPU without requiring support for compute shaders. It has limited support at the moment, but seems to fully support the PoseNet models used in this project. My testing shows that it is much more performant than using the `CSharpBurst` worker type for both the `MobileNet` and `ResNet50` versions of the model. Although, it is still not as performant as using compute shaders when those are available.

---

### Version 1

[GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/Version-1)

