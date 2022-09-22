# Barracuda PoseNet Tutorial 2nd Edition

## Update - 3/10/22
Added a new [branch](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/WebGL) and [tutorial](https://christianjmills.com/Barracuda-PoseNet-WebGL-Tutorial/) for adapting the project to run in a browser using WebGL.

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

### Introduction

This tutorial series provides step-by-step instructions for how to perform human [pose estimation](https://www.fritz.ai/pose-estimation/) in [Unity](https://unity.com/) with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@2.1/manual/index.html) inference library. We will be using a pretrained [PoseNet](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) model to estimate the 2D locations of key points on the bodies of one or more individuals in a [video frame](https://en.wikipedia.org/wiki/Film_frame). We will then use the output from the model to control the locations of [`GameObjects`](https://docs.unity3d.com/ScriptReference/GameObject.html) in a scene.

**Note:** I have only tested this project on Windows 10 with an Nvidia GPU. I do not have a Mac, but some readers have reported problems with using a webcam in Unity on Mac. There might also be some issues with GPU inference with the Metal Graphics API.

### Single Pose Estimation

https://user-images.githubusercontent.com/9126128/191867974-7ba156d7-40a7-4f0d-a509-ed4521fe4d0d.mp4




### Multi-Pose Estimation

https://user-images.githubusercontent.com/9126128/191868000-b07938e3-7bcd-415b-a6cf-b29b11860bdd.mp4


## Tutorial Links

[Part 1](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-1/): This post covers the process for installing the Barracuda package as well as importing the required video files and PoseNet models into the project.

[Part 2](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-2/): This post covers how to set up a video player and webcam in Unity. We'll be using the video player to check the accuracy of the PoseNet model.

[Part 3](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-3/): This post covers how to implement the preprocessing steps for the PoseNet models.

[Part 4](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-4/): This post covers how to load, modify, and execute the PoseNet models.

[Part 5](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-5/): This post covers how to implement the post processing steps for single pose estimation.

[Part 6](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-6/): This post covers how to implement the post processing steps for multi-pose estimation.

[Part 7](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-7/): This post covers how to create pose skeletons and manipulate them using output from the model.

[WebGL](https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/webgl/): This post covers how to modify the Barracuda PoseNet project to run in the browser using WebGL.


### Version 1

[GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/Version-1)

