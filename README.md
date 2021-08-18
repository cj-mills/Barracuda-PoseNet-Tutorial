# Barracuda PoseNet Tutorial 2nd Edition

### Introduction

This tutorial series provides step-by-step instructions for how to perform human [pose estimation](https://www.fritz.ai/pose-estimation/) in [Unity](https://unity.com/) with the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@2.1/manual/index.html) inference library. We will be using a pretrained [PoseNet](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5) model to estimate the 2D locations of key points on the bodies of one or more individuals in a [video frame](https://en.wikipedia.org/wiki/Film_frame). We will then use the output from the model to control the locations of [`GameObjects`](https://docs.unity3d.com/ScriptReference/GameObject.html) in a scene.

**Note:** I have only tested this project on Windows 10 with an Nvidia GPU. I do not have a Mac, but some readers have reported problems with using a webcam in Unity on Mac. There might also be some issues with GPU inference with the Metal Graphics API.

### Single Pose Estimation

![](https://github.com/cj-mills/christianjmills/raw/master/images/barracuda-posenet-tutorial-v2/part-7/single-pose-test.gif)

### Multi-Pose Estimation

![](https://github.com/cj-mills/christianjmills/raw/master/images/barracuda-posenet-tutorial-v2/part-7/multi-pose-test.gif)

### Demo Videos

* [Single Pose Estimation](https://youtu.be/KQyJgXss9NA)

* [Multi-Pose Estimation](https://youtu.be/F995ZadTZik)

## Tutorial Links

[Part 1](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-1/): This post covers the process for installing the Barracuda package as well as importing the required video files and PoseNet models into the project.

[Part 2](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-2/): This post covers how to set up a video player and webcam in Unity. We'll be using the video player to check the accuracy of the PoseNet model.

[Part 3](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-3/): This post covers how to implement the preprocessing steps for the PoseNet models.

[Part 4](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-4/): This post covers how to load, modify, and execute the PoseNet models.

[Part 5](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-5/): This post covers how to implement the post processing steps for single pose estimation.

[Part 6](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-6/): This post covers how to implement the post processing steps for multi-pose estimation.

[Part 7](https://christianjmills.com/Barracuda-PoseNet-Tutorial-V2-7/): This post covers how to create pose skeletons and manipulate them using output from the model.



### Version 1

[GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial/tree/Version-1)

