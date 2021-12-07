# PoseNet using Barracuda 2.3.1

The Unity project in this branch uses Barracuda version [2.3.1](https://docs.unity3d.com/Packages/com.unity.barracuda@2.3/changelog/CHANGELOG.html).  No code changes have been made so far.

Version 2.3.1 contains some improvements over version 2.1.0 used in the main branch that may be especially relevant when building for non-Windows platforms.

It has support for creating a tensor from a RenderTexture even when compute shaders or GPUs are not available.

It also now has a Pixel Shader worker. This allows models to run on GPU without requiring support for compute shaders. It has limited support at the moment, but seems to fully support the PoseNet models used in this project. My testing shows that it is much more performant than using the `CSharpBurst` worker type for both the `MobileNet` and `ResNet50` versions of the model.
