# Barracuda PoseNet Tutorial 2nd Edition Pt. 5

> [!NOTE]
> **Retired from [christianjmills.com](https://christianjmills.com) on 2026-10-09.** This post was first published on 2021-07-29 at <https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-5/>, which now redirects here. It is kept as it was written; its tools and steps may be out of date.
>
> Source: [`posts/barracuda-posenet-tutorial-v2/part-5/index.md@8229e99`](https://github.com/cj-mills/christianjmills/blob/8229e991c0be7f8c78a909f0d8a2ca74390dc1e6/posts/barracuda-posenet-tutorial-v2/part-5/index.md)

*Implement the post-processing steps for single pose estimation with PoseNet.*

- [Overview](#overview)
- [Update `Utils` Script](#update-utils-script)
- [Update `PoseEstimator` Script](#update-poseestimator-script)
- [Summary](#summary)

## Overview

In this post, we will cover how to implement the post processing steps for single pose estimation. This method is much simpler than what is required to perform multi-pose estimation. However, it should only be used when there is a single person in the input image.

## Update `Utils` Script

We will implement the methods for processing the model output in the `Utils` script.

### Add Required Namespace

First, we need to add the `Unity.Barracuda` namespace since we will be working with Tensors.

``` c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
```

### Add Public Variables

Each key point predicted by the model has a confidence score, position, and id number associated with it. For example a nose has the id number `0`. We will define a new `struct` to keep track of these values for each key point.

``` c#
/// <summary>
/// Stores the heatmap score, position, and partName index for a single keypoint
/// </summary>
public struct Keypoint
{
    public float score;
    public Vector2 position;
    public int id;

    public Keypoint(float score, Vector2 position, int id)
    {
        this.score = score;
        this.position = position;
        this.id = id;
    }
}
```

### Create `GetOffsetVector` Method

Next, we will create a new method to obtain the offset values associated with a given heatmap coordinate. The method will take in the X and Y values for a heatmap coordinate, the current key point id number, and the offset values from the model output.

``` c#
/// <summary>
/// Get the offset values for the provided heatmap indices
/// </summary>
/// <param name="y">Heatmap column index</param>
/// <param name="x">Heatmap row index</param>
/// <param name="keypoint">Heatmap channel index</param>
/// <param name="offsets">Offsets output tensor</param>
/// <returns></returns>
public static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
{
    // Get the offset values for the provided heatmap coordinates
    return new Vector2(offsets[0, y, x, keypoint + 17], offsets[0, y, x, keypoint]);
}
```

### Create `GetImageCoords` Method

We can calculate the estimated location of a key point in the input image by multiplying the heatmap coordinate by the stride value for the model and then adding the associated offset values. We will calculate the stride value for the current model in the `PoseEstimator` script.

``` c#
 /// <summary>
/// Calculate the position of the provided key point in the input image
/// </summary>
/// <param name="part"></param>
/// <param name="stride"></param>
/// <param name="offsets"></param>
/// <returns></returns>
public static Vector2 GetImageCoords(Keypoint part, int stride, Tensor offsets)
{
    // The accompanying offset vector for the current coords
    Vector2 offsetVector = GetOffsetVector((int)part.position.y, (int)part.position.x,
                                           part.id, offsets);

    // Scale the coordinates up to the input image resolution
    // Add the offset vectors to refine the key point location
    return (part.position * stride) + offsetVector;
}
```

### Create `DecodeSinglePose` Method

This is the method that will be called from the `PoseEstimator` script after executing the model. It will take in the heatmaps and offsets from the model output along with the stride value for the model as input.

For single pose estimation, we will iterate through the heatmaps from the model output and keep track of the indices with the highest confidence value for each key point. Once we have the heatmap location with the highest confidence value, we can call the `GetImageCoords` method to calculate the position of the key point in the input image. We will store each key point in a `Keypoint` array.

> **Note:** This approach should only be used when there is a single person in the input image. It is unlikely that the key points with the highest confidence scores will belong to the same body when multiple people are visible.

``` c#
/// <summary>
/// Determine the estimated key point locations using the heatmaps and offsets tensors
/// </summary>
/// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
/// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
/// <returns>An array of keypoints for a single pose</returns>
public static Keypoint[] DecodeSinglePose(Tensor heatmaps, Tensor offsets, int stride)
{
    Keypoint[] keypoints = new Keypoint[heatmaps.channels];

    // Iterate through heatmaps
    for (int c = 0; c < heatmaps.channels; c++)
    {
        Keypoint part = new Keypoint();
        part.id = c;

        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.width; x++)
            {
                if (heatmaps[0, y, x, c] > part.score)
                {
                    // Update the highest confidence for the current key point
                    part.score = heatmaps[0, y, x, c];

                    // Update the estimated key point coordinates
                    part.position.x = x;
                    part.position.y = y;
                }
            }
        }

        // Calcluate the position in the input image for the current (x, y) coordinates
        part.position = GetImageCoords(part, stride, offsets);

        // Add the current keypoint to the list
        keypoints[c] = part;
    }

    return keypoints;
}
```

## Update `PoseEstimator` Script

In the `PoseEstimator` script, we need to add some new variables before we can call the `DecodeSinglePose` method.

### Add Public Variables

First, we will define a new `public enum` so that we can choose whether to perform single or multi-pose estimation from the inspector tab.

``` c#
public enum EstimationType
{
    MultiPose,
    SinglePose
}

[Tooltip("The type of pose estimation to be performed")]
public EstimationType estimationType = EstimationType.SinglePose;
```

### Add Private Variables

We will store the `Keypoint` arrays returned by the post processing methods in an array of `Keypoint` arrays. There will only be one array stored for single pose estimation, but there will be several for multi-pose estimation.

``` c#
// Stores the current estimated 2D keypoint locations in videoTexture
private Utils.Keypoint[][] poses;
```

### Create `ProcessOutput` Method

We will call the postprocessing methods inside a new method called `ProcessOutput`. This method will take in the `IWorker` from `engine`.

#### Method Steps

1.  Get the four model outputs

2.  Calculate the stride for the current model

3.  Call the appropriate post processing method for the selected estimation type

    > **Note:** We will fill in the `else` statement when we implement the post processing steps for multi-pose estimation.

4.  Release the resources allocated for the output Tensors.

``` c#
/// <summary>
/// Obtains the model output and either decodes single or mutlple poses
/// </summary>
/// <param name="engine"></param>
private void ProcessOutput(IWorker engine)
{
    // Get the model output
    Tensor heatmaps = engine.PeekOutput(predictionLayer);
    Tensor offsets = engine.PeekOutput(offsetsLayer);
    Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
    Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

    // Calculate the stride used to scale down the inputImage
    int stride = (imageDims.y - 1) / (heatmaps.shape.height - 1);
    stride -= (stride % 8);

    if (estimationType == EstimationType.SinglePose)
    {
        // Initialize the array of Keypoint arrays
        poses = new Utils.Keypoint[1][];

        // Determine the key point locations
        poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
    }
    else
    {
        
    }
    
    // Release the resources allocated for the output Tensors
    heatmaps.Dispose();
    offsets.Dispose();
    displacementFWD.Dispose();
    displacementBWD.Dispose();
}
```

### Modify `Update` Method

We will call the `ProcessOutput` method at the end of the `Update` method.

``` c#
// Decode the keypoint coordinates from the model output
ProcessOutput(engine.worker);
```

#### Full Code

``` c#
void Update()
{
    // Copy webcamTexture to videoTexture if using webcam
    if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);

    // Prevent the input dimensions from going too low for the model
    imageDims.x = Mathf.Max(imageDims.x, 64);
    imageDims.y = Mathf.Max(imageDims.y, 64);

    // Update the input dimensions while maintaining the source aspect ratio
    if (imageDims.x != targetDims.x)
    {
        aspectRatioScale = (float)videoTexture.height / videoTexture.width;
        targetDims.y = (int)(imageDims.x * aspectRatioScale);
        imageDims.y = targetDims.y;
        targetDims.x = imageDims.x;
    }
    if (imageDims.y != targetDims.y)
    {
        aspectRatioScale = (float)videoTexture.width / videoTexture.height;
        targetDims.x = (int)(imageDims.y * aspectRatioScale);
        imageDims.x = targetDims.x;
        targetDims.y = imageDims.y;
    }

    // Update the rTex dimensions to the new input dimensions
    if (imageDims.x != rTex.width || imageDims.y != rTex.height)
    {
        RenderTexture.ReleaseTemporary(rTex);
        // Assign a temporary RenderTexture with the new dimensions
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, rTex.format);
    }

    // Copy the src RenderTexture to the new rTex RenderTexture
    Graphics.Blit(videoTexture, rTex);

    // Prepare the input image to be fed to the selected model
    ProcessImage(rTex);

    // Reinitialize Barracuda with the selected model and backend 
    if (engine.modelType != modelType || engine.workerType != workerType)
    {
        engine.worker.Dispose();
        InitializeBarracuda();
    }

    // Execute neural network with the provided input
    engine.worker.Execute(input);
    // Release GPU resources allocated for the Tensor
    input.Dispose();

    // Decode the keypoint coordinates from the model output
    ProcessOutput(engine.worker);
}
```

## Summary

That is all we need to perform pose estimation when there is a single person in the input image. In the next post, we will implement the post processing steps for multi-pose estimation.

**Previous:** [Part 4](../part-4/README.md)

**Previous:** [Part 6](../part-6/README.md)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)
