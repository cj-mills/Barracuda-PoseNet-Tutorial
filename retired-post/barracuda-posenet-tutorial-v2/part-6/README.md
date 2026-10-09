# Barracuda PoseNet Tutorial 2nd Edition Pt. 6

> [!NOTE]
> **Retired from [christianjmills.com](https://christianjmills.com) on 2026-10-09.** This post was first published on 2021-07-30 at <https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-6/>, which now redirects here. It is kept as it was written; its tools and steps may be out of date.
>
> Source: [`posts/barracuda-posenet-tutorial-v2/part-6/index.md@8229e99`](https://github.com/cj-mills/christianjmills/blob/8229e991c0be7f8c78a909f0d8a2ca74390dc1e6/posts/barracuda-posenet-tutorial-v2/part-6/index.md)

*Implement the post-processing steps for multi-pose estimation with PoseNet.*

- [Overview](#overview)
- [Update `Utils` Script](#update-utils-script)
- [Update `PoseEstimator` Script](#update-poseestimator-script)
- [Summary](#summary)

## Overview

In this post, we will cover how to implement the post processing steps for multi-pose estimation. This method is more complex than what is required to perform single pose estimation. However, it can produce more reliable results.

> **Note:** The original JavaScript code for decoding multiple poses can be found in the official [tfjs-models](https://github.com/tensorflow/tfjs-models/tree/master/posenet/src/multi_pose) repository on GitHub. The code has been modified for this tutorial to better take advantage of functionality provided by Unity and [.NET](https://docs.microsoft.com/en-us/dotnet/).

## Update `Utils` Script

There are a couple new variables and several methods that we will need to add to decode multiple poses from the model output.

### Add Required Namespace

First, we need to add the [`System`](https://docs.microsoft.com/en-us/dotnet/api/system?view=net-5.0) namespace to access the [`Tuple`](https://docs.microsoft.com/en-us/dotnet/api/system.tuple-2?view=net-5.0) class. We also need to access the [`System.Linq`](https://docs.microsoft.com/en-us/dotnet/api/system.linq?view=net-5.0) namespace to access classes and interfaces for querying data structures.

``` c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Linq;
```

### Add Public Variables

When iterating through the heatmaps from the model output, we will only be considering heatmap indices with the highest confidence score within a local radius called `kLocalMaximumRadius`. Naturally, setting this value to be larger than the dimensions of the heatmap would not do any good. The [original code](https://github.com/tensorflow/tfjs-models/blob/c3f5aa3ff74787457082f6683655c9d0c7cf3df7/posenet/src/multi_pose/decode_multiple_poses.ts#L54) sets this radius to a constant value of `1` so we will do the same.

When decoding the key point locations for a single body, we will need to traverse from the current key point to its neighboring key point. For example, the nose neighbors both the left eye and right eye. We will keep track of which key points neighbor each other in a `TupleTuple<int, int>` array, where the values are the key point id numbers.

``` c#
/// <summary>
/// Defines the size of the local window in the heatmap to look for
/// confidence scores higher than the one at the current heatmap coordinate
/// </summary>
const int kLocalMaximumRadius = 1;

/// <summary>
/// Defines the parent->child relationships used for multipose detection.
/// </summary>
public static Tuple<int, int>[] parentChildrenTuples = new Tuple<int, int>[]{
    // Nose to Left Eye
    Tuple.Create(0, 1),
    // Left Eye to Left Ear
    Tuple.Create(1, 3),
    // Nose to Right Eye
    Tuple.Create(0, 2),
    // Right Eye to Right Ear
    Tuple.Create(2, 4),
    // Nose to Left Shoulder
    Tuple.Create(0, 5),
    // Left Shoulder to Left Elbow
    Tuple.Create(5, 7),
    // Left Elbow to Left Wrist
    Tuple.Create(7, 9), 
    // Left Shoulder to Left Hip
    Tuple.Create(5, 11),
    // Left Hip to Left Knee
    Tuple.Create(11, 13), 
    // Left Knee to Left Ankle
    Tuple.Create(13, 15),
    // Nose to Right Shoulder
    Tuple.Create(0, 6), 
    // Right Shoulder to Right Elbow
    Tuple.Create(6, 8),
    // Right Elbow to Right Wrist
    Tuple.Create(8, 10), 
    // Right Shoulder to Right Hip
    Tuple.Create(6, 12),
    // Right Hip to Right Knee
    Tuple.Create(12, 14), 
    // Right Knee to Right Ankle
    Tuple.Create(14, 16)
};
```

### Create `GetStridedIndexNearPoint` Method

In order to traverse from a key point to its neighboring key point, we will need to downscale the key point position back down to the heatmap resolution. We can calculate the nearest heatmap indices by dividing the position by the stride value for the model and clamping the result.

``` c#
/// <summary>
/// Calculate the heatmap indices closest to the provided point
/// </summary>
/// <param name="point"></param>
/// <param name="stride"></param>
/// <param name="height"></param>
/// <param name="width"></param>
/// <returns>A vector with the nearest heatmap coordinates</returns>
static Vector2Int GetStridedIndexNearPoint(Vector2 point, int stride, int height, int width)
{
    // Downscale the point coordinates to the heatmap dimensions
    return new Vector2Int(
        (int)Mathf.Clamp(Mathf.Round(point.x / stride), 0, width - 1),
        (int)Mathf.Clamp(Mathf.Round(point.y / stride), 0, height - 1)
    );
}
```

### Create `GetDisplacement` Method

The displacement layers from the model output are used to find the location of the nearest neighboring key point. Much like the offset layer, they provide vectors that we then add to the current key point position.

``` c#
/// <summary>
/// Retrieve the displacement values for the provided point
/// </summary>
/// <param name="edgeId"></param>
/// <param name="point"></param>
/// <param name="displacements"></param>
/// <returns>A vector witht he displacement values for the provided point</returns>
static Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
{
    // Calculate the number of edges for the pose skeleton
    int numEdges = (int)(displacements.channels / 2);
    // Get the displacement values for the provided heatmap coordinates
    return new Vector2(
        displacements[0, point.y, point.x, numEdges + edgeId],
        displacements[0, point.y, point.x, edgeId]
    );
}
```

### Create `TraverseToTargetKeypoint` Method

We can use the `GetStridedIndexNearPoint` and `GetDisplacement` methods to find the location of the neighboring key point for a given `Keypoint`.

#### Method Steps

1.  Get the nearest heatmap indices for the current key point position

2.  Get the displacement vector for the nearest heatmap indices

3.  Calculate the position for a neighboring key point using the displacement vector

4.  Get the nearest heatmap indices for the displaced point

5.  Refine the location key point location with the associated offset vector

6.  Get the confidence score for the neighboring key point

7.  Return the neighboring `Keypoint`

``` c#
/// <summary>
/// Get a new keypoint along the provided edgeId for the pose instance.
/// </summary>
/// <param name="edgeId"></param>
/// <param name="sourceKeypoint"></param>
/// <param name="targetKeypointId"></param>
/// <param name="scores"></param>
/// <param name="offsets"></param>
/// <param name="stride"></param>
/// <param name="displacements"></param>
/// <returns>A new keypoint with the displaced coordinates</returns>
static Keypoint TraverseToTargetKeypoint(
    int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
    Tensor scores, Tensor offsets, int stride,
    Tensor displacements)
{
    // Get heatmap dimensions
    int height = scores.height;
    int width = scores.width;

    // Get neareast heatmap indices for source keypoint
    Vector2Int sourceKeypointIndices = GetStridedIndexNearPoint(
        sourceKeypoint.position, stride, height, width);
    // Retrieve the displacement values for the current indices
    Vector2 displacement = GetDisplacement(edgeId, sourceKeypointIndices, displacements);
    // Add the displacement values to the keypoint position
    Vector2 displacedPoint = sourceKeypoint.position + displacement;
    // Get neareast heatmap indices for displaced keypoint
    Vector2Int displacedPointIndices =
        GetStridedIndexNearPoint(displacedPoint, stride, height, width);
    // Get the offset vector for the displaced keypoint indices
    Vector2 offsetVector = GetOffsetVector(
        displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
        offsets);
    // Get the heatmap value at the displaced keypoint location
    float score = scores[0, displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];
    // Calculate the position for the displaced keypoint
    Vector2 targetKeypoint = (displacedPointIndices * stride) + offsetVector;

    return new Keypoint(score, targetKeypoint, targetKeypointId);
}
```

### Create `DecodePose` Method

We don’t know which key point (e.g. nose, left shoulder, right wrist) we will start from when decoding a single pose. Therefore, we will need to traverse the list of neighboring key points both forwards and backwards to get all 17 of the key points for an individual in the input image.

#### Method Steps

1.  Initialize a new `Keypoint` array

2.  Get the input image coordinates for the starting key point

3.  Store the starting key point in the `Keypoint` array

4.  Iterate upwards through the list of neighboring key points

    1.  Confirm that the current child key point has already been found and that the parent key point has *not* already been found.
        1.  Call the `TraverseToTargetKeypoint` method to obtain neighboring key points
        2.  Store each neighboring key point in the `Keypoint` array according to its id number

5.  Iterate downwards through the list of neighboring key points

    1.  Confirm that the current parent key point has already been found and that the child key point has *not* already been found.
        1.  Call the `TraverseToTargetKeypoint` method to obtain neighboring key points
        2.  Store each neighboring key point in the `Keypoint` array according to its id number

6.  Return the `Keypoint` array

``` c#
/// <summary>
/// Follows the displacement fields to decode the full pose of the object
/// instance given the position of a part that acts as root.
/// </summary>
/// <param name="root"></param>
/// <param name="scores"></param>
/// <param name="offsets"></param>
/// <param name="stride"></param>
/// <param name="displacementsFwd"></param>
/// <param name="displacementsBwd"></param>
/// <returns>An array of keypoints for a single pose</returns>
static Keypoint[] DecodePose(Keypoint root, Tensor scores, Tensor offsets,
                             int stride, Tensor displacementsFwd, Tensor displacementsBwd)
{

    Keypoint[] instanceKeypoints = new Keypoint[scores.channels];

    // Start a new detection instance at the position of the root.
    Vector2 rootPoint = GetImageCoords(root, stride, offsets);

    instanceKeypoints[root.id] = new Keypoint(root.score, rootPoint, root.id);

    int numEdges = parentChildrenTuples.Length;

    // Decode the part positions upwards in the tree, following the backward
    // displacements.
    for (int edge = numEdges - 1; edge >= 0; --edge)
    {
        int sourceKeypointId = parentChildrenTuples[edge].Item2;
        int targetKeypointId = parentChildrenTuples[edge].Item1;
        if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
            instanceKeypoints[targetKeypointId].score == 0.0f)
        {
            instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, stride, displacementsBwd);
        }
    }

    // Decode the part positions downwards in the tree, following the forward
    // displacements.
    for (int edge = 0; edge < numEdges; ++edge)
    {
        int sourceKeypointId = parentChildrenTuples[edge].Item1;
        int targetKeypointId = parentChildrenTuples[edge].Item2;
        if (instanceKeypoints[sourceKeypointId].score > 0.0f &&
            instanceKeypoints[targetKeypointId].score == 0.0f)
        {
            instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, stride, displacementsFwd);
        }
    }

    return instanceKeypoints;
}
```

### Create `ScoreIsMaximumInLocalWindow` Method

As mentioned earlier, we only consider key points with the highest confidence score in their local area as potential starting key points. We will determine whether a given key point has the highest score in a new method called `ScoreIsMaximumInLocalWindow`.

#### Method Steps

1.  Calculate the starting and ending indices for the local heatmap window

2.  Iterate through the heatmap indices within the local window

3.  Compare each confidence score in the local window to the score for the provided key point

    1.  Return `false` if any higher scores are found

``` c#
/// <summary>
/// Compare the value at the current heatmap location to the surrounding values
/// </summary>
/// <param name="keypointId"></param>
/// <param name="score"></param>
/// <param name="heatmapY"></param>
/// <param name="heatmapX"></param>
/// <param name="localMaximumRadius"></param>
/// <param name="scores"></param>
/// <returns>True if the value is the highest within a given radius</returns>
static bool ScoreIsMaximumInLocalWindow(int keypointId, float score, int heatmapY, int heatmapX,
                                        int localMaximumRadius, Tensor heatmaps)
{
    bool localMaximum = true;
    // Calculate the starting heatmap colummn index
    int yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
    // Calculate the ending heatmap colummn index
    int yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, heatmaps.height);

    // Iterate through calulated range of heatmap columns
    for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
    {
        // Calculate the starting heatmap row index
        int xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
        // Calculate the ending heatmap row index
        int xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, heatmaps.width);

        // Iterate through calulated range of heatmap rows
        for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
        {
            // Check if the score for at the current heatmap location
            // is the highest within the specified radius
            if (heatmaps[0, yCurrent, xCurrent, keypointId] > score)
            {
                localMaximum = false; 
                break;
            }
        }
        if (!localMaximum) break;
    }
    return localMaximum;
}
```

### Create `BuildPartList` Method

This is where we will build the list of potential starting key points that be passed to the `DecodePose` method.

Much like the `DecodeSinglePose` method, we need to iterate through the entire heatmap Tensor. This time, we will only consider heatmap indices with a value above the provided score threshold. When we get to an index with a value that meets this threshold, we will call the `ScoreIsMaximumInLocalWindow` method to confirm that it is the highest score in its local area. The heatmap indices with the highest local score will be added to a `Keypoint` [`List`](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.list-1?view=net-5.0).

``` c#
/// <summary>
/// Iterate through the heatmaps and create a list of indicies 
/// with the highest values within the provided radius.
/// </summary>
/// <param name="scoreThreshold"></param>
/// <param name="localMaximumRadius"></param>
/// <param name="scores"></param>
/// <returns>A list of keypoints with the highest values in their local area</returns>
static List<Keypoint> BuildPartList(float scoreThreshold, int localMaximumRadius, Tensor heatmaps)
{
    List<Keypoint> list = new List<Keypoint>();

    // Iterate through heatmaps
    for (int c = 0; c < heatmaps.channels; c++)
    {
        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.width; x++)
            {
                float score = heatmaps[0, y, x, c];

                // Skip parts with score less than the scoreThreshold
                if (score < scoreThreshold) continue;

                // Only add keypoints with the highest score in a local window.
                if (ScoreIsMaximumInLocalWindow(c, score, y, x, localMaximumRadius, heatmaps))
                {
                    list.Add(new Keypoint(score, new Vector2(x, y), c));
                }
            }
        }
    }

    return list;
}
```

### Create `WithinNmsRadiusOfCorrespondingPoint` Method

We want to make sure that any key points that have already been assigned to a body do not get used again. We can prevent this by only sending key points to the `DecodePose` method that are not too close to any key points in an existing `Keypoint` array.

``` c#
/// <summary>
/// Check if the provided image coordinates are too close to any keypoints in existing poses
/// </summary>
/// <param name="poses"></param>
/// <param name="squaredNmsRadius"></param>
/// <param name="vec"></param>
/// <param name="keypointId"></param>
/// <returns>True if there are any existing poses too close to the provided coords</returns>
static bool WithinNmsRadiusOfCorrespondingPoint(
    List<Keypoint[]> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
{
    // SquaredDistance
    return poses.Any(pose => (vec - pose[keypointId].position).sqrMagnitude <= squaredNmsRadius);
}
```

### Create `DecodeMultiplePoses` Method

This is the method that will be called from the `PoseEstimator` script after executing the model. It will take in all four output Tensors from the model output along with the stride value, max number poses to decode, a minimum confidence score threshold, and the radius for determining if a key point is too close to an existing pose.

#### Method Steps

1.  Initialize a new `List` of `Keypoint` arrays.
2.  Square the provided radius value
3.  Call the `BuildPartList` method to get the `List` of potential starting key points
4.  Sort the `List` in descending order based on the confidence scores for the key points
5.  Iterate through the `List` of starting key points
    1.  Create a copy of the key point with the highest score
    2.  Remove the key point from the `List`
    3.  Get the input image coordinates for the key point
    4.  Skip the key point if it is too close to an existing `Keypoint` array
    5.  Call the `DecodePose` method with the key point as the starting key point
    6.  Add the new `Keypoint` array to the `List`
6.  Return the `List` of `Keypoint` arrays as an array.

``` c#
/// <summary>
/// Detects multiple poses and finds their parts from part scores and displacement vectors. 
/// </summary>
/// <param name="heatmaps"></param>
/// <param name="offsets"></param>
/// <param name="displacementsFwd"></param>
/// <param name="displacementBwd"></param>
/// <param name="stride"></param>
/// <param name="maxPoseDetections"></param>
/// <param name="scoreThreshold"></param>
/// <param name="nmsRadius"></param>
/// <returns>An array of poses up to maxPoseDetections in size</returns>
public static Keypoint[][] DecodeMultiplePoses(
    Tensor heatmaps, Tensor offsets,
    Tensor displacementsFwd, Tensor displacementBwd,
    int stride, int maxPoseDetections,
    float scoreThreshold = 0.5f, int nmsRadius = 20)
{
    // Stores the final poses
    List<Keypoint[]> poses = new List<Keypoint[]>();
    // 
    float squaredNmsRadius = (float)nmsRadius * nmsRadius;

    // Get a list of indicies with the highest values within the provided radius.
    List<Keypoint> list = BuildPartList(scoreThreshold, kLocalMaximumRadius, heatmaps);
    // Order the list in descending order based on score
    list = list.OrderByDescending(x => x.score).ToList();

    // Decode poses until the max number of poses has been reach or the part list is empty
    while (poses.Count < maxPoseDetections && list.Count > 0)
    {
        // Get the part with the highest score in the list
        Keypoint root = list[0];
        // Remove the keypoint from the list
        list.RemoveAt(0);

        // Calculate the input image coordinates for the current part
        Vector2 rootImageCoords = GetImageCoords(root, stride, offsets);

        // Skip parts that are too close to existing poses
        if (WithinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.id))
        {
            continue;
        }

        // Find the keypoints in the same pose as the root part
        Keypoint[] keypoints = DecodePose(
            root, heatmaps, offsets, stride, displacementsFwd,
            displacementBwd);

        // The current list of keypoints
        poses.Add(keypoints);
    }

    return poses.ToArray();
}
```

## Update `PoseEstimator` Script

Now we can complete the `ProcessOutput` method in the `PoseEstimator` script.

### Add Public Variables

First, we will add some public variables so that we can adjust the max number of poses to decode, score threshold, and radius for determining whether a key point is too close to an existing pose from the Inspector tab.

``` c#
[Tooltip("The maximum number of posees to estimate")]
[Range(1, 20)]
public int maxPoses = 20;

[Tooltip("The score threshold for multipose estimation")]
[Range(0, 1.0f)]
public float scoreThreshold = 0.25f;

[Tooltip("Non-maximum suppression part distance")]
public int nmsRadius = 100;
```

### Modify `ProcessOutput` Method

We will assign the output from the `DecodeMultiplePoses` to the `poses` variable.

``` c#
// Determine the key point locations
poses = Utils.DecodeMultiplePoses(
    heatmaps, offsets,
    displacementFWD, displacementBWD,
    stride: stride, maxPoseDetections: maxPoses,
    scoreThreshold: scoreThreshold, 
    nmsRadius: nmsRadius);
```

#### Full Code

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
        poses = new Utils.Keypoint[1][];

        // Determine the key point locations
        poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
    }
    else
    {
        // Determine the key point locations
        poses = Utils.DecodeMultiplePoses(
            heatmaps, offsets,
            displacementFWD, displacementBWD,
            stride: stride, maxPoseDetections: maxPoses,
            scoreThreshold: scoreThreshold, 
            nmsRadius: nmsRadius);
    }

    heatmaps.Dispose();
    offsets.Dispose();
    displacementFWD.Dispose();
    displacementBWD.Dispose();
}
```

## Summary

We now have everything need to perform pose estimation. However, we cannot currently gauge the accuracy of the estimated poses. In the next post, we will demonstrate how to add pose skeletons so that we can compare the estimated key point locations to the source video feed.

**Previous:** [Part 5](../part-5/README.md)

**Next:** [Part 7](../part-7/README.md)

**Project Resources:** [GitHub Repository](https://github.com/cj-mills/Barracuda-PoseNet-Tutorial)
