using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Linq;

public class Utils
{
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

    /// <summary>
    /// Defines the size of the local window in the heatmap to look for
    /// confidence scores higher than the one at the current heatmap coordinate
    /// </summary>
    const int kLocalMaximumRadius = 1;

    // Defines the parent->child relationships used for multipose detection.
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


    /// <summary>
    /// Applies the preprocessing steps for the MobileNet model on the CPU
    /// </summary>
    /// <param name="tensor">Pixel data from the input tensor</param>
    public static void PreprocessMobileNet(float[] tensor)
    {
        // Normaliz the values to the range [-1, 1]
        System.Threading.Tasks.Parallel.For(0, tensor.Length, (int i) =>
        {
            tensor[i] = (float)(2.0f * tensor[i] / 1.0f) - 1.0f;
        });
    }

    ///// <summary>
    ///// Applies the preprocessing steps for the ResNet50 model on the CPU
    ///// </summary>
    ///// <param name="tensor">Pixel data from the input tensor</param>
    public static void PreprocessResNet(float[] tensor)
    {
        System.Threading.Tasks.Parallel.For(0, tensor.Length / 3, (int i) =>
        {
            tensor[i * 3 + 0] = (float)tensor[i * 3 + 0] * 255f - 123.15f;
            tensor[i * 3 + 1] = (float)tensor[i * 3 + 1] * 255f - 115.90f;
            tensor[i * 3 + 2] = (float)tensor[i * 3 + 2] * 255f - 103.06f;
        });
    }

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


    /// <summary>
    /// Calculate the position of the provided keypoint in the input image
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
}
