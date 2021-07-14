using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;

public class PoseNetClass
{
    public static string[] partNames = new string[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };

    public static int NUM_KEYPOINTS = partNames.Length;

    public static Tuple<int, int>[] connectedPartIndices = new Tuple<int, int>[] {
                Tuple.Create(11, 5), Tuple.Create(7, 5),
                Tuple.Create(7, 9), Tuple.Create(11, 13),
                Tuple.Create(13, 15), Tuple.Create(12, 6),
                Tuple.Create(8, 6), Tuple.Create(8, 10),
                Tuple.Create(12, 14), Tuple.Create(14, 16),
                Tuple.Create(5, 6), Tuple.Create(11, 12)
            };

    public static Tuple<int, int>[] parentChildrenTuples = new Tuple<int, int>[]{
                Tuple.Create(0, 1), Tuple.Create(1, 3),
                Tuple.Create(0, 2), Tuple.Create(2, 4),
                Tuple.Create(0, 5), Tuple.Create(5, 7),
                Tuple.Create(7, 9), Tuple.Create(5, 11),
                Tuple.Create(11, 13), Tuple.Create(13, 15),
                Tuple.Create(0, 6), Tuple.Create(6, 8),
                Tuple.Create(8, 10), Tuple.Create(6, 12),
                Tuple.Create(12, 14), Tuple.Create(14, 16)
            };


    public static int[] parentToChildEdges = parentChildrenTuples.Select(x => x.Item2).ToArray();
    public static int[] childToParentEdges = parentChildrenTuples.Select(x => x.Item1).ToArray();


    /// <summary>
    /// 
    /// </summary>
    public struct Part
    {
        public float score;
        public int heatmapX;
        public int heatmapY;
        public int id;

        public Part(float score, int heatmapX, int heatmapY, int id)
        {
            this.score = score;
            this.heatmapX = heatmapX;
            this.heatmapY = heatmapY;
            this.id = id;
        }

    }

    /// <summary>
    /// 
    /// </summary>
    public struct Keypoint
    {
        public float score;
        public Vector2 position;
        public string part;

        public Keypoint(float score, Vector2 position, string part)
        {
            this.score = score;
            this.position = position;
            this.part = part;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public struct Pose
    {
        public Keypoint[] keypoints;
        public float score;

        public Pose(Keypoint[] keypoints, float score)
        {
            this.keypoints = keypoints;
            this.score = score;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="y"></param>
    /// <param name="x"></param>
    /// <param name="keypoint"></param>
    /// <param name="offsets"></param>
    /// <returns></returns>
    public static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
    {
        return new Vector2(
            offsets[0, y, x, keypoint + NUM_KEYPOINTS],
            offsets[0, y, x, keypoint]
        );
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="part"></param>
    /// <param name="stride"></param>
    /// <param name="offsets"></param>
    /// <returns></returns>
    public static Vector2 GetImageCoords(Part part, int stride, Tensor offsets)
    {
        // The accompanying offset vector for the current coords
        Vector2 offsetVector = GetOffsetVector(part.heatmapY, part.heatmapX,
                                 part.id, offsets);

        Vector2 coords = new Vector2();
        // Scale the coordinates up to the inputImage resolution
        // Add the offset vectors to refine the key point location
        coords.x = (part.heatmapX * stride + offsetVector.x);
        coords.y = (part.heatmapY * stride + offsetVector.y);

        return coords;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="point"></param>
    /// <param name="stride"></param>
    /// <param name="height"></param>
    /// <param name="width"></param>
    /// <returns></returns>
    static Vector2Int GetStridedIndexNearPoint(
        Vector2 point, int stride, int height,
        int width)
    {

        return new Vector2Int(
            (int)Mathf.Clamp(Mathf.Round(point.x / stride), 0, width - 1),
            (int)Mathf.Clamp(Mathf.Round(point.y / stride), 0, height - 1)
        );
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="edgeId"></param>
    /// <param name="point"></param>
    /// <param name="displacements"></param>
    /// <returns></returns>
    static Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
    {

        int numEdges = (int)(displacements.channels / 2);

        return new Vector2(
            displacements[0, point.y, point.x, numEdges + edgeId],
            displacements[0, point.y, point.x, edgeId]
        );
    }

    /// <summary>
    /// We get a new keypoint along the `edgeId` for the pose instance, assuming
    /// that the position of the `idSource` part is already known. For this, we
    /// follow the displacement vector from the source to target part (stored in
    /// the `i`-t channel of the displacement tensor).
    /// </summary>
    /// <param name="edgeId"></param>
    /// <param name="sourceKeypoint"></param>
    /// <param name="targetKeypointId"></param>
    /// <param name="scores"></param>
    /// <param name="offsets"></param>
    /// <param name="stride"></param>
    /// <param name="displacements"></param>
    /// <returns></returns>
    static Keypoint TraverseToTargetKeypoint(
        int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
        Tensor scores, Tensor offsets, int stride,
        Tensor displacements)
    {

        int height = scores.height;
        int width = scores.width;

        // Nearest neighbor interpolation for the source->target displacements.
        Vector2Int sourceKeypointIndices = GetStridedIndexNearPoint(
            sourceKeypoint.position, stride, height, width);

        Vector2 displacement =
            GetDisplacement(edgeId, sourceKeypointIndices, displacements);

        Vector2 displacedPoint = sourceKeypoint.position + displacement;

        Vector2Int displacedPointIndices =
            GetStridedIndexNearPoint(displacedPoint, stride, height, width);

        Vector2 offsetVector = GetOffsetVector(
                displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
                offsets);

        float score = scores[0,
            displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];

        Vector2 targetKeypoint = new Vector2(
                    x: displacedPointIndices.x * stride,
                    y: displacedPointIndices.y * stride)
                + new Vector2(x: offsetVector.x, y: offsetVector.y);

        return new Keypoint(score, targetKeypoint, partNames[targetKeypointId]);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="root"></param>
    /// <param name="scores"></param>
    /// <param name="offsets"></param>
    /// <param name="stride"></param>
    /// <param name="displacementsFwd"></param>
    /// <param name="displacementsBwd"></param>
    /// <returns></returns>
    static Keypoint[] DecodePose(Part root, Tensor scores, Tensor offsets,
        int stride, Tensor displacementsFwd, Tensor displacementsBwd)
    {

        int numParts = scores.channels;
        int numEdges = parentToChildEdges.Length;

        Keypoint[] instanceKeypoints = new Keypoint[numParts];

        // Start a new detection instance at the position of the root.
        float rootScore = root.score;
        Vector2 rootPoint = GetImageCoords(root, stride, offsets);

        instanceKeypoints[root.id] = new Keypoint(
            rootScore,
            rootPoint,
            partNames[root.id]
        );

        // Decode the part positions upwards in the tree, following the backward
        // displacements.
        for (int edge = numEdges - 1; edge >= 0; --edge)
        {
            int sourceKeypointId = parentToChildEdges[edge];
            int targetKeypointId = childToParentEdges[edge];
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
            int sourceKeypointId = childToParentEdges[edge];
            int targetKeypointId = parentToChildEdges[edge];
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
    /// 
    /// </summary>
    /// <param name="poses"></param>
    /// <param name="squaredNmsRadius"></param>
    /// <param name="vec"></param>
    /// <param name="keypointId"></param>
    /// <returns></returns>
    static bool WithinNmsRadiusOfCorrespondingPoint(
        List<Pose> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
    {
        // SquaredDistance
        return poses.Any(pose => (vec - pose.keypoints[keypointId].position).sqrMagnitude <= squaredNmsRadius
        );
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="existingPoses"></param>
    /// <param name="squaredNmsRadius"></param>
    /// <param name="instanceKeypoints"></param>
    /// <returns></returns>
    static float GetInstanceScore(
        List<Pose> existingPoses, float squaredNmsRadius,
        Keypoint[] instanceKeypoints)
    {

        float notOverlappedKeypointScores = instanceKeypoints
           .Where((x, id) => 
           !WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
           .Sum(x => x.score);

        return notOverlappedKeypointScores / instanceKeypoints.Length;
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="keypointId"></param>
    /// <param name="score"></param>
    /// <param name="heatmapY"></param>
    /// <param name="heatmapX"></param>
    /// <param name="localMaximumRadius"></param>
    /// <param name="scores"></param>
    /// <returns></returns>
    static bool ScoreIsMaximumInLocalWindow(int keypointId, float score, int heatmapY, int heatmapX,
        int localMaximumRadius, Tensor heatmaps)
    {
        bool localMaximum = true;
        int yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
        int yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, heatmaps.height);

        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
        {
            int xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
            int xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, heatmaps.width);
            
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
            {
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
    /// 
    /// </summary>
    /// <param name="scoreThreshold"></param>
    /// <param name="localMaximumRadius"></param>
    /// <param name="scores"></param>
    /// <returns></returns>
    static List<Part> BuildPartQueue(float scoreThreshold, int localMaximumRadius, Tensor heatmaps)
    {
        List<Part> list = new List<Part>();

        for (int c = 0; c < heatmaps.channels; c++)
        {
            for (int y = 0; y < heatmaps.height; y++)
            {
                for (int x = 0; x < heatmaps.width; x++)
                {
                    float score = heatmaps[0, y, x, c];

                    // Only consider parts with score greater or equal to threshold as
                    // root candidates.
                    if (score < scoreThreshold) continue;

                    // Only consider keypoints whose score is maximum in a local window.
                    if (ScoreIsMaximumInLocalWindow(c, score, y, x, localMaximumRadius, heatmaps))
                    {
                        list.Add(new Part(score, x, y, c));
                    }
                }
            }
        }

        return list;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="heatmaps"></param>
    /// <param name="offsets"></param>
    /// <param name="displacementsFwd"></param>
    /// <param name="displacementBwd"></param>
    /// <param name="stride"></param>
    /// <param name="maxPoseDetections"></param>
    /// <param name="scoreThreshold"></param>
    /// <param name="nmsRadius"></param>
    /// <returns></returns>
    public static Pose[] DecodeMultiplePoses(
        Tensor heatmaps, Tensor offsets,
        Tensor displacementsFwd, Tensor displacementBwd,
        int stride, int maxPoseDetections,
        float scoreThreshold, int nmsRadius = 20, int kLocalMaximumRadius = 1)
    {
        // Stores the final poses
        List<Pose> poses = new List<Pose>();
        // 
        float squaredNmsRadius = (float)nmsRadius * nmsRadius;


        List<Part> list = BuildPartQueue(scoreThreshold, kLocalMaximumRadius, heatmaps);
        list = list.OrderByDescending(x => x.score).ToList();
        
        while (poses.Count < maxPoseDetections && list.Count > 0)
        {
            Part root = list[0];
            list.RemoveAt(0);

            // Part-based non-maximum suppression: We reject a root candidate if it
            // is within a disk of `nmsRadius` pixels from the corresponding part of
            // a previously detected instance.
            Vector2 rootImageCoords = GetImageCoords(root, stride, offsets);

            if (WithinNmsRadiusOfCorrespondingPoint(
                    poses, squaredNmsRadius, rootImageCoords, root.id))
            {
                continue;
            }

            // Start a new detection instance at the position of the root.
            Keypoint[] keypoints = DecodePose(
                root, heatmaps, offsets, stride, displacementsFwd,
                displacementBwd);

            float score = GetInstanceScore(poses, squaredNmsRadius, keypoints);
            poses.Add(new Pose(keypoints, score));
        }

        return poses.ToArray();
    }
}
