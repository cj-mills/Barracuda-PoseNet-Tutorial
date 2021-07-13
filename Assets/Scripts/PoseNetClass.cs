using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;

public class PoseNetClass
{
    const int kLocalMaximumRadius = 1;

    public static string[] partNames = new string[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };

    public static int NUM_KEYPOINTS = partNames.Length;

    public static Dictionary<String, int> partIds = partNames
        .Select((k, v) => new { k, v })
        .ToDictionary(p => p.k, p => p.v);

    public static Tuple<string, string>[] connectedPartNames = new Tuple<string, string>[] {
                Tuple.Create(partNames[11], partNames[5]), Tuple.Create(partNames[7], partNames[5]),
                Tuple.Create(partNames[7], partNames[9]), Tuple.Create(partNames[11], partNames[13]),
                Tuple.Create(partNames[13], partNames[15]), Tuple.Create(partNames[12], partNames[6]),
                Tuple.Create(partNames[8], partNames[6]), Tuple.Create(partNames[8], partNames[10]),
                Tuple.Create(partNames[12], partNames[14]), Tuple.Create(partNames[14], partNames[16]),
                Tuple.Create(partNames[5], partNames[6]), Tuple.Create(partNames[11], partNames[12])
            };

    public static Tuple<string, string>[] poseChain = new Tuple<string, string>[]{
                Tuple.Create(partNames[0], partNames[1]), Tuple.Create(partNames[1], partNames[3]), 
                Tuple.Create(partNames[0], partNames[2]), Tuple.Create(partNames[2], partNames[4]), 
                Tuple.Create(partNames[0], partNames[5]), Tuple.Create(partNames[5], partNames[7]), 
                Tuple.Create(partNames[7], partNames[9]), Tuple.Create(partNames[5], partNames[11]), 
                Tuple.Create(partNames[11], partNames[13]), Tuple.Create(partNames[13], partNames[15]), 
                Tuple.Create(partNames[0], partNames[6]), Tuple.Create(partNames[6], partNames[8]), 
                Tuple.Create(partNames[8], partNames[10]), Tuple.Create(partNames[6], partNames[12]), 
                Tuple.Create(partNames[12], partNames[14]), Tuple.Create(partNames[14], partNames[16])
            };

    public static Tuple<int, int>[] connectedPartIndices = connectedPartNames.Select(x =>
      new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
    ).ToArray();

    public static Tuple<int, int>[] parentChildrenTuples = poseChain.Select(x =>
      new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
    ).ToArray();

    public static int[] parentToChildEdges = parentChildrenTuples.Select(x => x.Item2).ToArray();
    public static int[] childToParentEdges = parentChildrenTuples.Select(x => x.Item1).ToArray();


    /// <summary>
    /// 
    /// </summary>
    public struct PartWithScore
    {
        public float score;
        public Part part;

        public PartWithScore(float score, Part part)
        {
            this.score = score;
            this.part = part;
        }

    }

    /// <summary>
    /// 
    /// </summary>
    public struct Part
    {
        public int heatmapX;
        public int heatmapY;
        public int id;
        public Part(int heatmapX, int heatmapY, int id)
        {
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
    public static Vector2 GetOffsetPoint(int y, int x, int keypoint, Tensor offsets)
    {
        return new Vector2(
            offsets[0, y, x, keypoint + NUM_KEYPOINTS],
            offsets[0, y, x, keypoint]
        );
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    static Vector2 AddVectors(Vector2 a, Vector2 b)
    {
        return new Vector2(x: a.x + b.x, y: a.y + b.y);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="part"></param>
    /// <param name="outputStride"></param>
    /// <param name="offsets"></param>
    /// <returns></returns>
    static Vector2 GetImageCoords(
        Part part, int outputStride, Tensor offsets)
    {
        Vector2 vec = GetOffsetPoint(part.heatmapY, part.heatmapX,
                                 part.id, offsets);
        return new Vector2(
            (float)(part.heatmapX * outputStride) + vec.x,
            (float)(part.heatmapY * outputStride) + vec.y
        );
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="minConfidence"></param>
    /// <returns></returns>
    static bool EitherPointDoesntMeetConfidence(
        float a, float b, float minConfidence)
    {
        return (a < minConfidence || b < minConfidence);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="keypoints"></param>
    /// <param name="minConfidence"></param>
    /// <returns></returns>
    public static Tuple<Keypoint, Keypoint>[] GetAdjacentKeyPoints(
           Keypoint[] keypoints, float minConfidence)
    {

        return connectedPartIndices
            .Where(x => !EitherPointDoesntMeetConfidence(
                keypoints[x.Item1].score, keypoints[x.Item2].score, minConfidence))
           .Select(x => new Tuple<Keypoint, Keypoint>(keypoints[x.Item1], keypoints[x.Item2])).ToArray();

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
    static bool ScoreIsMaximumInLocalWindow(
        int keypointId, float score, int heatmapY, int heatmapX,
        int localMaximumRadius, Tensor scores)
    {
        bool localMaximum = true;
        int yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
        int yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, scores.height);

        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
        {
            int xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
            int xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, scores.width);
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
            {
                if (scores[0, yCurrent, xCurrent, keypointId] > score)
                {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum)
            {
                break;
            }
        }
        return localMaximum;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="point"></param>
    /// <param name="outputStride"></param>
    /// <param name="height"></param>
    /// <param name="width"></param>
    /// <returns></returns>
    static Vector2Int GetStridedIndexNearPoint(
        Vector2 point, int outputStride, int height,
        int width)
    {

        return new Vector2Int(
            (int)Mathf.Clamp(Mathf.Round(point.x / outputStride), 0, width - 1),
            (int)Mathf.Clamp(Mathf.Round(point.y / outputStride), 0, height - 1)
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
    /// <param name="outputStride"></param>
    /// <param name="displacements"></param>
    /// <returns></returns>
    static Keypoint TraverseToTargetKeypoint(
        int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
        Tensor scores, Tensor offsets, int outputStride,
        Tensor displacements)
    {

        int height = scores.height;
        int width = scores.width;

        // Nearest neighbor interpolation for the source->target displacements.
        Vector2Int sourceKeypointIndices = GetStridedIndexNearPoint(
            sourceKeypoint.position, outputStride, height, width);

        Vector2 displacement =
            GetDisplacement(edgeId, sourceKeypointIndices, displacements);

        Vector2 displacedPoint = AddVectors(sourceKeypoint.position, displacement);

        Vector2Int displacedPointIndices =
            GetStridedIndexNearPoint(displacedPoint, outputStride, height, width);

        Vector2 offsetPoint = GetOffsetPoint(
                displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
                offsets);

        float score = scores[0,
            displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];

        Vector2 targetKeypoint =
            AddVectors(
                new Vector2(
                    x: displacedPointIndices.x * outputStride,
                    y: displacedPointIndices.y * outputStride)
                , new Vector2(x: offsetPoint.x, y: offsetPoint.y));

        return new Keypoint(score, targetKeypoint, partNames[targetKeypointId]);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="root"></param>
    /// <param name="scores"></param>
    /// <param name="offsets"></param>
    /// <param name="outputStride"></param>
    /// <param name="displacementsFwd"></param>
    /// <param name="displacementsBwd"></param>
    /// <returns></returns>
    static Keypoint[] DecodePose(PartWithScore root, Tensor scores, Tensor offsets,
        int outputStride, Tensor displacementsFwd,
        Tensor displacementsBwd)
    {

        int numParts = scores.channels;
        int numEdges = parentToChildEdges.Length;

        Keypoint[] instanceKeypoints = new Keypoint[numParts];

        // Start a new detection instance at the position of the root.
        Part rootPart = root.part;
        float rootScore = root.score;
        Vector2 rootPoint = GetImageCoords(rootPart, outputStride, offsets);

        instanceKeypoints[rootPart.id] = new Keypoint(
            rootScore,
            rootPoint,
            partNames[rootPart.id]
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
                    offsets, outputStride, displacementsBwd);
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
                    offsets, outputStride, displacementsFwd);
            }
        }

        return instanceKeypoints;

    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="position1"></param>
    /// <param name="position2"></param>
    /// <returns></returns>
    static float SquaredDistance(Vector2 position1, Vector2 position2)
    {
        return (position2 - position1).sqrMagnitude;
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
        return poses.Any(pose =>
            SquaredDistance(vec, pose.keypoints[keypointId].position) <= squaredNmsRadius
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
           .Where((x, id) => !WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
           .Sum(x => x.score);

        return notOverlappedKeypointScores / instanceKeypoints.Length;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="scoreThreshold"></param>
    /// <param name="localMaximumRadius"></param>
    /// <param name="scores"></param>
    /// <returns></returns>
    static PriorityQueue<float, PartWithScore> BuildPartWithScoreQueue(
        float scoreThreshold, int localMaximumRadius,
        Tensor heatmaps)
    {
        PriorityQueue<float, PartWithScore> queue = new PriorityQueue<float, PartWithScore>();

        for (int k = 0; k < heatmaps.channels; k++)
        {
            for (int y = 0; y < heatmaps.height; y++)
            {
                for (int x = 0; x < heatmaps.width; x++)
                {
                    float score = heatmaps[0, y, x, k];

                    // Only consider parts with score greater or equal to threshold as
                    // root candidates.
                    if (score < scoreThreshold) { continue; }

                    // Only consider keypoints whose score is maximum in a local window.
                    if (ScoreIsMaximumInLocalWindow(k, score, y, x, localMaximumRadius, heatmaps))
                    {
                        queue.Push(score, new PartWithScore(score, new Part(x, y, k)));
                    }
                }
            }
        }

        return queue;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="heatmaps"></param>
    /// <param name="offsets"></param>
    /// <param name="displacementsFwd"></param>
    /// <param name="displacementBwd"></param>
    /// <param name="outputStride"></param>
    /// <param name="maxPoseDetections"></param>
    /// <param name="scoreThreshold"></param>
    /// <param name="nmsRadius"></param>
    /// <returns></returns>
    public static Pose[] DecodeMultiplePoses(
        Tensor heatmaps, Tensor offsets,
        Tensor displacementsFwd, Tensor displacementBwd,
        int outputStride, int maxPoseDetections,
        float scoreThreshold, int nmsRadius = 20)
    {
        // Stores the final poses
        List<Pose> poses = new List<Pose>();
        // 
        float squaredNmsRadius = (float)nmsRadius * nmsRadius;

        PriorityQueue<float, PartWithScore> queue = BuildPartWithScoreQueue(
            scoreThreshold, kLocalMaximumRadius, heatmaps);

        while (poses.Count < maxPoseDetections && queue.Count > 0)
        {
            PartWithScore root = queue.Pop().Value;

            // Part-based non-maximum suppression: We reject a root candidate if it
            // is within a disk of `nmsRadius` pixels from the corresponding part of
            // a previously detected instance.
            Vector2 rootImageCoords =
                GetImageCoords(root.part, outputStride, offsets);

            if (WithinNmsRadiusOfCorrespondingPoint(
                    poses, squaredNmsRadius, rootImageCoords, root.part.id))
            {
                continue;
            }

            // Start a new detection instance at the position of the root.
            Keypoint[] keypoints = DecodePose(
                root, heatmaps, offsets, outputStride, displacementsFwd,
                displacementBwd);

            float score = GetInstanceScore(poses, squaredNmsRadius, keypoints);
            poses.Add(new Pose(keypoints, score));
        }

        return poses.ToArray();
    }
}
