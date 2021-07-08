using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;

public class PoseNetClass
{
    const int kLocalMaximumRadius = 1;
    public int NUM_KEYPOINTS = 0;
    public String[] partNames;
    public Dictionary<String, int> partIds;
    public Tuple<string, string>[] connectedPartNames;
    public Tuple<int, int>[] connectedPartIndices;
    public Tuple<string, string>[] poseChain;
    public Tuple<int, int>[] parentChildrenTuples;
    public int[] parentToChildEdges;
    public int[] childToParentEdges;


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


    Vector2 GetOffsetPoint(int y, int x, int keypoint, Tensor offsets)
    {
        return new Vector2(
            offsets[0, y, x, keypoint + NUM_KEYPOINTS],
            offsets[0, y, x, keypoint]
        );
    }

    float SquaredDistance(
        float y1, float x1, float y2, float x2)
    {
        var dy = y2 - y1;
        var dx = x2 - x1;
        return dy * dy + dx * dx;
    }

    Vector2 AddVectors(Vector2 a, Vector2 b)
    {
        return new Vector2(x: a.x + b.x, y: a.y + b.y);
    }

    Vector2 GetImageCoords(
        Part part, int outputStride, Tensor offsets)
    {
        var vec = GetOffsetPoint(part.heatmapY, part.heatmapX,
                                 part.id, offsets);
        return new Vector2(
            (float)(part.heatmapX * outputStride) + vec.x,
            (float)(part.heatmapY * outputStride) + vec.y
        );
    }

    public Tuple<Keypoint, Keypoint>[] GetAdjacentKeyPoints(
           Keypoint[] keypoints, float minConfidence)
    {

        return connectedPartIndices
            .Where(x => !EitherPointDoesntMeetConfidence(
                keypoints[x.Item1].score, keypoints[x.Item2].score, minConfidence))
           .Select(x => new Tuple<Keypoint, Keypoint>(keypoints[x.Item1], keypoints[x.Item2])).ToArray();

    }

    bool EitherPointDoesntMeetConfidence(
        float a, float b, float minConfidence)
    {
        return (a < minConfidence || b < minConfidence);
    }

    

    public PoseNetClass()
    {
        partNames = new String[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };

        NUM_KEYPOINTS = partNames.Length;

        partIds = partNames
            .Select((k, v) => new { k, v })
            .ToDictionary(p => p.k, p => p.v);

        connectedPartNames = new Tuple<string, string>[] {
            Tuple.Create("leftHip", "leftShoulder"), Tuple.Create("leftElbow", "leftShoulder"),
            Tuple.Create("leftElbow", "leftWrist"), Tuple.Create("leftHip", "leftKnee"),
            Tuple.Create("leftKnee", "leftAnkle"), Tuple.Create("rightHip", "rightShoulder"),
            Tuple.Create("rightElbow", "rightShoulder"), Tuple.Create("rightElbow", "rightWrist"),
            Tuple.Create("rightHip", "rightKnee"), Tuple.Create("rightKnee", "rightAnkle"),
            Tuple.Create("leftShoulder", "rightShoulder"), Tuple.Create("leftHip", "rightHip")
        };

        connectedPartIndices = connectedPartNames.Select(x =>
          new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
        ).ToArray();

        poseChain = new Tuple<string, string>[]{
            Tuple.Create("nose", "leftEye"), Tuple.Create("leftEye", "leftEar"), Tuple.Create("nose", "rightEye"),
            Tuple.Create("rightEye", "rightEar"), Tuple.Create("nose", "leftShoulder"),
            Tuple.Create("leftShoulder", "leftElbow"), Tuple.Create("leftElbow", "leftWrist"),
            Tuple.Create("leftShoulder", "leftHip"), Tuple.Create("leftHip", "leftKnee"),
            Tuple.Create("leftKnee", "leftAnkle"), Tuple.Create("nose", "rightShoulder"),
            Tuple.Create("rightShoulder", "rightElbow"), Tuple.Create("rightElbow", "rightWrist"),
            Tuple.Create("rightShoulder", "rightHip"), Tuple.Create("rightHip", "rightKnee"),
            Tuple.Create("rightKnee", "rightAnkle")
        };

        parentChildrenTuples = poseChain.Select(x =>
          new Tuple<int, int>(partIds[x.Item1], partIds[x.Item2])
        ).ToArray();

        parentToChildEdges = parentChildrenTuples.Select(x => x.Item2).ToArray();
        childToParentEdges = parentChildrenTuples.Select(x => x.Item1).ToArray();
    }

    bool ScoreIsMaximumInLocalWindow(
        int keypointId, float score, int heatmapY, int heatmapX,
        int localMaximumRadius, Tensor scores)
    {

        var height = scores.height;
        var width = scores.width;
        var localMaximum = true;
        var yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
        var yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, height);

        for (var yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
        {
            var xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
            var xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, width);
            for (var xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
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

    PriorityQueue<float, PartWithScore> BuildPartWithScoreQueue(
        float scoreThreshold, int localMaximumRadius,
        Tensor scores)
    {
        var queue = new PriorityQueue<float, PartWithScore>();

        var height = scores.height;
        var width = scores.width;
        var numKeypoints = scores.channels;

        for (int heatmapY = 0; heatmapY < height; ++heatmapY)
        {
            for (int heatmapX = 0; heatmapX < width; ++heatmapX)
            {
                for (int keypointId = 0; keypointId < numKeypoints; ++keypointId)
                {
                    float score = scores[0, heatmapY, heatmapX, keypointId];

                    // Only consider parts with score greater or equal to threshold as
                    // root candidates.
                    if (score < scoreThreshold)
                    {
                        continue;
                    }

                    // Only consider keypoints whose score is maximum in a local window.
                    if (ScoreIsMaximumInLocalWindow(
                            keypointId, score, heatmapY, heatmapX, localMaximumRadius,
                            scores))
                    {
                        queue.Push(score, new PartWithScore(score,
                            new Part(heatmapX, heatmapY, keypointId)
                        ));
                    }
                }
            }
        }

        return queue;
    }

    bool WithinNmsRadiusOfCorrespondingPoint(
        List<Pose> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
    {
        return poses.Any(pose =>
            SquaredDistance(vec.y, vec.x,
                pose.keypoints[keypointId].position.y,
                pose.keypoints[keypointId].position.x) <= squaredNmsRadius
        );
    }

    float GetInstanceScore(
        List<Pose> existingPoses, float squaredNmsRadius,
        Keypoint[] instanceKeypoints)
    {

        float notOverlappedKeypointScores = instanceKeypoints
           .Where((x, id) => !WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
           .Sum(x => x.score);

        return notOverlappedKeypointScores / instanceKeypoints.Length;
    }

    public Pose[] DecodeMultiplePoses(
        Tensor scores, Tensor offsets,
        Tensor displacementsFwd, Tensor displacementBwd,
        int outputStride, int maxPoseDetections,
        float scoreThreshold, int nmsRadius = 20)
    {
        var poses = new List<Pose>();
        var squaredNmsRadius = (float)nmsRadius * nmsRadius;

        PriorityQueue<float, PartWithScore> queue = BuildPartWithScoreQueue(
            scoreThreshold, kLocalMaximumRadius, scores);

        while (poses.Count < maxPoseDetections && queue.Count > 0)
        {
            var root = queue.Pop().Value;

            // Part-based non-maximum suppression: We reject a root candidate if it
            // is within a disk of `nmsRadius` pixels from the corresponding part of
            // a previously detected instance.
            var rootImageCoords =
                GetImageCoords(root.part, outputStride, offsets);

            if (WithinNmsRadiusOfCorrespondingPoint(
                    poses, squaredNmsRadius, rootImageCoords, root.part.id))
            {
                continue;
            }

            // Start a new detection instance at the position of the root.
            var keypoints = DecodePose(
                root, scores, offsets, outputStride, displacementsFwd,
                displacementBwd);

            var score = GetInstanceScore(poses, squaredNmsRadius, keypoints);
            poses.Add(new Pose(keypoints, score));
        }

        return poses.ToArray();
    }

    Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
    {

        var numEdges = (int)(displacements.channels / 2);

        return new Vector2(
            displacements[0, point.y, point.x, numEdges + edgeId],
            displacements[0, point.y, point.x, edgeId]
        );
    }

    Vector2Int GetStridedIndexNearPoint(
        Vector2 point, int outputStride, int height,
        int width)
    {

        return new Vector2Int(
            (int)Mathf.Clamp(Mathf.Round(point.x / outputStride), 0, width - 1),
            (int)Mathf.Clamp(Mathf.Round(point.y / outputStride), 0, height - 1)
        );
    }

    /**
     * We get a new keypoint along the `edgeId` for the pose instance, assuming
     * that the position of the `idSource` part is already known. For this, we
     * follow the displacement vector from the source to target part (stored in
     * the `i`-t channel of the displacement tensor).
     */

    Keypoint TraverseToTargetKeypoint(
        int edgeId, Keypoint sourceKeypoint, int targetKeypointId,
        Tensor scores, Tensor offsets, int outputStride,
        Tensor displacements)
    {

        var height = scores.height;
        var width = scores.width;

        // Nearest neighbor interpolation for the source->target displacements.
        var sourceKeypointIndices = GetStridedIndexNearPoint(
            sourceKeypoint.position, outputStride, height, width);

        var displacement =
            GetDisplacement(edgeId, sourceKeypointIndices, displacements);

        var displacedPoint = AddVectors(sourceKeypoint.position, displacement);

        var displacedPointIndices =
            GetStridedIndexNearPoint(displacedPoint, outputStride, height, width);

        var offsetPoint = GetOffsetPoint(
                displacedPointIndices.y, displacedPointIndices.x, targetKeypointId,
                offsets);

        var score = scores[0,
            displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];

        var targetKeypoint =
            AddVectors(
                new Vector2(
                    x: displacedPointIndices.x * outputStride,
                    y: displacedPointIndices.y * outputStride)
                , new Vector2(x: offsetPoint.x, y: offsetPoint.y));

        return new Keypoint(score, targetKeypoint, partNames[targetKeypointId]);
    }

    Keypoint[] DecodePose(PartWithScore root, Tensor scores, Tensor offsets,
        int outputStride, Tensor displacementsFwd,
        Tensor displacementsBwd)
    {

        var numParts = scores.channels;
        var numEdges = parentToChildEdges.Length;

        var instanceKeypoints = new Keypoint[numParts];

        // Start a new detection instance at the position of the root.
        var rootPart = root.part;
        var rootScore = root.score;
        var rootPoint = GetImageCoords(rootPart, outputStride, offsets);

        instanceKeypoints[rootPart.id] = new Keypoint(
            rootScore,
            rootPoint,
            partNames[rootPart.id]
        );

        // Decode the part positions upwards in the tree, following the backward
        // displacements.
        for (var edge = numEdges - 1; edge >= 0; --edge)
        {
            var sourceKeypointId = parentToChildEdges[edge];
            var targetKeypointId = childToParentEdges[edge];
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
        for (var edge = 0; edge < numEdges; ++edge)
        {
            var sourceKeypointId = childToParentEdges[edge];
            var targetKeypointId = parentToChildEdges[edge];
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

    
}
