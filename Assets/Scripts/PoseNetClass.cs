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

    public static double mean(Tensor tensor)
    {
        double sum = 0f;
        var x = tensor.height;
        var y = tensor.width;
        var z = tensor.channels;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < z; k++)
                {
                    sum += tensor[0, i, j, k];
                }
            }
        }
        var mean = sum / (x * y * z);
        return mean;
    }

    //Pose ScalePose(Pose pose, int scale) {

    //    var s = (float)scale;

    //    return new Pose(
    //        pose.keypoints.Select( x => 
    //            new Keypoint( 
    //                x.score,
    //                new Vector2(x.position.x * s, x.position.y * s),
    //                x.part)
    //         ).ToArray(),
    //         pose.score
    //     );
    //}

    //Pose[] ScalePoses(Pose[] poses, int scale) {
    //    if (scale == 1) {
    //        return poses;
    //    }
    //    return poses.Select(x => ScalePose(pose: x, scale: scale)).ToArray();
    //}

    //int GetValidResolution(float imageScaleFactor,
    //                       int inputDimension,
    //                       int outputStride) {
    //    var evenResolution = (int)(inputDimension * imageScaleFactor) - 1;
    //    return evenResolution - (evenResolution % outputStride) + 1;
    //}

    //int Half(int k)
    //{
    //    return (int)Mathf.Floor((float)(k / 2));
    //}


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
        //foreach (var pose in poses)
        //{
        //    if (SquaredDistance(vec.y, vec.x,
        //        pose.keypoints[keypointId].position.y,
        //        pose.keypoints[keypointId].position.x) <= squaredNmsRadius)
        //    {
        //        return true;
        //    }
        //}
        //return false;
    }

    float GetInstanceScore(
        List<Pose> existingPoses, float squaredNmsRadius,
        Keypoint[] instanceKeypoints)
    {

        float notOverlappedKeypointScores = instanceKeypoints
           .Where((x, id) => !WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
           .Sum(x => x.score);

        //int id = 0;
        //float notOverlappedKeypointScores = 0.0f;
        //foreach (var x in instanceKeypoints)
        //{
        //    if (!WithinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, x.position, id))
        //    {
        //        notOverlappedKeypointScores += x.score;
        //    }
        //}
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

    // Copyright (c) Microsoft Corporation. All rights reserved.
    // Licensed under the MIT License. See LICENSE in the project root for license information.
    // https://github.com/Microsoft/MixedRealityToolkit-Unity/blob/master/Assets/HoloToolkit/Utilities/Scripts/PriorityQueue.cs

    /// <summary>
    /// Min-heap priority queue. In other words, lower priorities will be removed from the queue first.
    /// See http://en.wikipedia.org/wiki/Binary_heap for more info.
    /// </summary>
    /// <typeparam name="TPriority">Type for the priority used for ordering.</typeparam>
    /// <typeparam name="TValue">Type of values in the queue.</typeparam>
    class PriorityQueue<TPriority, TValue> : IEnumerable<KeyValuePair<TPriority, TValue>>
    {
        public class ValueCollection : IEnumerable<TValue>
        {
            private readonly PriorityQueue<TPriority, TValue> parentCollection;

            public ValueCollection(PriorityQueue<TPriority, TValue> parentCollection)
            {
                this.parentCollection = parentCollection;
            }

            #region IEnumerable

            public IEnumerator<TValue> GetEnumerator()
            {
                foreach (KeyValuePair<TPriority, TValue> pair in parentCollection)
                {
                    yield return pair.Value;
                }
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            #endregion
        }

        private readonly IComparer<TPriority> priorityComparer;

        public PriorityQueue() : this(Comparer<TPriority>.Default) { }

        public PriorityQueue(IComparer<TPriority> comparer)
        {
            if (comparer == null)
            {
                throw new ArgumentNullException();
            }

            priorityComparer = comparer;
        }

        private readonly List<KeyValuePair<TPriority, TValue>> queue = new List<KeyValuePair<TPriority, TValue>>();
        private ValueCollection valueCollection;

        public ValueCollection Values
        {
            get
            {
                if (valueCollection == null)
                {
                    valueCollection = new ValueCollection(this);
                }

                return valueCollection;
            }
        }

        #region IEnumerable

        public IEnumerator<KeyValuePair<TPriority, TValue>> GetEnumerator()
        {
            return queue.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        /// <summary>
        /// Clears the priority queue
        /// </summary>
        public void Clear()
        {
            queue.Clear();
        }

        /// <summary>
        /// Add an element to the priority queue.
        /// </summary>
        /// <param name="priority">Priority of the element</param>
        /// <param name="value"></param>
        public void Push(TPriority priority, TValue value)
        {
            queue.Add(new KeyValuePair<TPriority, TValue>(priority, value));
            BubbleUp();
        }

        /// <summary>
        /// Number of elements in priority queue
        /// </summary>
        public int Count
        {
            get
            {
                return queue.Count;
            }
        }

        /// <summary>
        /// Get the element with the minimum priority in the queue. The Key in the return value is the priority.
        /// </summary>
        public KeyValuePair<TPriority, TValue> Top
        {
            get
            {
                return queue[0];
            }
        }

        /// <summary>
        /// Pop the minimal element of the queue. Will fail at runtime if queue is empty.
        /// </summary>
        /// <returns>The minimal element</returns>
        public KeyValuePair<TPriority, TValue> Pop()
        {
            KeyValuePair<TPriority, TValue> ret = queue[0];
            queue[0] = queue[queue.Count - 1];
            queue.RemoveAt(queue.Count - 1);
            BubbleDown();
            return ret;
        }

        /// <summary>
        /// Returns whether or not the value is contained in the queue
        /// </summary>
        public bool Contains(TValue value)
        {
            return queue.Any(itm => EqualityComparer<TValue>.Default.Equals(itm.Value, value));
        }

        /// <summary>
        /// Removes the first element that equals the value from the queue
        /// </summary>
        public bool Remove(TValue value)
        {
            int idx = queue.FindIndex(itm => EqualityComparer<TValue>.Default.Equals(itm.Value, value));
            if (idx == -1)
            {
                return false;
            }

            queue[idx] = queue[queue.Count - 1];
            queue.RemoveAt(queue.Count - 1);
            BubbleDown();

            return true;
        }

        /// <summary>
        /// Removes all elements with this priority from the queue.
        /// </summary>
        /// <returns>True if elements were removed</returns>
        public bool RemoveAtPriority(TPriority priority, Predicate<TValue> shouldRemove)
        {
            bool removed = false;

            for (int i = queue.Count - 1; i >= 0; --i)
            {
                // TODO: early out if key < priority
                if (queue[i].Key.Equals(priority) && (shouldRemove == null || shouldRemove(queue[i].Value)))
                {
                    queue[i] = queue[queue.Count - 1];
                    queue.RemoveAt(queue.Count - 1);
                    BubbleDown();

                    removed = true;
                }
            }

            return removed;
        }

        /// <summary>
        /// Bubble up the last element in the queue until it's in the correct spot.
        /// </summary>
        private void BubbleUp()
        {
            int node = queue.Count - 1;
            while (node > 0)
            {
                int parent = (node - 1) >> 1;
                if (Compare(queue[parent].Key, queue[node].Key) < 0)
                {
                    break; // we're in the right order, so we're done
                }
                KeyValuePair<TPriority, TValue> tmp = queue[parent];
                queue[parent] = queue[node];
                queue[node] = tmp;
                node = parent;
            }
        }

        /// <summary>
        /// Bubble down the first element until it's in the correct spot.
        /// </summary>
        private void BubbleDown()
        {
            int node = 0;
            while (true)
            {
                // Find smallest child
                int child0 = (node << 1) + 1;
                int child1 = (node << 1) + 2;
                int smallest;
                if (child0 < queue.Count && child1 < queue.Count)
                {
                    smallest = Compare(queue[child0].Key, queue[child1].Key) < 0 ? child0 : child1;
                }
                else if (child0 < queue.Count)
                {
                    smallest = child0;
                }
                else if (child1 < queue.Count)
                {
                    smallest = child1;
                }
                else
                {
                    break; // 'node' is a leaf, since both children are outside the array
                }

                if (Compare(queue[node].Key, queue[smallest].Key) < 0)
                {
                    break; // we're in the right order, so we're done.
                }

                KeyValuePair<TPriority, TValue> tmp = queue[node];
                queue[node] = queue[smallest];
                queue[smallest] = tmp;
                node = smallest;
            }
        }
        private int Compare(TPriority x, TPriority y)
        {
            return priorityComparer.Compare(y, x);
        }

    }
}
