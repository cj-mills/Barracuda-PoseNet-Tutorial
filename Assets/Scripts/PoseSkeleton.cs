using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class PoseSkeleton
{
    // The list of key point GameObjects that make up the pose skeleton
    public Transform[] keypoints;

    // The GameObjects that contain data for the lines between key points
    private GameObject[] lines;

    // The names of the body parts that will be detected by the PoseNet model
    private static string[] partNames = new string[]{
        "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
        "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
        "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    private static int NUM_KEYPOINTS = partNames.Length;

    // The pairs of key points that should be connected on a body
    private Tuple<int, int>[] jointPairs = new Tuple<int, int>[]{
        // Nose to Left Eye
        Tuple.Create(0, 1),
        // Nose to Right Eye
        Tuple.Create(0, 2),
        // Left Eye to Left Ear
        Tuple.Create(1, 3),
        // Right Eye to Right Ear
        Tuple.Create(2, 4),
        // Left Shoulder to Right Shoulder
        Tuple.Create(5, 6),
        // Left Shoulder to Left Hip
        Tuple.Create(5, 11),
        // Right Shoulder to Right Hip
        Tuple.Create(6, 12),
        // Left Shoulder to Right Hip
        Tuple.Create(5, 12),
        // Rigth Shoulder to Left Hip
        Tuple.Create(6, 11),
        // Left Hip to Right Hip
        Tuple.Create(11, 12),
        // Left Shoulder to Left Elbow
        Tuple.Create(5, 7),
        // Left Elbow to Left Wrist
        Tuple.Create(7, 9), 
        // Right Shoulder to Right Elbow
        Tuple.Create(6, 8),
        // Right Elbow to Right Wrist
        Tuple.Create(8, 10),
        // Left Hip to Left Knee
        Tuple.Create(11, 13), 
        // Left Knee to Left Ankle
        Tuple.Create(13, 15),
        // Right Hip to Right Knee
        Tuple.Create(12, 14), 
        // Right Knee to Right Ankle
        Tuple.Create(14, 16)
    };

    // Colors for the skeleton lines
    private Color[] colors = new Color[] {
        // Head
        Color.magenta, Color.magenta, Color.magenta, Color.magenta,
        // Torso
        Color.red, Color.red, Color.red, Color.red, Color.red, Color.red,
        // Arms
        Color.green, Color.green, Color.green, Color.green,
        // Legs
        Color.blue, Color.blue, Color.blue, Color.blue
    };

    // The width for the skeleton lines
    private float lineWidth;

    
    public PoseSkeleton(float pointScale = 10f, float lineWidth = 5f)
    {
        this.keypoints = new Transform[NUM_KEYPOINTS];

        Material keypointMat = new Material(Shader.Find("Unlit/Color"));
        keypointMat.color = Color.yellow;

        for (int i = 0; i < NUM_KEYPOINTS; i++)
        {
            this.keypoints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
            this.keypoints[i].position = new Vector3(0, 0, 0);
            this.keypoints[i].localScale = new Vector3(pointScale, pointScale, 0);
            this.keypoints[i].gameObject.GetComponent<MeshRenderer>().material = keypointMat;
            this.keypoints[i].gameObject.name = partNames[i];
        }

        this.lineWidth = lineWidth;

        // The number of joint pairs
        int numPairs = jointPairs.Length;
        // Initialize the lines array
        lines = new GameObject[numPairs];

        // Initialize the pose skeleton
        InitializeSkeleton();
    }

    /// <summary>
    /// Toggles visibility for the skeleton
    /// </summary>
    /// <param name="show"></param>
    public void ToggleSkeleton(bool show)
    {
        for (int i= 0; i < jointPairs.Length; i++)
        {
            lines[i].SetActive(show);
            keypoints[jointPairs[i].Item1].gameObject.SetActive(show);
            keypoints[jointPairs[i].Item2].gameObject.SetActive(show);
        }
    }

    /// <summary>
    /// Clean up skeleton GameObjects
    /// </summary>
    public void Cleanup()
    {

        for (int i = 0; i < jointPairs.Length; i++)
        {
            GameObject.Destroy(lines[i]);
            GameObject.Destroy(keypoints[jointPairs[i].Item1].gameObject);
            GameObject.Destroy(keypoints[jointPairs[i].Item2].gameObject);
        }
    }


    /// <summary>
    /// Create a line between the key point specified by the start and end point indices
    /// </summary>
    /// <param name="pairIndex"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <param name="width"></param>
    /// <param name="color"></param>
    private void InitializeLine(int pairIndex, float width, Color color)
    {
        int startIndex = jointPairs[pairIndex].Item1;
        int endIndex = jointPairs[pairIndex].Item2;

        // Create new line GameObject
        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);

        // Add LineRenderer component
        LineRenderer lineRenderer = lines[pairIndex].AddComponent<LineRenderer>();
        // Make LineRenderer Shader Unlit
        lineRenderer.material = new Material(Shader.Find("Unlit/Color"));
        // Set the material color
        lineRenderer.material.color = color;

        // The line will consist of two points
        lineRenderer.positionCount = 2;

        // Set the width from the start point
        lineRenderer.startWidth = width;
        // Set the width from the end point
        lineRenderer.endWidth = width;
    }

    /// <summary>
    /// Initialize the pose skeleton
    /// </summary>
    private void InitializeSkeleton()
    {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            InitializeLine(i, lineWidth, colors[i]);
        }
    }

    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    /// <param name="keypoints"></param>
    /// <param name="sourceScale"></param>
    /// <param name="sourceTexture"></param>
    /// <param name="mirrorImage"></param>
    /// <param name="minConfidence"></param>
    public void UpdateKeyPointPositions(Utils.Keypoint[] keypoints,
        float sourceScale, RenderTexture sourceTexture, bool mirrorImage, float minConfidence)
    {
        // Iterate through the key points
        for (int k = 0; k < keypoints.Length; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (keypoints[k].score >= minConfidence / 100f)
            {
                // Activate the current key point GameObject
                this.keypoints[k].GetComponent<MeshRenderer>().enabled = true;
            }
            else
            {
                // Deactivate the current key point GameObject
                this.keypoints[k].GetComponent<MeshRenderer>().enabled = false;
            }

            // Scale the keypoint position to the original resolution
            Vector2 coords = keypoints[k].position * sourceScale;

            // Flip the keypoint position vertically
            coords.y = sourceTexture.height - coords.y;

            // Mirror the x position if using a webcam
            if (mirrorImage) coords.x = sourceTexture.width - coords.x;

            // Update the current key point location
            // Set the z value to -1f to place it in front of the video screen
            this.keypoints[k].position = new Vector3(coords.x, coords.y, -1f);
        }
    }

    /// <summary>
    /// Draw the pose skeleton based on the latest location data
    /// </summary>
    public void UpdateLines()
    {
        // Iterate through the joint pairs
        for (int i = 0; i < jointPairs.Length; i++)
        {
            // Set the GameObject for the starting key point
            Transform startingKeyPoint = keypoints[jointPairs[i].Item1];
            // Set the GameObject for the ending key point
            Transform endingKeyPoint = keypoints[jointPairs[i].Item2];

            // Check if both the starting and ending key points are active
            if (startingKeyPoint.GetComponent<MeshRenderer>().enabled &&
                endingKeyPoint.GetComponent<MeshRenderer>().enabled)
            {
                // Activate the line
                lines[i].SetActive(true);

                LineRenderer lineRenderer = lines[i].GetComponent<LineRenderer>();
                // Update the starting position
                lineRenderer.SetPosition(0, startingKeyPoint.position);
                // Update the ending position
                lineRenderer.SetPosition(1, endingKeyPoint.position);
            }
            else
            {
                // Deactivate the line
                lines[i].SetActive(false);
            }
        }
    }
}
