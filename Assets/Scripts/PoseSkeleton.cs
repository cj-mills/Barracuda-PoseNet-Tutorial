using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PoseSkeleton
{
    public static string[] partNames = new string[]{
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };



    private int numKeypoints = partNames.Length;

    // The list of key point GameObjects that make up the pose skeleton
    public Transform[] keypoints;

    // The GameObjects that contain data for the lines between key points
    private GameObject[] lines;

    // The line renderers the draw the lines between key points
    private LineRenderer[] lineRenderers;
    
    // The pairs of key points that should be connected on a body
    private int[][] jointPairs;

    // The width for the skeleton lines
    private float lineWidth = 5.0f;

    Material keypointMat;

    // Start is called before the first frame update
    public PoseSkeleton()
    {
        this.keypoints = new Transform[numKeypoints];

        keypointMat = new Material(Shader.Find("Unlit/Color"));
        keypointMat.color = Color.yellow;

        for (int i = 0; i < numKeypoints; i++)
        {
            this.keypoints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
            this.keypoints[i].position = new Vector3(0, 0, 0);
            this.keypoints[i].localScale = new Vector3(10, 10, 0);
            this.keypoints[i].gameObject.GetComponent<MeshRenderer>().material = keypointMat;
            this.keypoints[i].gameObject.name = partNames[i];
        }


        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.transform.position = new Vector3(0, 1.5f, 0);

        // The number of joint pairs
        int numPairs = keypoints.Length + 1;
        // Initialize the lines array
        lines = new GameObject[numPairs];
        // Initialize the lineRenderers array
        lineRenderers = new LineRenderer[numPairs];
        // Initialize the jointPairs array
        jointPairs = new int[numPairs][];
        
        // Initialize the pose skeleton
        InitializeSkeleton();
    }


    public void ToggleKeypoints(bool show)
    {
        foreach(Transform transform in keypoints)
        {
            transform.GetComponent<MeshRenderer>().enabled = show;
        }
    }

    public void ToggleLines(bool show)
    {
        foreach (LineRenderer lineRenderer in lineRenderers)
        {
            lineRenderer.enabled = show;
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
    private void InitializeLine(int pairIndex, int startIndex, int endIndex, float width, Color color)
    {
        // Create a new joint pair with the specified start and end point indices
        jointPairs[pairIndex] = new int[] { startIndex, endIndex };

        // Create new line GameObject
        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);
        
        // Add LineRenderer component
        lineRenderers[pairIndex] = lines[pairIndex].AddComponent<LineRenderer>();
        // Make LineRenderer Shader Unlit
        lineRenderers[pairIndex].material = new Material(Shader.Find("Unlit/Color"));
        // Set the material color
        lineRenderers[pairIndex].material.color = color;
        
        // The line will consist of two points
        lineRenderers[pairIndex].positionCount = 2;

        // Set the width from the start point
        lineRenderers[pairIndex].startWidth = width;
        // Set the width from the end point
        lineRenderers[pairIndex].endWidth = width;
    }

    /// <summary>
    /// Initialize the pose skeleton
    /// </summary>
    private void InitializeSkeleton()
    {
        // Nose to left eye
        InitializeLine(0, 0, 1, lineWidth, Color.magenta);
        // Nose to right eye
        InitializeLine(1, 0, 2, lineWidth, Color.magenta);
        // Left eye to left ear
        InitializeLine(2, 1, 3, lineWidth, Color.magenta);
        // Right eye to right ear
        InitializeLine(3, 2, 4, lineWidth, Color.magenta);

        // Left shoulder to right shoulder
        InitializeLine(4, 5, 6, lineWidth, Color.red);
        // Left shoulder to left hip
        InitializeLine(5, 5, 11, lineWidth, Color.red);
        // Right shoulder to right hip
        InitializeLine(6, 6, 12, lineWidth, Color.red);
        // Left shoulder to right hip
        InitializeLine(7, 5, 12, lineWidth, Color.red);
        // Right shoulder to left hip
        InitializeLine(8, 6, 11, lineWidth, Color.red);
        // Left hip to right hip
        InitializeLine(9, 11, 12, lineWidth, Color.red);

        // Left Arm
        InitializeLine(10, 5, 7, lineWidth, Color.green);
        InitializeLine(11, 7, 9, lineWidth, Color.green);
        // Right Arm
        InitializeLine(12, 6, 8, lineWidth, Color.green);
        InitializeLine(13, 8, 10, lineWidth, Color.green);

        // Left Leg
        InitializeLine(14, 11, 13, lineWidth, Color.blue);
        InitializeLine(15, 13, 15, lineWidth, Color.blue);
        // Right Leg
        InitializeLine(16, 12, 14, lineWidth, Color.blue);
        InitializeLine(17, 14, 16, lineWidth, Color.blue);
    }

    /// <summary>
    /// Draw the pose skeleton based on the latest location data
    /// </summary>
    public void RenderSkeleton()
    {
        // Iterate through the joint pairs
        for (int i = 0; i < jointPairs.Length; i++)
        {
            // Set the start point index
            int startpointIndex = jointPairs[i][0];
            // Set the end poin indext
            int endpointIndex = jointPairs[i][1];

            // Set the GameObject for the starting key point
            GameObject startingKeyPoint = keypoints[startpointIndex].gameObject;
            // Set the GameObject for the ending key point
            GameObject endingKeyPoint = keypoints[endpointIndex].gameObject;

            // Get the starting position for the line
            Vector3 startPos = new Vector3(startingKeyPoint.transform.position.x,
                                           startingKeyPoint.transform.position.y,
                                           startingKeyPoint.transform.position.z);
            // Get the ending position for the line
            Vector3 endPos = new Vector3(endingKeyPoint.transform.position.x,
                                         endingKeyPoint.transform.position.y,
                                         endingKeyPoint.transform.position.z);

            // Check if both the starting and ending key points are active
            if (startingKeyPoint.GetComponent<MeshRenderer>().enabled && 
                endingKeyPoint.GetComponent<MeshRenderer>().enabled)
            {
                // Activate the line
                lineRenderers[i].gameObject.SetActive(true);
                // Update the starting position
                lineRenderers[i].SetPosition(0, startPos);
                // Update the ending position
                lineRenderers[i].SetPosition(1, endPos);
            }
            else
            {
                // Deactivate the line
                lineRenderers[i].gameObject.SetActive(false);
            }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="imageDimemsion"></param>
    /// <param name="position"></param>
    /// <returns></returns>
    private float FlipCoords(int imageDimemsion, float position)
    {
        return imageDimemsion - position;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="keypoint"></param>
    /// <param name="sourceScale"></param>
    /// <param name="unsqueezeScale"></param>
    /// <returns></returns>
    private Vector2 ScaleOutput(Vector2 coords, Vector2 sourceDims, float sourceScale, float unsqueezeScale)
    {
        // Scale the position up to the videoTexture resolution
        coords.x *= sourceScale;
        coords.y *= sourceScale;

        if (sourceDims.x > sourceDims.y)
        {
            coords.x *= unsqueezeScale;
        }
        else
        {
            coords.y *= unsqueezeScale;
        }

        return coords;
    }

    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    public void UpdateKeyPointPositions(PoseNetClass.Keypoint[] keypoints,
        float sourceScale, float unsqueezeScale, Vector2 sourceDims, bool mirrorImage, float minConfidence)
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

            Vector2 coords = ScaleOutput(keypoints[k].position, sourceDims, sourceScale, unsqueezeScale);
            coords.y = FlipCoords((int)sourceDims.y, coords.y);

            // Mirror the x position if using a webcam
            if (mirrorImage) coords.x = FlipCoords((int)sourceDims.x, coords.x);

            // Update the current key point location
            // Set the z value to -1f to place it in front of the video screen
            this.keypoints[k].position = new Vector3(coords.x, coords.y, -1f);
        }
    }

}
