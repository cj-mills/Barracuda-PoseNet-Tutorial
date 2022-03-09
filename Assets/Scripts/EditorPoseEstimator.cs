using System.Collections;
using System.Collections.Generic;
using UnityEngine;


#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(PoseEstimator))]
public class EditorPoseEstimator : Editor
{
    public override void OnInspectorGUI()
    {
        // Get a reference to the associated inference manager component
        PoseEstimator poseEstimator = (PoseEstimator)target;

        // Check for changes to the Models property field
        EditorGUI.BeginChangeCheck();
        // Draw the default editor user interface for the inference manager
        base.OnInspectorGUI();
        if (EditorGUI.EndChangeCheck())
        {
            poseEstimator.OnUserInput();
        }
    }
}
#endif