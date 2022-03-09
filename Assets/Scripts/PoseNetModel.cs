using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;


[CreateAssetMenu(menuName = "PoseNet Model")]
[System.Serializable]
public class PoseNetModel : ScriptableObject
{
    [Tooltip("The ONNX model asset file to use when performing inference")]
    public NNModel modelAsset;

    public Material preprocessingMaterial;

    public int heatmapLayerIndex = 0;
    public int offsetsLayerIndex = 1;
    public int displacementFWDLayerIndex;
    public int displacementBWDLayerIndex;

    public void OnEnable()
    {

    }
}
