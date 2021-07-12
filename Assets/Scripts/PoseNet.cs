using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Video;

public class PoseNet : MonoBehaviour
{
    public enum EstimationType
    {
        MultiPose,
        SinglePose
    }

    public enum ModelType
    {
        MobileNet,
        ResNet50
    }

    [Tooltip("The ComputeShader that will perform the model-specific preprocessing")]
    public ComputeShader posenetShader;

    [Tooltip("The requested webcam height")]
    public int webcamHeight = 720;

    [Tooltip("The requested webcam width")]
    public int webcamWidth = 1280;

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("The height of the image being fed to the model")]
    public int imageHeight = 360;
    
    [Tooltip("The width of the image being fed to the model")]
    public int imageWidth = 360;

    [Tooltip("Turn the InputScreen on or off")]
    public bool displayInput = false;

    [Tooltip("Use webcam feed as input")]
    public bool useWebcam = false;

    [Tooltip("Use GPU for preprocessing")]
    public bool useGPU = true;

    [Tooltip("The screen for viewing preprocessed images")]
    public GameObject inputScreen;


    [Tooltip("The type of pose estimation to be performed")]
    public EstimationType estimationType = EstimationType.SinglePose;

    [Tooltip("The maximum number of posees to estimate")]
    [Range(1, 15)]
    public int maxPoses = 15;

    [Tooltip("The score threshold for multipose estimation")]
    [Range(0, 1.0f)]
    public float scoreThreshold = 0.25f;

    [Tooltip("Non-maximum suppression part distance")]
    [Range(0, 100)]
    public int nmsRadius = 20;

    [Tooltip("The model architecture used")]
    public ModelType modelType = ModelType.ResNet50;

    [Tooltip("The MobileNet model asset file to use when performing inference")]
    public NNModel mobileNetModelAsset;

    [Tooltip("The ResNet50 model asset file to use when performing inference")]
    public NNModel resnetModelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The minimum confidence level required to display the key point")]
    [Range(0, 100)]
    public int minConfidence = 70;

    
    // The compiled model used for performing inference
    private Model m_RunTimeModel;

    // The interface used to execute the neural network
    private IWorker engine;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer;

    // The name for the offsets layer in the model asset
    private string offsetsLayer;

    private string displacementFWDLayer;

    private string displacementBWDLayer;

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";

    
    // The number of key points estimated by the model
    private const int numKeypoints = 17;

    // Stores the current estimated 2D keypoint locations in videoTexture
    // and their associated confidence values
    private PoseNetClass.Pose singlePose = new PoseNetClass.Pose(new PoseNetClass.Keypoint[numKeypoints], 0f);

    // Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The height of the current video source
    private int videoHeight;

    // The width of the current video source
    private int videoWidth;

    private RenderTexture videoTexture;
    private RenderTexture rTex;

    private Transform videoScreen;

    private string preProcessFuncName;

    private Tensor input;

    PoseNetClass posenet = new PoseNetClass();

    PoseSkeleton[] skeletons;


    // Start is called before the first frame update
    void Start()
    {
        skeletons = new PoseSkeleton[maxPoses];

        maxPoses = estimationType == EstimationType.SinglePose ? 1 : maxPoses;

        for (int i = 0; i < maxPoses; i++)
        {
            skeletons[i] = new PoseSkeleton();
        }


        // Get a reference to the Video Player GameObject
        GameObject videoPlayer = GameObject.Find("Video Player");
        
        // Get the Transform component for the VideoScreen GameObject
        videoScreen = GameObject.Find("VideoScreen").transform;

        if (useWebcam)
        {
            // Create a new WebCamTexture
            webcamTexture = new WebCamTexture(webcamWidth, webcamHeight, webcamFPS);

            // Flip the VideoScreen around the Y-Axis
            videoScreen.rotation = Quaternion.Euler(0, 180, 0);
            // Invert the scale value for the Z-Axis
            videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);

            // Start the Camera
            webcamTexture.Play();

            // Deactivate the Video Player
            videoPlayer.SetActive(false);

            // Update the videoHeight
            videoHeight = (int)webcamTexture.height;
            // Update the videoWidth
            videoWidth = (int)webcamTexture.width;

        }
        else
        {
            // Update the videoHeight
            videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            // Update the videoWidth
            videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;
        }

        // Create a new videoTexture using the current video dimensions
        videoTexture = RenderTexture.GetTemporary(videoWidth, videoHeight, 24, RenderTextureFormat.ARGBHalf);

        rTex = RenderTexture.GetTemporary(imageWidth, imageHeight, 24, RenderTextureFormat.ARGBHalf);


        // Use new videoTexture for Video Player
        videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

        // Apply the new videoTexture to the VideoScreen Gameobject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);
        // Adjust the VideoScreen dimensions for the new videoTexture
        videoScreen.localScale = new Vector3(videoWidth, videoHeight, videoScreen.localScale.z);
        // Adjust the VideoScreen position for the new videoTexture
        videoScreen.position = new Vector3(videoWidth / 2, videoHeight / 2, 1);

        // Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(videoWidth / 2, videoHeight / 2, -(videoWidth / 2));
        // Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoHeight/2;

        if (modelType == ModelType.ResNet50)
        {
            heatmapLayer = "float_heatmaps";
            offsetsLayer = "float_short_offsets";
            displacementFWDLayer = "resnet_v1_50/displacement_fwd_2/BiasAdd";
            displacementBWDLayer = "resnet_v1_50/displacement_bwd_2/BiasAdd";

            minConfidence = 80;

            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(resnetModelAsset);
            preProcessFuncName = "PreprocessResNet";
        }
        else
        {
            heatmapLayer = "heatmap_2";
            offsetsLayer = "offset_2";
            displacementFWDLayer = "displacement_fwd_2";
            displacementBWDLayer = "displacement_bwd_2";

            minConfidence = 65;

            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(mobileNetModelAsset);
            preProcessFuncName = "PreprocessMobileNet";
        }

        // Create a model builder to modify the m_RunTimeModel
        var modelBuilder = new ModelBuilder(m_RunTimeModel);

        // Add a new Sigmoid layer that takes the output of the heatmap layer
        modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
    }

    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        RenderTexture.ReleaseTemporary(videoTexture);
        RenderTexture.ReleaseTemporary(rTex);

        // Release the resources allocated for the inference engine
        engine.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        if (useWebcam)
        {
            // Copy webcamTexture to videoTexture
            Graphics.Blit(webcamTexture, videoTexture);
        }


        if (imageWidth != rTex.width || imageHeight != rTex.height)
        {
            RenderTexture.ReleaseTemporary(rTex);
            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(imageWidth, imageHeight, 24, RenderTextureFormat.ARGBHalf);
        }

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(videoTexture, rTex);


        if (useGPU)
        {
            // Apply preprocessing steps
            ProcessImage(rTex, preProcessFuncName);

            // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
            input = new Tensor(rTex, channels: 3);
        }
        else
        {
            input = new Tensor(rTex, channels: 3);
            float[] tensor_array = input.data.Download(input.shape);

            if (modelType == ModelType.MobileNet)
            {
                PreprocessMobilenet(tensor_array);
            }
            else
            {
                PreprocessResnet(tensor_array);
            }
            input = new Tensor(input.shape.batch, 
                               input.shape.height, 
                               input.shape.width, 
                               input.shape.channels, 
                               tensor_array);
        }

        // Execute neural network with the provided input
        engine.Execute(input);

        Tensor heatmap = engine.PeekOutput(predictionLayer);
        Tensor offsets = engine.PeekOutput(offsetsLayer);
        Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
        Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

        if (estimationType == EstimationType.SinglePose)
        {
            // Determine the key point locations
            ProcessOutput(engine.PeekOutput(predictionLayer), engine.PeekOutput(offsetsLayer));

            GameObject[] gameObjects = new GameObject[skeletons[0].keypoints.Length];

            for (int i = 0; i < gameObjects.Length; i++)
            {
                gameObjects[i] = skeletons[0].keypoints[i].gameObject;
            }

            // Update the positions for the key point GameObjects
            //UpdateKeyPointPositions(gameObjects);
            UpdateKeyPointPositions(singlePose, gameObjects);

            skeletons[0].RenderSkeleton();
        }
        else
        {
            // Calculate the stride used to scale down the inputImage
            float stride = (imageHeight - 1) / (heatmap.shape.height - 1);
            stride -= (stride % 8);

            // Determine the key point locations
            PoseNetClass.Pose[] poses = posenet.DecodeMultiplePoses(
                heatmap, offsets,
                displacementFWD,
                displacementBWD,
                outputStride: (int)stride, maxPoseDetections: maxPoses,
                scoreThreshold: scoreThreshold, nmsRadius: nmsRadius);


            int index = 0;
            foreach (PoseNetClass.Pose pose in poses)
            {
                GameObject[] gameObjects = new GameObject[skeletons[index].keypoints.Length];

                for (int i=0; i < gameObjects.Length; i++)
                {
                    gameObjects[i] = skeletons[index].keypoints[i].gameObject;
                }

                // Update the positions for the key point GameObjects
                UpdateKeyPointPositions2(pose, gameObjects);
                skeletons[index].RenderSkeleton();

                index++;
            }
        }

        // Release GPU resources allocated for the Tensor
        input.Dispose();

        heatmap.Dispose();
        offsets.Dispose();
        displacementFWD.Dispose();
        displacementBWD.Dispose();
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns></returns>
    private void ProcessImage(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    private void PreprocessMobilenet(float[] tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(2.0f * tensor[i] / 1.0f) - 1.0f;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    private void PreprocessResnet(float[] tensor)
    {
        float[] imagenetMean = new float[] { -123.15f, -115.90f, -103.06f };

        for (int i = 0; i < tensor.Length / 3; i++)
        {
            tensor[i * 3 + 0] = (float)tensor[i * 3 + 0] * 255f + imagenetMean[0];
            tensor[i * 3 + 1] = (float)tensor[i * 3 + 1] * 255f + imagenetMean[1];
            tensor[i * 3 + 2] = (float)tensor[i * 3 + 2] * 255f + imagenetMean[2];
        }
    }

    /// <summary>
    /// Determine the estimated key point locations using the heatmaps and offsets tensors
    /// </summary>
    /// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
    /// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
    private void ProcessOutput(Tensor heatmaps, Tensor offsets)
    {
        // Calculate the stride used to scale down the inputImage
        float stride = (imageHeight - 1) / (heatmaps.shape.height - 1);
        stride -= (stride % 8);

        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
        // The largest dimension of the videoTexture
        int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        float scale = (float) minDimension / (float) Mathf.Min(imageWidth, imageHeight);
        // The value used to compensate for resizing the source image to a square aspect ratio
        float unsqueezeScale = (float)maxDimension / (float)minDimension;

        

        // Iterate through heatmaps
        for (int k = 0; k < numKeypoints; k++)
        {
            // Get the location of the current key point and its associated confidence value
            PoseNetClass.Keypoint keypoint = LocateKeyPointIndex(heatmaps, k);

            // The accompanying offset vector for the current coords
            Vector2 offset_vector = PoseNetClass.GetOffsetPoint((int)keypoint.position.y, (int)keypoint.position.x, k, offsets);

            // Calcluate the X-axis position
            // Scale the X coordinate up to the inputImage resolution
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            // Compensate for any change in aspect ratio
            //float xPos = (keypoint.position.x*stride + offset_vector.x)*scale;
            keypoint.position.x = (keypoint.position.x * stride + offset_vector.x) * scale;

            // Calculate the Y-axis position
            // Scale the Y coordinate up to the inputImage resolution and subtract it from the imageHeight
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            //float yPos = (imageHeight - (keypoint.position.y*stride + offset_vector.y))*scale;
            keypoint.position.y = (imageHeight - (keypoint.position.y * stride + offset_vector.y)) * scale;

            if (videoTexture.width > videoTexture.height)
            {
                //xPos *= unsqueezeScale;
                keypoint.position.x *= unsqueezeScale;
            }
            else
            {
                //yPos *= unsqueezeScale;
                keypoint.position.y *= unsqueezeScale; 
            }

            // Flip the x position if using a webcam
            if (useWebcam)
            {
                //xPos = videoTexture.width - xPos;
                keypoint.position.x = videoTexture.width - keypoint.position.x;
            }

            // Update the estimated key point location in the source image
            //keypointLocations[k] = new float[] { xPos, yPos, keypoint.score };
            singlePose.keypoints[k] = keypoint;
        }
    }

    /// <summary>
    /// Find the heatmap index that contains the highest confidence value and the associated offset vector
    /// </summary>
    /// <param name="heatmaps"></param>
    /// <param name="offsets"></param>
    /// <param name="keypointIndex"></param>
    /// <returns>The heatmap index, offset vector, and associated confidence value</returns>
    private PoseNetClass.Keypoint LocateKeyPointIndex(Tensor heatmaps, int keypointIndex)
    {
        // Stores the highest confidence value found in the current heatmap
        float maxConfidence = 0f;

        // The (x, y) coordinates containing the confidence value in the current heatmap
        Vector2 coords = new Vector2();

        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.shape.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.shape.width; x++)
            {
                if (heatmaps[0, y, x, keypointIndex] > maxConfidence)
                {
                    // Update the highest confidence for the current key point
                    maxConfidence = heatmaps[0, y, x, keypointIndex];

                    // Update the estimated key point coordinates
                    coords.x = x;
                    coords.y = y;
                }
            }
        }

        PoseNetClass.Keypoint keypoint = new PoseNetClass.Keypoint(maxConfidence, coords, 
                                                                   PoseNetClass.partNames[keypointIndex]);
        return keypoint;
    }

    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    private void UpdateKeyPointPositions(PoseNetClass.Pose pose, GameObject[] keypoints)
    {
        // Iterate through the key points
        for (int k = 0; k < numKeypoints; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (pose.keypoints[k].score >= minConfidence / 100f)
            {
                // Activate the current key point GameObject
                keypoints[k].GetComponent<MeshRenderer>().enabled = true;
            }
            else
            {
                // Deactivate the current key point GameObject
                keypoints[k].GetComponent<MeshRenderer>().enabled = false;
            }

            // Create a new position Vector3
            // Set the z value to -1f to place it in front of the video screen
            Vector3 newPos = new Vector3(pose.keypoints[k].position.x, pose.keypoints[k].position.y, -1f);

            // Update the current key point location
            keypoints[k].transform.position = newPos;
        }
    }

    private void UpdateKeyPointPositions2(PoseNetClass.Pose pose, GameObject[] keypoints)
    {
        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
        // The largest dimension of the videoTexture
        int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        float scale = (float)minDimension / (float)Mathf.Min(imageWidth, imageHeight);
        // The value used to compensate for resizing the source image to a square aspect ratio
        float unsqueezeScale = (float)maxDimension / (float)minDimension;

        // Iterate through the key points
        for (int k = 0; k < numKeypoints; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (pose.keypoints[k].score >= minConfidence / 100f)
            {
                // Activate the current key point GameObject
                keypoints[k].GetComponent<MeshRenderer>().enabled = true;
            }
            else
            {
                // Deactivate the current key point GameObject
                keypoints[k].GetComponent<MeshRenderer>().enabled = false;
            }

            float xPos = pose.keypoints[k].position.x * scale;

            // Calculate the Y-axis position
            // Scale the Y coordinate up to the inputImage resolution and subtract it from the imageHeight
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            float yPos = (imageHeight - pose.keypoints[k].position.y) * scale;

            if (videoTexture.width > videoTexture.height)
            {
                xPos *= unsqueezeScale;
            }
            else
            {
                yPos *= unsqueezeScale;
            }

            // Create a new position Vector3
            // Set the z value to -1f to place it in front of the video screen
            Vector3 newPos = new Vector3(xPos, yPos, -1f);

            // Update the current key point location
            keypoints[k].transform.position = newPos;
        }
    }

}
