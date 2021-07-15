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
    //[Range(0, 100)]
    public int nmsRadius = 20;

    [Tooltip("A point (y, x) is considered as root part candidate if its score is a " + 
        "maximum in a window |y - y'| <= kLocalMaximumRadius, |x - x'| <=  kLocalMaximumRadius")]
    public int kLocalMaximumRadius = 1;

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

    PoseSkeleton[] skeletons;

    // Stores the current estimated 2D keypoint locations in videoTexture
    Utils.Keypoint[][] poses;

    // Start is called before the first frame update
    void Start()
    {
        if (estimationType == EstimationType.SinglePose) maxPoses = 1;
        skeletons = new PoseSkeleton[maxPoses];

        for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton();

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
        // Copy webcamTexture to videoTexture if using webcam
        if (useWebcam) Graphics.Blit(webcamTexture, videoTexture);


        if (imageWidth != rTex.width || imageHeight != rTex.height)
        {
            RenderTexture.ReleaseTemporary(rTex);
            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(imageWidth, imageHeight, 24, RenderTextureFormat.ARGBHalf);
        }

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(videoTexture, rTex);

        ProcessImage(rTex);

        // Execute neural network with the provided input
        engine.Execute(input);

        
        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
        // The largest dimension of the videoTexture
        int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        float sourceScale = (float)minDimension / Mathf.Min(input.width, input.height);
        // The value used to compensate for resizing the source image to a square aspect ratio
        float unsqueezeScale = (float)maxDimension / minDimension;


        Vector2Int sourceDims = new Vector2Int(videoTexture.width, videoTexture.height);

        ProcessOutput(engine);


        for (int i = 0; i < skeletons.Length; i++)
        {
            if (i <= poses.Length - 1)
            {
                skeletons[i].ToggleLines(true);

                // Update the positions for the key point GameObjects
                skeletons[i].UpdateKeyPointPositions(
                    poses[i], sourceScale, unsqueezeScale, 
                    sourceDims, useWebcam, minConfidence);
                skeletons[i].RenderSkeleton();
            }
            else
            {
                skeletons[i].ToggleKeypoints(false);
                skeletons[i].ToggleLines(false);
            }
        }

        // Release GPU resources allocated for the Tensor
        input.Dispose();
    }

    private void ProcessOutput(IWorker engine)
    {
        Tensor heatmaps = engine.PeekOutput(predictionLayer);
        Tensor offsets = engine.PeekOutput(offsetsLayer);
        Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
        Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

        // Calculate the stride used to scale down the inputImage
        int stride = (imageHeight - 1) / (heatmaps.shape.height - 1);
        stride -= (stride % 8);

        if (estimationType == EstimationType.SinglePose)
        {
            poses = new Utils.Keypoint[1][];

            // Determine the key point locations
            poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
        }
        else
        {
            // Determine the key point locations
            poses = Utils.DecodeMultiplePoses(
                heatmaps, offsets,
                displacementFWD, displacementBWD,
                stride: stride, maxPoseDetections: maxPoses,
                scoreThreshold: scoreThreshold, 
                nmsRadius: nmsRadius, kLocalMaximumRadius: kLocalMaximumRadius);
        }

        heatmaps.Dispose();
        offsets.Dispose();
        displacementFWD.Dispose();
        displacementBWD.Dispose();
    }

    private void ProcessImage(RenderTexture image)
    {
        if (useGPU)
        {
            // Apply preprocessing steps
            ProcessImageGPU(image, preProcessFuncName);

            // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
            input = new Tensor(image, channels: 3);
        }
        else
        {
            input = new Tensor(image, channels: 3);
            float[] tensor_array = input.data.Download(input.shape);

            if (modelType == ModelType.MobileNet)
            {
                Utils.PreprocessMobilenet(tensor_array);
            }
            else
            {
                Utils.PreprocessResnet(tensor_array);
            }
            input = new Tensor(input.shape.batch,
                               input.shape.height,
                               input.shape.width,
                               input.shape.channels,
                               tensor_array);
        }
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, string functionName)
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

}
