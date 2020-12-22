using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Video;

public class PoseNet : MonoBehaviour
{
    [Tooltip("The input image that will be fed to the model")]
    public RenderTexture videoTexture;

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

    [Tooltip("The screen for viewing preprocessed images")]
    public GameObject inputScreen;

    [Tooltip("Stores the preprocessed image")]
    public RenderTexture inputTexture;

    [Tooltip("The model asset file to use when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The minimum confidence level required to display the key point")]
    [Range(0, 100)]
    public int minConfidence = 70;

    [Tooltip("The list of key point GameObjects that make up the pose skeleton")]
    public GameObject[] keypoints;

    // The compiled model used for performing inference
    private Model m_RunTimeModel;

    // The interface used to execute the neural network
    private IWorker engine;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer = "float_heatmaps";

    // The name for the offsets layer in the model asset
    private string offsetsLayer = "float_short_offsets";

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";

    // The number of key points estimated by the model
    private const int numKeypoints = 17;

    // Stores the current estimated 2D keypoint locations in videoTexture
    // and their associated confidence values
    private float[][] keypointLocations = new float[numKeypoints][];

    // Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The height of the current video source
    private int videoHeight;

    // The width of the current video source
    private int videoWidth;


    // Start is called before the first frame update
    void Start()
    {
        // Get a reference to the Video Player GameObject
        GameObject videoPlayer = GameObject.Find("Video Player");
        
        // Get the Transform component for the VideoScreen GameObject
        Transform videoScreen = GameObject.Find("VideoScreen").transform;

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

        // Release the current videoTexture
        videoTexture.Release();
        // Create a new videoTexture using the current video dimensions
        videoTexture = new RenderTexture(videoWidth, videoHeight, 24, RenderTextureFormat.ARGB32);

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

        // Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(modelAsset);

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


        // Preprocess the image for the current frame
        Texture2D processedImage = PreprocessImage();

        if (displayInput)
        {
            // Activate the InputScreen GameObject
            inputScreen.SetActive(true);
            // Create a temporary Texture2D to store the rescaled input image
            Texture2D scaledInputImage = ScaleInputImage(processedImage);
            // Copy the data from the Texture2D to the RenderTexture
            Graphics.Blit(scaledInputImage, inputTexture);
            // Destroy the temporary Texture2D
            Destroy(scaledInputImage);
        }
        else
        {
            // Deactivate the InputScreen GameObject
            inputScreen.SetActive(false);
        }

        // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
        Tensor input = new Tensor(processedImage, channels: 3);

        // Execute neural network with the provided input
        engine.Execute(input);

        // Determine the key point locations
        ProcessOutput(engine.PeekOutput(predictionLayer), engine.PeekOutput(offsetsLayer));

        // Update the positions for the key point GameObjects
        UpdateKeyPointPositions();

        // Release GPU resources allocated for the Tensor
        input.Dispose();
        // Remove the processedImage variable
        Destroy(processedImage);
    }

    /// <summary>
    /// Prepare the image to be fed into the neural network
    /// </summary>
    /// <returns>The processed image</returns>
    private Texture2D PreprocessImage()
    {
        // Create a new Texture2D with the same dimensions as videoTexture
        Texture2D imageTexture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGBA32, false);

        // Copy the RenderTexture contents to the new Texture2D
        Graphics.CopyTexture(videoTexture, imageTexture);

        // Make a temporary Texture2D to store the resized image
        Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);
        // Remove the original imageTexture
        Destroy(imageTexture);

        // Apply model-specific preprocessing
        imageTexture = PreprocessResNet(tempTex);
        // Remove the temporary Texture2D
        Destroy(tempTex);

        return imageTexture;
    }

    /// <summary>
    /// Resize the provided Texture2D
    /// </summary>
    /// <param name="image">The image to be resized</param>
    /// <param name="newWidth">The new image width</param>
    /// <param name="newHeight">The new image height</param>
    /// <returns>The resized image</returns>
    private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
    {
        // Create a temporary RenderTexture
        RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
        // Make the temporary RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Copy the Texture2D to the temporary RenderTexture
        Graphics.Blit(image, rTex);
        // Create a new Texture2D with the new Dimensions
        Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

        // Copy the temporary RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);

        // Make the temporary RenderTexture not the active RenderTexture
        RenderTexture.active = null;

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);
        return nTex;
    }

    /// <summary>
    /// Perform model-specific preprocessing on the GPU
    /// </summary>
    /// <param name="inputImage">The image to be processed</param>
    /// <returns>The processed image</returns>
    private Texture2D PreprocessResNet(Texture2D inputImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the PreprocessResNet function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel("PreprocessResNet");
        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, inputImage.height / numthreads, inputImage.width / numthreads, 1);
        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);
        // Make the HDR RenderTexture not the active RenderTexture
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);
        return nTex;
    }

    /// <summary>
    /// Rescale the pixel values from [0, 255] to [0.0, 1.0]
    /// </summary>
    /// <param name="inputImage"></param>
    /// <returns>The rescaled image</returns>
    private Texture2D ScaleInputImage(Texture2D inputImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the ScaleInputImage function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel("ScaleInputImage");
        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, inputImage.height / numthreads, inputImage.width / numthreads, 1);
        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);
        // Make the HDR RenderTexture not the active RenderTexture
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);
        return nTex;
    }

    /// <summary>
    /// Determine the estimated key point locations using the heatmaps and offsets tensors
    /// </summary>
    /// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
    /// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
    private void ProcessOutput(Tensor heatmaps, Tensor offsets)
    {
        // Calculate the stride used to scale down the inputImage
        float stride = (imageHeight - 1) / (heatmaps.shape[1] - 1);
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
            var locationInfo = LocateKeyPointIndex(heatmaps, offsets, k);

            // The (x, y) coordinates containing the confidence value in the current heatmap
            var coords = locationInfo.Item1;
            // The accompanying offset vector for the current coords
            var offset_vector = locationInfo.Item2;
            // The associated confidence value
            var confidenceValue = locationInfo.Item3;

            // Calcluate the X-axis position
            // Scale the X coordinate up to the inputImage resolution
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            // Compensate for any change in aspect ratio
            float xPos = (coords[0]*stride + offset_vector[0])*scale;

            // Calculate the Y-axis position
            // Scale the Y coordinate up to the inputImage resolution and subtract it from the imageHeight
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            float yPos = (imageHeight - (coords[1]*stride + offset_vector[1]))*scale;

            if (videoTexture.width > videoTexture.height)
            {
                xPos *= unsqueezeScale;
            }
            else
            {
                yPos *= unsqueezeScale;
            }

            // Flip the x position if using a webcam
            if (useWebcam)
            {
                xPos = videoTexture.width - xPos;
            }

            // Update the estimated key point location in the source image
            keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };
        }
    }

    /// <summary>
    /// Find the heatmap index that contains the highest confidence value and the associated offset vector
    /// </summary>
    /// <param name="heatmaps"></param>
    /// <param name="offsets"></param>
    /// <param name="keypointIndex"></param>
    /// <returns>The heatmap index, offset vector, and associated confidence value</returns>
    private (float[], float[], float) LocateKeyPointIndex(Tensor heatmaps, Tensor offsets, int keypointIndex)
    {
        // Stores the highest confidence value found in the current heatmap
        float maxConfidence = 0f;

        // The (x, y) coordinates containing the confidence value in the current heatmap
        float[] coords = new float[2];
        // The accompanying offset vector for the current coords
        float[] offset_vector = new float[2];

        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.shape[1]; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.shape[2]; x++)
            {
                if (heatmaps[0, y, x, keypointIndex] > maxConfidence)
                {
                    // Update the highest confidence for the current key point
                    maxConfidence = heatmaps[0, y, x, keypointIndex];

                    // Update the estimated key point coordinates
                    coords = new float[] { x, y };

                    // Update the offset vector for the current key point location
                    offset_vector = new float[]
                    {
                            // X-axis offset
                            offsets[0, y, x, keypointIndex + numKeypoints],
                            // Y-axis offset
                            offsets[0, y, x, keypointIndex]
                    };
                }
            }
        }

        return ( coords, offset_vector, maxConfidence );
    }

    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    private void UpdateKeyPointPositions()
    {
        // Iterate through the key points
        for (int k = 0; k < numKeypoints; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (keypointLocations[k][2] >= minConfidence / 100f)
            {
                // Activate the current key point GameObject
                keypoints[k].SetActive(true);
                //keypoints[k].activeInHierarchy
            }
            else
            {
                // Deactivate the current key point GameObject
                keypoints[k].SetActive(false);
            }

            // Create a new position Vector3
            // Set the z value to -1f to place it in front of the video screen
            Vector3 newPos = new Vector3(keypointLocations[k][0], keypointLocations[k][1], -1f);

            // Update the current key point location
            keypoints[k].transform.position = newPos;
        }
    }

}
