# 1. Import the necessary library
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 2. Configure the client to connect to your local server
# The default URL for the local server is http://127.0.0.1:9001
# In your test.py file

client = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key="i3HLHjjhvWCpehLbgiSW"
)

model_id = "plastic-image-detection-ivewy/7"
image_path = "D:/VS Code/edi2/images/Underwater_garbage/test/images/images/Underwater_garbage/test/images/1bc7-iudfmpmn7245599_jpg.rf.c753c3518aa3bd30463da5e4b94a48be.jpg" # Fill in the path to an image

# 4. Run inference!
try:
    result = client.infer(image_path, model_id=model_id)
    # The first time you run this, the server will download your model weights.
    # This might take a moment.

    # 5. Print the prediction results
    print("Inference successful!")
    print(result)   

except Exception as e:
    print(f"An error occurred: {e}")