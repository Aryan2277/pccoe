# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your local server
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # Local server address
    api_key="**********"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="aryan-fqdxw",
    workflow_id="detect-count-and-visualize-3",
    images={
        "image": "D:\VS Code\edi2\images\Underwater_garbage\test\images\1bc7-iudfmpmn7245599_jpg.rf.c753c3518aa3bd30463da5e4b94a48be.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
