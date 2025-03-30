from ibm_watsonx_ai.foundation_models import ModelInference

# Your IBM Cloud credentials
apikey = "asd-"
project_id = "asd"
region = "us-south"  # or whichever region your instance is in

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

my_credentials = {
  "url": "https://us-south.ml.cloud.ibm.com",
  "apikey": {apikey},
}
client = APIClient(my_credentials)

# 2. Initialize the model inference
model_inference = ModelInference(
    model_id="meta-llama/llama-2-70b-chat",  # or another supported model
    project_id=project_id,
    credentials=my_credentials
)


# 3. Create a prompt
prompt_input = "Explain quantum computing in simple terms."

# 4. Generate a response
response = model_inference.generate_text(
    prompt=prompt_input,
    parameters={
        "decoding_method": "greedy",
        "max_new_tokens": 200,
        "temperature": 0.7
    }
)

# 5. Print the result
print(response["generated_text"])
