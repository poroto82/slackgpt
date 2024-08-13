import json
import boto3
from botocore.config import Config
from dotenv import load_dotenv

class TextGenerator:
    def __init__(self, logger, model_id, region_name='us-east-1'):
        """Initialize the TextGenerator with model_id and AWS region."""
        self.model_id = model_id
        self.logger = logger
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            config=Config(read_timeout=1000)
        )

    def _invoke_model(self, payload):
        """Invoke the Bedrock model with the provided payload."""
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
            contentType='application/json'
        )
        result = json.loads(response['body'].read().decode('utf-8'))
        return result['content'][0]['text']

    def generate_text(self, prompt, retrieved_data):
        """Generate text based on the provided prompt and retrieved data."""
        payload = {
            "max_tokens": 1024,
            "system": "You are a knowledgeable assistant that uses the provided data if relevant.",
             "messages": [
                {
                    "role": "user",
                    "content": f"Context: {retrieved_data}\n\nQuery: {prompt}"
                }
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
        return self._invoke_model(payload)
    
    def generate_summary(self, prompt):
        """Generate a summary based on the provided prompt."""
        payload = {
            "max_tokens": 1024,
            "system": "Use the retrieved data to generate a summary of the instructions given.",
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        }
        return self._invoke_model(payload)
