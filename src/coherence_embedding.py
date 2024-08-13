import json
import logging
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

class CoherenceEmbedding(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_id: str,
            bedrock_client
    ):
        self.model_id = model_id
        self.bedrock_client = bedrock_client

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return self.get_embeddings(input)
    
    def get_embeddings(self, texts):
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts debe ser una lista de cadenas.")

        try:
            # Serializar el cuerpo de la solicitud a JSON
            payload = json.dumps({'texts': texts, 'input_type':'search_query'})

            logging.info("Embed Payload: %s", payload)

            # Enviar la solicitud al modelo
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=payload,
                contentType='application/json'
            )

            logging.info("Embed response: %s", response)

            # Leer y procesar la respuesta
            response_body = response['body'].read().decode('utf-8')
            result = json.loads(response_body)

            embeddings = result.get('embeddings', [])
            if embeddings:
                return (embeddings)
            else:
                print("No se encontraron embeddings en la respuesta.")
                return None

        except json.JSONDecodeError as json_error:
            print(f"Error al procesar JSON: {json_error}")
            return None
        except KeyError as key_error:
            print(f"Error en la respuesta de la API: {key_error}")
            return None
        except Exception as e:
            print(f"Error al obtener embeddings: {e}")
            return None