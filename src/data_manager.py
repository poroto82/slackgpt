
import chromadb
import boto3
import uuid
from coherence_embedding import CoherenceEmbedding
from text_utils import extract_keywords_nltk

class DataManager:
    def __init__(self,logger, model_id, chroma_db_path, collection_name="default"):

        self.logger = logger
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = model_id
        self.bedrock_embedding_function = CoherenceEmbedding(self.model_id, self.bedrock_client)
        self.collection = self.chroma_client.get_or_create_collection(collection_name,embedding_function=self.bedrock_embedding_function)
        

    def retrieve_documents(self, query_texts, max_distance=1.0):
        """Retrieve documents based on query embeddings."""
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=10  # Número de resultados que deseas obtener
            )
            
            self.logger.info("Raw results: %s", results)

            # Extraer los IDs de documentos y las distancias
            document_ids = results.get("ids", [])
            distances = results.get("distances", [])

            # Filtrar los resultados en función de la distancia
            filtered_ids = []
            for doc_ids, dists in zip(document_ids, distances):
                filtered_ids.extend([doc_id for doc_id, dist in zip(doc_ids, dists) if dist <= float(max_distance)])

            self.logger.info("Filtered Document Ids: %s", filtered_ids)

            # Devolver el primer ID si hay alguno que cumpla con la condición
            return filtered_ids[0] if filtered_ids else ''

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    def get_document_content(self, document_id):
        results = self.collection.get(ids=[document_id])
        return results["documents"][0] if results["documents"] else None

    def store_documents(self, texts):
        """Store multiple documents with their embeddings."""
        try:
            document_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            self.logger.info('Generated document IDs: %s', document_ids)

            # Extraer palabras clave
            keywords = [extract_keywords_nltk(text) for text in texts]
            self.logger.info('Extracted keywords: %s', keywords)

            # Agregar los documentos a la colección
            self.collection.add(
                ids=document_ids,
                #embeddings=embeddings,
                documents=texts,
                metadatas=[{"keywords": ",".join(keyword)} for keyword in keywords]
            )

            print("Documents stored successfully.")
        except Exception as e:
            print(f"Error storing documents: {e}")
            raise e
