import os

CHUNK_SIZE = os.getenv("GPT_RESEARCHER_EMBEDDING_CHUNK_SIZE", 16)


class Memory:
    def __init__(self, embedding_provider, embedding_model, **kwargs):

        _embeddings = None
        match embedding_provider:
            case "ollama":
                from langchain.embeddings import OllamaEmbeddings

                _embeddings = OllamaEmbeddings(model="llama2")
            case "openai":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings()
            case "azureopenai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    deployment=os.environ["AZURE_EMBEDDING_MODEL"],
                    chunk_size=CHUNK_SIZE,
                )
            case "aim-loadbalancer":
                from model_loadbalancer import EmbeddingsBalancer

                _embeddings = EmbeddingsBalancer.from_engine(
                    engine=embedding_model,
                    chunk_size=CHUNK_SIZE,
                )

            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings

                _embeddings = HuggingFaceEmbeddings()

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
