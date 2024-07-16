import os


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
                    deployment=os.environ["AZURE_EMBEDDING_MODEL"], chunk_size=16
                )
            case "aim-loadbalancer":
                from model_loadbalancer import EmbeddingsBalancer

                _embeddings = EmbeddingsBalancer.from_engine(engine=embedding_model)

            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings

                _embeddings = HuggingFaceEmbeddings()

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings
