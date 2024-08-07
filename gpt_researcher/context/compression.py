import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from gpt_researcher.utils.multithreading import run_in_thread

from .retriever import SearchAPIRetriever

CHUNK_SIZE = os.getenv("GPT_RESEARCHER_CHUNK_SIZE", 10_000)
OVERLAP = os.getenv("GPT_RESEARCHER_OVERLAP", 100)
SIMILARITY_TH = float(os.getenv("GPT_RESEARCHER_SIMILARITY_TH", "0.5"))


class ContextCompressor:
    def __init__(self, documents, embeddings, max_results=5, **kwargs):
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = SIMILARITY_TH

    def _get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
        )
        relevance_filter = EmbeddingsFilter(
            embeddings=self.embeddings, similarity_threshold=self.similarity_threshold
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(pages=self.documents)
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def _pretty_print_docs(self, docs, top_n):
        return f"\n".join(
            f"Source: {d.metadata.get('source')}\n"
            f"Title: {d.metadata.get('title')}\n"
            f"Content: {d.page_content}\n"
            for i, d in enumerate(docs)
            if i < top_n
        )

    async def get_context(self, query, max_results=5):
        compressed_docs = self._get_contextual_retriever()
        relevant_docs = await run_in_thread(compressed_docs.get_relevant_documents, query)

        return self._pretty_print_docs(relevant_docs, max_results)
