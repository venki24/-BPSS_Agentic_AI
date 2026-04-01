"""
Single shared embeddings instance for the whole app.
All tools import from here — model loads exactly once.
"""
from langchain_core.embeddings import Embeddings
from app.config import settings

_instance: Embeddings = None


def get_embeddings() -> Embeddings:
    global _instance
    if _instance is not None:
        return _instance

    if settings.EMBEDDING_PROVIDER == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        _instance = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    else:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        _instance = HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _instance
