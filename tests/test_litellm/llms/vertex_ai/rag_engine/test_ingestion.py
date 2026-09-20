from typing import Final

import pytest

from litellm.llms.vertex_ai.rag_engine.ingestion import VertexAIRAGIngestion

CORPUS_ID: Final = "1234567890"
GCS_URI: Final = "gs://my-bucket/rag/notes.txt"


class _OfflineVertexAIRAGIngestion(VertexAIRAGIngestion):
    def __init__(self, ingest_options):
        super().__init__(ingest_options=ingest_options)
        self.imported_uris: list[str] = []

    async def _upload_file_to_gcs(self, file_content: bytes, filename: str, content_type: str) -> str:
        return GCS_URI

    async def _import_file_to_corpus_via_sdk(self, gcs_uri: str) -> None:
        self.imported_uris.append(gcs_uri)


@pytest.mark.asyncio
async def test_ingest_stores_file_data_in_the_corpus():
    ingestion = _OfflineVertexAIRAGIngestion(
        {
            "vector_store": {
                "custom_llm_provider": "vertex_ai",
                "vector_store_id": CORPUS_ID,
                "vertex_project": "my-project",
                "gcs_bucket": "my-bucket",
            }
        }
    )

    response = await ingestion.ingest(file_data=("notes.txt", b"hello vertex", "text/plain"))

    assert response == {
        "id": ingestion.ingest_id,
        "status": "completed",
        "vector_store_id": CORPUS_ID,
        "file_id": GCS_URI,
    }
    assert ingestion.imported_uris == [GCS_URI]
