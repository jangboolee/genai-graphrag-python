import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-5-nano", model_params={"reasoning_effort": "minimal"}
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


class TextLoader(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:

        # Process the file
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Return a PdfDocument
        return PdfDocument(
            text=text,
            document_info=DocumentInfo(path=str(filepath), metadata={}),
        )


data_loader = TextLoader()

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    embedder=embedder,
    from_pdf=True,
    pdf_loader=data_loader,
)


pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.txt"
print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
