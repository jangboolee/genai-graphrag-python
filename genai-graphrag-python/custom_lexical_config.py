import asyncio
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
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

config = LexicalGraphConfig(
    chunk_node_label="Section",
    document_node_label="Lesson",
    chunk_to_document_relationship_type="IN_LESSON",
    next_chunk_relationship_type="NEXT_SECTION",
    node_to_chunk_relationship_type="IN_SECTION",
    chunk_embedding_property="embeddings",
)

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    embedder=embedder,
    from_pdf=True,
    lexical_graph_config=config,
)

pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
