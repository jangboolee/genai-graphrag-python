import asyncio

from dotenv import load_dotenv
from neo4j_graphrag.experimental.components.schema import (
    SchemaFromTextExtractor,
)
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

schema_extractor = SchemaFromTextExtractor(
    llm=OpenAILLM(
        model_name="gpt-5-nano",
    ),
    use_structured_output=True,
)

texts = [
    "Neo4j is a graph database management system (GDBMS) developed by Neo4j Inc.",
    "Python is a programming language created by Guido van Rossum.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
]

# Extract the schema from the text
for text in texts:
    extracted_schema = asyncio.run(schema_extractor.run(text=text))
    print(f"Extracted schema for: {text}")
    for node in extracted_schema.node_types:
        print(node)
    for rel in extracted_schema.relationship_types:
        print(rel)
    for pattern in extracted_schema.patterns:
        print(pattern)
