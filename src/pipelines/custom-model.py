import asyncio
import os
import pathlib
import webbrowser
import xml.etree.ElementTree as ET
from typing import Optional, Union

from pydantic import BaseModel
from rdflib import Graph

from cognee import add, config, prune
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.pipelines import Task, cognee_pipeline
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import render_graph
from cognee.tasks.documents import (
    check_permissions_on_documents,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text


logger = get_logger("bide")

async def merge_ontology(catalog_path: str, output_path: str) -> str:
    """
    Parses an XML catalog file to find all .rdf ontology files,
    merges them into a single RDF graph, and serializes it to a new file.

    Args:
        catalog_path: The file path to the XML catalog.
        output_path: The file path where the merged ontology will be saved.

    Returns:
        The path to the newly created merged ontology file.
    """
    print(f"Starting ontology merge from catalog: {catalog_path}")
    merged_graph = Graph()
    catalog_dir = os.path.dirname(catalog_path)

    # Define the namespace for the catalog XML
    ns = {'catalog': 'urn:oasis:names:tc:entity:xmlns:xml:catalog'}

    # Parse the XML catalog file
    tree = ET.parse(catalog_path)
    root = tree.getroot()

    # Find all 'uri' elements and extract the 'uri' attribute
    rdf_files = [
        uri.get('uri') for uri in root.findall('catalog:uri', ns)
    ]

    print(f"Found {len(rdf_files)} RDF files to merge.")

    for rdf_file in rdf_files:
        if rdf_file:
            # Construct the full, absolute path for each RDF file
            file_path = os.path.join(catalog_dir, rdf_file)
            if os.path.exists(file_path):
                print(f"  -> Parsing {file_path}")
                # Parse the file and add its contents to the merged graph
                merged_graph.parse(file_path)
            else:
                print(f"  -> WARNING: File not found, skipping: {file_path}")

    # Serialize the entire merged graph to the output file
    merged_graph.serialize(destination=output_path, format="xml")
    print(f"Serialization complete. Saving merged ontology to: {output_path}")

    return output_path

async def my_cognify(
    datasets: Union[str, list[str]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
):

    tasks = await get_default_tasks(user, graph_model, chunker, chunk_size, ontology_file_path)

    return await cognee_pipeline(
        tasks=tasks, datasets=datasets, user=user, pipeline_name="cognify_pipeline"
    )


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_documents, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            task_config={"batch_size": 10},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return default_tasks


async def ingest(file_path: str):
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await add(file_path)


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage")
        ).resolve()
    )

    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system")
        ).resolve()
    )

    ia_file_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, "../data/legal/input/ia.pdf")
        ).resolve()
    )

    catalog_file_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, "../data/ontologies/fibo-master/catalog-v001.xml")
        ).resolve()
    )

    merged_ontology_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, "../data/ontologies/legal_vc.owl")
        ).resolve()
    )

    config.data_root_directory(data_directory_path)
    config.system_root_directory(cognee_directory_path)
    config.entity_extraction_prompt = """Extract key legal concepts and their relationships.

    Extreact the concept classess, ex. Company, Person, but also their instances, ex. Some Company sp. z o.o, or person John Smith.
    Extract their attributes, ex. Company name, Person name, etc.
    Extract their relationships, ex. Company is a parent of another Company, Person is a shareholder of a Company, etc.
    Extract their dates, ex. Company was founded in 2020, Person was born in 1990, The loan conversion has to be completed by 2025-01-01, etc.
    Extract legal terms, ex. Share, Shareholder, Drag-Along Clause, etc.
    Extract legal terms and their relationships, ex. Shareholder is a person who owns a share of a company, Drag-Along Clause is a clause in an agreement that allows a company to drag along another company to a transaction, etc.
    """


    if os.getenv("INGEST", "0") in ["1"]:
        await ingest(ia_file_path)

    if os.getenv("COGNIFY", "0") in ["1"]:
        await my_cognify()

    url = await render_graph()
    print(f"Graphistry URL: {url}")
    webbrowser.open(url)

    # Or use our simple graph preview
    # graph_file_path = str(
    #     pathlib.Path(
    #         os.path.join(pathlib.Path(__file__).parent, ".artifacts/graph_visualization.html")
    #     ).resolve()
    # )
    # await visualize_graph(graph_file_path)

    # Completion query that uses graph data to form context.
    # graph_completion = await search(query_text="Who represents SMOK Ventures?", query_type=SearchType.GRAPH_COMPLETION)
    # print("Graph completion result is:")
    # print(graph_completion)

    # # Completion query that uses document chunks to form context.
    # rag_completion = await search(query_text="Who represents SMOK Ventures?", query_type=SearchType.RAG_COMPLETION)
    # print("Completion result is:")
    # print(rag_completion)

    # # Query all summaries related to query.
    # summaries = await search(query_text="SMOK Ventures", query_type=SearchType.SUMMARIES)
    # print("Summary results are:")
    # for summary in summaries:
    #     print(summary)

    # chunks = await search(query_text="SMOK Ventures", query_type=SearchType.CHUNKS)
    # print("Chunk results are:")
    # for chunk in chunks:
    #     print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
