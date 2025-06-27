from asyncio.log import logger
import os
import asyncio
import pathlib
import webbrowser
import xml.etree.ElementTree as ET
from rdflib import Graph

from cognee import config, add, cognify, search, SearchType, prune, visualize_graph
from cognee.shared.utils import render_graph
from cognee.low_level import DataPoint


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
    # Set up the data directory. Cognee will store files here.
    config.data_root_directory(data_directory_path)

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

    # Set up the Cognee system directory. Cognee will store system files and databases here.
    config.system_root_directory(cognee_directory_path)
    config.entity_extraction_prompt = """Extract key legal concepts and their relationships.

    Extreact the concept classess, ex. Company, Person, but also their instances, ex. Some Company sp. z o.o, or person John Smith.
    Extract their attributes, ex. Company name, Person name, etc.
    Extract their relationships, ex. Company is a parent of another Company, Person is a shareholder of a Company, etc.
    Extract their dates, ex. Company was founded in 2020, Person was born in 1990, The loan conversion has to be completed by 2025-01-01, etc.
    Extract legal terms, ex. Share, Shareholder, Drag-Along Clause, etc.
    Extract legal terms and their relationships, ex. Shareholder is a person who owns a share of a company, Drag-Along Clause is a clause in an agreement that allows a company to drag along another company to a transaction, etc.
    """

    # Run merge_ontologies if the merged ontology file does not exist, or if MERGE_ONTOLOGY=1
    # if not os.path.exists(merged_ontology_path) or os.getenv("MERGE_ONTOLOGY", "0") in ["1"]:
    #     await merge_ontology(catalog_file_path, merged_ontology_path)


    if os.getenv("INGEST", "0") in ["1"]:
        await ingest(ia_file_path)

    if os.getenv("COGNIFY", "0") in ["1"]:
        await cognify() # ontology_file_path=merged_ontology_path)

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
