import asyncio
import os
import pathlib
import webbrowser
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL

from cognee import add, config, prune
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document
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

class OntologyProperty(BaseModel):
    """Represents a property/relationship in the ontology."""
    name: str = Field(description="Name of the property")
    domain: Optional[str] = Field(default=None, description="Domain class for the property")
    range: Optional[str] = Field(default=None, description="Range class for the property")

class OntologyIndividual(BaseModel):
    """Represents a named individual in the ontology."""
    name: str = Field(description="Name of the individual")
    class_type: str = Field(description="Class that this individual belongs to", alias="class")

class OntologyHierarchy(BaseModel):
    """Represents a class hierarchy relationship."""
    parent: str = Field(description="Parent class")
    child: str = Field(description="Child class")

class OntologyElement(BaseModel):
    """Represents an ontology element extracted from text."""
    classes: List[str] = Field(default_factory=list, description="Classes/concepts found in the text")
    properties: List[OntologyProperty] = Field(default_factory=list, description="Properties/relationships with source and target")
    individuals: List[OntologyIndividual] = Field(default_factory=list, description="Named individuals with their class")
    hierarchies: List[OntologyHierarchy] = Field(default_factory=list, description="Subclass relationships")

class AccumulatedOntology(BaseModel):
    """Accumulated ontology from all chunks."""
    classes: List[str] = Field(default_factory=list)
    properties: List[OntologyProperty] = Field(default_factory=list)
    individuals: List[OntologyIndividual] = Field(default_factory=list)
    hierarchies: List[OntologyHierarchy] = Field(default_factory=list)

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

async def build_ontology(chunks: list[DocumentChunk], custom_ontology_path: str = "./src/data/ontologies/custom.owl") -> list[DocumentChunk]:
    """
    Build a custom ontology from document chunks using LLM extraction.
    Accumulates ontological knowledge across all chunks and saves to custom.owl.
    """
    logger.info(f"Building ontology from {len(chunks)} chunks")
    
    # Get LLM client
    llm_client = get_llm_client()
    
    # Ontology extraction prompt
    ontology_prompt = """
Analyze this legal document chunk and extract ontology elements in the following categories:

1. CLASSES: Legal concepts, entity types, document types (e.g., "Company", "Agreement", "Investor")
2. PROPERTIES: Relationships and connections between entities (e.g., "hasInvestor", "governedBy", "hasAmount")
3. INDIVIDUALS: Specific named entities with their class (e.g., "Apple Inc." is a "Company")
4. HIERARCHIES: Parent-child relationships between classes (e.g., "InvestmentAgreement" is a subclass of "Agreement")

Extract comprehensive ontological knowledge focusing on:
- Legal entities and their types
- Contractual relationships
- Financial concepts and amounts
- Temporal relationships and dates
- Corporate structures and hierarchies
- Legal terms and their definitions

For properties, specify both the relationship name and what types of entities it connects.
For individuals, provide both the name and its most specific class.
For hierarchies, specify parent-child relationships clearly.
"""
    
    # Accumulate ontology across all chunks
    accumulated = AccumulatedOntology()
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        try:
            # Extract ontology elements from this chunk
            ontology_elements = await llm_client.acreate_structured_output(
                text_input=chunk.text,
                system_prompt=ontology_prompt,
                response_model=OntologyElement
            )
            
            # Accumulate unique elements
            for cls in ontology_elements.classes:
                if cls and cls not in accumulated.classes:
                    accumulated.classes.append(cls)
            
            for prop in ontology_elements.properties:
                if prop and prop not in accumulated.properties:
                    accumulated.properties.append(prop)
            
            for individual in ontology_elements.individuals:
                if individual and individual not in accumulated.individuals:
                    accumulated.individuals.append(individual)
            
            for hierarchy in ontology_elements.hierarchies:
                if hierarchy and hierarchy not in accumulated.hierarchies:
                    accumulated.hierarchies.append(hierarchy)
                    
        except Exception as e:
            logger.warning(f"Failed to extract ontology from chunk {i+1}: {e}")
            continue
    
    # Create RDF ontology
    await create_rdf_ontology(accumulated, custom_ontology_path)
    
    logger.info(f"Ontology built with {len(accumulated.classes)} classes, {len(accumulated.properties)} properties, {len(accumulated.individuals)} individuals")
    
    return chunks

async def create_rdf_ontology(ontology: AccumulatedOntology, output_path: str = "./src/data/ontologies/custom.owl"):
    """
    Create and serialize RDF/OWL ontology from accumulated ontology elements.
    """
    # Create RDF graph
    g = Graph()
    
    # Define namespaces
    CUSTOM = Namespace("http://cognee.ai/ontology/custom#")
    g.bind("custom", CUSTOM)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    
    # Add ontology header
    ontology_uri = URIRef("http://cognee.ai/ontology/custom")
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Custom Legal Document Ontology")))
    g.add((ontology_uri, RDFS.comment, Literal("Ontology built from legal document chunks")))
    
    # Add classes
    for cls_name in ontology.classes:
        if cls_name:
            cls_uri = CUSTOM[cls_name.replace(" ", "")]
            g.add((cls_uri, RDF.type, OWL.Class))
            g.add((cls_uri, RDFS.label, Literal(cls_name)))
    
    # Add hierarchies (subclass relationships)
    for hierarchy in ontology.hierarchies:
        if hierarchy.parent and hierarchy.child:
            parent_uri = CUSTOM[hierarchy.parent.replace(" ", "")]
            child_uri = CUSTOM[hierarchy.child.replace(" ", "")]
            g.add((child_uri, RDFS.subClassOf, parent_uri))
    
    # Add properties
    for prop in ontology.properties:
        if prop.name:
            prop_uri = CUSTOM[prop.name.replace(" ", "")]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label, Literal(prop.name)))
            
            # Add domain and range if specified
            if prop.domain:
                domain_uri = CUSTOM[prop.domain.replace(" ", "")]
                g.add((prop_uri, RDFS.domain, domain_uri))
            if prop.range:
                range_uri = CUSTOM[prop.range.replace(" ", "")]
                g.add((prop_uri, RDFS.range, range_uri))
    
    # Add individuals
    for individual in ontology.individuals:
        if individual.name and individual.class_type:
            ind_uri = CUSTOM[individual.name.replace(" ", "").replace(".", "")]
            cls_uri = CUSTOM[individual.class_type.replace(" ", "")]
            g.add((ind_uri, RDF.type, cls_uri))
            g.add((ind_uri, RDFS.label, Literal(individual.name)))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Serialize to file
    g.serialize(destination=output_path, format="xml")
    logger.info(f"Custom ontology saved to {output_path}")




async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    # Use custom ontology path built from chunks
    custom_ontology_path = "./src/data/ontologies/custom.owl"
    
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_documents, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),
        Task(build_ontology, custom_ontology_path=custom_ontology_path),
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=custom_ontology_path),
            task_config={"batch_size": 10},
        ),
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
