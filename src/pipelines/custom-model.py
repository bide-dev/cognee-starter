import asyncio
import os
import pathlib
import tempfile
import threading
import webbrowser
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, SKOS

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

# Thread lock for ontology file operations
_ontology_lock = threading.Lock()

def load_existing_ontology(ontology_path: str) -> Optional[AccumulatedOntology]:
    """
    Load existing ontology from file and convert to AccumulatedOntology format.
    Returns None if file doesn't exist or can't be loaded.
    """
    if not os.path.exists(ontology_path):
        return None
    
    try:
        # Load existing RDF graph
        existing_graph = Graph()
        existing_graph.parse(ontology_path)
        
        # Define namespaces
        CUSTOM = Namespace("http://cognee.ai/ontology/custom#")
        
        # Extract elements from existing ontology
        accumulated = AccumulatedOntology()
        
        # Extract classes
        for cls in existing_graph.subjects(RDF.type, OWL.Class):
            if str(cls).startswith(str(CUSTOM)):
                label = existing_graph.value(cls, RDFS.label)
                if label:
                    accumulated.classes.append(str(label))
        
        # Extract properties
        for prop in existing_graph.subjects(RDF.type, OWL.ObjectProperty):
            if str(prop).startswith(str(CUSTOM)):
                label = existing_graph.value(prop, RDFS.label)
                domain = existing_graph.value(prop, RDFS.domain)
                range_obj = existing_graph.value(prop, RDFS.range)
                
                if label:
                    prop_obj = OntologyProperty(
                        name=str(label),
                        domain=str(existing_graph.value(domain, RDFS.label)) if domain else None,
                        range=str(existing_graph.value(range_obj, RDFS.label)) if range_obj else None
                    )
                    accumulated.properties.append(prop_obj)
        
        # Extract individuals
        for individual in existing_graph.subjects(RDF.type, None):
            if str(individual).startswith(str(CUSTOM)):
                # Get the class this individual belongs to
                for cls in existing_graph.objects(individual, RDF.type):
                    if cls != OWL.Class and str(cls).startswith(str(CUSTOM)):
                        label = existing_graph.value(individual, RDFS.label)
                        cls_label = existing_graph.value(cls, RDFS.label)
                        
                        if label and cls_label:
                            ind_obj = OntologyIndividual(
                                name=str(label),
                                class_type=str(cls_label)
                            )
                            accumulated.individuals.append(ind_obj)
                        break
        
        # Extract hierarchies (subclass relationships)
        for child, parent in existing_graph.subject_objects(RDFS.subClassOf):
            if str(child).startswith(str(CUSTOM)) and str(parent).startswith(str(CUSTOM)):
                child_label = existing_graph.value(child, RDFS.label)
                parent_label = existing_graph.value(parent, RDFS.label)
                
                if child_label and parent_label:
                    hierarchy_obj = OntologyHierarchy(
                        parent=str(parent_label),
                        child=str(child_label)
                    )
                    accumulated.hierarchies.append(hierarchy_obj)
        
        logger.info(f"Loaded existing ontology with {len(accumulated.classes)} classes, "
                   f"{len(accumulated.properties)} properties, {len(accumulated.individuals)} individuals")
        return accumulated
        
    except Exception as e:
        logger.warning(f"Failed to load existing ontology from {ontology_path}: {e}")
        return None

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

    tasks = await get_tasks(user, graph_model, chunker, chunk_size, ontology_file_path)

    return await cognee_pipeline(
        tasks=tasks, datasets=datasets, user=user, pipeline_name="cognify_pipeline"
    )

async def build_ontology(chunks: list[DocumentChunk], custom_ontology_path: str = None) -> list[DocumentChunk]:
    """
    Build a custom ontology from document chunks using LLM extraction.
    Accumulates ontological knowledge across all chunks and saves to custom.owl.
    Now supports loading existing ontology and merging new knowledge.
    """
    # Set default path if not provided
    if custom_ontology_path is None:
        project_root = pathlib.Path(__file__).parent.parent.parent
        custom_ontology_path = str(project_root / "src" / "data" / "ontologies" / "custom.owl")
    
    logger.info(f"Building ontology from {len(chunks)} chunks")
    logger.info(f"Ontology will be saved to: {custom_ontology_path}")
    
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
    
    # Load existing ontology if it exists
    existing_ontology = load_existing_ontology(custom_ontology_path)
    
    # Initialize accumulated ontology with existing elements or create new
    if existing_ontology:
        accumulated = existing_ontology
        logger.info(f"Loaded existing ontology as base")
    else:
        accumulated = AccumulatedOntology()
        logger.info(f"Starting with empty ontology")
    
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
    
    # TODO: Add ontology validation for consistency and logical correctness
    # TODO: Implement ontology versioning for change tracking
    # TODO: Add metrics for ontology growth and quality
    # TODO: Consider streaming/chunked processing for very large ontologies
    
    logger.info(f"Ontology built with {len(accumulated.classes)} classes, {len(accumulated.properties)} properties, {len(accumulated.individuals)} individuals")
    
    return chunks

def resolve_entity_labels(entities: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Resolve conflicts between entity labels using canonical naming.
    Returns dict with canonical URIs as keys and label info as values.
    """
    entity_groups = {}
    
    for entity in entities:
        # Create canonical form (lowercase, no spaces, no special chars)
        canonical = entity.lower().replace(" ", "").replace("-", "").replace("_", "")
        
        if canonical not in entity_groups:
            entity_groups[canonical] = {
                "canonical_label": entity,  # First occurrence becomes canonical
                "alternative_labels": []
            }
        else:
            # Add as alternative if different from canonical
            if entity != entity_groups[canonical]["canonical_label"]:
                entity_groups[canonical]["alternative_labels"].append(entity)
    
    return entity_groups

async def create_rdf_ontology(ontology: AccumulatedOntology, output_path: str = "./src/data/ontologies/custom.owl"):
    """
    Create and serialize RDF/OWL ontology from accumulated ontology elements.
    Now supports SKOS vocabulary for conflict resolution and thread-safe operations.
    """
    # Create RDF graph
    g = Graph()
    
    # Define namespaces
    CUSTOM = Namespace("http://cognee.ai/ontology/custom#")
    g.bind("custom", CUSTOM)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    
    # Add ontology header
    ontology_uri = URIRef("http://cognee.ai/ontology/custom")
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Custom Legal Document Ontology")))
    g.add((ontology_uri, RDFS.comment, Literal("Ontology built from legal document chunks")))
    
    # Add classes with conflict resolution
    class_groups = resolve_entity_labels(ontology.classes)
    for canonical, label_info in class_groups.items():
        if label_info["canonical_label"]:
            cls_uri = CUSTOM[canonical.title()]  # Use canonical form for URI
            g.add((cls_uri, RDF.type, OWL.Class))
            
            # Add primary label using SKOS if there are alternatives, otherwise use rdfs:label
            if label_info["alternative_labels"]:
                g.add((cls_uri, SKOS.prefLabel, Literal(label_info["canonical_label"])))
                for alt_label in label_info["alternative_labels"]:
                    g.add((cls_uri, SKOS.altLabel, Literal(alt_label)))
            else:
                g.add((cls_uri, RDFS.label, Literal(label_info["canonical_label"])))
    
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
    
    # Thread-safe atomic write operation
    with _ontology_lock:
        # Create temporary file in same directory to ensure atomic move
        output_dir = os.path.dirname(output_path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.owl', 
                                       dir=output_dir, delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Serialize to temporary file
            g.serialize(destination=temp_path, format="xml")
            
            # Atomic move to final location
            os.rename(temp_path, output_path)
            logger.info(f"Custom ontology saved to {output_path}")
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e




async def prepare_documents_and_extract_all_chunks(
    documents: List[Document],
    user: User = None,
    max_chunk_size: int = None,
    chunker=TextChunker,
    permissions: List[str] = ["write"]
) -> List[DocumentChunk]:
    """
    Preparation task that runs document classification, permission checks, and chunk extraction.
    Collects ALL chunks before returning them as a batch for subsequent processing.
    This ensures the ontology is built from the complete set of chunks.
    """
    logger.info(f"Starting document preparation phase for {len(documents)} documents")
    
    # Phase 1: Classify documents
    logger.info("Phase 1: Classifying documents")
    classified_documents = await classify_documents(documents)
    
    # Phase 2: Check permissions
    logger.info("Phase 2: Checking permissions")
    permission_checked_documents = await check_permissions_on_documents(
        classified_documents, user=user, permissions=permissions
    )
    
    # Phase 3: Extract chunks and collect ALL of them
    logger.info("Phase 3: Extracting chunks from all documents")
    all_chunks = []
    
    # Process documents and collect all chunks
    async for chunk_batch in extract_chunks_from_documents(
        permission_checked_documents,
        max_chunk_size=max_chunk_size,
        chunker=chunker
    ):
        # chunk_batch might be a single chunk or list of chunks
        if isinstance(chunk_batch, list):
            all_chunks.extend(chunk_batch)
        else:
            all_chunks.append(chunk_batch)
    
    logger.info(f"Document preparation complete. Collected {len(all_chunks)} chunks from all documents")
    return all_chunks


async def get_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
    # Use custom ontology path built from chunks
    # Use absolute path to ensure consistent location regardless of working directory
    project_root = pathlib.Path(__file__).parent.parent.parent
    custom_ontology_path = str(project_root / "src" / "data" / "ontologies" / "custom.owl")
    
    default_tasks = [
        # Phase 1: Preparation - collect ALL chunks before proceeding
        Task(
            prepare_documents_and_extract_all_chunks,
            user=user,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
            permissions=["write"],
        ),
        # Phase 2: Process all chunks with complete ontology
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
    
    # Clean up ontology files for fresh start
    # Use absolute path to ensure consistent location regardless of working directory
    project_root = pathlib.Path(__file__).parent.parent.parent
    custom_ontology_path = str(project_root / "src" / "data" / "ontologies" / "custom.owl")
    if os.path.exists(custom_ontology_path):
        os.remove(custom_ontology_path)
        logger.info(f"Removed existing ontology file: {custom_ontology_path}")
    
    # Clean up any backup or temporary ontology files
    ontology_dir = os.path.dirname(custom_ontology_path)
    if os.path.exists(ontology_dir):
        for filename in os.listdir(ontology_dir):
            if filename.endswith(('.owl.bak', '.owl.tmp', '.owl~')):
                backup_path = os.path.join(ontology_dir, filename)
                os.remove(backup_path)
                logger.info(f"Removed backup ontology file: {backup_path}")

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
