#!/usr/bin/env python3
"""
Ontology Database Viewer Script

This script connects to the cognee graph database and displays the current ontology
stored in the database (not from the RDF file).

Usage:
    python src/scripts/print_ontology.py [--stats-only] [--full] [--search TERM]

Options:
    --stats-only    Show only quick statistics
    --full          Show full detailed view (default)
    --search TERM   Search for entities containing the term
    --help          Show this help message

Examples:
    python src/scripts/print_ontology.py
    python src/scripts/print_ontology.py --stats-only
    python src/scripts/print_ontology.py --search "Company"
"""

import asyncio
import argparse
import os
import sys
import pathlib

# Add the project root to the path so we can import cognee
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee import config

logger = get_logger("ontology_viewer")


async def print_ontology_from_db(search_term: str = None):
    """
    Query and print the current ontology stored in the graph database.
    Shows classes, properties, individuals, and hierarchies.
    
    Args:
        search_term: Optional term to filter entities
    """
    try:
        graph_engine = await get_graph_engine()
        
        print("\n" + "="*80)
        print("🗄️  CURRENT ONTOLOGY FROM DATABASE")
        if search_term:
            print(f"🔍  Searching for: '{search_term}'")
        print("="*80)
        
        # Build search filter
        search_filter = ""
        if search_term:
            search_filter = f"AND toLower(n.name) CONTAINS toLower('{search_term}')"
        
        # Query for all nodes and their types
        nodes_query = f"""
        MATCH (n)
        WHERE n.name IS NOT NULL {search_filter}
        RETURN DISTINCT 
            labels(n) as node_labels,
            n.name as name,
            n.id as node_id,
            properties(n) as properties
        ORDER BY n.name
        """
        
        nodes_result = await graph_engine.query(nodes_query, {})
        
        if not nodes_result:
            print(f"\n❌ No entities found{' matching search term' if search_term else ''}")
            return
        
        # Organize by node types
        classes = []
        individuals = []
        other_entities = []
        
        for record in nodes_result:
            labels = record.get('node_labels', [])
            name = record.get('name', '')
            properties = record.get('properties', {})
            
            # Categorize based on labels and properties
            if 'Class' in labels or properties.get('type') == 'Class':
                classes.append({
                    'name': name,
                    'labels': labels,
                    'properties': properties
                })
            elif any(label in ['Individual', 'Person', 'Company', 'Entity', 'Organization'] for label in labels):
                individuals.append({
                    'name': name,
                    'labels': labels,
                    'properties': properties
                })
            else:
                other_entities.append({
                    'name': name,
                    'labels': labels,
                    'properties': properties
                })
        
        # Print Classes
        print(f"\n📚 CLASSES ({len(classes)} found):")
        print("-" * 40)
        for cls in classes[:50]:  # Limit to first 50
            labels_str = ", ".join(cls['labels']) if cls['labels'] else "No labels"
            print(f"  • {cls['name']} [{labels_str}]")
        if len(classes) > 50:
            print(f"  ... and {len(classes) - 50} more classes")
        
        # Print Individuals  
        print(f"\n👤 INDIVIDUALS ({len(individuals)} found):")
        print("-" * 40)
        for ind in individuals[:50]:  # Limit to first 50
            labels_str = ", ".join(ind['labels']) if ind['labels'] else "No labels"
            print(f"  • {ind['name']} [{labels_str}]")
        if len(individuals) > 50:
            print(f"  ... and {len(individuals) - 50} more individuals")
        
        # Print Other Entities
        print(f"\n🔗 OTHER ENTITIES ({len(other_entities)} found):")
        print("-" * 40)
        for ent in other_entities[:50]:  # Limit to first 50
            labels_str = ", ".join(ent['labels']) if ent['labels'] else "No labels"
            print(f"  • {ent['name']} [{labels_str}]")
        if len(other_entities) > 50:
            print(f"  ... and {len(other_entities) - 50} more entities")
        
        # Query for relationships/properties
        relationships_query = f"""
        MATCH (a)-[r]->(b)
        WHERE a.name IS NOT NULL AND b.name IS NOT NULL
        {f"AND (toLower(a.name) CONTAINS toLower('{search_term}') OR toLower(b.name) CONTAINS toLower('{search_term}'))" if search_term else ""}
        RETURN DISTINCT 
            type(r) as relationship_type,
            a.name as source_name,
            b.name as target_name,
            properties(r) as rel_properties
        ORDER BY type(r), a.name
        """
        
        relationships_result = await graph_engine.query(relationships_query, {})
        
        # Group relationships by type
        relationship_types = {}
        for record in relationships_result:
            rel_type = record.get('relationship_type', 'UNKNOWN')
            source = record.get('source_name', '')
            target = record.get('target_name', '')
            
            if rel_type not in relationship_types:
                relationship_types[rel_type] = []
            
            relationship_types[rel_type].append(f"{source} → {target}")
        
        print(f"\n🔗 RELATIONSHIPS ({len(relationships_result)} total):")
        print("-" * 40)
        for rel_type, relations in relationship_types.items():
            print(f"\n  📌 {rel_type} ({len(relations)} instances):")
            for relation in relations[:15]:  # Limit to first 15 per type
                print(f"    • {relation}")
            if len(relations) > 15:
                print(f"    ... and {len(relations) - 15} more")
        
        # Summary statistics
        total_nodes = len(classes) + len(individuals) + len(other_entities)
        total_relationships = len(relationships_result)
        
        print(f"\n📊 ONTOLOGY SUMMARY:")
        print("-" * 40)
        print(f"  • Total Nodes: {total_nodes}")
        print(f"  • Classes: {len(classes)}")
        print(f"  • Individuals: {len(individuals)}")
        print(f"  • Other Entities: {len(other_entities)}")
        print(f"  • Total Relationships: {total_relationships}")
        print(f"  • Relationship Types: {len(relationship_types)}")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error querying ontology from database: {e}")
        logger.error(f"Failed to query ontology from database: {e}")


async def print_ontology_stats():
    """
    Print quick statistics about the ontology in the database.
    """
    try:
        graph_engine = await get_graph_engine()
        
        print("\n📊 QUICK ONTOLOGY STATS:")
        print("="*50)
        
        # Get total node count
        total_nodes_query = "MATCH (n) RETURN count(n) as total_nodes"
        total_nodes_result = await graph_engine.query(total_nodes_query, {})
        total_nodes = total_nodes_result[0]['total_nodes'] if total_nodes_result else 0
        
        # Get total relationship count
        total_rels_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
        total_rels_result = await graph_engine.query(total_rels_query, {})
        total_rels = total_rels_result[0]['total_relationships'] if total_rels_result else 0
        
        print(f"🎯 Overview: {total_nodes} nodes, {total_rels} relationships")
        print()
        
        # Get node count by labels
        node_stats_query = """
        MATCH (n)
        WITH labels(n) as node_labels
        UNWIND node_labels as label
        RETURN label, count(*) as count
        ORDER BY count DESC
        LIMIT 20
        """
        
        if total_nodes == 0:
            print("❌ Database is empty! No ontology data found.")
            print("\n💡 To populate the database, run:")
            print("   INGEST=1 COGNIFY=1 python src/pipelines/custom-model.py")
            return
        
        node_stats = await graph_engine.query(node_stats_query, {})
        
        # Get relationship count by type
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(*) as count
        ORDER BY count DESC
        LIMIT 20
        """
        
        rel_stats = await graph_engine.query(rel_stats_query, {})
        
        print("📋 Node Labels (Top 20):")
        print("-" * 30)
        if node_stats:
            for record in node_stats:
                print(f"  • {record['label']}: {record['count']}")
        else:
            print("  No node labels found")
        
        print("\n🔗 Relationship Types (Top 20):")
        print("-" * 30)
        if rel_stats:
            for record in rel_stats:
                print(f"  • {record['relationship_type']}: {record['count']}")
        else:
            print("  No relationships found")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error getting ontology stats: {e}")
        logger.error(f"Failed to get ontology stats: {e}")


async def search_entities(search_term: str):
    """
    Search for specific entities in the ontology.
    
    Args:
        search_term: Term to search for in entity names
    """
    try:
        graph_engine = await get_graph_engine()
        
        print(f"\n🔍 SEARCHING FOR: '{search_term}'")
        print("="*60)
        
        # Search in node names
        search_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($search_term)
        RETURN DISTINCT 
            labels(n) as node_labels,
            n.name as name,
            n.id as node_id,
            properties(n) as properties
        ORDER BY n.name
        LIMIT 100
        """
        
        results = await graph_engine.query(search_query, {"search_term": search_term})
        
        if not results:
            print(f"❌ No entities found containing '{search_term}'")
            return
        
        print(f"✅ Found {len(results)} entities:")
        print("-" * 40)
        
        for record in results:
            labels = record.get('node_labels', [])
            name = record.get('name', '')
            labels_str = ", ".join(labels) if labels else "No labels"
            print(f"  • {name} [{labels_str}]")
        
        # Also search in relationships
        rel_search_query = """
        MATCH (a)-[r]->(b)
        WHERE toLower(a.name) CONTAINS toLower($search_term) 
           OR toLower(b.name) CONTAINS toLower($search_term)
        RETURN DISTINCT 
            type(r) as relationship_type,
            a.name as source_name,
            b.name as target_name
        ORDER BY a.name
        LIMIT 50
        """
        
        rel_results = await graph_engine.query(rel_search_query, {"search_term": search_term})
        
        if rel_results:
            print(f"\n🔗 Related relationships ({len(rel_results)} found):")
            print("-" * 40)
            for record in rel_results:
                rel_type = record.get('relationship_type', 'UNKNOWN')
                source = record.get('source_name', '')
                target = record.get('target_name', '')
                print(f"  • {source} --[{rel_type}]--> {target}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error searching entities: {e}")
        logger.error(f"Failed to search entities: {e}")


def setup_cognee_config():
    """Set up cognee configuration using the same paths as the main script."""
    try:
        # Use the same paths as in custom-model.py
        data_directory_path = str(
            pathlib.Path(
                os.path.join(project_root, "src/pipelines/.data_storage")
            ).resolve()
        )
        
        cognee_directory_path = str(
            pathlib.Path(
                os.path.join(project_root, "src/pipelines/.cognee_system")
            ).resolve()
        )
        
        config.data_root_directory(data_directory_path)
        config.system_root_directory(cognee_directory_path)
        
        logger.info(f"Using data directory: {data_directory_path}")
        logger.info(f"Using cognee system directory: {cognee_directory_path}")
        
    except Exception as e:
        logger.warning(f"Failed to set up cognee config: {e}")
        print(f"⚠️  Warning: Could not configure cognee paths: {e}")


async def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Print ontology stored in the cognee graph database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/print_ontology.py                    # Full detailed view
  python src/scripts/print_ontology.py --stats-only       # Quick stats only
  python src/scripts/print_ontology.py --search Company   # Search for 'Company'
        """
    )
    
    parser.add_argument(
        "--stats-only", 
        action="store_true", 
        help="Show only quick statistics"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Show full detailed view (default)"
    )
    
    parser.add_argument(
        "--search", 
        type=str, 
        help="Search for entities containing the specified term"
    )
    
    args = parser.parse_args()
    
    # Set up cognee configuration
    setup_cognee_config()
    
    try:
        if args.search:
            await search_entities(args.search)
        elif args.stats_only:
            await print_ontology_stats()
        else:
            # Default: show full view
            await print_ontology_from_db()
            print("\n" + "💡 Tip: Use --stats-only for quick overview or --search TERM to find specific entities")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())