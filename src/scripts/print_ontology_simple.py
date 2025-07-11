#!/usr/bin/env python3
"""
Simple Ontology Viewer - Run from project root with venv activated

Usage:
    python src/scripts/print_ontology_simple.py
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

try:
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.shared.logging_utils import get_logger
    from cognee import config
    import pathlib
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n💡 Make sure you're in the project root and virtual environment is activated:")
    print("   cd /Users/maciejlotkowski/development/bide/cognee-starter")
    print("   source .venv/bin/activate")
    print("   python print_ontology_simple.py")
    sys.exit(1)

logger = get_logger("ontology_viewer")

async def print_quick_stats():
    """Print quick statistics about the ontology in the database."""
    try:
        # Set up cognee configuration
        data_directory_path = str(pathlib.Path("src/pipelines/.data_storage").resolve())
        cognee_directory_path = str(pathlib.Path("src/pipelines/.cognee_system").resolve())
        
        config.data_root_directory(data_directory_path)
        config.system_root_directory(cognee_directory_path)
        
        graph_engine = await get_graph_engine()
        
        print("\n📊 ONTOLOGY DATABASE STATS:")
        print("="*50)
        
        # Get total counts
        total_nodes_result = await graph_engine.query("MATCH (n) RETURN count(n) as total_nodes", {})
        total_nodes = total_nodes_result[0]['total_nodes'] if total_nodes_result else 0
        
        total_rels_result = await graph_engine.query("MATCH ()-[r]->() RETURN count(r) as total_relationships", {})
        total_rels = total_rels_result[0]['total_relationships'] if total_rels_result else 0
        
        print(f"🎯 Overview: {total_nodes} nodes, {total_rels} relationships")
        
        if total_nodes == 0:
            print("\n❌ Database is empty! No ontology data found.")
            print("\n💡 To populate the database, run:")
            print("   INGEST=1 COGNIFY=1 python src/pipelines/custom-model.py")
            return
        
        # Get sample nodes
        sample_query = """
        MATCH (n) 
        WHERE n.name IS NOT NULL 
        RETURN labels(n) as labels, n.name as name 
        ORDER BY n.name 
        LIMIT 10
        """
        sample_result = await graph_engine.query(sample_query, {})
        
        print(f"\n📋 Sample Entities (showing 10 of {total_nodes}):")
        print("-" * 40)
        for record in sample_result:
            labels = ", ".join(record.get('labels', []))
            name = record.get('name', 'Unknown')
            print(f"  • {name} [{labels}]")
        
        # Get relationship types
        rel_types_query = """
        MATCH ()-[r]->() 
        RETURN type(r) as rel_type, count(*) as count 
        ORDER BY count DESC 
        LIMIT 5
        """
        rel_types_result = await graph_engine.query(rel_types_query, {})
        
        print(f"\n🔗 Top Relationship Types:")
        print("-" * 30)
        for record in rel_types_result:
            print(f"  • {record['rel_type']}: {record['count']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Failed to get ontology stats: {e}")

async def main():
    await print_quick_stats()

if __name__ == "__main__":
    asyncio.run(main())