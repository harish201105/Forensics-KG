"""Initialize Neo4j database with ontology schema."""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema


async def init_database(clear: bool = False):
    settings = get_settings()

    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await client.connect()

    if clear:
        print("Clearing database...")
        await client.clear_database()

    # Load ontology
    schema = ForensicsOntologySchema(settings.ontology_path)
    schema.load()

    # Apply constraints and indexes
    constraints = schema.generate_neo4j_constraints()
    indexes = schema.generate_neo4j_indexes()
    print(f"Applying {len(constraints)} constraints and {len(indexes)} indexes...")
    await client.initialize_schema(constraints, indexes)

    # Print stats
    stats = await client.get_graph_stats()
    print(f"\nDatabase initialized successfully!")
    print(f"Node counts: {stats['node_counts']}")
    print(f"Relationship counts: {stats['relationship_counts']}")
    print(f"Total: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships")

    await client.disconnect()


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    asyncio.run(init_database(clear=clear))
