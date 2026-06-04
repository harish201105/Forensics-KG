from typing import Dict, List, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from loguru import logger

# Internal properties that should never be sent to visualization/search clients.
_INTERNAL_PROPS = ("embedding", "embed_text")


def _strip_internal(props: Any) -> Any:
    """Remove heavy/internal properties (e.g. embedding vectors) from a node's props."""
    if isinstance(props, dict):
        return {k: v for k, v in props.items() if k not in _INTERNAL_PROPS}
    return props


class Neo4jClient:
    """Async Neo4j client using official driver."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._username, self._password)
        )
        await self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {self._uri}")

    async def disconnect(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("Disconnected from Neo4j")

    def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")
        return self._driver

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            result = await session.run(query, parameters or {})  # type: ignore[arg-type]
            records = await result.data()
            return records

    async def execute_write(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            result = await session.run(query, parameters or {})  # type: ignore[arg-type]
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
            }

    async def execute_transaction(
        self, queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute multiple queries in a single transaction for atomicity."""
        driver = self._get_driver()
        totals = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}
        async with driver.session(database=self._database) as session:
            async with await session.begin_transaction() as tx:
                for q in queries:
                    result = await tx.run(q["query"], q.get("parameters", {}))
                    summary = await result.consume()
                    totals["nodes_created"] += summary.counters.nodes_created
                    totals["relationships_created"] += summary.counters.relationships_created
                    totals["properties_set"] += summary.counters.properties_set
                await tx.commit()
        return totals

    async def merge_node(
        self,
        label: str,
        match_props: Dict[str, Any],
        set_props: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        match_clause = ", ".join(f"{k}: ${k}" for k in match_props)
        set_clause = ""
        params = {**match_props}
        if set_props:
            set_items = ", ".join(f"n.{k} = $set_{k}" for k in set_props)
            set_clause = f" ON CREATE SET {set_items} ON MATCH SET {set_items}"
            params.update({f"set_{k}": v for k, v in set_props.items()})

        query = f"MERGE (n:{label} {{{match_clause}}}){set_clause} RETURN n"
        records = await self.execute_query(query, params)
        return records[0]["n"] if records else {}

    async def merge_relationship(
        self,
        source_label: str,
        source_key: str,
        source_value: Any,
        target_label: str,
        target_key: str,
        target_value: Any,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        props_clause = ""
        params = {"source_val": source_value, "target_val": target_value}
        if properties:
            prop_items = ", ".join(f"{k}: $rel_{k}" for k in properties)
            props_clause = f" {{{prop_items}}}"
            params.update({f"rel_{k}": v for k, v in properties.items()})

        query = (
            f"MATCH (a:{source_label} {{{source_key}: $source_val}}) "
            f"MATCH (b:{target_label} {{{target_key}: $target_val}}) "
            f"MERGE (a)-[r:{rel_type}{props_clause}]->(b) "
            f"RETURN type(r) as type"
        )
        result = await self.execute_write(query, params)
        return result

    async def get_node_by_property(
        self, label: str, prop_name: str, prop_value: Any
    ) -> Optional[Dict[str, Any]]:
        query = f"MATCH (n:{label} {{{prop_name}: $value}}) RETURN n LIMIT 1"
        records = await self.execute_query(query, {"value": prop_value})
        return _strip_internal(records[0]["n"]) if records else None

    async def get_graph_stats(self) -> Dict[str, Any]:
        node_query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
        RETURN label, value.count as count
        """
        rel_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL apoc.cypher.run('MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) as count', {}) YIELD value
        RETURN relationshipType, value.count as count
        """
        try:
            node_records = await self.execute_query(node_query)
            rel_records = await self.execute_query(rel_query)
        except Exception:
            # Fallback without APOC. Pick the first NON-internal label so nodes
            # whose first label happens to be ":Embedded" are still counted under
            # their real type.
            node_records = await self.execute_query(
                "MATCH (n) "
                "WITH [l IN labels(n) WHERE l <> 'Embedded'][0] AS label "
                "WHERE label IS NOT NULL "
                "RETURN label, count(*) as count"
            )
            rel_records = await self.execute_query(
                "MATCH ()-[r]->() RETURN type(r) as relationshipType, count(r) as count"
            )

        # Exclude the internal ":Embedded" helper label used for the vector index.
        node_counts = {
            r["label"]: r["count"]
            for r in node_records
            if r["label"] != "Embedded"
        }
        rel_counts = {r["relationshipType"]: r["count"] for r in rel_records}
        return {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }

    async def get_subgraph(
        self, center_node_id: str, label: str, key: str, depth: int = 2
    ) -> Dict[str, Any]:
        query = f"""
        MATCH (center:{label} {{{key}: $node_id}})
        CALL apoc.path.subgraphAll(center, {{maxLevel: $depth}}) YIELD nodes, relationships
        UNWIND nodes as n
        WITH collect(DISTINCT {{
            id: elementId(n),
            labels: labels(n),
            properties: n {{.*, embedding: NULL, embed_text: NULL}}
        }}) as nodeList, relationships
        UNWIND relationships as r
        RETURN nodeList as nodes, collect(DISTINCT {{
            id: elementId(r),
            source: elementId(startNode(r)),
            target: elementId(endNode(r)),
            type: type(r),
            properties: properties(r)
        }}) as edges
        """
        try:
            records = await self.execute_query(
                query, {"node_id": center_node_id, "depth": depth}
            )
        except Exception:
            # Fallback without APOC. Collect DISTINCT nodes (not paths) to avoid the
            # combinatorial path-explosion that blows transaction memory, and exclude
            # the heavy `embedding` vector from each node's properties.
            fallback_query = f"""
            MATCH (center:{label} {{{key}: $node_id}})
            OPTIONAL MATCH (center)-[*1..{depth}]-(connected)
            WITH center, collect(DISTINCT connected) AS others
            WITH [center] + [x IN others WHERE x IS NOT NULL] AS nodeset
            UNWIND nodeset AS n
            WITH collect(DISTINCT n) AS nodes
            UNWIND nodes AS a
            OPTIONAL MATCH (a)-[r]-(b) WHERE b IN nodes
            WITH nodes, collect(DISTINCT r) AS rels
            RETURN
              [n IN nodes | {{
                id: elementId(n), labels: labels(n),
                properties: n {{.*, embedding: NULL, embed_text: NULL}}
              }}] AS nodes,
              [r IN rels WHERE r IS NOT NULL | {{
                id: elementId(r), source: elementId(startNode(r)),
                target: elementId(endNode(r)), type: type(r), properties: properties(r)
              }}] AS edges
            """
            records = await self.execute_query(
                fallback_query, {"node_id": center_node_id}
            )

        if records:
            nodes = records[0]["nodes"]
            for n in nodes:
                if isinstance(n, dict):
                    n["properties"] = _strip_internal(n.get("properties"))
            return {"nodes": nodes, "edges": records[0]["edges"]}
        return {"nodes": [], "edges": []}

    async def get_full_graph(self, limit: int = 500) -> Dict[str, Any]:
        query = """
        MATCH (n)
        WITH n LIMIT $limit
        OPTIONAL MATCH (n)-[r]->(m)
        WITH collect(DISTINCT {
            id: elementId(n), labels: labels(n), properties: n {.*, embedding: NULL, embed_text: NULL}
        }) + collect(DISTINCT {
            id: elementId(m), labels: labels(m), properties: m {.*, embedding: NULL, embed_text: NULL}
        }) as allNodes,
        collect(DISTINCT {
            id: elementId(r), source: elementId(startNode(r)),
            target: elementId(endNode(r)), type: type(r), properties: properties(r)
        }) as edges
        UNWIND allNodes as node
        WITH collect(DISTINCT node) as nodes, edges
        RETURN nodes, edges
        """
        records = await self.execute_query(query, {"limit": limit})
        if records:
            # Filter out null nodes and strip internal embedding properties
            nodes = []
            for n in records[0].get("nodes", []):
                if n.get("id"):
                    n["properties"] = _strip_internal(n.get("properties"))
                    nodes.append(n)
            edges = [e for e in records[0].get("edges", []) if e.get("id")]
            return {"nodes": nodes, "edges": edges}
        return {"nodes": [], "edges": []}

    async def search_nodes(
        self, query_text: str, labels: Optional[List[str]] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        # Skip internal/embedding props when scanning text.
        scan = (
            "any(prop in keys(n) WHERE NOT prop IN ['embedding','embed_text'] "
            "AND toString(n[prop]) CONTAINS $q)"
        )
        if labels:
            label_filter = " OR ".join(f"(n:{l} AND {scan})" for l in labels)
        else:
            label_filter = scan

        query = f"""
        MATCH (n) WHERE {label_filter}
        RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        rows = await self.execute_query(query, {"q": query_text, "limit": limit})
        for r in rows:
            r["properties"] = _strip_internal(r.get("properties"))
        return rows

    async def initialize_schema(
        self, constraints: List[str], indexes: List[str]
    ) -> None:
        for stmt in constraints:
            try:
                await self.execute_write(stmt)
                logger.debug(f"Applied: {stmt[:80]}...")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")

        for stmt in indexes:
            try:
                await self.execute_write(stmt)
                logger.debug(f"Applied: {stmt[:80]}...")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")

        logger.info(
            f"Schema initialized: {len(constraints)} constraints, {len(indexes)} indexes"
        )

    async def clear_database(self) -> Dict[str, Any]:
        result = await self.execute_write("MATCH (n) DETACH DELETE n")
        logger.warning("Database cleared!")
        return result
