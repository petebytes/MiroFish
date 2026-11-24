"""Drop-in replacement for ``zep_cloud.client.Zep``.

Translates every Zep Cloud SDK call that MiroFish makes into the equivalent
graphiti-core + Neo4j operation.  MiroFish is Flask (sync); graphiti-core is
async — an ``AsyncBridge`` runs a dedicated event-loop thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

logger = logging.getLogger("zep_compat")


# ---------------------------------------------------------------------------
# Async bridge — runs one event-loop on a daemon thread
# ---------------------------------------------------------------------------

class AsyncBridge:
    """Submit coroutines from sync code and block for the result."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:  # noqa: ANN401
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


# Singleton bridge shared by all Zep instances in the process.
_bridge = AsyncBridge()


# ---------------------------------------------------------------------------
# Wrapper objects — make graphiti objects look like Zep SDK return types
# ---------------------------------------------------------------------------

class ZepNodeWrapper:
    """Wraps a graphiti ``EntityNode`` to match the Zep SDK interface."""

    def __init__(self, node: EntityNode) -> None:
        self._node = node

    def __getattr__(self, name: str) -> Any:
        return getattr(self._node, name)

    @property
    def uuid_(self) -> str:
        return self._node.uuid

    @property
    def uuid(self) -> str:
        return self._node.uuid


class ZepEdgeWrapper:
    """Wraps a graphiti ``EntityEdge`` to match the Zep SDK interface."""

    def __init__(self, edge: EntityEdge) -> None:
        self._edge = edge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._edge, name)

    @property
    def uuid_(self) -> str:
        return self._edge.uuid

    @property
    def uuid(self) -> str:
        return self._edge.uuid

    @property
    def fact_type(self) -> str:
        return self._edge.name or ""


@dataclass
class ZepEpisodeResult:
    uuid_: str
    processed: bool = True
    data: str = ""


@dataclass
class ZepSearchResult:
    edges: list[Any] = field(default_factory=list)
    nodes: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Namespace helpers — ``client.graph.node.*``, ``client.graph.edge.*``, etc.
# ---------------------------------------------------------------------------

class _EpisodeNamespace:
    def __init__(self, parent: "Zep") -> None:
        self._p = parent

    def get(self, uuid_: str) -> ZepEpisodeResult:
        return self._p._episode_get(uuid_)


class _NodeNamespace:
    def __init__(self, parent: "Zep") -> None:
        self._p = parent

    def get(self, uuid_: str) -> ZepNodeWrapper:
        return self._p._node_get(uuid_)

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: str | None = None,
    ) -> list[ZepNodeWrapper]:
        return self._p._node_get_by_graph_id(graph_id, limit, uuid_cursor)

    def get_entity_edges(self, node_uuid: str) -> list[ZepEdgeWrapper]:
        return self._p._node_get_entity_edges(node_uuid)


class _EdgeNamespace:
    def __init__(self, parent: "Zep") -> None:
        self._p = parent

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: str | None = None,
    ) -> list[ZepEdgeWrapper]:
        return self._p._edge_get_by_graph_id(graph_id, limit, uuid_cursor)


class GraphNamespace:
    """``client.graph`` — exposes the sub-namespaces that MiroFish uses."""

    def __init__(self, parent: "Zep") -> None:
        self._p = parent
        self.node = _NodeNamespace(parent)
        self.edge = _EdgeNamespace(parent)
        self.episode = _EpisodeNamespace(parent)

    # -- top-level graph methods --

    def create(self, graph_id: str, name: str = "", description: str = "") -> None:
        self._p._graph_create(graph_id, name, description)

    def delete(self, graph_id: str) -> None:
        self._p._graph_delete(graph_id)

    def set_ontology(
        self,
        graph_ids: list[str],
        entities: dict[str, Any] | None = None,
        edges: dict[str, Any] | None = None,
    ) -> None:
        self._p._graph_set_ontology(graph_ids, entities, edges)

    def add_batch(self, graph_id: str, episodes: list[Any]) -> list[ZepEpisodeResult]:
        return self._p._graph_add_batch(graph_id, episodes)

    def add(self, graph_id: str, type: str = "text", data: str = "") -> None:
        self._p._graph_add(graph_id, type, data)

    def search(
        self,
        graph_id: str = "",
        query: str = "",
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "rrf",
        **kwargs: Any,
    ) -> ZepSearchResult:
        return self._p._graph_search(graph_id, query, limit, scope)


# ---------------------------------------------------------------------------
# Main Zep client
# ---------------------------------------------------------------------------

class Zep:
    """Drop-in replacement for ``zep_cloud.client.Zep``.

    Initialises a graphiti-core ``Graphiti`` instance backed by Neo4j.
    All parameters are read from environment variables — the ``api_key``
    constructor argument is accepted but ignored.
    """

    def __init__(self, api_key: str | None = None) -> None:
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://mirofish-neo4j:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "mirofish-dev")

        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        llm_model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

        llm_config = LLMConfig(
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            small_model=llm_model,
        )
        llm_client = OpenAIClient(config=llm_config)

        embedder_config = OpenAIEmbedderConfig(
            api_key=llm_api_key,
            base_url=llm_base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        self._graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )

        # Ensure DB indices exist (idempotent).
        _bridge.run(self._graphiti.build_indices_and_constraints())

        # In-memory stores
        self._ontologies: dict[str, dict[str, Any]] = {}
        self._graphs: dict[str, dict[str, str]] = {}
        self._episodes: dict[str, ZepEpisodeResult] = {}

        self.graph = GraphNamespace(self)

        logger.info("Zep adapter initialised (graphiti-core → %s)", neo4j_uri)

    # ------------------------------------------------------------------
    # graph.create / graph.delete
    # ------------------------------------------------------------------

    def _graph_create(self, graph_id: str, name: str, description: str) -> None:
        self._graphs[graph_id] = {"name": name, "description": description}
        logger.info("Graph created (in-memory): %s", graph_id)

    def _graph_delete(self, graph_id: str) -> None:
        async def _delete() -> None:
            driver = self._graphiti.driver
            await driver.execute_query(
                "MATCH (n) WHERE n.group_id = $gid DETACH DELETE n",
                params={"gid": graph_id},
            )

        _bridge.run(_delete())
        self._graphs.pop(graph_id, None)
        self._ontologies.pop(graph_id, None)
        logger.info("Graph deleted: %s", graph_id)

    # ------------------------------------------------------------------
    # graph.set_ontology
    # ------------------------------------------------------------------

    def _graph_set_ontology(
        self,
        graph_ids: list[str],
        entities: dict[str, Any] | None,
        edges: dict[str, Any] | None,
    ) -> None:
        ont: dict[str, Any] = {}
        if entities:
            ont["entity_types"] = entities
        if edges:
            ont["edge_types"] = {
                k: (v[0] if isinstance(v, tuple) else v) for k, v in edges.items()
            }
        for gid in graph_ids:
            self._ontologies[gid] = ont
        logger.info("Ontology stored for graph_ids=%s", graph_ids)

    # ------------------------------------------------------------------
    # graph.add_batch / graph.add
    # ------------------------------------------------------------------

    def _graph_add_batch(
        self, graph_id: str, episodes: list[Any]
    ) -> list[ZepEpisodeResult]:
        ont = self._ontologies.get(graph_id, {})
        entity_types = ont.get("entity_types")
        edge_types = ont.get("edge_types")

        raw_episodes: list[RawEpisode] = []
        for ep in episodes:
            ep_uuid = _uuid.uuid4().hex
            raw_episodes.append(
                RawEpisode(
                    name=f"ep-{ep_uuid[:8]}",
                    content=ep.data if hasattr(ep, "data") else str(ep),
                    source_description="MiroFish import",
                    source=EpisodeType.text,
                    reference_time=datetime.now(timezone.utc),
                    uuid=ep_uuid,
                )
            )

        _bridge.run(
            self._graphiti.add_episode_bulk(
                bulk_episodes=raw_episodes,
                group_id=graph_id,
                entity_types=entity_types,
                edge_types=edge_types,
            )
        )

        results: list[ZepEpisodeResult] = []
        for raw in raw_episodes:
            ep_result = ZepEpisodeResult(
                uuid_=raw.uuid or _uuid.uuid4().hex,
                processed=True,
                data=raw.content,
            )
            self._episodes[ep_result.uuid_] = ep_result
            results.append(ep_result)

        return results

    def _graph_add(self, graph_id: str, type: str, data: str) -> None:
        ont = self._ontologies.get(graph_id, {})
        entity_types = ont.get("entity_types")
        edge_types = ont.get("edge_types")

        ep_uuid = _uuid.uuid4().hex
        _bridge.run(
            self._graphiti.add_episode(
                name=f"ep-{ep_uuid[:8]}",
                episode_body=data,
                source_description="MiroFish runtime",
                source=EpisodeType.text,
                reference_time=datetime.now(timezone.utc),
                group_id=graph_id,
                uuid=ep_uuid,
                entity_types=entity_types,
                edge_types=edge_types,
            )
        )

        self._episodes[ep_uuid] = ZepEpisodeResult(
            uuid_=ep_uuid, processed=True, data=data
        )

    # ------------------------------------------------------------------
    # graph.episode.get
    # ------------------------------------------------------------------

    def _episode_get(self, uuid_: str) -> ZepEpisodeResult:
        if uuid_ in self._episodes:
            return self._episodes[uuid_]
        # Episodes added via graphiti are processed synchronously.
        return ZepEpisodeResult(uuid_=uuid_, processed=True)

    # ------------------------------------------------------------------
    # graph.search
    # ------------------------------------------------------------------

    def _graph_search(
        self, graph_id: str, query: str, limit: int, scope: str
    ) -> ZepSearchResult:
        raw_edges: list[EntityEdge] = _bridge.run(
            self._graphiti.search(
                query=query,
                group_ids=[graph_id],
                num_results=limit,
            )
        )

        result = ZepSearchResult()

        if scope in ("edges", "both"):
            result.edges = [ZepEdgeWrapper(e) for e in raw_edges]

        if scope in ("nodes", "both"):
            # Collect unique node UUIDs from the returned edges
            node_uuids: set[str] = set()
            for e in raw_edges:
                node_uuids.add(e.source_node_uuid)
                node_uuids.add(e.target_node_uuid)

            if node_uuids:
                nodes: list[EntityNode] = _bridge.run(
                    EntityNode.get_by_uuids(
                        self._graphiti.driver, list(node_uuids)
                    )
                )
                result.nodes = [ZepNodeWrapper(n) for n in nodes]

        return result

    # ------------------------------------------------------------------
    # graph.node.*
    # ------------------------------------------------------------------

    def _node_get(self, uuid_: str) -> ZepNodeWrapper:
        node: EntityNode = _bridge.run(
            EntityNode.get_by_uuid(self._graphiti.driver, uuid_)
        )
        return ZepNodeWrapper(node)

    def _node_get_by_graph_id(
        self, graph_id: str, limit: int = 100, uuid_cursor: str | None = None
    ) -> list[ZepNodeWrapper]:
        nodes: list[EntityNode] = _bridge.run(
            EntityNode.get_by_group_ids(
                self._graphiti.driver,
                group_ids=[graph_id],
                limit=limit,
                uuid_cursor=uuid_cursor,
            )
        )
        return [ZepNodeWrapper(n) for n in nodes]

    def _node_get_entity_edges(self, node_uuid: str) -> list[ZepEdgeWrapper]:
        edges: list[EntityEdge] = _bridge.run(
            EntityEdge.get_by_node_uuid(self._graphiti.driver, node_uuid)
        )
        return [ZepEdgeWrapper(e) for e in edges]

    # ------------------------------------------------------------------
    # graph.edge.*
    # ------------------------------------------------------------------

    def _edge_get_by_graph_id(
        self, graph_id: str, limit: int = 100, uuid_cursor: str | None = None
    ) -> list[ZepEdgeWrapper]:
        edges: list[EntityEdge] = _bridge.run(
            EntityEdge.get_by_group_ids(
                self._graphiti.driver,
                group_ids=[graph_id],
                limit=limit,
                uuid_cursor=uuid_cursor,
            )
        )
        return [ZepEdgeWrapper(e) for e in edges]
