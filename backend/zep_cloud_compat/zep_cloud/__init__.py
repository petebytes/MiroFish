"""Zep Cloud compatibility shim — backed by graphiti-core + Neo4j."""

from dataclasses import dataclass


@dataclass
class EpisodeData:
    """Mirrors zep_cloud.EpisodeData used by graph_builder.py."""

    data: str
    type: str = "text"


@dataclass
class EntityEdgeSourceTarget:
    """Mirrors zep_cloud.EntityEdgeSourceTarget used by graph_builder.py."""

    source: str = "Entity"
    target: str = "Entity"


class InternalServerError(Exception):
    """Mirrors zep_cloud.InternalServerError used by zep_paging.py."""

    pass
