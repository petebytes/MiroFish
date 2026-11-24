"""Shims for zep_cloud.external_clients.ontology.

graph_builder.py dynamically subclasses EntityModel / EdgeModel and annotates
fields with Optional[EntityText].  The actual ontology validation is handled by
graphiti-core — these classes only need to be valid Pydantic BaseModels so that
the dynamic ``type(name, (EntityModel,), attrs)`` calls succeed.
"""

from pydantic import BaseModel


class EntityModel(BaseModel):
    """Base class for dynamically-created entity type definitions."""

    class Config:
        arbitrary_types_allowed = True


class EdgeModel(BaseModel):
    """Base class for dynamically-created edge type definitions."""

    class Config:
        arbitrary_types_allowed = True


# EntityText is used only as a type annotation (Optional[EntityText]).
# A plain str alias satisfies Pydantic validation.
EntityText = str
