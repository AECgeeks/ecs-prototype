# generated by datamodel-codegen:
#   filename:  dir.json
#   timestamp: 2024-01-10T12:47:54+00:00

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Assembly:
    ref: str


@dataclass
class Classification:
    name: str


@dataclass
class Contain:
    ref: str


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Polyline:
    points: List[Point]


@dataclass
class Geometry:
    ref: str


@dataclass
class Relativeplacement:
    location: List[float]
    axes: Optional[List[List[float]]] = None
    ref: Optional[str] = None


@dataclass
class Trianglemesh:
    positions: List
    indices: List
    normals: Optional[List] = None


@dataclass
class Type:
    ref: str


@dataclass
class Component:
    type: str
    entity: str
    tag: Optional[str] = None


@dataclass
class Extrusion:
    profile: Polyline
    depth: float
