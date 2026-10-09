"""Table-driven engineering-model builder (phase F2, geometry kernel).

Turns a sourced machine database (``machines/<name>/dimensions.yaml`` +
``profiles/*.csv``, or a delta variant ``machines/<name>/deltas.yaml`` with
``base:``) into per-component meshes with instance transforms, material tags
and per-value provenance.  Pure numpy -- nothing here imports ``bpy``.

    from tokviz.machine import load_machine, build_machine
    db = load_machine("machines/jet1975")
    result = build_machine(db)
"""
from .schema import load_machine, MachineDB, Dim, ComponentSpec, SchemaError  # noqa: F401
from .build import build_machine, write_bake, load_component  # noqa: F401
