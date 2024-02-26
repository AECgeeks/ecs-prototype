import dataclasses
import functools
import itertools
import operator
import os
import glob
import json
import re

import jsonschema
import numpy as np

import schema

import occ

fns = glob.glob("schemas/**/*.json", recursive=True)
fns = filter(lambda fn: os.path.basename(fn) != "dir.json", fns)
schemas = dict(
    map(
        lambda s: (s["$id"], s),
        (json.load(open(fn)) for fn in fns if os.path.getsize(fn)),
    )
)


class entity:
    def __init__(self, model, k):
        self.model, self.id = model, k

    def __str__(self):
        return self.id

    @property
    def components(self):
        return self.model.tree[self.id]


class model:
    def __init__(self, fn):
        self.components = list(self.parse(fn))
        self.validate()
        self.typed_components = self.apply_schema()
        self.tree = self.resolve()

    def parse(self, fn):
        s = open(fn, encoding="utf-8").read()
        while s:
            s = re.sub("^\w+", "", s)
            if s.startswith("#"):
                s = s[s.find("\n") + 1 :]
            try:
                yield json.loads(s)
                s = None
            except json.decoder.JSONDecodeError as e:
                yield json.loads(s[0 : e.pos])
                s = s[e.pos :]

    def validate(self):
        def jsonschema_validate(*, instance, schema):
            validator = jsonschema.Draft7Validator(
                schema=schema,
                resolver=jsonschema.RefResolver(base_uri="", referrer=schema, store=schemas),
            )
            validator.validate(instance=instance)

        for inst in self.components:
            jsonschema_validate(instance=inst, schema=schemas.get("urn:bsi:ifc5:component"))
            jsonschema_validate(instance=inst, schema=schemas.get(inst["type"]))

    def apply_schema(self):
        def apply(inst):
            comp = dict(inst.items())
            ty = comp.pop("type")
            ent = comp.pop("entity")
            tag = comp.pop("tag", None)
            class_name = ty.split(":")[-1].replace("-", "").title()
            return ent, tag, getattr(schema, class_name)(**comp)

        by_entity = operator.itemgetter(0)
        typed = map(apply, self.components)
        grouped = itertools.groupby(sorted(typed, key=by_entity), key=by_entity)
        return dict((k, [v[1:] for v in vs]) for k, vs in grouped)

    def resolve(self):
        def apply_types(k):
            for tag, comp in self.typed_components[k]:
                if isinstance(comp, schema.Type):
                    yield from apply_types(comp.ref)
                yield tag, comp

        # d = dict((k, list(apply_types(k))) for k in self.typed_components)
        d = self.typed_components

        def resolve_references(comp):
            if "ref" in map(operator.attrgetter("name"), dataclasses.fields(comp)):
                if comp.ref is not None and not isinstance(comp.ref, entity):
                    comp.ref = entity(self, comp.ref)
            return comp

        return dict(
            (
                entity_id,
                [(tag_comp[0], resolve_references(tag_comp[1])) for tag_comp in comps],
            )
            for entity_id, comps in d.items()
        )

    def component_by_type(self, Ty, tag=None, ent=None):
        if ent is None:
            ents = self.tree.items()
        else:
            if isinstance(ent, entity):
                ent = ent.id
            ents = [(ent, self.tree[ent])]
        for k, vss in ents:
            for tg, comp in vss:
                if isinstance(comp, Ty) and (tag is None or tg == tag):
                    tup = ()
                    if ent is None:
                        tup = tup + (k,)
                    if tag is None:
                        tup = tup + (tg,)
                    tup += (comp,)
                    yield tup

    def traverse(self, ent, *, component_type=None, types=(schema.Assembly,), tag=None, path=()):
        if isinstance(ent, entity):
            ent = ent.id
        for tg, comp in self.tree[ent]:
            if isinstance(comp, types):
                yield from self.traverse(
                    comp.ref.id,
                    component_type=component_type,
                    types=types,
                    tag=tag,
                    path=path + (entity(self, ent),),
                )
            elif (component_type is None or isinstance(comp, component_type)) and (tag is None or tg == tag):
                yield path + (entity(self, ent),), comp


class converter:
    def evaluate_Relativeplacement(self, entity):
        comp = list(entity.model.component_by_type(schema.Relativeplacement, ent=entity))[0][1]
        axes = comp.axes
        if not axes:
            axes = np.eye(3)
        m4 = np.hstack((np.vstack((np.array(axes).T, comp.location)), np.transpose([(0, 0, 0, 1)]))).T
        if comp.ref:
            yield from self.evaluate_Relativeplacement(comp.ref)
        yield entity.id, m4

    evaluate_entity = evaluate_Relativeplacement

    @staticmethod
    def mesh_occt(shp):
        tris = np.array(list(occ.mesh(shp)))
        flat = tris.flatten().reshape((-1, 3))
        ps, idxs = np.unique(flat, axis=0, return_inverse=True)
        idxs = idxs.reshape((-1, 3))
        return ps, idxs

    def evaluate_Extrusion(self, comp: schema.Extrusion):
        ps = comp.profile["points"]
        ps3d = np.hstack((ps, np.zeros((len(ps), 1))))
        shp = occ.extrude(occ.make_polygon(ps3d), comp.depth)
        return self.mesh_occt(shp)

    def evaluate_Trianglemesh(self, comp: schema.Trianglemesh):
        return np.array(comp.positions), np.array(comp.indices)

    def evaluate(self, x):
        return getattr(self, f"evaluate_{x.__class__.__name__}")(x)


class app:
    def __init__(self, model):
        self.model = model
        self.converter = converter()
        self.entities_part_of_type_tree = set()
        for comp in map(operator.itemgetter(2), self.model.component_by_type(schema.Type)):
            self.entities_part_of_type_tree.update(map(lambda p: p[0][-1].id, self.model.traverse(comp.ref)))

    def to_obj(self, fn):
        with open(fn, "w") as f:
            nv = 1
            for ent, placements in itertools.groupby(
                sorted(
                    self.model.component_by_type(schema.Relativeplacement),
                    key=operator.itemgetter(0),
                ),
                key=operator.itemgetter(0),
            ):
                if ent in self.entities_part_of_type_tree:
                    continue
                geometries = list(self.model.traverse(ent, component_type=schema.Geometry, tag="body"))
                voids = list(
                    filter(
                        lambda tup: len(tup[0]) > 1,
                        self.model.traverse(
                            ent,
                            component_type=schema.Geometry,
                            types=(schema.Assembly, schema.Contain),
                            tag="void",
                        ),
                    )
                )
                placements = list(
                    map(
                        lambda ab: (ab[0], list(self.converter.evaluate(ab[0][-1]))),
                        self.model.traverse(
                            ent,
                            component_type=schema.Relativeplacement,
                            types=(schema.Assembly, schema.Contain),
                        ),
                    )
                )

                def get_child_placement(path):
                    return functools.reduce(
                        operator.matmul,
                        dict(
                            sum(
                                map(
                                    operator.itemgetter(1),
                                    sorted(
                                        (
                                            (x, p)
                                            for x, p in placements
                                            if set(map(operator.attrgetter("id"), x))
                                            <= set(map(operator.attrgetter("id"), path))
                                        ),
                                        key=lambda xp: len(xp[0]),
                                    ),
                                ),
                                [],
                            )
                        ).values(),
                    )

                def transform_points(m4, ps):
                    return np.array([(m4 @ np.concatenate((p, [1])))[0:3] for p in ps])

                for path, geometry in geometries:
                    item = geometry.ref.components[0][1]
                    ps, idxs = self.converter.evaluate(item)
                    ps = transform_points(get_child_placement(path), ps)

                    if voids:

                        def vds():
                            for p, v in voids:
                                item = v.ref.components[0][1]
                                ps, idxs = self.converter.evaluate(item)
                                ps = transform_points(get_child_placement(p), ps)
                                yield occ.make_shell(ps, idxs)

                        ps, idxs = self.converter.mesh_occt(
                            occ.boolean_op("Cut", occ.make_shell(ps, idxs), list(vds()))
                        )

                    idxs += nv
                    nv += len(ps)

                    print("g", "_".join(map(str, path)), file=f)
                    for p in ps:
                        print("v", *p, file=f)
                    for fa in idxs:
                        print("f", *fa, file=f)

    def to_dot(self, fn):
        with open(fn, "w") as f:
            f.write("digraph G {\n")
            f.write("  compound=true;\n")
            f.write("  node [shape=plaintext];\n")
            for k, vss in self.model.tree.items():
                f.write(f"  subgraph cluster_{k} {{\n")
                f.write(f'    label="{k}";')
                for i, (tg, comp) in enumerate(vss):
                    fields = list(
                        map(
                            lambda nm: (nm, getattr(comp, nm)),
                            map(operator.attrgetter("name"), dataclasses.fields(comp)),
                        )
                    )
                    attributes_html = "".join(
                        f"<tr><td>{nm}</td><td><i>{v}</i></td></tr>" for nm, v in fields if not isinstance(v, entity)
                    )
                    nm = comp.__class__.__name__
                    if tg:
                        nm = f"{nm} ({tg})"
                    label = f"<<table border='0'><tr><td colspan='2'><b>{nm}</b></td></tr>{attributes_html}</table>>"
                    f.write(f"    {k}_{i}_{comp.__class__.__name__} [label={label}];\n")
                f.write("  }\n")
            for k, vss in self.model.tree.items():
                for i, (tg, comp) in enumerate(vss):
                    fields = list(
                        map(
                            lambda nm: (nm, getattr(comp, nm)),
                            map(operator.attrgetter("name"), dataclasses.fields(comp)),
                        )
                    )
                    for nm, val in fields:
                        if isinstance(val, entity):
                            f.write(
                                f"  {k}_{i}_{comp.__class__.__name__} -> {val.id}_{0}_{self.model.tree[val.id][0][1].__class__.__name__} [lhead=cluster_{val.id}];\n"
                            )
            f.write("}\n")


if __name__ == "__main__":
    import sys

    fn = sys.argv[1]
    a = app(model(fn))
    a.to_obj(fn + ".obj")
    a.to_dot(fn + ".dot")
