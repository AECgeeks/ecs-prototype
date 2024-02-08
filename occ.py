import OCC.Core.gp
import OCC.Core.BRepBuilderAPI
import OCC.Core.BRepAlgoAPI
import OCC.Core.BRepPrimAPI
import OCC.Core.BRepOffsetAPI
import OCC.Core.BRepMesh
import OCC.Core.BRep
import OCC.Core.TopTools
import OCC.Core.TopLoc
import OCC.Core.TopoDS
import OCC.Core.TopExp
import OCC.Core.TopAbs
import OCC.Core.ShapeFix
import OCC.Core.ShapeUpgrade

def unify(shp):
    usd = OCC.Core.ShapeUpgrade.ShapeUpgrade_UnifySameDomain(shp)
    try:
        usd.Build()
    except:
        return shp
    return usd.Shape()

def shape_list(shps):
    sl = OCC.Core.TopTools.TopTools_ListOfShape()
    for shp in shps:
        sl.Append(shp)
    return sl


def to_tuple(gp):
    xy = gp.X(), gp.Y()
    if hasattr(gp, "Z"):
        return xy + (gp.Z(),)
    else:
        return xy

def fix(shp):
    sfs = OCC.Core.ShapeFix.ShapeFix_Shape(shp)
    sfs.Perform()
    return sfs.Shape()


def boolean_op(operation, operand1, operands2):
    builder = getattr(OCC.Core.BRepAlgoAPI, f"BRepAlgoAPI_{operation}")()
    builder.SetNonDestructive(True)
    builder.SetFuzzyValue(1.e-5)
    builder.SetArguments(shape_list([operand1]))
    builder.SetTools(shape_list(operands2))
    builder.Build()
    if builder.IsDone():
        return fix(builder.Shape())


def make_polygon(pts):
    mp = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
    for pt in map(lambda arr: OCC.Core.gp.gp_Pnt(*arr), pts):
        mp.Add(pt)
        
    mp.Close()
    
    wr = mp.Wire()
    fa = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakeFace(wr).Face()
    return fa

def make_shell(pts, idxs):
    builder = OCC.Core.BRepOffsetAPI.BRepOffsetAPI_Sewing()
    for poly in map(make_polygon, pts[idxs]):
        builder.Add(poly)
    builder.Perform()
    return OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakeSolid(unify(builder.SewedShape())).Solid()

def extrude(fa, depth):
    v = OCC.Core.gp.gp_Vec(0,0,depth)
    return OCC.Core.BRepPrimAPI.BRepPrimAPI_MakePrism(fa, v).Shape()

def mesh(shape, deflection=0.01):
    OCC.Core.BRepMesh.BRepMesh_IncrementalMesh(shape, deflection)
    bt = OCC.Core.BRep.BRep_Tool()

    exp = OCC.Core.TopExp.TopExp_Explorer(shape, OCC.Core.TopAbs.TopAbs_FACE)
    while exp.More():
        face = OCC.Core.TopoDS.topods_Face(exp.Current())
        loc = OCC.Core.TopLoc.TopLoc_Location()
        triangulation = bt.Triangulation(face, loc)
        trsf = loc.Transformation()

        if triangulation:
            vs = [triangulation.Node(i + 1).Transformed(trsf) for i in range(triangulation.NbNodes())]
            vs = list(map(to_tuple, vs))

            tris = triangulation.Triangles()
            for i in range(triangulation.NbTriangles()):
                tri = tris.Value(i + 1)
                pts = tuple(map(lambda i: vs[i - 1], tri.Get()))
                if face.Orientation() == OCC.Core.TopAbs.TopAbs_REVERSED:
                    pts = pts[::-1]
                yield pts

        exp.Next()