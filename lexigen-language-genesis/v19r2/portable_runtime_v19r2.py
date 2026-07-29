from __future__ import annotations
from collections import Counter
from typing import Any

SCHEMA = "lexigen-v19r2-executable-production-v1"
BANNED = {"complete_marker_reflection", "reflect_marker_object", "symmetry_completion",
          "move_singleton_towards", "edge_project", "recolour", "decode_regular_linegrid",
          "overlay_equal_tiles", "canonical_rectangular_layers", "fill_internal_blank_axis",
          "extend_corner_marked_rays"}
class PortableV19R2Error(RuntimeError): pass

def _grid(value):
    g=tuple(tuple(int(x) for x in row) for row in value)
    if not g or not g[0] or any(len(row)!=len(g[0]) for row in g): raise PortableV19R2Error("invalid grid")
    return g

def _ops(value):
    if isinstance(value,dict):
        if "op" in value: yield str(value["op"])
        for child in value.values(): yield from _ops(child)
    elif isinstance(value,list):
        for child in value: yield from _ops(child)

def _coord(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2: raise PortableV19R2Error("expected pair")
    return int(value[0]),int(value[1])
def _box(points):
    pts=list(points)
    if not pts: raise PortableV19R2Error("empty bbox")
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return {"min_row":min(rs),"max_row":max(rs),"min_col":min(cs),"max_col":max(cs),
            "height":max(rs)-min(rs)+1,"width":max(cs)-min(cs)+1}
def _eval(e,g,s):
    if isinstance(e,(int,bool)) or e is None: return e
    if isinstance(e,str): return e
    if isinstance(e,list): return [_eval(x,g,s) for x in e]
    if not isinstance(e,dict) or "op" not in e: return e
    op=str(e["op"])
    if op=="var":
        n=str(e["name"])
        if n not in s: raise PortableV19R2Error("unbound "+n)
        return s[n]
    if op=="input": return g
    if op=="height": return len(g)
    if op=="width": return len(g[0])
    if op=="mode":
        if "__mode" not in s:
            c=Counter(v for row in g for v in row); s["__mode"]=min(c,key=lambda x:(-c[x],x))
        return s["__mode"]
    if op=="palette": return tuple(sorted({v for row in g for v in row}))
    if op=="pair": return int(_eval(e["row"],g,s)),int(_eval(e["col"],g,s))
    if op=="first": return _coord(_eval(e["value"],g,s))[0]
    if op=="second": return _coord(_eval(e["value"],g,s))[1]
    if op=="coords_colour":
        colour=int(_eval(e["colour"],g,s)); return frozenset((r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==colour)
    if op=="bbox": return _box(frozenset(_eval(e["coords"],g,s)))
    if op=="bbox_field": return int(_eval(e["bbox"],g,s)[str(e["name"])])
    if op in {"add","sub","mul"}:
        a=int(_eval(e["left"],g,s)); b=int(_eval(e["right"],g,s))
        return a+b if op=="add" else a-b if op=="sub" else a*b
    if op in {"eq","ne","lt","le","gt","ge"}:
        a=_eval(e["left"],g,s); b=_eval(e["right"],g,s)
        if op=="eq": return a==b
        if op=="ne": return a!=b
        if op=="lt": return a<b
        if op=="le": return a<=b
        if op=="gt": return a>b
        return a>=b
    if op=="and": return all(bool(_eval(x,g,s)) for x in e["items"])
    if op=="or": return any(bool(_eval(x,g,s)) for x in e["items"])
    if op=="not": return not bool(_eval(e["value"],g,s))
    if op=="if": return _eval(e["then"] if bool(_eval(e["condition"],g,s)) else e["else"],g,s)
    if op in {"filter","map"}:
        result=[]; name=str(e["var"]); existed=name in s; old=s.get(name)
        try:
            for item in list(_eval(e["items"],g,s)):
                s[name]=item
                if op=="map": result.append(_eval(e["body"],g,s))
                elif bool(_eval(e["predicate"],g,s)): result.append(item)
        finally:
            if existed: s[name]=old
            else: s.pop(name,None)
        return frozenset(result) if op=="map" else tuple(result)
    if op=="unique":
        items=list(_eval(e["items"],g,s))
        if len(items)!=1: raise PortableV19R2Error(f"unique got {len(items)}")
        return items[0]
    if op=="set_union":
        out=set()
        for child in e["items"]: out.update(_coord(v) for v in _eval(child,g,s))
        return frozenset(out)
    if op=="set_intersection":
        groups=[set(_coord(v) for v in _eval(child,g,s)) for child in e["items"]]
        return frozenset(set.intersection(*groups)) if groups else frozenset()
    if op=="set_difference":
        return frozenset(set(_coord(v) for v in _eval(e["left"],g,s))-set(_coord(v) for v in _eval(e["right"],g,s)))
    if op=="canvas":
        h=int(_eval(e["rows"],g,s)); w=int(_eval(e["cols"],g,s)); fill=int(_eval(e["fill"],g,s))
        if not(1<=h<=60 and 1<=w<=60): raise PortableV19R2Error("shape")
        return tuple(tuple(fill for _ in range(w)) for _ in range(h))
    if op=="paint":
        base=_grid(_eval(e["grid"],g,s)); pts=frozenset(_coord(v) for v in _eval(e["coords"],g,s)); colour=int(_eval(e["colour"],g,s))
        return tuple(tuple(colour if (r,c) in pts else v for c,v in enumerate(row)) for r,row in enumerate(base))
    raise PortableV19R2Error("unknown opcode "+op)
def execute_portable(program:dict[str,Any],value:Any):
    if program.get("schema")!=SCHEMA: raise PortableV19R2Error("schema")
    hits=sorted(set(_ops(program))&BANNED)
    if hits: raise PortableV19R2Error("banned "+str(hits))
    g=_grid(value); state={}
    for binding in program.get("bindings",[]): state[str(binding["name"])]=_eval(binding["expr"],g,state)
    return _grid(_eval(program["body"],g,state))
