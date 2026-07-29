from __future__ import annotations
from typing import Any
from runtime_v19r2 import SCHEMA, Grid, RuntimeV19R2Error, canonical, execute, node_count, sha256_json

PRODUCTION_SCHEMA = "lexigen-v19r2-invented-production-v1"

def v(name:str): return {"op":"var","name":name}
def b(op:str,left:Any,right:Any): return {"op":op,"left":left,"right":right}
def box_field(name:str,bbox:Any): return {"op":"bbox_field","name":name,"bbox":bbox}
def pair(row:Any,col:Any): return {"op":"pair","row":row,"col":col}
def param(name:str): return {"op":"param","name":name}

def build_program(marker_colour:Any, output_background:Any) -> dict[str,Any]:
    colour=v("colour"); point=v("point")
    payload_box=v("payload_box"); marker_box=v("marker_box")
    payload=v("payload_coords")
    palette_filtered={
        "op":"filter","items":{"op":"palette"},"var":"colour",
        "predicate":{"op":"and","items":[
            b("ne",colour,{"op":"mode"}),
            b("ne",colour,marker_colour),
        ]},
    }
    payload_colour={"op":"unique","items":palette_filtered}
    left=b("lt",box_field("max_col",marker_box),box_field("min_col",payload_box))
    right=b("gt",box_field("min_col",marker_box),box_field("max_col",payload_box))
    top=b("lt",box_field("max_row",marker_box),box_field("min_row",payload_box))
    reflected_col={"op":"if","condition":left,
        "then":b("sub",b("sub",b("mul",2,box_field("min_col",payload_box)),1),{"op":"second","value":point}),
        "else":{"op":"if","condition":right,
            "then":b("sub",b("add",b("mul",2,box_field("max_col",payload_box)),1),{"op":"second","value":point}),
            "else":{"op":"second","value":point}}}
    reflected_row={"op":"if","condition":top,
        "then":b("sub",b("sub",b("mul",2,box_field("min_row",payload_box)),1),{"op":"first","value":point}),
        "else":{"op":"if","condition":{"op":"or","items":[left,right]},
            "then":{"op":"first","value":point},
            "else":b("sub",b("add",b("mul",2,box_field("max_row",payload_box)),1),{"op":"first","value":point})}}
    reflected={"op":"map","items":payload,"var":"point","body":pair(reflected_row,reflected_col)}
    return {
        "schema":SCHEMA,
        "bindings":[
            {"name":"payload_colour","expr":payload_colour},
            {"name":"payload_coords","expr":{"op":"coords_colour","colour":v("payload_colour")}},
            {"name":"marker_coords","expr":{"op":"coords_colour","colour":marker_colour}},
            {"name":"payload_box","expr":{"op":"bbox","coords":payload}},
            {"name":"marker_box","expr":{"op":"bbox","coords":v("marker_coords")}},
            {"name":"reflected_coords","expr":reflected},
        ],
        "body":{"op":"paint",
            "grid":{"op":"canvas","rows":{"op":"height"},"cols":{"op":"width"},"fill":output_background},
            "coords":{"op":"set_union","items":[payload,v("reflected_coords")]},
            "colour":v("payload_colour")},
    }

def _substitute(value:Any,args:dict[str,Any])->Any:
    if isinstance(value,dict):
        if value.get("op")=="param":
            name=str(value["name"])
            if name not in args: raise RuntimeV19R2Error("missing parameter "+name)
            return args[name]
        return {k:_substitute(v,args) for k,v in value.items()}
    if isinstance(value,list): return [_substitute(v,args) for v in value]
    return value

def expand_production(production:dict[str,Any],arguments:dict[str,Any])->dict[str,Any]:
    if production.get("schema")!=PRODUCTION_SCHEMA: raise RuntimeV19R2Error("production schema")
    expected=sorted(str(x["name"]) for x in production["parameters"])
    if expected!=sorted(arguments): raise RuntimeV19R2Error("argument mismatch")
    return _substitute(production["body"],arguments)

def invent(examples:list[tuple[Grid,Grid]]):
    survivors=[]; evaluated=invalid=0
    for marker_colour in range(10):
        for output_background in range(10):
            program=build_program(marker_colour,output_background); evaluated+=1
            try: exact=all(execute(program,source)==target for source,target in examples)
            except RuntimeV19R2Error: invalid+=1; continue
            if exact: survivors.append((node_count(program),canonical(program),marker_colour,output_background,program))
    if not survivors: raise RuntimeError("fixed affine meta-grammar found no exact production")
    size,_,marker_colour,output_background,program=min(survivors)
    production={
        "schema":PRODUCTION_SCHEMA,
        "parameters":[{"name":"marker_colour","type":"colour"},{"name":"output_background","type":"colour"}],
        "body":build_program(param("marker_colour"),param("output_background")),
        "origin":{"method":"typed_argument_enumeration_and_abstraction","concrete_program_sha256":sha256_json(program)},
    }
    production["name"]="generated_"+sha256_json(production)[:16]
    arguments={"marker_colour":marker_colour,"output_background":output_background}
    return production,arguments,program,{
        "candidate_productions_evaluated":evaluated,"runtime_invalid_candidates":invalid,
        "exact_survivors":len(survivors),"selected_node_count":size,
        "selected_arguments":arguments,"concrete_program_sha256":sha256_json(program),
        "production_sha256":sha256_json(production),"named_task_operator_used":False,
    }
