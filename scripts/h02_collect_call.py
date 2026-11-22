import sys, json, modal
call_id, out = sys.argv[1], sys.argv[2]
res = modal.FunctionCall.from_id(call_id).get(timeout=4*3600)
open(out,'w').write(json.dumps(res, indent=1))
print(json.dumps({k:v for k,v in res.items() if k not in ('clips',)}, indent=1))
