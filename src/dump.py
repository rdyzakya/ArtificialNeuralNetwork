
import json
from json import JSONEncoder


class ComplexEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj,'reprJSON'):
                return obj.reprJSON()
            else:
                return json.JSONEncoder.default(self, obj)

def dump(obj,filename):
    with open(filename,"w",encoding='utf-8') as f:
        dumpObject = json.loads(json.dumps(obj.reprJSON(), cls=ComplexEncoder))
        print("dump object = ",dumpObject)
        json.dump(dumpObject,f)

def load(filename):
    f = open(filename)
    data = json.load(f)
    print("load line : ",type(data))
    for i in data['layers']:
        print(i)
    f.close()
    return data

