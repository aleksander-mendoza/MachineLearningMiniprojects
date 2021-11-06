import json


def incr(m, e):
    if e in m:
        m[e] += 1
    else:
        m[e] = 1


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(json_object, file):
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    with open(file, 'w+', encoding='utf-8') as fd:
        json.dump(json_object, fd, ensure_ascii=False, indent=4, cls=SetEncoder)
