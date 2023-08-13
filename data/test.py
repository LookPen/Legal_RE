import json

path = "./re.json"

data = json.load(open(path, "r"))["data"]

rels = []
for line in data:
    for spo in line["spo_list"]:
        rels.append(spo["predicate"])


rels = list(set(rels))

json.dump(rels, open("./schemas.json", "w"),ensure_ascii=False)


json.dump(data[:1100], open("./train.json", "w"), ensure_ascii=False, indent=4)
json.dump(data[1100:], open("./test.json", "w"), ensure_ascii=False, indent=4)

