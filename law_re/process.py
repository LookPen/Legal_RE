import json

data = []
for line in open("./test.json").readlines():
    line = json.loads(line)
    new_line = {
        "text": line["sentText"],
        "spo_list":[
        ]
    }

    for spo in line["relationMentions"]:
        new_line["spo_list"].append(
            {
                "subject": spo["em1Text"],
                "predicate": spo["label"],
                "object": spo["em2Text"]
            }
        )

    data.append(new_line)

json.dump({"data": data}, open("./test_re.json", "w"), ensure_ascii=False, indent=4)
