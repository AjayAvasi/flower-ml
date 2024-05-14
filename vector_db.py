import json
import numpy
import pandas as pd


def add_vector(vector, image_path):
    with open("./vector_data.json", "r") as f:
        db = json.load(f)
    db.append({"vector": vector, "img_path": image_path})
    with open("./vector_data.json", "w") as f:
        json.dump(db, f, indent=1)


def add_vectors(data):
    with open("./vector_data.json", "r") as f:
        db = json.load(f)
    db = db + data
    with open("./vector_data.json", "w") as f:
        json.dump(db, f, indent=1)


def get_k_closest(k, vector):
    with open("./vector_data.json", "r") as f:
        db = json.load(f)
        data = {
            "distances": [],
            "img_path": []
        }
        for row in db:
            data["distances"].append(numpy.sum(numpy.abs(numpy.array(row["vector"]) - numpy.array(vector))))
            data["img_path"].append(row["img_path"])
        df = pd.DataFrame(data)
        df = df.sort_values(by="distances", ascending=True)
        return df.head(k)
