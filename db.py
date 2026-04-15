from pymongo import MongoClient

def get_collection():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["sentiment_db"]
        return db["results"]
    except:
        return None


def save_result(data):
    collection = get_collection()

    if collection is not None:
        try:
            collection.insert_one(data)
        except:
            pass