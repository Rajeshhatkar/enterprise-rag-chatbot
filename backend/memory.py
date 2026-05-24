memory_store = {}

def save_memory(user, query, answer):

    if user not in memory_store:
        memory_store[user] = []

    memory_store[user].append({
        "query": query,
        "answer": answer
    })


def get_memory(user):

    return memory_store.get(user, [])