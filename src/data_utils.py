import pickle

def load_pickle(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def save_pickle(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


