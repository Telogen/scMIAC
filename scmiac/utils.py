import pickle

def savepkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def readpkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data



