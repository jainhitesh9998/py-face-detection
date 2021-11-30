from os.path import dirname, realpath, exists


def generate(file_name=None):
    if not file_name:
        return dirname(realpath(__file__))
    path = generate() + "/" + file_name
    return path


def get(file_name=None):
    path = generate(file_name)
    if exists(path):
        return path
    return None
