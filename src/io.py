from pathlib import Path

def check(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def copen(path, *args, **kwargs):
    check(path) 
    return open(path, *args, **kwargs)

class FileIOSTR():
    def __init__(self, sys_name, subfolder):
        self.name = f"{sys_name}/{subfolder}"

    def file(self, name, type=None, verbose=1):
        if type is None:
            filename = f"../generated/{self.name}/{name}"
        else:
            filename = f"../generated/{self.name}/{type}/{name}"
        self.vprint(verbose, filename=filename)
        check(filename)
        return filename
        
    def fig(self, name, **kwargs):
        return self.file(name, type="figure", **kwargs)

    def traj(self, name, **kwargs):
        return self.file(name, type="traj", **kwargs)

    def data(self, name, **kwargs):
        return self.file(name, type="datafile", **kwargs)

    def state_dict(self, name, **kwargs):
        return self.file(name, type="state_dict", **kwargs)
    
    def tmp(self, name, verbose=1):
        filename = f"../tmp/{self.name}/{name}"
        check(filename)
        self.vprint(verbose, filename=filename)
        return filename

    def vprint(self, verbose, filename=None, **kwargs):
        if verbose==1:
            print(filename)


class FileIO():
    def __init__(self, sys_name, subfolder):
        self.str = FileIOSTR(sys_name, subfolder)

    def file(self, *args, w="wb+", **kwargs):
        return copen(self.str.file(*args, **kwargs), w)

    def fig(self, name, **kwargs):
        return self.file(name, type="figure", **kwargs)

    def traj(self, name, **kwargs):
        return self.file(name, type="traj", **kwargs)

    def data(self, name, **kwargs):
        return self.file(name, type="datafile", **kwargs)

    def state_dict(self, name, **kwargs):
        return self.file(name, type="state_dict", **kwargs)
    
    def tmp(self, *args, w="wb+", **kwargs):
        return copen(self.str.tmp(*args, **kwargs), w)
