from astropy import units as U

def to_preferred(a, units):
    return a.decompose(bases=units)

def convert(units, *args):
    if len(args) > 1:
        out = []
        for a in args:
            out += [to_preferred(a, units)]
        return out
    else:
        return to_preferred(args[0], units)

def ustrip(*args):
    if len(args) > 1:
        return [a.value for a in args]
    else:
        return args[0].value
