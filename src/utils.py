import einops


def bmult(x,y):
    if isinstance(y,float):
        return x*y
    return einops.einsum(x, y, 'b ..., b -> b ...')