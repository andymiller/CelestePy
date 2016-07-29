import autograd.numpy as np

def eval_random_dir(th, fun, zmin=-5, zmax=5, num_z=100):
    rdir = np.random.randn(th.shape[0])
    rdir /= np.sqrt(np.sum(rdir*rdir))
    funz = lambda z: fun(rdir*z + th)
    zgrid = np.linspace(zmin, zmax, num_z)
    return np.array([funz(z) for z in zgrid])

