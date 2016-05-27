# Simple optimizers swiped from
# https://github.com/HIPS/hypergrad/blob/master/hypergrad/optimizers.py
# Thanks Dougal and David!
import numpy as np
from scipy.optimize import minimize
import autograd.numpy as np
from builtins import range


def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x += step_size * velocity
    return x


def rmsprop(grad, x, callback=None, num_iters=100, step_size=0.1, gamma=0.9, eps = 10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x


def adam(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return x


def bfgs(obj_and_grad, x, callback=None, num_iters=100):
    def epoch_counter():
        epoch = 0
        while True:
            yield epoch
            epoch += 1
    ec = epoch_counter()

    wrapped_callback=None
    if callback:
        def wrapped_callback(x):
            callback(x, next(ec))

    res =  minimize(fun=obj_and_grad, x0=x, jac =True, callback=wrapped_callback,
                    options = {'maxiter':num_iters, 'disp':True})
    return res.x

def minimize_chunk(fun, jac, x0, method, max_iter, 
                   chunk_size = 25,
                   callback   = None,
                   verbose    = True):
    """ minimize function that saves every few iterations """
    num_chunks = int(np.ceil(max_iter / float(chunk_size)))
    for chunk_i in range(num_chunks):
        print "optimizing chunk %d of %d (curr_ll = %2.7g)"%(chunk_i, num_chunks, fun(x0))
        res = minimize(fun = fun, jac = jac, x0 = x0, method = method,
                       options = {'maxiter': chunk_size, 'disp': verbose})
        x0  = res.x

        # perform callback at certain iter
        if callback is not None:
            callback(x0, chunk_i)
    return res

