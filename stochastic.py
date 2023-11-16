import numpy as np
from scipy.optimize import OptimizeResult

def sgd(
    fun,
    x0,
    args=(),
    learning_rate=0.001,
    mass=0.9,
    startiter=0,
    maxiter=1000,
    use_proj=False,
    verbose=False,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of stochastic
    gradient descent with momentum.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        f, g = fun(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

        if use_proj:
            x[-1] = x[-1].clip(min = 1e-9)
        
        if verbose:
            print('sgd iter {}, step = {:.8f}, obj = {}'.format(i, learning_rate, f))
            print(x[:20])
            print(x[-1])

    i += 1
    return OptimizeResult(x=x, fun=f, jac=g, nit=i, nfev=i, success=True)


def rmsprop(
    fun,
    x0,
    args=(),
    learning_rate=0.1,
    gamma=0.9,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    use_proj=False,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of root mean
    squared prop: See Adagrad paper for details.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    avg_sq_grad = np.ones_like(x)

    for i in range(startiter, startiter + maxiter):
        f, g = fun(x)

        if callback and callback(x):
            break

        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)
        if use_proj:
            x[-1] = x[-1].clip(min = 1e-9)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def adam(
    func_grad,
    x0,
    args=(),
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    use_proj=False,
    verbose=False,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].

    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    lr = learning_rate
    for i in range(startiter, startiter + maxiter):

        f, g = func_grad(x)
        
        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        dt = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - dt

        if use_proj:
            x[-1] = x[-1].clip(min = 1e-9)
            # x = x - min(0, x[-1]) / dt[-1] * dt
            
        if verbose:
            print('adam iter {}, step = {:.8f}, obj = {}'.format(i, lr, f))
            print(x[:20])
            print(x[-1])

    i += 1

    return OptimizeResult(x=x, fun=f, jac=g, nit=i, nfev=i, success=True)

