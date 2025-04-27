from .models import TrussGraphModel
import optax
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
import matplotlib.pyplot as plt
import jax.numpy as jnp

def train(*args, loss=None, **kwargs):
    def fn(
        model: TrussGraphModel,
        trainloader,
        testloader,
        optim: optax.GradientTransformation,
        steps: int,
        print_every: int,
        ) -> TrussGraphModel:
        # Just like earlier: It only makes sense to train the arrays in our model,
        # so filter out everything else.
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # Always wrap everything -- computing gradients, running the optimiser, updating
        # the model -- into a single JIT region. This ensures things run as fast as
        # possible.
        @eqx.filter_jit
        def make_step(
            model: TrussGraphModel,
            opt_state: PyTree,
            trainloader
        ):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, trainloader)
            updates, opt_state = optim.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        # Loop over our training dataset as many times as we need.
        def infinite_trainloader():
            while True:
                yield from trainloader

        Loss = []
        for step in range(steps):
            model, opt_state, train_loss = make_step(model, opt_state, trainloader)
            if (step % print_every) == 0 or (step == steps - 1):
                print(
                    f"{step=}, train_loss={train_loss.item()}, "
                )
            if train_loss.item() < 1.0e-10:
                print("Done.")
                break
            Loss.append(train_loss.item())

        extra = (Loss, )
        return model, extra
    return fn


def mse(y, y_, M=None): 
    dy = (y - y_)
    A = jnp.mean(jnp.square(dy), axis=0)
    if M is None:
        return A.mean()
    return (M @ A).mean() / M.mean()
    
def loss(model, data, M=None, trN=None, sensors=None):
    print("recompile loss")
    L1 = 0.0
    for (x, y) in data:
        y = y[:trN, :]
        y_ = model(x)[2].reshape(trN, -1)
        L1 += mse(y[:trN][:, sensors], y_[:trN][:, sensors], M=M if M is None else M[sensors, sensors])
    return L1 
