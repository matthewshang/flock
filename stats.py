import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnames=['next_fn', 'n_steps', 'J_fn'])
def time_average_with_history(
        initial_state, 
        next_fn,
        n_steps, 
        J_fn):

    def step(state, _):
        next_state = next_fn(state)
        return next_state, J_fn(next_state)

    _, J = jax.lax.scan(step, initial_state, None, n_steps)
    cumsum = jnp.cumsum(J, axis=0)
    dims = (-1, 1) if J.ndim > 1 else -1
    correction = jnp.arange(1, n_steps + 1).reshape(dims)
    return cumsum / correction

def time_average(initial_state, next_fn, n_steps, J_fn):
    def step(carry, _):
        state, J = carry
        next_state = next_fn(state)
        return (next_state, J + J_fn(next_state)), None
    
    J = jax.lax.scan(step, (initial_state, 0.0), None, n_steps)[0][1]
    return J / n_steps

@functools.partial(jax.jit, static_argnames=['next_fn', 'n_steps', 'pos_fn'])
def trajectory(initial_state, next_fn, n_steps, pos_fn):
    def step(state, _):
        next_state = next_fn(state)
        return next_state, pos_fn(next_state)
    
    _, pos = jax.lax.scan(step, initial_state, None, n_steps)
    return pos

@functools.partial(jax.jit, static_argnames=['next_fn', 'n_steps'])
def advance(initial_state, next_fn, n_steps):
    def step(state, _):
        return next_fn(state), None
    
    return jax.lax.scan(step, initial_state, None, n_steps)[0]