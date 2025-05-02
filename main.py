import jax
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
import sympy2jax
import pickle
import functools as f
import jax.numpy as jnp
from jax.flatten_util  import ravel_pytree
from jax import random
import tqdm
jax.config.update("jax_enable_x64", True)



def readPolytopes(fname="polytopes.txt"):
    polytopes = []
    symbols = set()
    with open(fname,"r") as file:
        for line in file:
            eqs = line.split(";")
            polytope = True
            print ("------")
            for eq in eqs:
                if eq.count ("<") <= 1:
                    print (eq)
                    expr = sp.parse_expr(eq, transformations=standard_transformations+(implicit_multiplication,))
                    symbols = symbols.union(expr.atoms(sp.Symbol))
                    print (expr)
                    polytope = polytope & expr
                else:
                    t1,t2,t3 = [sp.parse_expr(ti, transformations=standard_transformations+(implicit_multiplication,)) for ti in eq.split("<")]
                    symbols = symbols.union(t1.atoms(sp.Symbol))
                    symbols = symbols.union(t2.atoms(sp.Symbol))
                    symbols = symbols.union(t3.atoms(sp.Symbol))
                    print (t1 < t2)
                    print (t2 < t3)
                    polytope = polytope & (t1 < t2) & (t2 < t3)


            polytopes.append (polytope)
            #polytopes.append (sympy2jax.SymbolicModule(polytope))
    polytopes = sympy2jax.SymbolicModule(polytopes)
    y0 = {str(sym): jnp.array(0) for sym in symbols}
    _, f = ravel_pytree(y0)

    return jax.jit(lambda x: polytopes(**f(x)))


def test(key, dim, npoints, polytopes, box_min, box_max):
    points = random.uniform(key, [npoints,dim])*(box_max-box_min) + box_min #double not rational
    batched_polytopes = jax.vmap(polytopes)

    ret = jnp.array(batched_polytopes(points))
    return ret.sum(axis=1)


def batched_tests(seed, dim, nepoch, batch_size, polytopes, box_min, box_max):
    x0 = jnp.zeros(dim)
    x0 = polytopes(x0)
    ret = jnp.array([False for _ in x0])


    key = random.key(seed)
    for i in tqdm.tqdm(range(nepoch)):
        key, subkey = random.split(key)
        ret  += test(key, dim, batch_size, polytopes, box_min, box_max)

    return ret    #return ret/(nepoch*batch_size)*(box_max-box_min)**2, ret.min(), ret.max()


    
    

