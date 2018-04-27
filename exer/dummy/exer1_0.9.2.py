# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:40:22 2018

@author: Bany
"""
import operator
import math
import random
import logging
import sys
import time
import numpy
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# system method
def stringify(expr):
    """Evaluate the expression *expr* into a string.
    """
    string = ""
    stack = []
    for node in expr:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = prim.format(*args)
            if len(stack) == 0:
                break   # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)

    return string


# Define new functions
def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0
def safeLog(uni):
    try:
        return math.log(uni)
    except ValueError:
        return 0
            
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(safeLog, 1)
pset.addEphemeralConstant(ephemeral=lambda: random.randint(-5,6))
pset.renameArguments(ARG0 = 'x')
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genRamped, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.lambdify, pset=pset)
'''
y={}
for i in range(-10,10):
    y[i/10.] = numpy.random.randint(-5,5)
    
g = pd.read_csv('data.csv')
x = g['x']
t = list(range(100))
f = {}
for i in t:
    f[i] = x[i]'''

t = list(range(-200,200))


f = {}
for i in t:
    #f[i]=math.cos(i/10)-2*math.sin(i/10)
     f[i]=i**5 - 3*i**4 + 5*i**2 + 2

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.lambdify(expr=individual)
    # Evaluate the sum of squared difference between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #values = (x/10. for x in range(-10,10))
    start1 = time.time()
    diff_func = lambda x: (func(x)-f[x])**2
    start2 = time.time()

    diff = sum(map(diff_func, t))+(start2-start1)*10

    return diff,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut)

def main():
    random.seed(318)
    
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", tools.mean)
    stats.register("Std", tools.std)
    stats.register("Min", min)
    stats.register("Max", max)
    try:
        algorithms.eaSimple(pop,toolbox,  0.5, 0.1, 1000, stats, halloffame=hof)
    except MemoryError:
        logging.info("Best individual is %s, %s", stringify(hof[0]), hof[0].fitness)
    except KeyboardInterrupt:
        logging.info("Best individual is %s, %s", stringify(hof[0]), hof[0].fitness)
    logging.info("Best individual is %s, %s", stringify(hof[0]), hof[0].fitness)
    return pop, stats, hof

if __name__ == "__main__":
    main()
