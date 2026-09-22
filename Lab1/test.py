import numpy as np
import matplotlib.pyplot as plt
import dlfbt_lab1 as solution
import dlfbt

# 1. TO_DO
dg = dlfbt.DataGeneratorLinear(a=[-5.0, 2.0, -3.0, 2.0])
dg.create_dataset(noise=0.0, n=500)

linrm = solution.LinearRegressionModel()
linrm.w = dg.a
linrm.b = dg.b

y = linrm.predict(dg.x)

tol = 1.0e-15
assert y.shape == (500, 1)
assert np.abs(y - dg.t).max() < tol
assert linrm.get_loss(dg.x, dg.t) < tol

linrm = solution.LinearRegressionModel()
linrm.w = np.array([[2.0]])
linrm.b = np.array([[1.0]])

x = np.array([[3.0]])
t = np.array([[6.0]])
db, dw = linrm.compute_gradients(x, t)

assert db.shape == (1, 1)
assert db == [[1.0]]
assert dw.shape == (1, 1)
assert dw == [[3.0]]
