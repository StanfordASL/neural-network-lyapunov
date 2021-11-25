"""
Verify that the dynamics for the pole/end-effector system is correct.
"""
import sympy

xA, yA, zA = sympy.symbols("xA yA zA")
length = sympy.symbols("l")
me, ms = sympy.symbols("me ms")
g = sympy.symbols("g")
alpha = sympy.symbols("alpha")
beta = sympy.symbols("beta")
xAdot, yAdot, zAdot = sympy.symbols("xAd yAd zAd")
alphadot, betadot = sympy.symbols("alphad betad")
xAddot, yAddot, zAddot = sympy.symbols("xAdd yAdd zAdd")
alphaddot, betaddot = sympy.symbols("alphadd betadd")
p_WS = sympy.Matrix([
    xA + length * sympy.cos(beta) * sympy.cos(alpha),
    yA + length * sympy.cos(beta) * sympy.sin(alpha),
    zA + length * sympy.sin(beta)
])
q = [xA, yA, zA, alpha, beta]
qdot = [xAdot, yAdot, zAdot, alphadot, betadot]
qddot = [xAddot, yAddot, zAddot, alphaddot, betaddot]
v_WS = sympy.Matrix([
    sum([sympy.diff(p_WS[i], var) * vardot for (var, vardot) in zip(q, qdot)])
    for i in range(3)
])
T = 0.5 * me * (xAdot**2 + yAdot**2 +
                zAdot**2) + 0.5 * ms * (v_WS[0]**2 + v_WS[1]**2 + v_WS[2]**2)
U = me * g * zA + ms * g * (zA + length * sympy.sin(beta))
L = T - U
dLdqdot = [sympy.diff(L, qdot[i]) for i in range(5)]
dLdq = [sympy.diff(L, q[i]) for i in range(5)]
lhs = [
    sum([sympy.diff(dLdqdot[row], q[i]) * qdot[i] for i in range(5)]) +
    sum([sympy.diff(dLdqdot[row], qdot[i]) * qddot[i]
         for i in range(5)]) - dLdq[row] for row in range(5)
]
for row in range(3):
    print(f"row {row}:")
    print(sympy.simplify(lhs[row]))

print("simplify row 3:")
print(sympy.simplify(lhs[3] / (ms * length)))
print("simplify row 4:")
print(sympy.simplify(lhs[4] / (ms * length)))
