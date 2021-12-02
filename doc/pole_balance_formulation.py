"""
Verify that the dynamics for the pole/end-effector system is correct.
"""
import sympy

xA, yA, zA = sympy.symbols("xA yA zA")
length = sympy.symbols("l")
me, ms = sympy.symbols("me ms")
g = sympy.symbols("g")
xAdot, yAdot, zAdot = sympy.symbols("xAd yAd zAd")
xAddot, yAddot, zAddot = sympy.symbols("xAdd yAdd zAdd")
xAB, yAB, zAB = sympy.symbols("xAB yAB zAB")
xABdot, yABdot = sympy.symbols("xABd yABd")
xABddot, yABddot = sympy.symbols("xABdd yABdd")
p_WS = sympy.Matrix([
    xA + xAB, yA + yAB, zA + sympy.sqrt(length**2 - xAB**2 - yAB**2)
])
q = [xA, yA, zA, xAB, yAB]
qdot = [xAdot, yAdot, zAdot, xABdot, yABdot]
qddot = [xAddot, yAddot, zAddot, xABddot, yABddot]
v_WS = sympy.Matrix([
    sum([sympy.diff(p_WS[i], var) * vardot for (var, vardot) in zip(q, qdot)])
    for i in range(3)
])
for i in range(3):
    print(f"v_WS[{i}]")
    print(v_WS[i])

T = 0.5 * me * (xAdot**2 + yAdot**2 +
                zAdot**2) + 0.5 * ms * (v_WS[0]**2 + v_WS[1]**2 + v_WS[2]**2)
U = me * g * zA + ms * g * p_WS[2]
L = T - U
dLdqdot = [sympy.diff(L, qdot[i]) for i in range(5)]
dLdq = [sympy.diff(L, q[i]) for i in range(5)]
lhs = [
    sum([sympy.diff(dLdqdot[row], q[i]) * qdot[i] for i in range(5)]) +
    sum([sympy.diff(dLdqdot[row], qdot[i]) * qddot[i]
         for i in range(5)]) - dLdq[row] for row in range(5)
]
for row in range(2):
    print(f"row {row}:")
    print(sympy.simplify(lhs[row]))
for row in range(2, 5):
    print(f"row {row}:")
    print(
        sympy.simplify(lhs[row].subs(sympy.sqrt(length**2 - xAB**2 - yAB**2),
                                     zAB)))
