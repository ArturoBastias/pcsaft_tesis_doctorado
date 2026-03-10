# Script to solve for pressure varying density
import pyomo.environ as pyo
from numpy import pi

PI = pi
KBOLTZMANN = 1.380649e-23  # [J/K] Boltzmann constant
NAVOGADRO = 6.02214076e23  # [mol-1] Avogadro's number
# %% MODEL
m = pyo.ConcreteModel(name="PCSAFT")
# SETS
m.set03 = pyo.RangeSet(0, 3)  # for zeta
m.set02 = pyo.RangeSet(0, 2)  # for disp_const
m.set06 = pyo.RangeSet(0, 6)  # for disp_const, I1, eta_aux
m.const_index = m.set02 * m.set06  # for disp_const

# PARAMETERS
m.mseg = pyo.Param(initialize=86.177*.0354812325)  # Hexane
m.sigma = pyo.Param(initialize=3.79829291e-10)  # [m], not [A]
m.epskb = pyo.Param(initialize=236.769054)
m.easskb = pyo.Param(initialize=1)  # dummy for now
m.kass = pyo.Param(initialize=1)  # dummy for now

# VARIABLES
m.tmprt = pyo.Var(name="Temperature (K)", within=pyo.PositiveReals)
m.press = pyo.Var(name="Pressure (Pa)", within=pyo.PositiveReals)

# Choose molecular density or reduced density as variable
# m.ndens = pyo.Var(name="Density (1/m3)", within=pyo.PositiveReals)
# m.eta = None
m.eta = pyo.Var(name="Reduced density (-)", within=pyo.PositiveReals, bounds=(0, 0.7405))
m.ndens = None

# EQUATIONS
m.d = pyo.Expression(expr=m.sigma * (1 - 0.12 * pyo.exp(-3 * m.epskb / m.tmprt)))
m.aux_xi = pyo.Expression(m.set03, rule=lambda m_, n: m_.mseg * PI / 6 * m_.d ** n)
# m.deta_drho = pyo.Expression(expr=m.aux_xi[m.set03.last()])
m.deta_drho = pyo.Expression(expr=m.aux_xi[3])
if m.ndens is not None:
    # molecular density is variable, reduced density is fixed
    del m.eta
    m.xi = pyo.Expression(m.set03, rule=lambda m_, n: m_.ndens * m_.aux_xi[n])
    m.eta = pyo.Expression(expr=m.xi[3])
else:
    # reduced density is variable, molecular density is fixed
    del m.ndens
    m.ndens = pyo.Expression(expr=m.eta / m.aux_xi[3])
    m.xi = pyo.Expression(m.set03, rule=lambda m_, n: m_.ndens * m_.aux_xi[n])

# Monomer
m.ahs = pyo.Expression(expr=((3 * m.xi[1] * m.xi[2]) / (1 - m.xi[3]) + m.xi[2] ** 3 / ((1 - m.xi[3]) ** 2 * m.xi[3]) +
                             (-m.xi[0] + m.xi[2] ** 3 / m.xi[3] ** 2) * pyo.log(1 - m.xi[3])) / m.xi[0])
m.dahs_deta = pyo.Expression(expr=(m.xi[2] ** 3 * (-3 + m.xi[3]) + 3 * m.xi[1] * m.xi[2] * (-1 + m.xi[3]) -
                                   m.xi[0] * (-1 + m.xi[3]) ** 2 * m.xi[3]) / (m.xi[0] * (-1 + m.xi[3]) ** 3 * m.xi[3]))
# Chain
m.ghs = pyo.Expression(expr=(m.d ** 2 * m.xi[2] ** 2) / (2 * (1 - m.xi[3]) ** 3) + (3 * m.d * m.xi[2]) /
                            (2 * (1 - m.xi[3]) ** 2) + 1 / (1 - m.xi[3]))
m.dghs_deta = pyo.Expression(expr=(2 * (-1 + m.xi[3]) ** 2 * m.xi[3] + m.d ** 2 * m.xi[2] ** 2 * (2 + m.xi[3]) -
                                   3 * m.d * m.xi[2] * (-1 + m.xi[3] ** 2)) / (2 * (-1 + m.xi[3]) ** 4 * m.xi[3]))
m.ahc = pyo.Expression(expr=m.ahs * m.mseg - (-1 + m.mseg) * pyo.log(m.ghs))
m.dahc_deta = pyo.Expression(expr=-((m.dghs_deta * (-1 + m.mseg)) / m.ghs) + m.dahs_deta * m.mseg)

# Dispersion
a_const = {(0, 0): 0.91056314451539, (1, 0): -0.30840169182720, (2, 0): -0.09061483509767, (0, 1): 0.63612814494991,
           (1, 1): 0.18605311591713, (2, 1): 0.45278428063920, (0, 2): 2.68613478913903, (1, 2): -2.50300472586548,
           (2, 2): 0.59627007280101, (0, 3): -26.5473624914884, (1, 3): 21.41979362966688, (2, 3): -1.72418291311787,
           (0, 4): 97.7592087835073, (1, 4): -65.2558853303492, (2, 4): -4.13021125311661, (0, 5): -159.591540865600,
           (1, 5): 83.3186804808856, (2, 5): 13.7766318697211, (0, 6): 91.2977740839123, (1, 6): -33.7469229297323,
           (2, 6): -8.67284703679646}
m.a_const = pyo.Param(m.const_index, initialize=a_const)
b_const = {(0, 0): 0.72409469413165, (1, 0): -0.57554980753450, (2, 0): 0.09768831158356, (0, 1): 1.11913959304690 * 2,
           (1, 1): 0.34975477607218 * 2, (2, 1): -0.12787874908050 * 2, (0, 2): -1.33419498282114 * 3,
           (1, 2): 1.29752244631769 * 3, (2, 2): -3.05195205099107 * 3, (0, 3): -5.25089420371162 * 4,
           (1, 3): -4.30386791194303 * 4, (2, 3): 5.16051899359931 * 4, (0, 4): 5.37112827253230 * 5,
           (1, 4): 38.5344528930499 * 5, (2, 4): -7.76088601041257 * 5, (0, 5): 34.4252230677698 * 6,
           (1, 5): -26.9710769414608 * 6, (2, 5): 15.6044623461691 * 6, (0, 6): -50.8003365888685 * 7,
           (1, 6): -23.6010990650801 * 7, (2, 6): -4.23812936930675 * 7}
m.b_const = pyo.Param(m.const_index, initialize=b_const)
m.aa = pyo.Expression(m.set06, rule=lambda m_, i: m_.a_const[0, i]+((m_.mseg-1)/m_.mseg)*m_.a_const[1, i]+((m_.mseg-1)/m_.mseg)*((m_.mseg-2)/m_.mseg)*m_.a_const[2, i])
m.bb = pyo.Expression(m.set06, rule=lambda m_, i: m_.b_const[0, i]+((m_.mseg-1)/m_.mseg)*m_.b_const[1, i]+((m_.mseg-1)/m_.mseg)*((m_.mseg-2)/m_.mseg)*m_.b_const[2, i])
m.mes3 = pyo.Expression(expr=m.mseg * (m.sigma ** 3) * m.epskb / m.tmprt)
m.m2e2s3 = pyo.Expression(expr=(m.mseg ** 2) * (m.sigma ** 3) * (m.epskb / m.tmprt) ** 2)

m.c1 = pyo.Expression(expr=1 / (1 + (m.mseg * (8 * m.eta - 2 * m.eta ** 2)) / (1 - m.eta) ** 4 + (
        (1 - m.mseg) * (20 * m.eta - 27 * m.eta ** 2 + 12 * m.eta ** 3 - 2 * m.eta ** 4)) / (
                                        (1 - m.eta) ** 2 * (2 - m.eta) ** 2)))
m.dc1_deta = pyo.Expression(expr=(-2 * (-2 + m.eta) * (-1 + m.eta) ** 3 * (
        (-1 + m.eta) ** 2 * (20 + m.eta * (-24 + m.eta * (6 + m.eta))) + m.mseg * (
        12 + (-1 + m.eta) * m.eta * (-96 + m.eta * (90 + (-25 + m.eta) * m.eta))))) / (4 + m.eta * (
        -(m.eta * (26 + m.eta * (-42 + m.eta * (27 + (-8 + m.eta) * m.eta)))) + m.mseg * (
        12 + m.eta * (27 + m.eta * (-70 + m.eta * (51 + 2 * (-8 + m.eta) * m.eta)))))) ** 2)
m.eta_aux1 = pyo.Expression(m.set06, rule=lambda m_, i: m.eta ** i)
m.I1 = pyo.Expression(expr=pyo.sum_product(m.aa, m.eta_aux1))
m.I2 = pyo.Expression(expr=pyo.sum_product(m.bb, m.eta_aux1))
m.eta_aux2 = pyo.Expression(m.set06, rule=lambda m_, i: i * m.eta ** (i-1))
m.dI1_deta = pyo.Expression(expr=pyo.sum_product(m.aa, m.eta_aux2))
m.dI2_deta = pyo.Expression(expr=pyo.sum_product(m.bb, m.eta_aux2))
m.adisp = pyo.Expression(expr=-m.xi[0]*(12*m.I1*m.mes3 + 6*m.c1*m.I2*m.m2e2s3))
m.dadisp_deta = pyo.Expression(expr=-(m.xi[3]*(6*m.c1*m.dI2_deta*m.m2e2s3 + 6*m.dc1_deta*m.I2*m.m2e2s3 + 12*m.dI1_deta*m.mes3) + (6*m.c1*m.I2*m.m2e2s3 + 12*m.I1*m.mes3))/m.d**3)

# Association
# Total residual
m.ares = pyo.Expression(expr=m.ahc + m.adisp)
m.dares_drho = pyo.Expression(expr=m.deta_drho * (m.dahc_deta + m.dadisp_deta))

# Ideal TODO: Only include a when calculating T derivatives.
h = 6.626070150e-34  # J s
me = 9.10938291e-31  # 1/Kg TODO: Would be better to use Mw/Na of substance.
m.beta = pyo.Expression(expr=1./(m.tmprt * KBOLTZMANN))
m.broglie_vol = pyo.Expression(expr=h / pyo.sqrt(2*PI * me * m.beta))
m.aideal = pyo.Expression(expr=pyo.log(m.ndens * m.broglie_vol**3) - 1)
m.daideal_drho = pyo.Expression(expr=1./m.ndens)

# Total Helhmholtz
m.atotal = pyo.Expression(expr=(m.ares + m.aideal)*(NAVOGADRO/m.beta))
m.datotal_drho = pyo.Expression(expr=(m.dares_drho + m.daideal_drho)*(NAVOGADRO/m.beta))

# Pressure
m.press_calc = pyo.Expression(expr=m.ndens**2 * m.datotal_drho / NAVOGADRO)

# CONSTRAINTS
# OBJECTIVE
m.obj = pyo.Objective(expr=(m.press - m.press_calc)**2/1e10, sense=pyo.minimize)

#
m.tmprt.fix(298.15)
m.press.fix(1e5)
mrho_hexane = 1000  # kg/m3
nrho_hexane = mrho_hexane*1000/86.177*NAVOGADRO
# m.ndens = nrho_hexane
held_eta = 0.3851883096498
m.eta = 0.6
# Save initial values
obj_init = pyo.value(m.obj)
eta_init = pyo.value(m.eta)
ndens_init = pyo.value(m.ndens)

# SOLVE
pardisopath = "/Users/arturobastias/Downloads/panua-pardiso-20240630-mac_x86/lib/libpardiso.dylib"
hslpath = "/Users/arturobastias/Downloads/CoinHSL.v2023.11.17.x86_64" \
          "-apple-darwin-libgfortran5/lib/libcoinhsl.dylib"
kwds = {"tee": True, "options": {
#    "linear_solver": "ma57", "hsllib": hslpath,
#    "linear_solver": "pardiso", "pardisolib": pardisopath,
    "linear_solver": "mumps",
                                #"max_iter": 0
                                 }}
pyo.SolverFactory('ipopt').solve(m, **kwds)
# using ma57 as linear_solver may throw error (a bug, it is too fast)
# using mpardiso as linear_solver may throw error at 1st iteration (a bug, idk)

print("----- Initial values -----")
print("objective: ", obj_init)
print("eta: ", eta_init)
print("rho: ", ndens_init)
print("-----  Final values  -----")
print("objective: ", pyo.value(m.obj))
print("eta: ", pyo.value(m.eta))
print("rho: ", pyo.value(m.ndens))
# print(pyo.value(m.ahs))
# print(pyo.value(m.ahc))
# print(pyo.value(m.adisp))
# print(pyo.value(m.dahs_deta))
# print(pyo.value(m.dahc_deta))
# print(pyo.value(m.dadisp_deta))
# print(pyo.value(m.datotal_drho))
print("Calculated pressure: ", pyo.value(m.press_calc))
