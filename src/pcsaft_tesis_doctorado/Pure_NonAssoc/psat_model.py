# Script to solve for pressure varying density
import pyomo.environ as pyo
from numpy import pi as PI

KBOLTZMANN = 1.380649e-23  # [J/K] Boltzmann constant
NAVOGADRO = 6.02214076e23  # [mol-1] Avogadro's number


# %% MODEL
def create_pcsaft():
    m = pyo.ConcreteModel(name="PCSAFT")
    # SETS
    m.set03 = pyo.RangeSet(0, 3)  # for zeta
    m.set02 = pyo.RangeSet(0, 2)  # for disp_const
    m.set06 = pyo.RangeSet(0, 6)  # for disp_const, I1, eta_aux
    m.const_index = m.set02 * m.set06  # for disp_const

    # PARAMETERS
    m.mseg = pyo.Param(initialize=86.177 * .0354812325)  # Hexane
    m.sigma = pyo.Param(initialize=3.79829291e-10)  # [m], not [A]
    m.epskb = pyo.Param(initialize=236.769054)
    m.easskb = pyo.Param(initialize=1)
    m.kass = pyo.Param(initialize=1)

    # VARIABLES
    m.tmprt = pyo.Var(name="Temperature (K)", within=pyo.PositiveReals)
    m.press = pyo.Var(name="Pressure (Pa)", within=pyo.PositiveReals)

    # EQUATIONS
    m.d = pyo.Expression(expr=m.sigma * (1 - 0.12 * pyo.exp(-3 * m.epskb / m.tmprt)))
    m.aux_xi = pyo.Expression(m.set03, rule=lambda m_, n: m_.mseg * PI / 6 * m_.d ** n)
    # m.deta_drho = pyo.Expression(expr=m.aux_xi[m.set03.last()])
    m.deta_drho = pyo.Expression(expr=m.aux_xi[3])

    # Dispersion
    a_const = {(0, 0): 0.91056314451539, (1, 0): -0.30840169182720, (2, 0): -0.09061483509767, (0, 1): 0.63612814494991,
               (1, 1): 0.18605311591713, (2, 1): 0.45278428063920, (0, 2): 2.68613478913903, (1, 2): -2.50300472586548,
               (2, 2): 0.59627007280101, (0, 3): -26.5473624914884, (1, 3): 21.41979362966688,
               (2, 3): -1.72418291311787,
               (0, 4): 97.7592087835073, (1, 4): -65.2558853303492, (2, 4): -4.13021125311661,
               (0, 5): -159.591540865600,
               (1, 5): 83.3186804808856, (2, 5): 13.7766318697211, (0, 6): 91.2977740839123, (1, 6): -33.7469229297323,
               (2, 6): -8.67284703679646}
    m.a_const = pyo.Param(m.const_index, initialize=a_const)
    b_const = {(0, 0): 0.72409469413165, (1, 0): -0.57554980753450, (2, 0): 0.09768831158356,
               (0, 1): 1.11913959304690 * 2,
               (1, 1): 0.34975477607218 * 2, (2, 1): -0.12787874908050 * 2, (0, 2): -1.33419498282114 * 3,
               (1, 2): 1.29752244631769 * 3, (2, 2): -3.05195205099107 * 3, (0, 3): -5.25089420371162 * 4,
               (1, 3): -4.30386791194303 * 4, (2, 3): 5.16051899359931 * 4, (0, 4): 5.37112827253230 * 5,
               (1, 4): 38.5344528930499 * 5, (2, 4): -7.76088601041257 * 5, (0, 5): 34.4252230677698 * 6,
               (1, 5): -26.9710769414608 * 6, (2, 5): 15.6044623461691 * 6, (0, 6): -50.8003365888685 * 7,
               (1, 6): -23.6010990650801 * 7, (2, 6): -4.23812936930675 * 7}
    m.b_const = pyo.Param(m.const_index, initialize=b_const)
    m.aa = pyo.Expression(m.set06,
                          rule=lambda m_, i: m_.a_const[0, i] + ((m_.mseg - 1) / m_.mseg) * m_.a_const[1, i] + (
                                  (m_.mseg - 1) / m_.mseg) * ((m_.mseg - 2) / m_.mseg) * m_.a_const[2, i])
    m.bb = pyo.Expression(m.set06,
                          rule=lambda m_, i: m_.b_const[0, i] + ((m_.mseg - 1) / m_.mseg) * m_.b_const[1, i] + (
                                  (m_.mseg - 1) / m_.mseg) * ((m_.mseg - 2) / m_.mseg) * m_.b_const[2, i])
    m.mes3 = pyo.Expression(expr=m.mseg * (m.sigma ** 3) * m.epskb / m.tmprt)
    m.m2e2s3 = pyo.Expression(expr=(m.mseg ** 2) * (m.sigma ** 3) * (m.epskb / m.tmprt) ** 2)

    # Ideal TODO: Only include a_ideal when calculating T derivatives.
    h = 6.626070150e-34  # J s
    me = 9.10938291e-31  # 1/Kg TODO: Would be better to use Mw/Na of substance.
    m.beta = pyo.Expression(expr=1. / (m.tmprt * KBOLTZMANN))
    m.broglie_vol = pyo.Expression(expr=h / pyo.sqrt(2 * PI * me * m.beta))
    return m


def create_pcsaft_phase_v1(m, p_name="unknown"):
    # creates phase where only eta (reduced density) is different.
    m.p = pyo.Block()  # p: phase
    m.p.eta = pyo.Var(name="Reduced density (-), phase: " + p_name, within=pyo.PositiveReals, bounds=(0, 0.7405))

    # EQUATIONS
    m.p.ndens = pyo.Expression(expr=m.p.eta / m.aux_xi[3])
    m.p.xi = pyo.Expression(m.set03, rule=lambda m_, n: m_.p.ndens * m_.aux_xi[n])
    # Monomer
    m.p.ahs = pyo.Expression(
        expr=((3 * m.p.xi[1] * m.p.xi[2]) / (1 - m.p.xi[3]) + m.p.xi[2] ** 3 / ((1 - m.p.xi[3]) ** 2 * m.p.xi[3]) +
              (-m.p.xi[0] + m.p.xi[2] ** 3 / m.p.xi[3] ** 2) * pyo.log(1 - m.p.xi[3])) / m.p.xi[0])
    m.p.dahs_deta = pyo.Expression(
        expr=(m.p.xi[2] ** 3 * (-3 + m.p.xi[3]) + 3 * m.p.xi[1] * m.p.xi[2] * (-1 + m.p.xi[3]) -
              m.p.xi[0] * (-1 + m.p.xi[3]) ** 2 * m.p.xi[3]) / (
                     m.p.xi[0] * (-1 + m.p.xi[3]) ** 3 * m.p.xi[3]))

    # Chain
    m.p.ghs = pyo.Expression(expr=(m.d ** 2 * m.p.xi[2] ** 2) / (2 * (1 - m.p.xi[3]) ** 3) + (3 * m.d * m.p.xi[2]) /
                                  (2 * (1 - m.p.xi[3]) ** 2) + 1 / (1 - m.p.xi[3]))
    m.p.dghs_deta = pyo.Expression(
        expr=(2 * (-1 + m.p.xi[3]) ** 2 * m.p.xi[3] + m.d ** 2 * m.p.xi[2] ** 2 * (2 + m.p.xi[3]) -
              3 * m.d * m.p.xi[2] * (-1 + m.p.xi[3] ** 2)) / (2 * (-1 + m.p.xi[3]) ** 4 * m.p.xi[3]))
    m.p.ahc = pyo.Expression(expr=m.p.ahs * m.mseg - (-1 + m.mseg) * pyo.log(m.p.ghs))
    m.p.dahc_deta = pyo.Expression(expr=-((m.p.dghs_deta * (-1 + m.mseg)) / m.p.ghs) + m.p.dahs_deta * m.mseg)

    # Dispersion
    m.p.c1 = pyo.Expression(expr=1 / (1 + (m.mseg * (8 * m.p.eta - 2 * m.p.eta ** 2)) / (1 - m.p.eta) ** 4 + (
            (1 - m.mseg) * (20 * m.p.eta - 27 * m.p.eta ** 2 + 12 * m.p.eta ** 3 - 2 * m.p.eta ** 4)) / (
                                              (1 - m.p.eta) ** 2 * (2 - m.p.eta) ** 2)))
    m.p.dc1_deta = pyo.Expression(expr=(-2 * (-2 + m.p.eta) * (-1 + m.p.eta) ** 3 * (
            (-1 + m.p.eta) ** 2 * (20 + m.p.eta * (-24 + m.p.eta * (6 + m.p.eta))) + m.mseg * (
            12 + (-1 + m.p.eta) * m.p.eta * (-96 + m.p.eta * (90 + (-25 + m.p.eta) * m.p.eta))))) / (4 + m.p.eta * (
            -(m.p.eta * (26 + m.p.eta * (-42 + m.p.eta * (27 + (-8 + m.p.eta) * m.p.eta)))) + m.mseg * (
            12 + m.p.eta * (27 + m.p.eta * (-70 + m.p.eta * (51 + 2 * (-8 + m.p.eta) * m.p.eta)))))) ** 2)
    m.p.eta_aux1 = pyo.Expression(m.set06, rule=lambda m_, i: m.p.eta ** i)
    m.p.I1 = pyo.Expression(expr=pyo.sum_product(m.aa, m.p.eta_aux1))
    m.p.I2 = pyo.Expression(expr=pyo.sum_product(m.bb, m.p.eta_aux1))
    m.p.eta_aux2 = pyo.Expression(m.set06, rule=lambda m_, i: i * m.p.eta ** (i - 1))
    m.p.dI1_deta = pyo.Expression(expr=pyo.sum_product(m.aa, m.p.eta_aux2))
    m.p.dI2_deta = pyo.Expression(expr=pyo.sum_product(m.bb, m.p.eta_aux2))
    m.p.adisp = pyo.Expression(expr=-m.p.xi[0] * (12 * m.p.I1 * m.mes3 + 6 * m.p.c1 * m.p.I2 * m.m2e2s3))
    m.p.dadisp_deta = pyo.Expression(expr=-(m.p.xi[3] * (
            6 * m.p.c1 * m.p.dI2_deta * m.m2e2s3 + 6 * m.p.dc1_deta * m.p.I2 * m.m2e2s3 + 12 * m.p.dI1_deta * m.mes3) + (
                                                    6 * m.p.c1 * m.p.I2 * m.m2e2s3 + 12 * m.p.I1 * m.mes3)) / m.d ** 3)
    # Total residual
    m.p.ares = pyo.Expression(expr=m.p.ahc + m.p.adisp)
    m.p.dares_drho = pyo.Expression(expr=m.deta_drho * (m.p.dahc_deta + m.p.dadisp_deta))

    # Ideal
    m.p.aideal = pyo.Expression(expr=pyo.log(m.ndens * m.broglie_vol ** 3) - 1)
    m.p.daideal_drho = pyo.Expression(expr=1. / m.ndens)

    # Total Helhmholtz
    m.p.atotal = pyo.Expression(expr=(m.p.ares + m.p.aideal) * (NAVOGADRO / m.beta))
    m.p.datotal_drho = pyo.Expression(expr=(m.p.dares_drho + m.p.daideal_drho) * (NAVOGADRO / m.beta))

    # Pressure
    m.p.press_calc = pyo.Expression(expr=m.ndens ** 2 * m.p.datotal_drho / NAVOGADRO)

    # Compressibility factor (use given pressure or calculated pressure?)
    m.p.compress_factor = pyo.Expression(expr=m.p.press_calc * m.beta / m.p.ndens)

    # Fugacity coefficient (natural logarithm)
    m.p.lnphi = pyo.Expression(expr=m.p.ares + (m.p.compress_factor - 1.) - pyo.log(m.p.compress_factor))

    # CONSTRAINT or OBJECTIVE
    m.p.pressure_constraint = pyo.Constraint(expr=(m.press - m.p.press_calc) ** 2 / 1e10 == 0)
    return None


def create_pcsaft_phase_v2(b, p_name):
    # creates phase where only eta (reduced density) is different.

    # Recover data from parent model
    set03 = b.model().set03
    set06 = b.model().set06
    mseg = b.model().mseg
    d = b.model().d
    aux_xi = b.model().aux_xi
    deta_drho = b.model().deta_drho
    aa = b.model().aa
    bb = b.model().bb
    mes3 = b.model().mes3
    m2e2s3 = b.model().m2e2s3
    beta = b.model().beta

    p = pyo.ConcreteModel(name="PCSAFT for phase " + p_name)
    p.eta = pyo.Var(name="Reduced density (-), phase: " + p_name, within=pyo.PositiveReals, bounds=(0, 0.7405))

    # EQUATIONS
    p.ndens = pyo.Expression(expr=p.eta / aux_xi[3])
    p.xi = pyo.Expression(set03, rule=lambda m_, n: p.ndens * aux_xi[n])
    # Monomer
    p.ahs = pyo.Expression(
        expr=((3 * p.xi[1] * p.xi[2]) / (1 - p.xi[3]) + p.xi[2] ** 3 / ((1 - p.xi[3]) ** 2 * p.xi[3]) +
              (-p.xi[0] + p.xi[2] ** 3 / p.xi[3] ** 2) * pyo.log(1 - p.xi[3])) / p.xi[0])
    p.dahs_deta = pyo.Expression(
        expr=(p.xi[2] ** 3 * (-3 + p.xi[3]) + 3 * p.xi[1] * p.xi[2] * (-1 + p.xi[3]) -
              p.xi[0] * (-1 + p.xi[3]) ** 2 * p.xi[3]) / (
                     p.xi[0] * (-1 + p.xi[3]) ** 3 * p.xi[3]))

    # Chain
    p.ghs = pyo.Expression(expr=(d ** 2 * p.xi[2] ** 2) / (2 * (1 - p.xi[3]) ** 3) + (3 * d * p.xi[2]) /
                                (2 * (1 - p.xi[3]) ** 2) + 1 / (1 - p.xi[3]))
    p.dghs_deta = pyo.Expression(
        expr=(2 * (-1 + p.xi[3]) ** 2 * p.xi[3] + d ** 2 * p.xi[2] ** 2 * (2 + p.xi[3]) -
              3 * d * p.xi[2] * (-1 + p.xi[3] ** 2)) / (2 * (-1 + p.xi[3]) ** 4 * p.xi[3]))
    p.ahc = pyo.Expression(expr=p.ahs * mseg - (-1 + mseg) * pyo.log(p.ghs))
    p.dahc_deta = pyo.Expression(expr=-((p.dghs_deta * (-1 + mseg)) / p.ghs) + p.dahs_deta * mseg)

    # Dispersion
    p.c1 = pyo.Expression(expr=1 / (1 + (mseg * (8 * p.eta - 2 * p.eta ** 2)) / (1 - p.eta) ** 4 + (
            (1 - mseg) * (20 * p.eta - 27 * p.eta ** 2 + 12 * p.eta ** 3 - 2 * p.eta ** 4)) / (
                                            (1 - p.eta) ** 2 * (2 - p.eta) ** 2)))
    p.dc1_deta = pyo.Expression(expr=(-2 * (-2 + p.eta) * (-1 + p.eta) ** 3 * (
            (-1 + p.eta) ** 2 * (20 + p.eta * (-24 + p.eta * (6 + p.eta))) + mseg * (
            12 + (-1 + p.eta) * p.eta * (-96 + p.eta * (90 + (-25 + p.eta) * p.eta))))) / (4 + p.eta * (
            -(p.eta * (26 + p.eta * (-42 + p.eta * (27 + (-8 + p.eta) * p.eta)))) + mseg * (
            12 + p.eta * (27 + p.eta * (-70 + p.eta * (51 + 2 * (-8 + p.eta) * p.eta)))))) ** 2)
    p.eta_aux1 = pyo.Expression(set06, rule=lambda m_, i: p.eta ** i)
    p.I1 = pyo.Expression(expr=pyo.sum_product(aa, p.eta_aux1))
    p.I2 = pyo.Expression(expr=pyo.sum_product(bb, p.eta_aux1))
    p.eta_aux2 = pyo.Expression(set06, rule=lambda m_, i: i * p.eta ** (i - 1))
    p.dI1_deta = pyo.Expression(expr=pyo.sum_product(aa, p.eta_aux2))
    p.dI2_deta = pyo.Expression(expr=pyo.sum_product(bb, p.eta_aux2))
    p.adisp = pyo.Expression(expr=-p.xi[0] * (12 * p.I1 * mes3 + 6 * p.c1 * p.I2 * m2e2s3))
    p.dadisp_deta = pyo.Expression(expr=-(
            p.xi[3] * (6 * p.c1 * p.dI2_deta * m2e2s3 + 6 * p.dc1_deta * p.I2 * m2e2s3 + 12 * p.dI1_deta * mes3) + (
            6 * p.c1 * p.I2 * m2e2s3 + 12 * p.I1 * mes3)) / d ** 3)
    # Total residual
    p.ares = pyo.Expression(expr=p.ahc + p.adisp)
    p.dares_drho = pyo.Expression(expr=deta_drho * (p.dahc_deta + p.dadisp_deta))

    # Ideal
    p.daideal_drho = pyo.Expression(expr=1. / p.ndens)

    # Total Helhmholtz
    p.datotal_drho = pyo.Expression(expr=(p.dares_drho + p.daideal_drho) * (NAVOGADRO / beta))

    # Pressure
    p.press_calc = pyo.Expression(expr=p.ndens ** 2 * p.datotal_drho / NAVOGADRO)

    # Compressibility factor (use given pressure or calculated pressure?)
    p.compress_factor = pyo.Expression(expr=p.press_calc * beta / p.ndens)

    # Fugacity coefficient (natural logarithm)
    p.lnphi = pyo.Expression(expr=p.ares + (p.compress_factor - 1.) - pyo.log(p.compress_factor))

    # CONSTRAINT or OBJECTIVE
    press = b.model().press
    p.pressure_constraint = pyo.Constraint(expr=(press - p.press_calc) ** 2 / 1e10 == 0)
    return p


model = create_pcsaft()
model.phases_names = pyo.Set(initialize=["Liquid", "Vapor"])
model.phases = pyo.Block(model.phases_names, rule=create_pcsaft_phase_v2)

# CONSTRAINTS
# model.pressure_constraint = pyo.Constraint(expr=(model.press - model.press_calc)**2/1e10, sense=pyo.minimize)
# OBJECTIVE
model.obj = pyo.Objective(expr=(model.phases[model.phases_names.first()].lnphi -
                                model.phases[model.phases_names.last()].lnphi) ** 2, sense=pyo.minimize)

model.tmprt.fix(298.15)
# Initial conditions:
model.phases["Liquid"].eta = 0.5
model.phases["Vapor"].eta = 1e-10

# SOLVE
options = {"linear_solver": "mumps", "halt_on_ampl_error": "no"}
# options = {"max_iter": 0}  # for testing
pyo.SolverFactory('ipopt').solve(model, tee=True, options=options)

print("eta1 [-]   : ", pyo.value(model.phases["Liquid"].eta))
print("rho1 [mol/m3]: ", pyo.value(model.phases["Liquid"].ndens) / NAVOGADRO)
print("eta2 [-]   : ", pyo.value(model.phases["Vapor"].eta))
print("rho2 [mol/m3]: ", pyo.value(model.phases["Vapor"].ndens) / NAVOGADRO)
print("P    [Pa]  : ", pyo.value(model.press))
