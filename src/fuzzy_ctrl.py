import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

deg=ctrl.Antecedent(np.arange(0,361,1),"degree")
dist=ctrl.Antecedent(np.arange(0,1.025,0.025),"dist")

lin=ctrl.Consequent(np.arange(-1,1.1,0.1),"linear")
ang=ctrl.Consequent(np.arange(-1,1.1,0.1),"angular")

deg["r0"]=fuzz.trapmf(deg.universe,[1,1,89,90])
deg["r1"]=fuzz.trapmf(deg.universe,[90,90,179,180])
deg["r2"]=fuzz.trapmf(deg.universe,[180,180,269,270])
deg["r3"]=fuzz.trapmf(deg.universe,[270,270,359,360])

dist["near"]=fuzz.trapmf(dist.universe,[0,0,0.075,0.1])
dist["medium"]=fuzz.trapmf(dist.universe,[0.1,0.1025,0.5,0.5])
dist["far"]=fuzz.trapmf(dist.universe,[0.5,0.5025,1,1])

lin["slow"]=fuzz.trapmf(lin.universe,[-1,-1,-0.7,-0.6])
lin["moderate"]=fuzz.trimf(lin.universe,[-0.6,0,0.6])
lin["fast"]=fuzz.trapmf(lin.universe,[0.6,0.7,1,1])

ang["right"]=fuzz.trapmf(ang.universe,[-1,-1,-0.6,-0.5])
ang["straight"]=fuzz.trimf(ang.universe,[-0.5,0,0.5])
ang["left"]=fuzz.trapmf(ang.universe,[0.5,0.6,1,1])

r00=ctrl.Rule(deg["r0"]&dist["near"],lin["slow"])
r01=ctrl.Rule(deg["r0"]&dist["near"],ang["right"])
r10=ctrl.Rule(deg["r0"]&dist["medium"],lin["moderate"])
r11=ctrl.Rule(deg["r0"]&dist["medium"],ang["right"])
r20=ctrl.Rule(deg["r0"]&dist["far"],lin["fast"])
r21=ctrl.Rule(deg["r0"]&dist["far"],ang["right"])

r30=ctrl.Rule(deg["r1"]&dist["near"],lin["fast"])
r31=ctrl.Rule(deg["r1"]&dist["near"],ang["straight"])
r40=ctrl.Rule(deg["r1"]&dist["medium"],lin["fast"])
r41=ctrl.Rule(deg["r1"]&dist["medium"],ang["straight"])
r50=ctrl.Rule(deg["r1"]&dist["far"],lin["fast"])
r51=ctrl.Rule(deg["r1"]&dist["far"],ang["straight"])

r60=ctrl.Rule(deg["r2"]&dist["near"],lin["fast"])
r61=ctrl.Rule(deg["r2"]&dist["near"],ang["straight"])
r70=ctrl.Rule(deg["r2"]&dist["medium"],lin["fast"])
r71=ctrl.Rule(deg["r2"]&dist["medium"],ang["straight"])
r80=ctrl.Rule(deg["r2"]&dist["far"],lin["fast"])
r81=ctrl.Rule(deg["r2"]&dist["far"],ang["straight"])

r90=ctrl.Rule(deg["r3"]&dist["near"],lin["slow"])
r91=ctrl.Rule(deg["r3"]&dist["near"],ang["left"])
r100=ctrl.Rule(deg["r3"]&dist["medium"],lin["moderate"])
r101=ctrl.Rule(deg["r3"]&dist["medium"],ang["left"])
r110=ctrl.Rule(deg["r3"]&dist["far"],lin["fast"])
r111=ctrl.Rule(deg["r3"]&dist["far"],ang["left"])

obs_rule=ctrl.ControlSystem([r00,r01,r10,r11,r20,r21,r30,r31,
                             r40,r41,r50,r51,r60,r61,r70,r71,
                             r80,r81,r90,r91,r100,r101,r110,
                             r111])

obs_ctrl=ctrl.ControlSystemSimulation(obs_rule)

obs_ctrl.input["degree"]=340
obs_ctrl.input["dist"]=0.03

obs_ctrl.compute()
print([obs_ctrl.output["linear"],obs_ctrl.output["angular"]])
