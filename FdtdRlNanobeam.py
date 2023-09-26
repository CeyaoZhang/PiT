""" this code establishes the real-time port b/t FDTD and Python to enable state and reward passes
    for optimization of photonic crystals by RL. NOEL, Renjie Li, March 2023
"""

import numpy as np
import random
#import gym
import sys
import os
# sys.path.append("C:\\Program Files\\Lumerical\\v202\\api\\python\\")   # Default windows lumapi path
# sys.path.append(os.path.dirname(__file__))   # Current directory
# os.add_dll_directory('C:\\Program Files\\Lumerical\\v202\\api\\python\\')
# import lumapi as lp
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d

class FdtdRlNanobeam():

    def __init__(self):

        self.circles = 9
        self.rectangles = 4
        self.leng = 0.52
        self.a = 400E-9

    def addgeometry(self, l3):
        """ This function constructs the PCSEL geometry by running a setup script in FDTD
        """
        l3.switchtolayout()
        l3.unselectall()
        l3.addstructuregroup()
        l3.set("name", "pcsel")
        l3.set('x', 0)
        l3.set('y', 0)
        l3.set('z', 0)
        l3.adduserprop("t", 2, 450E-9)
        l3.adduserprop("len", 2, 2000E-9)
        l3.adduserprop("t_2", 2, 0)
        l3.adduserprop("t_3", 2, 315E-9)
        l3.adduserprop("t_4", 2, 5000E-9)
        l3.adduserprop("n_1", 0, 3.2035)
        l3.adduserprop("n_2", 0, 3.4038)
        l3.adduserprop("n_3", 0, 3.415)
        l3.adduserprop("n_4", 0, 3.2035)
        l3.adduserprop("index", 0, 1)
        l3.adduserprop("material_air", 5, "etch")
        l3.adduserprop("material", 5, "<Object defined dielectric>")
        l3.adduserprop("layer", 0, 1)
        l3.adduserprop("leng", 0, 0.52)
        l3.adduserprop("leng2", 0, 0.406)
        l3.adduserprop("a", 2, 400E-9)
        l3.adduserprop("t_1", 2, 100E-9)
        l3.set("construction group", 0)

        t = 450E-9
        len = 2000E-9
        t_2 = 0
        t_3 = 315E-9
        t_4 = 5000E-9
        n_1 = 3.2035
        n_2 = 3.4038 #GaAs (from https://refractiveindex.info)
        n_3 = 3.415   #active layer
        n_4 = 3.2035
        material_air = "etch"
        material = "<Object defined dielectric>"
        layer = 1
        leng = 0.52
        leng2 = 0.406 #for triangular holes
        a = 400E-9
        t_1 = 100E-9
        index = 1


        l3.addrect()
        l3.addtogroup("::model::pcsel")
        l3.set("x",0)
        l3.set("y",0)
        l3.set("z",t_1/2+t+t_2+t_3/2)
        l3.set("x span", len)
        l3.set("y span", len)
        l3.set("z span", t_1)
        l3.set("material", material)
        l3.set("index", n_2)


        l3.addrect()
        l3.addtogroup("::model::pcsel")
        l3.set("x",0)
        l3.set("y",0)
        l3.set("z", t/2+t_2+t_3/2)
        l3.set("x span", len)
        l3.set("y span", len)
        l3.set("z span", t)
        l3.set("material", material)
        l3.set("index", n_1)
        l3.set("alpha", 0.7)


        l3.addrect()
        l3.addtogroup("::model::pcsel")
        l3.set("x",0)
        l3.set("y",0)
        l3.set("z", 0)
        l3.set("x span", len)
        l3.set("y span", len)
        l3.set("z span", t_3)
        l3.set("material", material)
        l3.set("index", n_3)
        l3.set("alpha", 0.5)

        l3.addrect()
        l3.addtogroup("::model::pcsel")
        l3.set("x",0)
        l3.set("y",0)
        l3.set("z",-t_3/2-t_4/2)
        l3.set("x span", len)
        l3.set("y span", len)
        l3.set("z span", t_4)
        l3.set("material", material)
        l3.set("index", n_4)
        l3.set("alpha", 0.3)

  # ---------------------if circular holes---------------------------
        for i in range(-layer, layer+1):   
            for j in range(-layer, layer+1):
                x_c=j*a
                y_c=i*a
                
                l3.addcircle()
                l3.addtogroup("::model::pcsel")
                l3.set('radius', leng*a/2)
                l3.set('x',x_c)
                l3.set('y',y_c)
                l3.set("z", t_1/2+t/2+t_2+t_3/2)
                l3.set("z span", t_1+t)
                l3.set("material", material_air)
                #l3.set("index", index)
            

  # ---------------------if triangular holes---------------------------
        # for i in range(-layer, layer+1):   
        #     for j in range(-layer, layer+1):
        #         x_c=j*a
        #         y_c=i*a

        #         V=[[x_c-leng2*a/2, y_c+leng2*a/2],
        #             [x_c+leng2*a/2, y_c+leng2*a/2],
        #             [x_c+leng2*a/2, y_c-leng2*a/2]]
        #         l3.addpoly()
        #         l3.addtogroup("::model::pcsel")
        #         l3.set("vertices", V)
        #         l3.set("z", t/2+t_2+t_3/2)
        #         l3.set("z span", t)
        #         l3.set("material", material_air)

        l3.selectall()
        #l3.set("z", 0)
        # if l3.get("material") == "<Object defined dielectric>":
            # l3.set("index", index)
        l3.runsetup()

    def index_to_xdata(self, xdata, indices):
        "interpolate the values from signal.peak_widths to xdata"
        #ind = np.arange(len(xdata))
        ind = np.arange(200)
        f = interp1d(ind,xdata)
        return f(indices)

    def adjustdesignparams(self, dlen, dt, dt1, dt3, dn1, dn3, dleng, da):
        """ This function makes is convenient to reconstruct the simulation;
                while changing the design parameters, a brand new FDTD session will start
                and close within this function. Symmetry of the geometry is taken into account.
        """
        #netDLen, netDT, netDT1, netDT3, netDN1, netDN3, netDLeng, netDA
        
        # print("starting new FDTD session... ")

        # with lp.FDTD() as l3:
        #     #l3.load("C:/Users/Administrator/OneDrive - CUHK-Shenzhen/Desktop/Renjie/nanobeam/short_InP/Nanobeam-Short-InP_Q83637.fsp")  # for QW case
        #     l3.load("PCSEL-1310-GaAs100.fsp")    # for PCSEL

        #     # create the structure setup script
        #     self.addgeometry(l3)

        #     # rectangular layers            
        #     for i in range(1, self.rectangles+1):
        #         if i == 1:
        #             t1 = l3.getnamed("::model::pcsel::rectangle", "z span", i)
        #             l3.setnamed("::model::pcsel::rectangle", "z span", float(t1 + dt1), i)
        #         if i == 2:
        #             n = l3.getnamed("::model::pcsel::rectangle", "index", i)
        #             l3.setnamed("::model::pcsel::rectangle", "index", float(float(n) + dn1), i)
        #             t = l3.getnamed("::model::pcsel::rectangle", "z span", i)
        #             l3.setnamed("::model::pcsel::rectangle", "z span", float(t + dt), i)
        #         elif i == 3:
        #             n = l3.getnamed("::model::pcsel::rectangle", "index", i)
        #             l3.setnamed("::model::pcsel::rectangle", "index", float(float(n) + dn3), i)
        #             t3 = l3.getnamed("::model::pcsel::rectangle", "z span", i)
        #             l3.setnamed("::model::pcsel::rectangle", "z span", float(t3 + dt3), i)
        #         elif i == 4:
        #             n4 = l3.getnamed("::model::pcsel::rectangle", "index", i)
        #             l3.setnamed("::model::pcsel::rectangle", "index", float(float(n4) + dn1), i)

        #         x_len = l3.getnamed("::model::pcsel::rectangle", "x span", i)
        #         l3.setnamed("::model::pcsel::rectangle", "x span", float(x_len + dlen), i)
        #         y_len = l3.getnamed("::model::pcsel::rectangle", "y span", i)
        #         l3.setnamed("::model::pcsel::rectangle", "y span", float(y_len + dlen), i)
                
        #     #circles
        #     for i in range(1, self.circles+1):
        #         #rad = l3.getnamed("::model::pcsel::circle", "radius", i)
        #         l3.setnamed("::model::pcsel::circle", "radius", float((self.leng+dleng)*(self.a+da)/2.0), i)
                
                
        #     l3.run()

        #     #print(pxNew, radNew, pxNew1, radNew1)

        #     l3.runanalysis()

        #     #Qraw1 = l3.getresult("::model::Q::Qanalysis 3", "Q")
        #     #Qmax1 = max(Qraw1['Q'])

        #     Qraw5 = l3.getresult("::model::Q::Qanalysis 5", "Q")
        #     Qmax5 = max(Qraw5['Q'])

        #     #lam = l3.getresult("::model::Q::Qanalysis", "spectrum.lambda")
        #     lam = l3.getresult("::model::Q::Qanalysis 5", "Q.peak_lam")
        #     res_wavelength = np.mean(lam['peak_lam'])   #resonance wavelength in nm

        #     powerArray = l3.getresult("::model::Top", "power")
        #     power = max(np.real(powerArray))
        #     dipole_power = 3.98265e-14      #dipole source power
        #     power = power[0]/dipole_power  #output power/injecting power

        #     areaArray = l3.getresult("::model::mode_area", "A")
        #     area  = max(areaArray)[0]  #in terms of m^2

        #     E2 = l3.getresult("::model::ffp", "farfield.E2")   #get farfield data
        #     #print(np.shape(E2['E2']))
        #     #print('E2 is type', type(E2),' with keys', str(E2.keys()) )
        #     uy = E2['uy']
        #     uy = np.reshape(uy, 200)
        #     #print(np.shape(uy))
        #     #print(uy)
        #     E = np.squeeze(E2['E2'])
        #     E = E[100,0:199]
        #     Emax = np.max(E)
        #     #print(E[100])
        #     peaks, _ = find_peaks(E, height = Emax/2)
        #     widths, width_heights, left_ips, right_ips = peak_widths(E, peaks, rel_height=0.5)
        #     #fwhm = self.index_to_xdata(uy, widths)/2
        #     left_ips = self.index_to_xdata(uy, left_ips)
        #     right_ips = self.index_to_xdata(uy, right_ips)
        #     #print(fwhm, width_heights, left_ips, right_ips)
        #     true_fwhm = (right_ips[0] - left_ips[0])/2
        #     div_angle = np.arcsin(true_fwhm)*2*180/np.pi #divergence angle in deg

        #     l3.switchtolayout()
        #     l3.select("::model::pcsel")
        #     l3.delete()
        #     l3.save()
        #todo analsys lumerical need money
        Qmax5, res_wavelength, power, area, div_angle=[0,0,0,0,0]
        return Qmax5, res_wavelength, power, area, div_angle

