import numpy as np
from scipy.io import FortranFile
from hybrid_read import Hybrid_read

class test_part:

    def __init__(self,dir,Npart,Nstep):
        self.dir = dir
        self.Npart = Npart
        self.Nstep = Nstep
        self.x = np.zeros([self.Nstep,self.Npart])
        self.y = np.zeros([self.Nstep,self.Npart])
        self.z = np.zeros([self.Nstep,self.Npart])
        self.vx = np.zeros([self.Nstep,self.Npart])
        self.vy = np.zeros([self.Nstep,self.Npart])
        self.vz = np.zeros([self.Nstep,self.Npart])
        self.Bx = np.zeros([self.Nstep,self.Npart])
        self.By = np.zeros([self.Nstep,self.Npart])
        self.Bz = np.zeros([self.Nstep,self.Npart])
        self.Ex = np.zeros([self.Nstep,self.Npart])
        self.Ey = np.zeros([self.Nstep,self.Npart])
        self.Ez = np.zeros([self.Nstep,self.Npart])
        self.ux = np.zeros([self.Nstep,self.Npart])
        self.uy = np.zeros([self.Nstep,self.Npart])
        self.uz = np.zeros([self.Nstep,self.Npart])
        self.Jx = np.zeros([self.Nstep,self.Npart])
        self.Jy = np.zeros([self.Nstep,self.Npart])
        self.Jz = np.zeros([self.Nstep,self.Npart])
        self.gradPx = np.zeros([self.Nstep,self.Npart])
        self.gradPy = np.zeros([self.Nstep,self.Npart])
        self.gradPz = np.zeros([self.Nstep,self.Npart])
        self.mu = np.zeros([self.Nstep,self.Npart])
        self.mu0 = np.zeros(self.Npart)
        self.alpha = np.zeros([self.Nstep,self.Npart])
        self.temp = np.zeros(self.Nstep)
        self.w = int(10)  #gyroperiod 1/omega
        self.dt = 2*np.pi/0.1
        
    def read_part(self):
        file = self.dir+'c.test_part'+'.dat'
        f = FortranFile(file,'r')
        for i in range(self.Nstep):
            for j in range(self.Npart):
                x = f.read_reals('f4')
#                print(i,j)
                self.x[i,j] = x[0]
                self.y[i,j] = x[1]
                self.z[i,j] = x[2]
                self.vx[i,j] = x[3]
                self.vy[i,j] = x[4]
                self.vz[i,j] = x[5]
                self.Ex[i,j] = x[6]
                self.Ey[i,j] = x[7]
                self.Ez[i,j] = x[8]
                self.Bx[i,j] = x[9]
                self.By[i,j] = x[10]
                self.Bz[i,j] = x[11]
                self.ux[i,j] = x[12]
                self.uy[i,j] = x[13]
                self.uz[i,j] = x[14]
                self.Jx[i,j] = x[15]
                self.Jy[i,j] = x[16]
                self.Jz[i,j] = x[17]
                self.gradPx[i,j] = x[18]
                self.gradPy[i,j] = x[19]
                self.gradPz[i,j] = x[20]
                
        f.close()
        return

#---------------------------------------------------------------------------
dir = '/data/KH3D/run_heating_beta_3/'
Npart = 1000
Nstep = 3000
#st = 1000    #starting time for mu normalization
p = Hybrid_read(dir)
p.read_para()
h = test_part(dir,Npart,Nstep)
h.read_part()
