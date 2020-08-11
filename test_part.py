import numpy as np
from scipy.io import FortranFile
from scipy.stats import gmean
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

    def get_mu(self,st):  #vperp^2/B
        mp = 1
        B = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
        vx = self.vx - self.ux
        vy = self.vy - self.uy
        vz = self.vz - self.uz
        v2 = vx**2 + vy**2 + vz**2
        bx = self.Bx/B
        by = self.By/B
        bz = self.Bz/B
        vpar = vx*bx + vy*by + vz*bz
        vperp2 = v2 - vpar**2
        self.mu0 = gmean(0.5*mp*vperp2[st,:]/B[st,:])
        self.mu = (0.5*mp*vperp2/B)/self.mu0

    def get_alpha(self,st):  #pitch angle
        mp = 1
        B = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
        vx = self.vx - self.ux
        vy = self.vy - self.uy
        vz = self.vz - self.uz
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        bx = self.Bx/B
        by = self.By/B
        bz = self.Bz/B
        vx = vx/v
        vy = vy/v
        vz = vz/v
        self.alpha = np.arccos(vx*bx + vy*by + vz*bz)
        have, bins = np.histogram(self.alpha[self.Nstep-1,:],bins=20)
        for i in range(1,2000):
            h1, bins = np.histogram(self.alpha[self.Nstep-1-i,:],bins=20)
            have = have + h1
        h1 = have/2001    
        h0, bins = np.histogram(self.alpha[1,:],bins=20)
        bins = bins*(180/np.pi)
        plt.plot(bins[0:-1],h1-h0)
        hhat = savgol_filter(h1-h0,5,3)
        plt.plot(bins[0:-1],hhat)
        plt.xlabel('$\\alpha$')
        plt.ylabel('h1-h0')
        plt.show()
        
    def plot_hist(self):
        a = h.mu[h.Nstep-1,:]
        plt.hist(a,bins='auto')
        plt.title(str(gmean(a)))
        plt.xlim(0.0,4.0)
        #plt.xscale('log')
        plt.show()
        
    def plot_correlation(self,y,st,rl):
        dmu = self.mu - np.roll(self.mu,-rl,axis=0)
        dy = y - np.roll(y,-rl,axis=0)
        print(dmu[:,:].max(),self.mu[:,:])
        #for i in range(20,900):
        plt.plot(dmu[st:-2-rl,:],dy[st:-2-rl,:],'.',markersize=2)
        plt.xlabel('$\delta \mu$')
        plt.ylabel('$\delta E_\perp$')
        plt.show()

    def get_perp_par(self,x,y,z):
        B = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
        bx = self.Bx/B
        by = self.By/B
        bz = self.Bz/B
        vecpar = x*bx + y*by + z*bz
        vecperp = np.sqrt((x**2 + y**2 + z**2) - vecpar**2)
        return vecperp,vecpar

    def plot_phase_space(self):
        vperp, vpar = self.get_perp_par(self.vx-self.ux,self.vy-self.uy,self.vz-self.uz)
        hvperp0, bins = np.histogram(vperp[0,:],bins=20)
        hvperp1, bins = np.histogram(vperp[self.Nstep-1,:],bins=20)
        vth = np.std(bins)
        vbar = gmean(bins)
        v = np.linspace(0,np.max(bins),100)
        fv = np.max(hvperp0)*np.exp(-(v-vbar)**2/(vth/2)**2)
        plt.plot(v,fv)
        plt.hist(vperp[self.Nstep-1,:],bins=20)
        plt.xlabel('$v_\perp$')
        plt.ylabel('f(v)')
        plt.show()
        plt.plot(bins[0:-1],hvperp0-hvperp1)
        plt.xlabel('$v_\perp$')
        plt.ylabel('h1-h0')
        plt.show()
        hvpar0, bins = np.histogram(vpar[0,:],bins=20)
        hvpar1, bins = np.histogram(vpar[self.Nstep-1,:],bins=20)
        vth = np.std(bins)
        vbar = np.mean(bins)
        v = np.linspace(-np.max(bins),np.max(bins),100)
        fv = np.max(hvpar0)*np.exp(-(v-vbar)**2/(vth/2)**2)
        plt.plot(v,fv)
        plt.hist(vpar[self.Nstep-1,:],bins=20)
        plt.xlabel('$v_\parallel$')
        plt.ylabel('f(v)')
        plt.show()
        plt.plot(bins[0:-1],hvpar0-hvpar1)
        plt.xlabel('$v_\parallel$')
        plt.ylabel('h1-h0')
        plt.show()
        
        plt.plot(h.y[st:,0:100],h.vy[st:,0:100],'.',markersize=1.5)
        plt.show()
    
#---------------------------------------------------------        
dir = '/data/KH3D/run_test/'
Npart = 1000
Nstep = 100
st = 0    #starting time for mu normalization
h = test_part(dir,Npart,Nstep)
h.read_part()
h.get_mu(st)

h.plot_hist()
#vperp,vpar = h.get_perp_par(h.Ex,h.Ey,h.Ez)
#vperp = np.sqrt(h.Bx**2 + h.Bz*2)
#h.plot_correlation(vperp,st,10)
#h.plot_phase_space()
#h.get_alpha(st)
