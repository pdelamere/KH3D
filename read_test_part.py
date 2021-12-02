import numpy as np
from scipy.io import FortranFile
from scipy.stats import gmean
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from hybrid_read import Hybrid_read
import math
import pickle
#from statsmodels.graphics import tsaplots
import statsmodels.api as sm

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
        self.w = int(2*np.pi*10)  #gyroperiod
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
        self.mu0 = (0.5*mp*vperp2[st,:]/B[st,:])
        self.mu = (0.5*mp*vperp2/B)/self.mu0
        return vpar, vperp2

    def get_mu_bar(self,st):  #vperp^2/B
        w2 = int(self.w/2)
        mp = 1
        vxbar = self.get_gyro_average(self.vx)
        vybar = self.get_gyro_average(self.vy)
        vzbar = self.get_gyro_average(self.vz)
        Bxbar = self.get_gyro_average(self.Bx)
        Bybar = self.get_gyro_average(self.By)
        Bzbar = self.get_gyro_average(self.Bz)
#        vxbar = np.zeros([self.Nstep,self.Npart])
#        vybar = np.zeros([self.Nstep,self.Npart])
#        vzbar = np.zeros([self.Nstep,self.Npart])
#        Bxbar = np.zeros([self.Nstep,self.Npart])
#        Bybar = np.zeros([self.Nstep,self.Npart])
#        Bzbar = np.zeros([self.Nstep,self.Npart])
#        for i in range(len(self.Bx[0])):
#            vxbar[:,i] = np.convolve(self.vx[w2:-w2+1,i],np.ones(self.w))/self.w
#            vybar[:,i] = np.convolve(self.vy[w2:-w2+1,i],np.ones(self.w))/self.w
#            vzbar[:,i] = np.convolve(self.vz[w2:-w2+1,i],np.ones(self.w))/self.w
#            Bxbar[:,i] = np.convolve(self.Bx[w2:-w2+1,i],np.ones(self.w))/self.w
#            Bybar[:,i] = np.convolve(self.By[w2:-w2+1,i],np.ones(self.w))/self.w
#            Bzbar[:,i] = np.convolve(self.Bz[w2:-w2+1,i],np.ones(self.w))/self.w
        B = np.sqrt(Bxbar**2 + Bybar**2 + Bzbar**2)        
        #vx = self.vx - self.ux
        #vy = self.vy - self.uy
        #vz = self.vz - self.uz
        v2 = vxbar**2 + vybar**2 + vzbar**2
        bx = Bxbar/B
        by = Bybar/B
        bz = Bzbar/B
        vpar = vxbar*bx + vybar*by + vzbar*bz
        vperp2 = v2 - vpar**2
        self.mu0 = (0.5*mp*vperp2[st,:]/B[st,:])
        self.mu = (0.5*mp*vperp2/B)/self.mu0
        plt.plot(self.mu[self.w:-self.w+1,:])
        plt.show()
        return vpar, vperp2
        
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
        wh = np.logical_and((self.z[Nstep-1,:] > h.z[10]), (self.z[Nstep-1,:] < h.z[int(p.nz/2)-10]))
        a = self.mu[-self.w+1,wh]
        plt.hist(a,bins='auto')
        plt.title(str(np.mean(a)))
        plt.xlabel('$\mu$')
        plt.xlim(0.0,10.0)
        #plt.xscale('log')
        plt.show()

    def get_temp(self):
        #get temperature
        #plt.figure()
        for i in range(Nstep):            
            v2 = (self.vx[i,:]**2 + self.vy[i,:]**2 + self.vz[i,:]**2).sum()/self.Npart
            u = (self.vx[i,:] + self.vy[i,:] + self.vz[i,:]).sum()/self.Npart
            self.temp[i] = 0.5*1.67e-27*(v2 - u**2)*1e6/1.6e-19
        plt.figure()
        plt.plot(self.temp)
        plt.show()
        
    def plot_dist(self,vperp,vpar):
        plt.figure(1)
        plt.plot(vpar[Nstep-1,:],vperp[Nstep-1,:],'.',markersize=1.0)
        plt.xlabel('vperp')
        plt.ylabel('vpar')
        plt.show()
        plt.figure(2)
        H, xedges, yedges = np.histogram2d(vpar[Nstep-1,:],vperp[Nstep-1,:],bins=20)
        H = H.T
        plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect= 1)

        #get temperature
        v2 = (self.vx[Nstep-1,:]**2 + self.vy[Nstep-1,:]**2 + self.vz[Nstep-1,:]**2).sum()/self.Npart
        u = (self.vx[Nstep-1,:] + self.vy[Nstep-1,:] + self.vz[Nstep-1,:]).sum()/self.Npart
        aveE = 0.5*1.67e-27*(v2 - u**2)*1e6/1.6e-19
        print('aveE...',aveE)
        plt.title(str(aveE)+' eV')
        plt.xlabel('vperp')
        plt.ylabel('vpar')
        plt.show()
        
    def plot_correlation(self,st,rl,Nstep):
        Bperp, Bpar = self.get_delta_B_perp()
        vperp,vpar = self.get_perp_par(h.vx,h.vy,h.vz)
        vdotE = self.get_v_dot_E()
        va= 5e-9/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
        """
        Bmax = []
        mu_std = []
        dmu = []
        for i in range(len(Bperp[0])):
            Bmax.append(Bperp[st:Nstep,i].max())
            mu_std.append(abs(self.mu[st,i] - self.mu[Nstep-1,i])/self.mu[Nstep-1,i])
            dmu.append((abs(self.mu[st,i] - self.mu[st:Nstep,i])/self.mu[st:Nstep,i]).max())
        plt.figure()
        plt.plot(dmu,Bmax,'.')
        plt.show()
        """
        
        #plt.plot(self.mu[:,1:10]/self.mu[0,1:10])
        #plt.plot(y[:,1:10])
        #print(y[:,1])
        wh = np.logical_and((self.z[Nstep-1,:] > h.z[50]), (self.z[Nstep-1,:] < h.z[int(p.nz-50)]))
        dmu = (self.mu[1:,:]-self.mu[0:-1,:])/self.mu[0:-1,:]#/np.roll(self.mu,-rl,axis=0)
        dvdotE = (vdotE[1:,:]-vdotE[0:-1,:])/vdotE[0:-1,:]#/np.roll(self.mu,-rl,axis=0)
        #dmu = (self.mu[0:-1,:])/self.mu[0,:]#/np.roll(self.mu,-rl,axis=0)
        dBperp2 = ((Bperp[1:,:]-Bperp[0:-1,:])/Bperp[0:-1,:])**2
        #dvperp = (vperp[1:,:]-vperp[0:-1,:])/vperp[0:-1,:]#/np.roll(vperp,-rl,axis=0)
        Btot = np.sqrt(h.Bx**2 + h.By**2 + h.Bz**2)
        dB = (Btot[1:,:]-Btot[0:-1,:])/Btot[0:-1,:]
        #for i in range(20,900):
        #plt.plot(dmu[st:-2-rl,:],dy[st:-2-rl,:],'.',markersize=2)
        plt.plot(dmu[st:,wh],vpar[st:-1,wh]/va,'.',markersize=2)
        plt.xlabel('$\delta \mu$')
        plt.ylabel('$\delta B_\perp^2$')
        #plt.xlim(-1.0,10.0)
        #plt.ylim(-1.0,10.0)
        plt.show()
                
    def get_perp_par(self,x,y,z):
        B = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
        bx = self.Bx/B
        by = self.By/B
        bz = self.Bz/B
        vecpar = x*bx + y*by + z*bz
        vecperp = np.sqrt((x**2 + y**2 + z**2) - vecpar**2)
        return vecperp,vecpar

    def get_gyro_average(self,x):
        w2 = int(self.w/2)
        xbar = np.zeros([self.Nstep,self.Npart])
        for i in range(len(self.Bx[0])):
            xbar[:,i] = np.convolve(x[w2:-w2+1,i],np.ones(self.w))/self.w
        return xbar    
    
    def get_delta_B_perp(self):
        plt.figure()
        w = self.w
        w2 = int(self.w/2)
        B = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
#        Bxbar = np.zeros([self.Nstep,self.Npart])
#        Bybar = np.zeros([self.Nstep,self.Npart])
#        Bzbar = np.zeros([self.Nstep,self.Npart])
        Bpar = np.zeros([self.Nstep,self.Npart])
        Bperp = np.zeros([self.Nstep,self.Npart])
        Bxbar = self.get_gyro_average(self.Bx)
        Bybar = self.get_gyro_average(self.By)
        Bzbar = self.get_gyro_average(self.Bz)        
#        for i in range(len(B[0])):
#            Bxbar[:,i] = np.convolve(self.Bx[w2:-w2+1,i],np.ones(w))/w
#            Bybar[:,i] = np.convolve(self.By[w2:-w2+1,i],np.ones(w))/w
#            Bzbar[:,i] = np.convolve(self.Bz[w2:-w2+1,i],np.ones(w))/w
        Bbar = np.sqrt(Bxbar**2 + Bybar**2 + Bzbar**2)
        Bxbar = Bxbar/Bbar   #direction unit vectors
        Bybar = Bybar/Bbar
        Bzbar = Bzbar/Bbar
        print(Bxbar**2 + Bybar**2 + Bzbar**2)
        Bpar = self.By #Bxbar*self.Bx + Bybar*self.By + Bzbar*self.Bz
        Bperp = np.sqrt(self.Bx**2 + self.Bz**2) #np.sqrt(B*B - Bpar*Bpar)
        plt.plot(B[w:-w+1,6],'-')
        plt.plot(Bbar[w:-w+1,6],'.')
        plt.plot(Bpar[w:-w+1,6],'.')
        #plt.plot(Bbar**2 - Bpar**2)
        plt.plot(Bperp[w:-w+1,6],'.')
        plt.show()
        return Bperp, Bpar

    def get_v_dot_E(self):
        E = np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)
        ex = self.Ex/E
        ey = self.Ey/E
        ez = self.Ez/E
        vdotE = self.vx*ex + self.vy*ey + self.vz*ez
        return vdotE

    def get_E_dot_J(self):
        E = np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)
        ex = self.Ex/E
        ey = self.Ey/E
        ez = self.Ez/E
        J = np.sqrt(self.Jx**2 + self.Jy**2 + self.Jz**2)
        jx = self.Jx/J
        jy = self.Jy/J
        jz = self.Jz/J
        EdotJ = jx*ex + jy*ey + jz*ez
        return EdotJ
#--------------------------------------------------------------

dir = '/data/KH3D/run_heating/'
Npart = 1000
Nstep = 2100
st = 1500    #starting time for mu normalization
p = Hybrid_read(dir)
p.read_para()
h = test_part(dir,Npart,Nstep)
h.read_part()
    
