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


def get_cff(h):
    for i in range(1,10):
        arr = sm.tsa.stattools.ccf(h.mu[t0:t1,i]/h.mu[st,i], vdotEbar[t0:t1,i])
        wh = np.where(arr[0:30] == arr[0:30].max())
        plt.plot(h.mu[t0:t1,i]/h.mu[st,i], vdotEbar[t0-wh[0]:t1-wh[0],i])
        plt.show()
    
def plot_phase_space(h):
    B = np.sqrt(h.Bx**2 + h.By**2 + h.Bz**2)
    wh = np.logical_and((h.z[Nstep-1,:] > h.z[20]), (h.z[Nstep-1,:] < h.z[int(p.nz-20)]))
    plt.figure(1)
    vperp, vpar = h.get_perp_par(h.vx-h.ux,h.vy-h.uy,h.vz-h.uz)
    hvperp0, bins = np.histogram(vperp[0,wh],bins=20)
    bins=bins[:-1]
    dbin = bins[1]-bins[0]
    vth0 = np.sum(bins*hvperp0*dbin)/np.sum(hvperp0*dbin)
    print('vth0...',vth0)
    v0 = np.linspace(0,np.max(bins),100)
    fv0 = v0*np.exp(-(v0)**2/(vth0)**2)
    fv0 = (np.max(hvperp0)/np.max(fv0))*fv0
    plt.hist(vperp[0,wh],bins=20)
    plt.plot(v0,fv0)
    plt.xlabel('$v_\perp$')
    plt.ylabel('f(v)')

    plt.figure(2)   
    hvperp1, bins = np.histogram(vperp[2999,wh],bins=20)
    #        vth = np.std(bins)
    dbin = bins[1]-bins[0]
    bins=bins[:-1]
    vth = np.sum(bins*hvperp1*dbin)/np.sum(hvperp1*dbin)
    print('vth...',vth)
    vbar = gmean(bins)
    v = np.linspace(0,np.max(bins),100)
    fv = v*np.exp(-(v)**2/(vth)**2)
    fv = (np.max(hvperp1)/np.max(fv))*fv
    plt.plot(v,fv)
    plt.plot(v0,fv0)
    #plt.hist(vperp[h.Nstep-1,:],bins=40)
    plt.hist(vperp[2999,wh],alpha=0.5,bins=20)
    plt.xlabel('$v_\perp$')
    plt.ylabel('f(v)')
    plt.show()
    plt.plot(bins[0:-1],hvperp0-hvperp1)
    plt.xlabel('$v_\perp$')
    plt.ylabel('h1-h0')
    plt.show()
    hvpar0, bins = np.histogram(vpar[0,:],bins=40)
    hvpar1, bins = np.histogram(vpar[2000,:],bins=40)
    vth = np.std(bins)
    vbar = np.mean(bins)
    v = np.linspace(-np.max(bins),np.max(bins),100)
    fv = np.max(hvpar1)*np.exp(-(v-vbar)**2/(vth/2)**2)
    plt.plot(v,fv)
    plt.hist(vpar[h.Nstep-1,:],bins=40)
    plt.xlabel('$v_\parallel$')
    plt.ylabel('f(v)')
    plt.show()
    plt.plot(bins[0:-1],hvpar0-hvpar1)
    plt.xlabel('$v_\parallel$')
    plt.ylabel('h1-h0')
    plt.show()
    
    plt.plot(B[st:,0:10],h.mu[st:,0:10],'-.',markersize=1.5)
    plt.xlabel('y')
    plt.ylabel('$\mu$')
    plt.show()

#---------------------------------------------------------        
dir = '/data/KH3D/run_heating/'
Npart = 1000
Nstep = 2100
st = 1000    #starting time for mu normalization
p = Hybrid_read(dir)
p.read_para()
h = test_part(dir,Npart,Nstep)
h.read_part()
    
vpar, vperp2 = h.get_mu_bar(st)
#vpar, vperp2 = h.get_mu(st)

h.get_temp()
h.plot_hist()
#h.plot_dist(np.sqrt(vperp2),vpar)
#vperp,vpar = h.get_perp_par(h.vx,h.vy,h.vz)
#Btot = np.sqrt(h.Bx**2 + h.By**2 + h.Bz**2)
Bperp, Bpar = h.get_delta_B_perp()
vdotE = h.get_v_dot_E()
EdotJ = h.get_E_dot_J()
h.plot_correlation(st,1,Nstep)
mubar = h.get_gyro_average(h.mu)
dmu = (h.mu[1:,:]-h.mu[0:-1,:])
dvperp2 = vperp2[1:,:]-vperp2[0:-1,:]
##plot_phase_space(h)


va= 5e-9/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
myatan2 = np.vectorize(math.atan2)
fig,ax = plt.subplots(6)
w = 10
st = 1000
t0 = 1000
t1 = 2000
part0 = 1
part1 = 2
tm = np.linspace(0,t1-t0,t1-t0)/(2*np.pi*10)
#tm = np.linspace(0,t1-t0,t1-t0)/(10)
#ax[0].plot(np.convolve(h.mu[1500:1800,part]/h.mu[st,part],np.ones(w))/w,'-.')
#ax[0].set(ylabel='$\mu$')
for i in range(6,7):
    h.w = int(2*np.pi*10)
    Ezave = h.get_gyro_average(h.Ez)
    Exave = h.get_gyro_average(h.Ex)
    Bzave = h.get_gyro_average(h.Bz)
    Bxave = h.get_gyro_average(h.Bx)
    Ez1 = h.Ez[t0:t1,i]-Ezave[t0:t1,i]
    Ex1 = h.Ex[t0:t1,i]-Exave[t0:t1,i]
    Bx1 = h.Bx[t0:t1,i]-Bxave[t0:t1,i]
    Bz1 = h.Bz[t0:t1,i]-Bzave[t0:t1,i]
    S = -(Ex1*Bz1 - Ez1*Bx1)
    #ax[0].plot(Bperp[t0:t1,i],'-')
    ax[0].plot(tm,S,'-')
    ax[0].plot(tm,np.zeros(t1-t0))
    ax[0].set(ylabel = 'S')
    #ax[1].plot(np.abs((Ez1/Bx1)/va))
    #ax[1].plot(np.abs((Ex1/Bz1)/va))
    #dedbave = h.get_gyro_average((np.sqrt(Ex1**2 + Ez1**2)/np.sqrt(Bx1**2 + Bz1**2))/va)
    ax[1].plot(tm,(np.sqrt(Ex1**2 + Ez1**2)/np.sqrt(Bx1**2 + Bz1**2))/va)
    #ax[1].plot(tm,dedbave)
    ax[1].plot(tm,np.ones(t1-t0))
    ax[1].set(ylabel= '$(E_\perp/B_\perp)/v_A$')
    ax[1].set_ylim([0,20])
    #ax[1].plot(EdotJ[t0:t1,i],'-.')
    #ax[1].plot(np.ones(t1-t0)*EdotJ[t0:t1,i].mean())
    #ax[1].set(ylabel = 'vperp2')
    #ax[0].plot(Bpar[t0:t1,i],'-')
    #ax[2].plot(h.mu[t0:t1,i]/h.mu[st,i],'-.')

    #arr = sm.tsa.stattools.ccf(h.mu[t0:t1,i]/h.mu[st,i], vdotEbar[t0:t1,i])
    #ax[5].plot(arr,'.')
    #ax[5].plot(tm,h.vy[t0:t1,i]/va)
    h.w = int(10)
    Jybar = h.get_gyro_average(h.Jy)
    #ax[2].plot(tm,h.Jy[t0:t1,i])
    ax[2].plot(tm,h.mu[t0:t1,i]/h.mu[st,i])
    ax[2].plot(tm,np.ones(t1-t0))
#    ax[2].plot(tm,Bperp[t0:t1,i])
#    ax[2].plot(tm,Bpar[t0:t1,i])
#    ax[2].plot(tm,np.sqrt(Bpar[t0:t1,i]**2 + Bperp[t0:t1,i]**2))
    ax[2].set(ylabel='$\mu$')
    #ax[2].set(xlabel='time ($2 \pi \Omega_i^{-1}$)')
    #ax[4].plot(vdotE[t0:t1,i])
    #ax[4].plot(np.ones(t1-t0)*vdotE[t0:t1,i].mean())


    ax[3].plot(tm,dmu[t0:t1,i])
    ax[3].plot(tm,np.zeros(t1-t0))
    #ax[2].plot(vperp2[t0:t1,i]/va**2,'-.')
    #ax[2].plot((vpar[t0:t1,i] + np.sqrt(vperp2[t0:t1,i]))/va)
    #Ax[1].plot(h.mu[t0:t1,i]/h.mu[st,i],'o')
    ax[3].set(ylabel='$d \mu/dt$')
    #ax[3].plot(myatan2(h.Bx[1500:1800,i],h.Bz[1500:1800,i])*180./np.pi,'-.')
    Exbar = h.Ex[t0:t1,i]-h.Ex[t0:t1,i].mean()
    Ezbar = h.Ez[t0:t1,i]-h.Ez[t0:t1,i].mean()
    #ax[3].plot(myatan2(h.Ex[t0:t1,i]-h.Ex[t0:t1,i].mean(),h.Ez[t0:t1,i]-h.Ez[t0:t1,i].mean())*180/np.pi,'-.')
    ax[5].plot(tm,myatan2(Ez1,Ex1)*180/np.pi,'.')
    ax[5].plot(tm,myatan2(Bz1,Bx1)*180/np.pi,'.')
    ax[5].set(ylabel='$atan(Ez/Ex)$')
    ax[5].set(xlabel='time ($2 \pi \Omega_i^{-1}$)')
    #ax[3].plot(h.Ex[t0:t0+10,i],h.Ez[t0:t0+10,i],'o')
    vxbar = h.vx[t0:t1,i]-h.vx[t0:t1,i].mean()
    vzbar = h.vz[t0:t1,i]-h.vz[t0:t1,i].mean()
    #ax[4].plot(myatan2(h.vx[t0:t1,i]-h.vx[t0:t1,i].mean(),h.vz[t0:t1,i]-h.vz[t0:t1,i].mean())*180/np.pi,'-.')
    #vdotE = vxbar*Exbar + vzbar*Ezbar
    h.w = int(10)
    vdotEbar = h.get_gyro_average(vdotE)
    vdotE_sum = np.cumsum(vdotE[t0:t1,i])
    
    #ax[4].plot(vdotE_sum)
    ax[4].plot(tm,vdotEbar[t0:t1,i])
    #ax[4].plot(vdotE_sum)
    ax[4].plot(tm,np.zeros(t1-t0))
    ax[4].set(ylabel='$v \cdot E$')
    
    #ax[4].plot(vdotE.mean()*np.ones(t1-t0))
#ax[4].plot(h.vx[t0:t0+10,i],h.vz[t0:t0+10,i],'o')

#ax[2].plot(Bperp[t0:t1,i]/Btot[t0:t1,i],h.mu[t0:t1,i]/h.mu[st,i],'.')
#ax[2].set(ylabel='$J_\perp$')
plt.show()

plt.figure()
#plt.scatter(vdotEbar[t0:t1,1:10],h.mu[t0:t1,1:10]/h.mu[st,1:10])
plt.scatter(h.y[t0:t1,6],h.mu[t0:t1,6]/h.mu[st,6])
plt.show()

"""
for i in range(1,10):
    dmu = (h.mu[t0:t1,i]-h.mu[t0-1:t1-1,i])/h.mu[t0-1:t1-1,i]
    munorm = (h.mu[t0:t1,i]/h.mu[st,i])
    munorm = munorm/munorm.max()
    arr = sm.tsa.stattools.ccf(munorm, vdotEbar[t0:t1,i])
    a1 = arr[0:30]
    wh = np.where(a1 == a1.max())
    wh = np.squeeze(wh)
    vdotEbarnorm = vdotEbar[t0-wh:t1-wh,i]
    #vdotEbarnorm = vdotEbarnorm/(abs(vdotEbarnorm)).max()
    plt.plot(munorm,vdotEbarnorm,'.')
    #plt.plot(dmu,vdotE[t0:t1,i],'.')
    plt.show()

#h.get_alpha(st)
"""
