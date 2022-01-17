import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from scipy.stats import gmean

#plt.rc('font', family='Helvetica')

def get_gyro_average(h,x):
    print('w...',h.w)
    w2 = int(h.w/2)
    xbar = np.zeros([h.Nstep,h.Npart])
    for i in range(len(h.Bx[0])):
        xbar[:,i] = np.convolve(x[w2:-w2+1,i],np.ones(h.w))/h.w
    return xbar    

def get_v_dot_E(h,Ex,Ey,Ez):
    #Exbar = get_gyro_average(h,h.Ex)
    #Eybar = get_gyro_average(h,h.Ey)
    #Ezbar = get_gyro_average(h,h.Ez)
    #vxbar = get_gyro_average(h,h.vx)
    #vybar = get_gyro_average(h,h.vy)
    #vzbar = get_gyro_average(h,h.vz)

    E = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    ex = Ex/E
    ey = Ey/E
    ez = Ez/E
    vdotE = h.vx*ex + h.vy*ey + h.vz*ez
    return vdotE

def get_mu_bar(h,st,va,b0):  #vperp^2/B
    print('w...',h.w)
    w2 = int(h.w/2)
    mp = 1
    vxbar = get_gyro_average(h,h.vx)
    vybar = get_gyro_average(h,h.vy)
    vzbar = get_gyro_average(h,h.vz)
    Bxbar = get_gyro_average(h,h.Bx)
    Bybar = get_gyro_average(h,h.By)
    Bzbar = get_gyro_average(h,h.Bz)
    B = np.sqrt(Bxbar**2 + Bybar**2 + Bzbar**2)        
    v2 = vxbar**2 + vybar**2 + vzbar**2
    bx = Bxbar/B
    by = Bybar/B
    bz = Bzbar/B
    vpar = vxbar*bx + vybar*by + vzbar*bz
    vperp2 = v2 - vpar**2
    #h.mu0 = (0.5*mp*vperp2[st,:]/B[st,:])
    #print('B...',B,1.6e-19*b0/1.67e-27)
    h.mu0 = (1.6e-19*b0/1.67e-27)/va**2              #normalization
    h.mu = (0.5*mp*vperp2/B)*h.mu0
    #plt.plot(h.mu[h.w:-h.w+1,:])
    #plt.show()
    return vpar, vperp2

def plot_6_panel(h,p):

    b0 = 5e-9
    va= b0/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
    myatan2 = np.vectorize(math.atan2)
    fig,ax = plt.subplots(6)
    #h.w = int(2*np.pi*10)
    h.w = 62
    st = 500
    t0 = 1500
    t1 = 2000
    part0 = 1
    part1 = 2
#    tm = np.linspace(0,t1-t0,t1-t0)/(2*np.pi*10)
    tm = np.linspace(t0,t1,(t1-t0))/(2*np.pi*10)

    ngyro = int(2*np.pi*10)  #number of time steps per gyroperiod
    
    vpar, vperp2 = get_mu_bar(h,st,va,b0)

    Ezave = get_gyro_average(h,h.Ez)
    Eyave = get_gyro_average(h,h.Ey)
    Exave = get_gyro_average(h,h.Ex)
    Bzave = get_gyro_average(h,h.Bz)
    Byave = get_gyro_average(h,h.By)
    Bxave = get_gyro_average(h,h.Bx)
    dEz1 = h.Ez-Ezave
    dEy1 = h.Ey-Eyave
    dEx1 = h.Ex-Exave
    dBx1 = h.Bx-Bxave
    dBy1 = h.By-Byave
    dBz1 = h.Bz-Bzave

    
    for i in range(6,7):
        #h.w = int(2*np.pi*10)
        #h.w=10

        Ex1 = dEx1[t0:t1,i]
        Ey1 = dEy1[t0:t1,i]
        Ez1 = dEz1[t0:t1,i]
        Bx1 = dBx1[t0:t1,i]
        By1 = dBy1[t0:t1,i]
        Bz1 = dBz1[t0:t1,i]
        Btot = h.By[t0:t1,i]
        
        S = -(Ex1*Bz1 - Ez1*Bx1)

        ax[0].plot(tm,S,'-')
        #ax[0].plot(tm,np.sqrt(Bx1**2 + Bz1**2)/(1.6e-19*b0/1.67e-27))
        ax[0].plot(tm,np.zeros(t1-t0))
        ax[0].set(ylabel = 'Poynting Flux')
        ax[0].xaxis.set_ticklabels([])
        vkaw=(np.sqrt(Ex1**2 + Ez1**2)/np.sqrt(Bx1**2 + Bz1**2))/va
        ax[1].plot(tm,(np.sqrt(Ex1**2 + Ez1**2)/np.sqrt(Bx1**2 + Bz1**2))/va)
        ax[1].plot(tm,np.ones(t1-t0))
        ax[1].set(ylabel= '$(E_\perp/B_\perp)/v_A$')
        ax[1].set_ylim([0,20])
        ax[1].xaxis.set_ticklabels([])
        #h.w=int(10)
        

        #h.w = int(10)
        vzvx = myatan2(h.vz[t0:t1,i],h.vx[t0:t1,i])
        wh = np.where(abs(vzvx[1:]-vzvx[0:-1]) > np.pi)
        #print('wh...',wh)
        muarr = h.mu[t0:t1,i]
        d2mu = muarr[2:] - 2*muarr[1:-1] + muarr[0:-2]
        dmu = muarr[2:] - muarr[0:-2]
        Jybar = get_gyro_average(h,h.Jy)
        ax[3].plot(tm,h.mu[t0:t1,i])
        #ax[2].plot(tm[::ngyro],h.mu[t0:t1:ngyro,i],'o')
        ax[3].plot(tm[wh],muarr[wh],'o')
        #ax[2].plot(tm[1:-1],abs(d2mu))
        #ax[2].plot(tm[::ngyro],h.mu[t0:t1:ngyro,i],'o')
        #ax[2].plot(tm[wh],d2mu[wh],'o')
        #ax[2].plot(tm,np.ones(t1-t0))
        ax[3].set(ylabel='$\mu$')
        ax[3].xaxis.set_ticklabels([])
        #h.w=4
        vdotE1 = get_v_dot_E(h,Exave,Eyave,Ezave)
        #vdotE = get_v_dot_E(h,dEx1,dEy1,dEz1)
        vdotE = get_v_dot_E(h,h.Ex,h.Ey,h.Ez)
        vdotEbar = get_gyro_average(h,vdotE)
        vdotE_sum = np.cumsum(vdotEbar[t0:t1,i])
        vpar, vperp2 = get_mu_bar(h,st,va,b0)
        dvperp2 = (vperp2[1:,:]-vperp2[0:-1,:])
        #arr = sm.tsa.stattools.ccf(h.mu[t0:t1,i]/h.mu[st,i], vdotEbar[t0:t1,i])
        ax[2].plot(tm,np.sqrt(Bx1**2 + Bz1**2)/0.479)
        ax[2].plot(tm,np.ones(t1-t0)*0.05)

        print(Btot)
        
#        ax[3].plot(tm,dvperp2[t0:t1,i])
#        ax[3].plot(tm,np.zeros(t1-t0))
#        ax[3].set(ylabel='$dv_\perp^2/dt$')
        ax[2].set(ylabel='$\delta B_\perp/B_0$')
        ax[2].xaxis.set_ticklabels([])

        #ax[4].plot(tm,vdotE[t0:t1,i])
        #ax[4].plot(tm,vdotE1[t0:t1,i])
        #ax[4].plot(tm,vdotE_sum)
        ax[4].plot(tm,vdotEbar[t0:t1,i])
        #ax[4].plot(tm[::ngyro],vdotE[t0:t1:ngyro,i],'o')
        vdotEarr = vdotEbar[t0:t1,i]
        ax[4].plot(tm[wh],vdotEarr[wh],'.')
        #ax[4].plot(tm[::ngyro],vdotE_sum[::ngyro],'-o')
        
        
        ax[4].plot(tm,np.zeros(t1-t0))
        ax[4].set(ylabel='$v \cdot E$')
        ax[4].xaxis.set_ticklabels([])
        
#        Exbar = h.Ex[t0:t1,i]-h.Ex[t0:t1,i].mean()
#        Ezbar = h.Ez[t0:t1,i]-h.Ez[t0:t1,i].mean()

        ax[5].plot(tm,vdotE_sum/vdotE_sum.max())
        ax[5].plot(tm,np.zeros(t1-t0))
        ax[5].set(ylabel='$\Sigma v \cdot E$')

#        ax[5].plot(tm,myatan2(Ez1,Ex1)*180/np.pi,'.')
#        ax[5].plot(tm[wh],myatan2(Ez1[wh],Ex1[wh])*180/np.pi,'o')
#        #ax[5].plot(tm,myatan2(Bz1,Bx1)*180/np.pi,'.')
#        ax[5].set(ylabel='$atan(Ez/Ex)$')
        ax[5].set(xlabel='time ($2 \pi \Omega_i^{-1}$)')
        vxbar = h.vx[t0:t1,i]-h.vx[t0:t1,i].mean()
        vzbar = h.vz[t0:t1,i]-h.vz[t0:t1,i].mean()

          
        plt.show()


    #plt.figure()
    #plt.plot(abs(d2mu),vkaw[1:-1],'.')
    #plt.show()
        
def plot_poincare(h,p):

    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    
    b0 = 5e-9
    va= b0/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
    myatan2 = np.vectorize(math.atan2)
    #fig,ax = plt.subplots(1)
    h.w = int(2*np.pi*10)
    st = 100
    t0 = 100
    t1 = 1500
    part0 = 1
    part1 = 2
    #tm = np.linspace(0,t1-t0,t1-t0)/(2*np.pi*10)
    tm = np.linspace(t0,t1,(t1-t0))/(2*np.pi*10)
    
    ngyro = int(2*np.pi*10)  #number of time steps per gyroperiod
    
    vpar, vperp2 = get_mu_bar(h,st,va,b0)
    vdotE = get_v_dot_E(h,h.Ex,h.Ey,h.Ez)
    dmu = (h.mu[1:,:]-h.mu[0:-1,:])
    
    for i in range(1,1000):
        if ((h.z[t0,i] > p.qz[40]) and (h.z[t0,i] < p.qz[int(p.nz-40)])): 
            h.w = int(2*np.pi*10)
            Ezave = get_gyro_average(h,h.Ez)
            Exave = get_gyro_average(h,h.Ex)
            Bzave = get_gyro_average(h,h.Bz)
            Bxave = get_gyro_average(h,h.Bx)
            Ez1 = h.Ez[t0:t1,i]-Ezave[t0:t1,i]
            Ex1 = h.Ex[t0:t1,i]-Exave[t0:t1,i]
            Bx1 = h.Bx[t0:t1,i]-Bxave[t0:t1,i]
            Bz1 = h.Bz[t0:t1,i]-Bzave[t0:t1,i]
            muave = get_gyro_average(h,h.mu)
            """        
            ax[0].plot(tm,h.mu[t0:t1,i])
            ax[0].plot(tm,muave[t0:t1,i])
            ax[0].plot(tm[::ngyro],muave[t0:t1:ngyro,i],'o')
            ax[0].plot(tm,np.ones(t1-t0))
            ax[0].set_ylabel='$\mu$'
            
            ax[1].plot(tm,myatan2(Ez1,Ex1)*180/np.pi,'.')
            ax[1].plot(tm,myatan2(Bz1,Bx1)*180/np.pi,'.')
            ax[1].set(ylabel='$atan(Ez/Ex)$')
            ax[1].set(xlabel='time ($2 \pi \Omega_i^{-1}$)')
            """
            phi = myatan2(Ez1,Ex1)*180/np.pi
            vzvx = myatan2(h.vz[t0:t1,i],h.vx[t0:t1,i])
            wh = np.where(abs(vzvx[1:]-vzvx[0:-1]) > np.pi)
            muarr = h.mu[t0:t1,i]
            #clr = np.cos(np.linspace(0,2*np.pi,len(phi[::ngyro])))
            #plt.plot(phi[wh],muarr[wh],':o')
            plt.plot(tm[wh],muarr[wh],'.',color=cm(1-1.*muarr[0]/0.4))
            #plt.plot(phi[::ngyro],h.mu[t0:t1:ngyro,i],':o')
            #plt.xlabel('$\phi ~(atan(E_z/E_x))$')
            #plt.xlabel('$\phi$')
            plt.xlabel('time ($2 \pi \Omega_i^{-1}$)',fontsize=14)
            plt.ylabel('$\mu$',fontsize=14)
        
    plt.show()

def get_perp_par(h,x,y,z):
    B = np.sqrt(h.Bx**2 + h.By**2 + h.Bz**2)
    bx = h.Bx/B
    by = h.By/B
    bz = h.Bz/B
    vecpar = x*bx + y*by + z*bz
    vecperp = np.sqrt((x**2 + y**2 + z**2) - vecpar**2)
    return vecperp,vecpar    
    
def plot_phase_space(h,p,Nstep):
    b0 = 5e-9
    va= b0/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3

    B = np.sqrt(h.Bx**2 + h.By**2 + h.Bz**2)
    wh = np.logical_and((h.z[Nstep-1,:] > p.qz[40]), (h.z[Nstep-1,:] < p.qz[int(p.nz-40)]))
    plt.figure(1)
    vperp, vpar = get_perp_par(h,h.vx-h.ux,h.vy-h.uy,h.vz-h.uz)
    hvperp0, bins = np.histogram(vperp[100,wh],bins=20)
    bins=bins[:-1]
    dbin = bins[1]-bins[0]
    vth0 = np.sum(bins*hvperp0*dbin)/np.sum(hvperp0*dbin)
    print('vth0...',vth0)
    v0 = np.linspace(0,np.max(bins),100)
    fv0 = v0*np.exp(-(v0)**2/(vth0)**2)
    fv0 = (np.max(hvperp0)/np.max(fv0))*fv0
    plt.hist(vperp[100,wh]/va,bins=20)
    plt.plot(v0/va,fv0)
    plt.xlabel('$v_{0\perp}/v_A$',fontsize=14)
    plt.ylabel('$v_\perp f(v_\perp)$',fontsize=14)
    plt.xlim([0,3])


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
    plt.plot(v/va,fv)
    #plt.plot(v0,fv0)
    #plt.hist(vperp[h.Nstep-1,:],bins=40)
    plt.hist(vperp[2999,wh]/va,bins=20)
    plt.xlabel('$v_\perp/v_A$',fontsize=14)
    plt.ylabel('$v_\perp f(v_\perp)$',fontsize=14)
    plt.xlim([0,3])
    plt.show()
    plt.plot(bins[0:],hvperp0-hvperp1)
    plt.xlabel('$v_\perp$')
    plt.ylabel('h1-h0')
    plt.show()
    hvpar0, bins = np.histogram(vpar[0,:],bins=40)
    hvpar1, bins = np.histogram(vpar[2000,:],bins=40)
    vth = np.std(bins)
    vbar = np.mean(bins)
    v = np.linspace(-np.max(bins),np.max(bins),100)
    fv = np.max(hvpar1)*np.exp(-(v-vbar)**2/(vth/2)**2)
#    fv = (np.max(hvpar1)/np.max(fv))*fv
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
  
def plot_dist(h,Nstep):
    vperp, vpar = get_perp_par(h,h.vx-h.ux,h.vy-h.uy,h.vz-h.uz)
    plt.figure(1)
    plt.plot(vpar[Nstep-1,:],vperp[Nstep-1,:],'.',markersize=1.0)
    plt.xlabel('vperp')
    plt.ylabel('vpar')
    plt.show()
    plt.figure(2)
    H1, xedges, yedges = np.histogram2d(vpar[100,:],vperp[100,:],bins=20)
    H1 = H1.T
    H, xedges, yedges = np.histogram2d(vpar[Nstep-1,:],vperp[Nstep-1,:],bins=20)
    H = H.T
    plt.imshow(H-H1, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect= 1)
    plt.show()

def plot_dmu_dvdotE(h,p):

    h.w = int(62)  
    t0 = 500
    t1 = 2000
    st = 1000
    b0 = 5e-9
    va= b0/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
    vpar, vperp2 = get_mu_bar(h,st,va,b0)
    vdotE = get_v_dot_E(h,h.Ex,h.Ey,h.Ez)

    dB = np.sqrt(h.Bx**2 + h.Bz**2)/h.By
    deltaB_ave = get_gyro_average(h,dB)
    
    #muave = get_gyro_average(h,h.mu)
    vdotEave = get_gyro_average(h,vdotE)

    wh = np.logical_and((h.z[st,:] > p.qz[40]), (h.z[st,:] < p.qz[int(p.nz-40)]))
    
    dmu = (h.mu[1:,:]-h.mu[0:-1,:])
    dvperp2 = (vperp2[1:,:]-vperp2[0:-1,:])
    #dvdotE = vdotEave[1:,:] - vdotEave[0:-1,:]
    plt.plot(dmu[t0:t1,wh]/abs(dmu[t0:t1,wh]).max(),vdotEave[t0:t1,wh]/abs(vdotEave[t0:t1,wh]).max(),'.')
    plt.xlabel('$\delta \mu$')
    plt.ylabel('$<v \cdot E>$')
    arr = sm.tsa.stattools.ccf(dmu[t0:t1,wh].flatten(), vdotEave[t0:t1,wh].flatten())
    print(arr[0])
    plt.show()
    
def get_temp(h,Nstep):
    #get temperature
    #plt.figure()
    for i in range(Nstep):            
        v2 = (h.vx[i,:]**2 + h.vy[i,:]**2 + h.vz[i,:]**2).sum()/h.Npart
        u = (h.vx[i,:] + h.vy[i,:] + h.vz[i,:]).sum()/h.Npart
        h.temp[i] = 0.5*1.67e-27*(v2 - u**2)*1e6/1.6e-19
    plt.figure()
    plt.plot(h.temp[:Nstep-1])
    plt.show()


def get_mu_std(h,p):

    b0 = 5e-9
    va= b0/np.sqrt(np.pi*4e-7*1.67e-27*0.4e6)/1e3
    myatan2 = np.vectorize(math.atan2)
    #fig,ax = plt.subplots(1)
    h.w = int(2*np.pi*10)
    st = 1000
    t0 = 500
    t1 = 2000
    part0 = 1
    part1 = 2
    tm = np.linspace(0,t1-t0,t1-t0)/(2*np.pi*10)

    
    
    ngyro = int(2*np.pi*10)  #number of time steps per gyroperiod
    
    vpar, vperp2 = get_mu_bar(h,st,va,b0)
    vdotE = get_v_dot_E(h,h.Ex,h.Ey,h.Ez)
    dmu = (h.mu[1:,:]-h.mu[0:-1,:])

    mustd = []
    
    for i in range(1,50):
        if ((h.z[t0,i] > p.qz[40]) and (h.z[t0,i] < p.qz[int(p.nz-40)])): 
            h.w = int(2*np.pi*10)
            #Ezave = get_gyro_average(h,h.Ez)
            #Exave = get_gyro_average(h,h.Ex)
            #Bzave = get_gyro_average(h,h.Bz)
            #Bxave = get_gyro_average(h,h.Bx)
            #Ez1 = h.Ez[t0:t1,i]-Ezave[t0:t1,i]
            #Ex1 = h.Ex[t0:t1,i]-Exave[t0:t1,i]
            #Bx1 = h.Bx[t0:t1,i]-Bxave[t0:t1,i]
            #Bz1 = h.Bz[t0:t1,i]-Bzave[t0:t1,i]
            muave = get_gyro_average(h,h.mu)

            vzvx = myatan2(h.vz[t0:t1,i],h.vx[t0:t1,i])
            wh = np.where(abs(vzvx[1:]-vzvx[0:-1]) > 1.5*np.pi)
            muarr = h.mu[t0:t1,i]
            #w2 = int(h.w/2)
            #mustd = np.zeros([h.Nstep,h.Npart])
            #for i in range(len(h.Bx[0])):
            #    mustd[:,i] = np.convolve(muarr[w2:-w2+1,i].std(),np.ones(h.w))/h.w
            tm1 = tm[wh]
            muarr1 = muarr[wh]
            mustd.append(muarr[wh].std())
            #plt.plot(tm1,muarr1,'.')
            print(muarr[wh].std())
    plt.plot(mustd)
    plt.xlabel('$\phi$')
    plt.ylabel('$\mu$')
        
    plt.show()
