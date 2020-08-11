import numpy as np
from scipy.io import FortranFile
import matplotlib.pyplot as plt
import mayavi
from mayavi import mlab
from hybrid_read import Hybrid_read
import viscid

def get_stream(arr,i,j,k): #i,j,k is the seed location
    complete = False
    b = viscid.zeros([h.x,h.y,h.z]/h.di*1e3,nr_comps=3,layout='interlaced')
    X,Y,Z = b.get_crds('xyz',shaped=False)
    b['x'] = arr[1:,1:,1:,0]
    b['y'] = arr[1:,1:,1:,1]
    b['z'] = arr[1:,1:,1:,2]
    obound0 = np.array([X.min(),Y.min(),Z.min()])
    obound1 = np.array([X.max(),Y.max(),Z.max()])
    #seeds = viscid.Line((X[1], Y[int(h.ny/2)], Z[int(h.nz/2)]),
    #                    (X[-2],Y[int(h.ny/2)], Z[int(h.nz/2)]),10)
#    while (complete = False): # check to see if trace reached top/bottom boundary
    seeds = viscid.Point([X[i],Y[j],Z[k]])
    b_l = []
    nit = 0
    while (complete == False):
        b_lines, topo = viscid.calc_streamlines(b,seeds,obound0=obound0,obound1=obound1,
                                                stream_dir = viscid.DIR_FORWARD, method=viscid.RK4,
                                                output=viscid.OUTPUT_BOTH)
        b_l.append(b_lines)
        bl = b_lines[0]
        if (bl[1,:].max() > Y[-2]):
            print('trace complete',bl[1,:].max())
            complete=True
        elif (bl[0,-1] >= X[-1]):
            print('trace hit max X',X[-1],bl[1,-1],bl[2,-1])
            seeds = viscid.Point([X[2],bl[1,-1],bl[2,-1]])
            complete=False
        elif (bl[0,-1] <= X[1]):
            print('trace hit mix X',X[1])
            seeds = viscid.Point([X[-2],bl[1,-1],bl[2,-1]])
            complete=False
        nit += 1
        print(nit,complete)
    return b_l,topo


def plot_field_lines(b_lines):
    fig = mlab.figure(size=(800,600),bgcolor=(1,1,1),
                      fgcolor=(0,0,0))
    fig.scene.renderer.use_depth_peeling = True    
    clr = 'Spectral'
    x,y,z = np.mgrid[0:np.max(h.x):h.dx,
                     np.min(h.y):np.max(h.y):h.dy,
                     np.min(h.z):np.max(h.z):h.delz]/(h.di/1e3)
    
    mlab.volume_slice(x,y,z,arr,plane_orientation='y_axes',
                      slice_index=0,colormap=clr)
#                      transparent=True,opacity=0.5)
    mlab.volume_slice(x,y,z,arr,plane_orientation='y_axes',
                      slice_index=np.int(h.ny/2),colormap=clr)
#                      transparent=True,opacity=0.5)
    mlab.volume_slice(x,y,z,arr,plane_orientation='y_axes',
                      slice_index=h.ny-1,colormap=clr)
#                      transparent=True,opacity=0.5)
    mlab.outline(color=(0,0,0))
    mlab.axes(xlabel='x (di)',ylabel='y (di)',zlabel='z (di)',color=(0,0,0))
    for i in range(len(b_lines)):
        print('segment...',i)
        bl = b_lines[i]
        b = bl[0]
        xb = b[0,:]
        yb = b[1,:]
        zb = b[2,:]
        ii = np.round((xb*h.di/1e3)/h.dx).astype(int)-1
        jj = np.round((yb*h.di/1e3)/h.dy).astype(int)-1
        kk = np.round((zb*h.di/1e3)/h.delz).astype(int)-1
        s = arr[ii,jj,kk]
        mlab.plot3d(xb,yb,zb,s,tube_radius=0.5,tube_sides=12,
                    colormap='Spectral')
   
    mlab.show()
        
nfrm = 3
dir = './run_va_0.8_beta_1/'
h = Hybrid_read(dir)
h.read_para()
h.read_coords()

#xp = h.read_part('c.xp_ 1',3)
#vp = h.read_part('c.vp_ 1',3)
b = h.read_vector('c.b1',nfrm*4)
arr = h.read_scalar('c.mixed',nfrm*4)
arr[arr>1] = 1.0

b_lines,topo = get_stream(b,int(h.nx/2),1,int(h.nz/2))

print(np.shape(b_lines[0]))
b = b_lines[0]
type(b[0])

print('number of trace segments....',len(b_lines))

plot_field_lines(b_lines)

