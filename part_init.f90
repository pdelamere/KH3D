module part_init
      implicit none
      save
      contains

      subroutine Energy_diag(Evp,Euf,EB1,EB1x,EB1y,EB1z,EE,EeP)
            use dimensions
            use mpi
            use mult_proc, only: my_rank
            use grid, only: dx_cell,dy_cell,dz_cell
            use inputs, only: mion,q,mu0,mO,km_to_m,epsilon
            use var_arrays, only: vp,b0,b1,E,nu,up,np,Ni_tot,beta,beta_p,input_E,prev_Etot,bndry_Eflux,m_arr
            implicit none
            real, intent(out):: Euf,EB1,EB1x,EB1y,EB1z,EE,EeP,Evp
            real:: denf,m_q,recvbuf,total_E,aveEvp,norm_E,vol
            real:: S_Evp,S_input_E
            integer:: count, ierr
            integer:: i,j,k,m,l
            
            count = 1
            m_q = mion/q
            
            Euf = 0.0
            EB1 = 0.0
            EB1x = 0.0
            EB1y = 0.0
            EB1z = 0.0
            EE = 0.0
            EeP = 0.0
            
            do i=1,nx-1
!                  do j = 1,ny-1
                   j=2
                        do k=1,nz-1
                              vol= dx_cell(i)*dy_cell(j)*dz_cell(k)*km_to_m**3
                              EB1x=EB1x + (vol/(2.0*mu0))*(m_q*b1(i,j,k,1))**2
                              EB1y=EB1y + (vol/(2.0*mu0))*(m_q*b1(i,j,k,2))**2
                              EB1z=EB1z + (vol/(2.0*mu0))*(m_q*b1(i,j,k,3))**2
                                    do m=1,3
                                          denf = np(i,j,k)/(km_to_m**3)
                                          Euf = Euf + 0.5*mO*denf*vol*(up(i,j,k,m)*km_to_m)**2
                                          EB1 = EB1 + (vol/(2.0*mu0))*(m_q*(b1(i,j,k,m)-b0(i,j,k,m)))**2
                                          EE = EE + (epsilon*vol/2.0)*(m_q*E(i,j,k,m)*km_to_m)**2
                                    enddo
                        enddo
!                  enddo
            enddo
            
            Evp = 0.0
            do l=1, Ni_tot
                  do m=1,3
                        Evp = Evp + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2 / (beta*beta_p(l))
                  enddo
            enddo
            
            
            call MPI_BARRIER(MPI_COMM_WORLD,ierr)
            
            call MPI_ALLREDUCE(Evp,recvbuf,count,MPI_REAL,MPI_SUM,MPI_COMM_WORLD,ierr)
            S_Evp = recvbuf
            
            call MPI_ALLREDUCE(input_E,recvbuf,count,MPI_REAL,MPI_SUM,MPI_COMM_WORLD,ierr)
            S_input_E = recvbuf
            
            total_E = S_Evp + EE + EB1
            aveEvp = S_Evp/S_input_E
            
            
            if (my_rank .eq. 0) then
                  write(*,*) 'Normalized energy.................',total_E/S_input_E, my_rank
                  write(*,*) 'Normalized energy (bndry).........',total_E/(S_input_E+bndry_Eflux)
            endif
            
            norm_E = total_E/S_input_E
            prev_Etot = norm_E

      end subroutine Energy_diag
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine load_Maxwellian(vth,Ni_tot_1,mass,mratio)
            use dimensions
            use boundary
            use inputs, only: PI, vsw, dx, dy, km_to_m, beta_particle, kboltz, mion, amp, grad, nf_init,b0_init,mu0,boundx, Lo, q, mO
            use grid, only: qx,qy,qz,dz_grid
            use gutsp
            use var_arrays, only: np,vp,vp1,xp,input_p,up,Ni_tot,input_E,ijkp,m_arr,mrat,beta,beta_p,wght,grav,temp_p,mix_ind,b0
            implicit none
            integer(4), intent(in):: Ni_tot_1 
            real, intent(in):: mratio, mass, vth
            real:: Lo_y
                                  
            integer:: disp
            real:: vth2, vx, vy, vz, va, va_x, Temp, Tempcalc, pl_beta(nx,ny,nz)
            integer:: l,m,i,j,k,ii,kk
            
            disp = 0 !Displacement of gradient
!            amp = 100.0
!            grad = 100.0 ! density gradient (larger = more gradual
            
            do i=1,nx
                  do j=1,ny
                        do k=1,nz
                              pl_beta(i,j,k) = 1.0 !2.0 + 1.0*exp(-(real(i-nx/2)**2+real(k-nz/2)**2)/(10**2))
                        enddo
                  enddo
            enddo
            va_x = (mO/q)*b0(1,1,1,1)/sqrt(mu0*mion*nf_init/1e9)/1e3
            va = b0_init/sqrt(mu0*mion*nf_init/1e9)/1e3

!            write(*,*) 'Va....',va
!            stop
            
            do l = Ni_tot_1,Ni_tot
                  xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                  xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
                  if (boundx .eq. 1) then
                     xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                  else
                     xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz)-qz(1))
                  endif
                  
                  if (xp(l,3) .gt. qz(nz/2)) mix_ind(l) = 1
                  if (xp(l,3) .le. qz(nz/2)) mix_ind(l) = 0
            
                  m_arr(l) = mass
                  mrat(l) = mratio

!                  beta_p(l) = 1.0/(beta_particle+beta_particle*amp*exp(-((xp(l,3)-qz(nz/2-disp))/ &
!                        (grad*dz_grid(nz/2-disp)))**2))
                  beta_p(l) = beta_particle
!!!!!!!!!!!!!Get P-index!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                  i=1
!                  do 
!                        if (xp(l,1) .le. qx(i)) exit
!                        i = i+1
!                  enddo
!                  i=i-1
                  
!                  ijkp(l,1) = i
!                  ijkp(l,2) = floor(xp(l,2)/dy)
                  
!                  k=1
!                  do
!                        if (xp(l,3) .le. qz(k)) exit
!                        k=k+1
!                  enddo
!                  k=k-1
                  
!                  ijkp(l,3) = k
!!!!!!!!!!!!!End get P-index!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                  call get_pindex(i,j,k,l)
!                  vth2=sqrt(vth*vth*beta_p(l)) !thermal speed dependent on np to set up pressure balance for density gradient

!                  vth2=va*sqrt(pl_beta(ijkp(l,1),ijkp(l,2),ijkp(l,3)))

                  vth2 = vth - va_x*cosh((qz(nz/2)-qz(k))/Lo)**(-2) 

                  
                  vx = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf()) !remember to add in vsw to get the flow velocity
                  vy = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vz = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  
                  ii = ijkp(l,1)
                  kk = ijkp(l,3)

!                  Lo_y = 2*Lo*(1.0-exp(-(xp(l,2)-qy(ny/2))**2/(5*dy)**2)) + Lo
                  
!                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2)
!               x        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx
                  vp(l,1) =  0.8*va*(tanh((qz(k)-qz(nz/2))/(Lo))) + vx !+ &
!                       -(0.1*20*dx/(PI*Lo))*va*cosh((qz(nz/2)-qz(k))/Lo)**(-2)*tanh((qz(nz/2)-qz(k))/Lo)*cos(qx(i)*PI/(20*dx))
!vx!+57.0*exp(-(xp(l,3)-qz(nz/2))**2/(5*dz_grid(nz/2))**2) !Gaussian velocity perturbation (20)
                  vp(l,2) = vy! +57.0*(1+0.5*cos(8*pi*qx(ii)/qx(nx-1)))* &
                       !(1+0.5*cos(8*pi*qz(kk)/qz(nz)))* &
                       !exp(-((qx(ii)-qx(nx/2))**2 + (qz(kk)-qz(nz/2))**2)/(10*dx)**2)
                  vp(l,3) = vz !+0.1*0.5*va*cosh((qz(nz/2)-qz(k))/Lo)**(-2)*sin(PI*qx(i)/(20*dx))
                  
                  do m=1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2/(beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l) * vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            
            call get_interp_weights()
            call update_np()
            call update_up(vp)

            
            ! Add a centrifugal gravity term to keep the plasma confined to the torus.  Use T * dn/dz = nmg.  
            ! Depends on the density gradient.  Currently set as a gaussian.
            
!            Temp = vth**2/(3*kboltz)*mion*1.48*10-23!8.61738e-5
!            write(*,*) 'vth.................', vth
!            write(*,*) 'boltzman............', kboltz
!            write(*,*) 'temperature(analytic)..', Temp
!            call get_temperature()
!            Tempcalc = sum(temp_p(2,2,1:(nz-1)))/1e6/(nz-1) !in kg km^2/s^2
!            write(*,*) 'temperature (2,2,100)..', temp_p(2,2,2:10)/1.6e-19
!            stop
            
            do i=1,nx
            do j=1,ny
            do k=1,nz
                  ! Gravity is based on the analytical expression for the density profile (look at beta_p)
                  ! np = const/(beta*beta_p), and grav = const * (dn/dx) / n
                  

                        
                       
                        
!                  grav(i,j,k) = -2.0*Tempcalc/(mion*(grad*dz_grid(nz/2-disp))**2 &
!                        *(1.0+amp*exp(-((qz(k)-qz(nz/2-disp))/(grad*dz_grid(nz/2-disp)))**2))) &
!                        *amp*(qz(k)-qz(nz/2-disp))*exp(-((qz(k)-qz(nz/2-disp))/(grad*dz_grid(nz/2-disp)))**2)   
                 
       
                 grav(i,j,k) = 0.0
                  
                  
            enddo
            enddo
            enddo
           ! write(*,*) 'gravity...', grav(2,2,nz/2+50), grav(2,2,nz/2-50)
           ! stop

            
      end subroutine load_Maxwellian
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine load_Maxwellian_KH(vth,Ni_tot_1,Ni_tot_2,mass,mratio,beta_part,rL,region)
!!!!!! Adds particles to bottom if .true. else add to top.

            use dimensions
            use boundary
            use inputs, only: PI, va_f, dx, dy, delz,km_to_m, beta_particle, kboltz, mion, amp, grad, nf_init,b0_init,mu0,boundx, Lo, q, mO
            use grid, only: qx,qy,qz,dz_grid
            use gutsp
            use var_arrays, only: np,vp,vp1,xp,input_p,up,Ni_tot,input_E,ijkp,m_arr,mrat,beta,beta_p,wght,grav,temp_p,mix_ind,b0
            implicit none
            integer(4), intent(in):: Ni_tot_1, Ni_tot_2
            real, intent(in):: beta_part
            real, intent(in):: mratio, mass, vth, rL
            real:: Lo_y
                                  
            integer:: disp
            real:: vth2, vx, vy, vz, va, va_x, Temp, Tempcalc, pl_beta(nx,ny,nz)
            integer:: l,m,i,j,k,ii,kk
            integer:: region
            real:: rnd
            integer:: flg
            real:: v1_xuan,kx,ky
            
            
            disp = 0 !Displacement of gradient
!            amp = 100.0
!            grad = 100.0 ! density gradient (larger = more gradual
            
            v1_xuan = 0.0 !-0.8/40  !was 0.1*va
            kx = PI/(nx*dx)
            ky = PI/(ny*dy)

            do i=1,nx
                  do j=1,ny
                        do k=1,nz
                              pl_beta(i,j,k) = 1.0 !2.0 + 1.0*exp(-(real(i-nx/2)**2+real(k-nz/2)**2)/(10**2))
                        enddo
                  enddo
            enddo
            va_x = (mO/q)*abs(b0(1,1,1,1))/sqrt(mu0*mion*nf_init/1e9)/1e3
            va = b0_init/sqrt(mu0*mion*nf_init/1e9)/1e3

            do l = Ni_tot_1,Ni_tot_2
               xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
               xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
               
               flg = 0
               do 20 while (flg .eq. 0)
                  if (boundx .eq. 1) then
                     xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                  else
                     xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz)-qz(1))
                  endif

                  if (region .eq. 1) then
!                     rnd = ((1.0-tanh((xp(l,3)-qz(nz/2)-delz*0.5)/(Lo)))/2.0)  !for bottom
                     rnd = ((1.0-tanh((xp(l,3)-qz(nz/2)-delz*0.5)/(rL)))/2.0)  !for bottom
                  endif
                  if (region .eq. 2) then
!                     rnd = (1.0+tanh((xp(l,3)-qz(nz/2)-delz*0.5)/(Lo)))/2.0 !for top
                     rnd = (1.0+tanh((xp(l,3)-qz(nz/2)-delz*0.5)/(rL)))/2.0 !for top
                  endif
                  if (pad_ranf() .le. rnd) flg = 1
                  
20                continue
                  
                  if (xp(l,3) .gt. qz(nz/2)) mix_ind(l) = 1
                  if (xp(l,3) .le. qz(nz/2)) mix_ind(l) = 0
            
                  m_arr(l) = mass
                  mrat(l) = mratio

!                  beta_p(l) = 1.0/(beta_particle+beta_particle*amp*exp(-((xp(l,3)-qz(nz/2-disp))/ &
!                        (grad*dz_grid(nz/2-disp)))**2))
                  beta_p(l) = beta_part

                  call get_pindex(i,j,k,l)

!                  vth2=sqrt(vth*vth*beta_p(l)) !thermal speed dependent on np to set up pressure balance for density gradient

!                  vth2=va*sqrt(pl_beta(ijkp(l,1),ijkp(l,2),ijkp(l,3)))

                  vth2 = vth !+ va_x*cosh((qz(nz/2)-qz(k))/Lo)**(-2) 

                  
                  vx = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf()) !remember to add in vsw to get the flow velocity
                  vy = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vz = vth2*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  
                  ii = ijkp(l,1)
                  kk = ijkp(l,3)

!                  Lo_y = 2*Lo*(1.0-exp(-(xp(l,2)-qy(ny/2))**2/(5*dy)**2)) + Lo
                  
!                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2)
!               x        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx
                  vp(l,1) =  va_f*va*(tanh((qz(k)-qz(nz/2))/(Lo))) + vx + &
                       (v1_xuan/(kx*Lo))*cosh((qz(k)-qz(nz/2))/(2*Lo))**(-2)*tanh((qz(k)-qz(nz/2))/(2*Lo))*cos((qx(i)-qx(nx/2))*kx)*cos((qy(j)-qy(ny/2))*ky)
!vx!+57.0*exp(-(xp(l,3)-qz(nz/2))**2/(5*dz_grid(nz/2))**2) !Gaussian velocity perturbation (20)
                  vp(l,2) = vy! +57.0*(1+0.5*cos(8*pi*qx(ii)/qx(nx-1)))* &
                       !(1+0.5*cos(8*pi*qz(kk)/qz(nz)))* &
                       !exp(-((qx(ii)-qx(nx/2))**2 + (qz(kk)-qz(nz/2))**2)/(10*dx)**2)
                  vp(l,3) = vz - v1_xuan*cosh((qz(k)-qz(nz/2))/(2*Lo))**(-2)*sin((qx(i)-qx(nx/2))*kx)*cos((qy(j)-qy(ny/2))*ky)
                  
                  do m=1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2/(beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l) * vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            
            call get_interp_weights()
            call update_np()
            call update_up(vp)

                       
            do i=1,nx
               do j=1,ny
                  do k=1,nz
                     grav(i,j,k) = 0.0
                  enddo
               enddo
            enddo
            
      end subroutine load_Maxwellian_KH
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine init_KH_part()
        
        use Var_Arrays
        use inputs

        integer(4):: N_1, N_2, N_3 !particle partitioning
        real:: vth1, vth2, vth3    !Initialize for 3 species (Maxwellian populations)
        real:: m1, m2, m3
        real:: beta1, beta2, beta3          !macro scaling parameters
        real:: N1, N2, N3    !actual number of particles
        real:: d1, d2, d3    !actual number density
        real:: rL1, rL2, rL3 !gyroradius

!        n1 = 0.4
!        n2 = 0.2
!        n3 = 0.01
        

        N_1 = Ni_tot
        N_2 = nint(2.0*Ni_tot)
        Ni_tot = N_2
!        N_3 = nint(2.0*Ni_tot)
!        Ni_tot = N_3

        N1 = real(N_1)
        N2 = real(N_2) - real(N_1)
        N3 = real(N_3) - real(N_2)

        beta1 = 1.0 
        beta2 = (N2/N1)/dens2
        beta3 = (N3/N1)/dens3

        write(*,*) 'beta...',beta1,beta2,beta3
!        stop
        
        vth1 = vth
        vth2 = vth
        vth3 = 4.0*vth

        m1 = 1.0
        m2 = 1.0
!        m3 = m_pu
        m3 = 1.1
        
!!        vth1 = sqrt((m2/m1)*(N2/N1)*(beta1/beta2)*vth2*vth2 + (m3/m1)*(N3/N1)*(beta1/beta3)*vth3*vth3)
!        vth2 = sqrt((m1/m2)*(N1/N2)*(beta2/beta1)*vth1*vth1 - (m3/m2)*(N3/N2)*(beta2/beta3)*vth3*vth3)
!        vth3 = sqrt(-(m2/m3)*(N2/N3)*(beta3/beta2)*vth2*vth2 + (m1/m3)*(N1/N3)*(beta3/beta1)*vth1*vth1)

        rL1 = m1*mion*vth1/(q*B0_init)  !gryoradius
        rL2 = m2*mion*vth2/(q*B0_init)
        rL3 = m3*mion*vth3/(q*B0_init)

!        write(*,*) 'rL...',rL1,rL2,rL3,lambda_i
!        stop
        
        write(*,*) 'vth...',vth1,vth2,vth3
!        stop

        call load_Maxwellian_KH(vth1,1,N_1,m1*mion,1/m1,beta1,rL1,1)
        call load_Maxwellian_KH(vth2,N_1+1,N_2,m2*mion,1/m2,beta2,rL2,2)
!        call load_Maxwellian_KH(vth3,N_2+1,N_3,m3*mion,1/m3,beta3,rL3,2)
        
      end subroutine init_KH_part
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine init_KH_heavy()
        
        use Var_Arrays
        use inputs

        integer(4):: N_1, N_2, N_3 !particle partitioning
        real:: vth1, vth2, vth3    !Initialize for 3 species (Maxwellian populations)
        real:: m1, m2, m3
        real:: beta1, beta2, beta3          !macro scaling parameters
        real:: N1, N2, N3    !actual number of particles
        real:: d1, d2, d3    !actual number density
        real:: rL1, rL2, rL3 !gyroradius

!        n1 = 0.4
!        n2 = 0.2
!        n3 = 0.01
        
        N_1 = Ni_tot
        N_2 = nint(2.0*Ni_tot)
        Ni_tot = N_2
        N_3 = nint(2.0*Ni_tot)
        Ni_tot = N_3

        N1 = real(N_1)
        N2 = real(N_2) - real(N_1)
        N3 = real(N_3) - real(N_2)

        beta1 = 1.0 
        beta2 = (N2/N1)/dens2
        beta3 = (N3/N1)/dens3

        write(*,*) 'beta...',beta1,beta2,beta3
!        stop
        
        vth1 = vth
        vth2 = vth
        vth3 = vth

        m1 = 1.0
        m2 = 1.0
!        m3 = m_pu
        m3 = 16.0
        
!        vth1 = sqrt((m2/m1)*(N2/N1)*(beta1/beta2)*vth2*vth2 + (m3/m1)*(N3/N1)*(beta1/beta3)*vth3*vth3)
        vth2 = sqrt((m1/m2)*(N1/N2)*(beta2/beta1)*vth1*vth1 - (m3/m2)*(N3/N2)*(beta2/beta3)*vth3*vth3)
        vth3 = sqrt(-(m2/m3)*(N2/N3)*(beta3/beta2)*vth2*vth2 + (m1/m3)*(N1/N3)*(beta3/beta1)*vth1*vth1)

        rL1 = m1*mion*vth1/(q*B0_init)  !gryoradius
        rL2 = m2*mion*vth2/(q*B0_init)
        rL3 = m3*mion*vth3/(q*B0_init)

!        write(*,*) 'rL...',rL1,rL2,rL3,lambda_i
!        stop
        
        write(*,*) 'vth...',vth1,vth2,vth3
!        stop

        call load_Maxwellian_KH(vth1,1,N_1,m1*mion,1/m1,beta1,rL1,1)
        call load_Maxwellian_KH(vth2,N_1+1,N_2,m2*mion,1/m2,beta2,rL2,2)
        call load_Maxwellian_KH(vth3,N_2+1,N_3,m3*mion,1/m3,beta3,rL3,2)
        
      end subroutine init_KH_heavy
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine load_ring_beam(vring,dNi,mass,mratio)
            use dimensions
            use boundary
            use inputs, only: PI, vsw, dx, dy, km_to_m, beta_pu, ion_amu,m_pu,beta_particle, amp, grad
            use grid, only: qx,qy,qz,dz_grid
            use gutsp
            use var_arrays, only: np,vp,vp1,xp,input_p,up,Ni_tot,input_E,ijkp,m_arr,mrat,beta,beta_p,wght,mix_ind
            implicit none
            integer(4), intent(in):: dNi 
            real, intent(in):: vring, mass,mratio
                                  
            integer:: disp, flg, l1
            real:: vth2, vx, vy, vz, rand1, theta2
            integer:: i,j,k,l,m
            
            disp = 0 !Displacement of gradient
!            amp = 100.0
!            grad = 400.0 ! density gradient (larger = more gradual
            
!            v1=1.0
            l1=Ni_tot+1
            
            do l = l1, l1+dni-1
                 ! xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                  xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
                  flg=0
                        do while (flg .eq. 0)
                            !  xp(l,3) = qz(nz/2-20) + (1.0-pad_ranf())*(qz(nz/2+20)-qz(nz/2-20))
                              xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                              rand1=pad_ranf()
                              if (exp(-(xp(l,3)-qz(nz/2))**2/(10*dz_grid(nz/2)**2)) .gt. rand1) then
                                    flg = 1
                              endif
                        enddo
                        
                  flg=0
                        do while (flg .eq. 0)
                              xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                              rand1=pad_ranf()
                              if (exp(-(xp(l,1)-qx(nx/2))**2/(10*dx**2)) .gt. rand1) then
                                    flg = 1
                              endif
                        enddo
                        
                        beta_p(l) = beta_particle/10.0
                        m_arr(l) = mass
                        mrat(l) = mratio
                  
                  
                  call get_pindex(i,j,k,l)
                  
                  
!                  ii = ijkp(l,1)
!                  kk = ijkp(l,3)
                  
!                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2)
!               x        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx

            !   Ring beam velocity initializtion
!                  theta2 = pad_ranf()*2*PI
!                  vp(l,1) = vring*cos(theta2)
!                  vp(l,2) = vring*sin(theta2)
!                  vp(l,3) = 0.0
                  
            !   Maxwellian thermal distribution 
            
                  vth2=100.0;
                  
                  vx = vth2*sqrt(-log(pad_ranf()))*cos(2*PI*pad_ranf()) !remember to add in vsw to get the flow velocity
                  vy = vth2*sqrt(-log(pad_ranf()))*cos(2*PI*pad_ranf())
                  vz = vth2*sqrt(-log(pad_ranf()))*cos(2*PI*pad_ranf())
                  
                  
                  do m=1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2/(beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l) * vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            Ni_tot=Ni_tot+dNi
            
            call get_interp_weights()
            call update_np()
            call update_up(vp)
            
     end subroutine load_ring_beam
      
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine load_aniso_Maxwellian(vth,Ni_tot_1)
            use dimensions
            use boundary
            use grid, only: qx,qy,qz,dz_grid
            use inputs, only: mion, dx, dy,delz, vsw, km_to_m, PI, beta_particle
            use gutsp
            use var_arrays, only: np,vp,vp1,xp,input_p,up,Ni_tot,input_E,ijkp,m_arr,mrat,beta,beta_p,wght
            implicit none
            real, intent(in):: vth
            integer(4), intent(in):: Ni_tot_1
            real:: vx,vy,vz
            real:: aniso_frac, vthx, vthy, vthz
            integer:: i,j,k,l,m
            
            aniso_frac = 0.06
            
            
            do l = 1, Ni_tot_1
                  xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                  xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
                  xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                  m_arr(l) = mion
                  mrat(l) = 1.0
                  beta_p(l) = beta_particle
                  
                  call get_pindex(i,j,k,l)
                  
                  vx = vsw+vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vy = vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vz = vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())

!                  ii = ijkp(l,1)
!                  kk = ijkp(l,3)
                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2) &
                        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx
                  vp(l,2) = vy 
                  vp(l,3) = vz 
                  
                  do m = 1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2 / (beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l)*vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            
            vthx = 1200.0
            vthy = 1200.0
            vthz = 500.0
            
            do l = Ni_tot_1+1, Ni_tot_1 + aniso_frac*Ni_tot_1
                  xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                  xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
                  xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                  m_arr(l) = mion
                  mrat(l) = 1.0
                  beta_p(l) = beta_particle
                  
                  call get_pindex(i,j,k,l)
                  
                  vx = vsw+vthx*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vy = vthy*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vz = vthz*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())

!                  ii = ijkp(l,1)
!                  kk = ijkp(l,3)
                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2) &
                        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx
                  vp(l,2) = vy 
                  vp(l,3) = vz 
                  
                  do m = 1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2 / (beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l)*vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            
            Ni_tot = Ni_tot_1 + aniso_frac*Ni_tot_1
            
            do l = Ni_tot + 1, Ni_tot + aniso_frac*Ni_tot
                  xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
                  xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
                  xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
                  m_arr(l) = 2.0*mion
                  mrat(l) = 0.5
                  beta_p(l) = beta_particle
                  
                  call get_pindex(i,j,k,l)
                  
                  vx = vsw+vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vy = vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())
                  vz = vth*sqrt(-log(pad_ranf()))*cos(PI*pad_ranf())

!                  ii = ijkp(l,1)
!                  kk = ijkp(l,3)
                  vp(l,1) = -0.0*(exp(-(xp(l,3)-qz(nz/2))**2/(10.*delz)**2) &
                        *exp(-(xp(l,1)-qx(nx/2))**2/(10.*dx)**2))+vx
                  vp(l,2) = vy 
                  vp(l,3) = vz 
                  
                  do m = 1,3
                        vp1(l,m) = vp(l,m)
                        input_E = input_E + 0.5*m_arr(l)*(vp(l,m)*km_to_m)**2 / (beta * beta_p(l))
                        input_p(m) = input_p(m) + m_arr(l)*vp(l,m) / (beta * beta_p(l))
                  enddo
                  
            enddo
            
            Ni_tot = Ni_tot + aniso_frac*Ni_tot
            
            call get_interp_weights()
            call update_np()
            call update_up(vp)
      
      end subroutine load_aniso_Maxwellian
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine sw_part_setup_maxwl_mix()
            use dimensions
            use inputs, only: mion, vth_bottom
            use gutsp
            use var_arrays, only: np,vp,vp1,xp,input_p,up,np_t_flg,np_b_flg,Ni_tot,input_E,ijkp,m_arr,mrat,beta,beta_p,wght
            implicit none
            integer(4):: Ni_tot_1
            
            
            np_t_flg = 0
            np_b_flg = 0
            
!       Add cold populations first

            Ni_tot_1 = 1
            call load_Maxwellian(vth_bottom,Ni_tot_1,mion,1.0) !mass ratio
! add He++

!      Ni_tot_1 = Ni_tot + 1
!      Ni_tot = 2.0*Ni_tot_0
      
!      call load_Maxwellian(np,vp,vp1,xp,input_p,up,
!     x     vth_bottom, Ni_tot_1, 2.0*mion, 1.0/2.0, 10.0) !inverse ration, this is a mass 2 particle (q/m)
         

! add pickup distribution

!         Ni_tot_1 = Ni_tot + 1
!         Ni_tot = 3*Ni_tot_0

!         do 69 l = Ni_tot_1,Ni_tot
!
!            xp(l,1) = qx(1)+(1.0-pad_ranf())*(qx(nx-1)-qx(1))
!            xp(l,2) = qy(1)+(1.0-pad_ranf())*(qy(ny-1)-qy(1))
!
!           xp(l,3) = qz(1)+(1.0-pad_ranf())*(qz(nz-1)-qz(1))
!
!            m_arr(l) = mion
!            mrat(l) = 1.0
!            beta_p(l) = beta_pu

!            i=0
! 71         continue
!            i = i + 1
!            if (xp(l,1) .gt. qx(i)) go to 71 !find i on non-uniform 
!            i = i-1
!            ijkp(l,1)= i


!            ijkp(l,2) = floor(xp(l,2)/dy) 
            
!            k=0
! 70         continue
!            k = k + 1
!            if (xp(l,3) .gt. qz(k)) go to 70 !find k on non-uniform 
!            k = k-1
!            ijkp(l,3)= k

!            theta = pad_ranf()*2*PI
            
!            vp(l,1) = vsw+vsw*cos(theta) !+dvx
!            vp(l,2) = vsw*sin(theta) !+dvz 
!            vp(l,3) = 0.0

!            if (xp(l,3) .gt. qz(nz/2)) mix_ind(l) = 1
!            if (xp(l,3) .le. qz(nz/2)) mix_ind(l) = 0
            
!            do  m=1,3
!               vp1(l,m) = vp(l,m)
!               input_E = input_E + 
!     x              0.5*m_arr(l)*(vp(l,m)*km_to_m)**2 /(beta*beta_p(l))
!               input_p(m) = input_p(m) + m_arr(l)*vp(l,m)/(beta*beta_p(l))
!            enddo
            

! 69      enddo
            call get_interp_weights()
            call update_np()
            call update_up(vp)
            
      end subroutine sw_part_setup_maxwl_mix
      

end module part_init
