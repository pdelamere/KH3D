x = findgen(200)/2

  k1 = 2*!pi/20
  k2 = 2*!pi/10.
  k3 = 2*!pi/20.
  k4 = 2*!pi/50.
  
  b = 1e-10*cos(k1*x); + cos(k2*x) + cos(k3*x) + cos(k4*x)

  !p.multi=[0,1,2]
  plot,x,b
  print,total(b^2)/n_elements(b)
  

  ;f =fft_powerspectrum(b,10,freq=k)

  ff = fft(b,-1)
  !p.charsize=2.0
  plot,k,abs(ff)
  ;plot,x,fft(ff,1),linestyle=2
  
  end
