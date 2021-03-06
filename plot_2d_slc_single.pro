pro plot_image,img,x,y,sclf,loc,tit

   ctbl = colortable(15,/reverse)  
   im = image(img(*,*),x,y,/current,rgb_table=ctbl,layout=[3,1,loc],font_size=14,aspect_ratio=1.0)

   xax = axis('x',axis_range=[min(x),max(x)],location=[0,min(y(*))],thick=1,tickdir=1,target=im,tickfont_size=14)
   yax = axis('y',axis_range=[min(y),max(y)],location=[0,0],thick=1,tickdir=1,target=im,tickfont_size=14)

   im.xtitle='$x (c/\omega_{pi})$'
   im.ytitle='$y (c/\omega_{pi})$'

;   ct = colorbar(target = im,title=tit,orientation=1,textpos=1,font_size=12);,$
;                 position=[max(x), min(y),
;                 0.05*(max(x)-min(x)),max(y)],/data)
   im.scale,sclf,sclf
   ct = colorbar(target = im,title=tit,textpos=1,font_size=14,orientation=1,$
                 position=[1.04,0,1.11,1.0],/relative)

   

end


pro plot_2d_slc,nf,slc,pln
  w = window()
  @clr_win
;1 = xy
;2 = yz
;3 = xz


  sclf = 0.8
  xsz = 1100.
  ysz = 600.
  file = 'ion_cyclotron.mp4'
  width = xsz
  height = ysz
  frames = 180
  fps = 30
  speed = 2
  samplerate = 22050L
 
device,decomposed=0
loadct,27

;dir = '/Volumes/Scratch/hybrid/KH_new/run_3d_24/'
;dir = '/Volumes/Scratch/hybrid/KH3D/run_1_periodic/'
;dir = './run_36/'
dir = './run_va_0.8_beta_3/'

nrestart = ''

nframe=nf
read_para,dir
restore,filename=dir+'para.sav'
read_coords,dir,x,y,z
@get_const
;stop

;XINTERANIMATE, SET=[xsz,ysz, nframe], /SHOWLOAD 
w = window(dimensions=[xsz,ysz])   

;video_file = 'KAW.mp4'
;video = idlffvideowrite(video_file)
;framerate = 7.5
;;wdims = w.window.dimensions
;stream = video.addvideostream(xsz, ysz, framerate)

;openr,1,'./tmp1/c.test_part.dat',/f77_unformatted
;xpart = fltarr(3,3)
;readu,1,xpart
;xp = xpart
;while not(eof(1)) do begin
;   readu,1,xpart
;   xp = [xp,xpart]
;endwhile
;close,1

case pln of
   1: begin
      x = x 
      y = y
      dy = dy
   end
   2: begin
      x = y
      y = z
      dy = delz
   end
   3: begin 
      x = x 
      y = z
      dy = delz
   end
endcase

wpi = sqrt(q*q*(np_top/1e9)/(epo*mproton))
coverwpi = 3e8/wpi/1e3
print,coverwpi
x = x/coverwpi
y = y/(coverwpi)
z = z/coverwpi


;for nfrm = 1,nframe,1 do begin
nfrm = nframe

   c_read_3d_vec_m_32,dir,'c.b1'+nrestart,nfrm,b1
   c_read_3d_vec_m_32,dir,'c.up'+nrestart,nfrm,up
   c_read_3d_m_32,dir,'c.np'+nrestart,nfrm,np
   c_read_3d_m_32,dir,'c.mixed'+nrestart,nfrm,mixed
   c_read_3d_m_32,dir,'c.temp_p'+nrestart,nfrm,tp

   comegapi = (z(1)-z(0))*2

;   bx = (reform(b1(*,*,1,1))*16*mp/q)/1e-9
;   b1(0,0,0,0) = 120.
;   print,max(bx)

   w.erase

   case pln of
      1: begin ;xy
         
         bx = (reform(b1(*,*,slc,0))*mp/q)/1e-9
         by = (reform(b1(*,*,slc,1))*mp/q)/1e-9
         bz = (reform(b1(*,*,slc,2))*mp/q)/1e-9
         
         nparr = reform(np(*,*,slc))/1e15
         upx = reform(up(*,*,slc,0))
         upy = reform(up(*,*,slc,1))
         upz = reform(up(*,*,slc,2))
         mixarr = reform(mixed(*,*,slc))
         tparr = reform(tp(*,*,slc))
         
      end
      2: begin ;yz
         
         bx = (reform(b1(slc,*,*,0))*mp/q)/1e-9
         by = (reform(b1(slc,*,*,1))*mp/q)/1e-9
         bz = (reform(b1(slc,*,*,2))*mp/q)/1e-9
         
         nparr = reform(np(slc,*,*))/1e15
         upx = reform(up(slc,*,*,0))
         upy = reform(up(slc,*,*,1))
         upz = reform(up(slc,*,*,2))
         mixarr = reform(mixed(slc,*,*))
         tparr = reform(tp(slc,*,*))
         
      end
      3: begin ;xz
         
         bx = (reform(b1(*,slc,*,0))*mp/q)/1e-9
         by = (reform(b1(*,slc,*,1))*mp/q)/1e-9
         bz = (reform(b1(*,slc,*,2))*mp/q)/1e-9
         
         nparr = reform(np(*,slc,*))/1e15
         upx = reform(up(*,slc,*,0))
         upy = reform(up(*,slc,*,1))
         upz = reform(up(*,slc,*,2))
         mixarr = reform(mixed(*,slc,*))
         tparr = reform(tp(*,slc,*))
         
      end
   endcase


   plot_image,bx,x,y,sclf,1,'Bx (nT)'

;   plot_image,by,x,y,sclf,2,'By (nT)'

;   plot_image,bz,x,y,sclf,3,'Bz (nT)'

   plot_image,upx,x,y,sclf,2,'ux (km/s)'

;   plot_image,upy,x,y,sclf,5,'uy (km/s)'      

;   plot_image,upz,x,y,sclf,6,'uz (km/s)'

;   plot_image,tparr,x,y,sclf,7,'Tp (eV)'

;   plot_image,nparr,x,y,sclf,3,'np (cm$^{-3}$)'

   wh = where(mixarr gt 1.0)
   if(wh(0) gt -1.0) then begin
      mixarr(wh) = 1.0
   endif

   plot_image,mixarr,x,y,sclf,3,'mixing'

;   img = w.CopyWindow()

;   print, 'Time:', video.put(stream, w.copywindow())
;   xinteranimate, frame = nfrm-1, image = img


;endfor

;video.cleanup
;xinteranimate,/keep_pixmaps

return
end
