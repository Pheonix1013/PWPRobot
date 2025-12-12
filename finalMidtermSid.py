#importing nesesary fucntions
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Line detection variables for tuning in testing
FRAME_W, FRAME_H = 640, 480
BLUR_K = 9; TH_BLOCK = 51; TH_C = 7; MORPH_K = 7
MIN_ARCLEN = 150; MIN_AREA = 200
NUM_SAMPLES = 60; SMOOTH_WIN = 7
CROP_X,CROP_Y,CROP_W,CROP_H = 160,120,320,240
LINE_THICK = 3

#function for getting line of best fit
def pleaseWork(contour_pts):
    pts = contour_pts.reshape(-1,2).astype(float)
    xs = pts[:,0]; ys=pts[:,1]; best=("v",0,0,1e12)
    try:
        m_v,b_v=np.polyfit(ys,xs,1)
        err_v=np.mean((xs-(m_v*ys+b_v))**2)
        best=("v",m_v,b_v,err_v)
    except: pass
    try:
        m_h,b_h=np.polyfit(xs,ys,1)
        err_h=np.mean((ys-(m_h*xs+b_h))**2)
        if err_h<best[3]: best=("h",m_h,b_h,err_h)
    except: pass
    return best

def checkIfMoving(a,n): return np.convolve(a,np.ones(n)/n,mode='same') if len(a)>=n else a
def extendLines(mode,m,b,w,h):
    if mode=="v": return (int(round(m*0+b)),0),(int(round(m*(h-1)+b)),h-1)
    else: return (0,int(round(m*0+b))),(w-1,int(round(m*(w-1)+b)))

#getting cam feed
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

line_frame = None
raw_frame = None

def camera_loop():
    global line_frame, raw_frame
    while True:
        ret, frame = cap.read()
        if not ret: continue
        raw_copy = frame.copy()

        #line detection code
        x2,y2=CROP_X+CROP_W,CROP_Y+CROP_H
        cropped = frame[CROP_Y:y2,CROP_X:x2].copy()
        h,w=cropped.shape[:2]
        
        #code for grayscale, blurring, and cropping
        gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(BLUR_K,BLUR_K),0)
        mask=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,TH_BLOCK,TH_C)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_K,MORPH_K))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)

        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        good=[c for c in contours if cv2.arcLength(c,False)>=MIN_ARCLEN and cv2.contourArea(c)>=MIN_AREA]
        good=sorted(good,key=cv2.contourArea,reverse=True)[:2]

        #checking if the length of the line is enough to get all the midpoints
        if len(good)>=2:
            cntA,cntB=good[0],good[1]
            mean_x=lambda c:c.reshape(-1,2)[:,0].mean()
            left_cnt,right_cnt=(cntA,cntB) if mean_x(cntA)<=mean_x(cntB) else (cntB,cntA)
            modeL,mL,bL,_=pleaseWork(left_cnt)
            modeR,mR,bR,_=pleaseWork(right_cnt)
            cv2.drawContours(cropped,[left_cnt],-1,(0,0,255),2)
            cv2.drawContours(cropped,[right_cnt],-1,(255,0,0),2)

            #actually getting midpoint coordinates
            midpoints=[]
            if modeL=="v" and modeR=="v":
                ys=np.linspace(0,h-1,NUM_SAMPLES)
                for yy in ys: midpoints.append(((mL*yy+bL+mR*yy+bR)/2,yy))
            if len(midpoints)>=3:
                pts=np.array(midpoints)
                xs=checkIfMoving(pts[:,0],SMOOTH_WIN)
                ys=checkIfMoving(pts[:,1],SMOOTH_WIN)
                mid_s=np.vstack((xs,ys)).T
                try: mcv,bcv=np.polyfit(mid_s[:,1],mid_s[:,0],1)
                except: mcv,bcv=0,0
                p_top=(int(round(mcv*0+bcv)),0)
                p_bot=(int(round(mcv*(h-1)+bcv)),h-1)
                cv2.line(cropped,p_top,p_bot,(0,255,0),LINE_THICK)

        line_frame = cropped

        # Draw red box on the region of interest in the raw frame
        cv2.rectangle(raw_copy,(CROP_X,CROP_Y),(CROP_X+CROP_W,CROP_Y+CROP_H),(0,0,255),2)
        raw_frame = raw_copy

threading.Thread(target=camera_loop, daemon=True).start()

# tkinter code
root = tk.Tk()
root.title("Robot GUI")

canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

#self explanatory - every 0.3 seconds, refresh gui with new frame and new lines on the frame
def update_gui():
    if line_frame is not None and raw_frame is not None:
        # resize frames
        line_img = cv2.cvtColor(line_frame, cv2.COLOR_BGR2RGB)
        raw_img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        line_img = cv2.resize(line_img,(canvas_width//2,canvas_height//2))
        raw_img = cv2.resize(raw_img,(canvas_width//2,canvas_height//2))

        # convert to ImageTk
        line_pil = ImageTk.PhotoImage(Image.fromarray(line_img))
        raw_pil = ImageTk.PhotoImage(Image.fromarray(raw_img))

        # draw top-left
        canvas.create_image(0,0,anchor="nw",image=line_pil)
        # draw bottom-left
        canvas.create_image(0,canvas_height//2,anchor="nw",image=raw_pil)

        # draw top-right buttons (nonfunctional)
        canvas.create_rectangle(canvas_width//2,0,canvas_width,canvas_height//2,fill="#ccc")
        canvas.create_text(canvas_width*3//4,50,text="↑",font=("Arial",24))
        canvas.create_text(canvas_width*3//4,100,text="←  ■  →",font=("Arial",24))
        canvas.create_text(canvas_width*3//4,150,text="↓",font=("Arial",24))

        # draw bottom-right console
        canvas.create_rectangle(canvas_width//2,canvas_height//2,canvas_width,canvas_height,fill="#eee")
        canvas.create_text(canvas_width*3//4,canvas_height*3//4,text="Line detection running...",font=("Arial",16))

        # keep references
        canvas.image_line = line_pil
        canvas.image_raw = raw_pil

    root.after(30,update_gui)

update_gui()
root.mainloop()
