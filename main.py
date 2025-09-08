from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


root = Tk()
root.title("Robot GUI")
root.geometry("1920x1080")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)

videoStream1 = Frame(root, bd=2, relief="groove")
controller = Frame(root, bd=2, relief="groove")
videoStream2 = Frame(root, bd=2, relief="groove")
userLog = Frame(root, bd=2, relief="groove")

videoStream1.grid(row = 0, column = 0, sticky = "nsew")
controller.grid(row = 0, column = 1, sticky = "nsew")
videoStream2.grid(row = 1, column = 0, sticky = "nsew")
userLog.grid(row = 1, column = 1, sticky = "nsew")

Label(videoStream1, text="Video Stream 1", font=("Arial", 16)).pack(side="top", anchor="n", pady=5)
Label(controller, text="Controller", font=("Arial", 16)).pack(side="top", anchor="n", pady=5)
Label(videoStream2, text="Video Stream 2", font=("Arial", 16)).pack(side="top", anchor="n", pady=5)
Label(userLog, text="User Log", font=("Arial", 16)).pack(side="top", anchor="n", pady=5)

junkArrowImage = Image.open(r"C:\Users\siddh\OneDrive\Documents\Siddharth data\Sid's epic stuff\arrow_no_bg.png")

resizedArrow = junkArrowImage.resize((200,300))
resizedArrow90 = resizedArrow.rotate(90)
resizedArrow180 = resizedArrow.rotate(180)
resizedArrow270 = resizedArrow.rotate(270)


arrowImage = ImageTk.PhotoImage(resizedArrow)
arrowImage90 = ImageTk.PhotoImage(resizedArrow90)
arrowImage180 = ImageTk.PhotoImage(resizedArrow180)
arrowImage270 = ImageTk.PhotoImage(resizedArrow270)



Label(controller, image=arrowImage).pack(side="right", ipadx=0, pady=0)
Label(controller, image=arrowImage90).pack(side="top", padx=0, pady=0)
Label(controller, image=arrowImage180).pack(side="left", padx=0, pady=0)
Label(controller, image=arrowImage270).pack(side="bottom", padx=0, pady=0)

playButton = Button(controller, text="Play", command=None)
playButton.pack(anchor="center", padx=0, pady=0)
stopButton = Button(controller, text="Stop", command=None)
stopButton.pack(anchor="center", padx=0, pady=0)

root.mainloop()