from tkinter import *

root = Tk()
root.title("Robot GUI")
root.geometry("1200x800")

#Whole root grid
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

#Frames
videoStream1 = Frame(root, bd=2, relief="groove")
controller = Frame(root, bd=2, relief="groove")
videoStream2 = Frame(root, bd=2, relief="groove")
userLog = Frame(root, bd=2, relief="groove")

videoStream1.grid(row=0, column=0, sticky="nsew")
controller.grid(row=0, column=1, sticky="nsew")
videoStream2.grid(row=1, column=0, sticky="nsew")
userLog.grid(row=1, column=1, sticky="nsew")

#Video Stram 1
videoStream1.grid_rowconfigure(1, weight=1)
videoStream1.grid_columnconfigure(0, weight=1)
Label(videoStream1, text="Video Stream 1", font=("Arial", 16)).grid(row=0, column=0, pady=5, sticky="n")

#Controller
controller.grid_rowconfigure(1, weight=1)
controller.grid_columnconfigure(0, weight=1)
Label(controller, text="Controller", font=("Arial", 16)).grid(row=0, column=0, pady=5, sticky="n")
controlPanel = Frame(controller)
controlPanel.grid(row=1, column=0)

arrow_font = ("Arial", 32)
btn_font   = ("Arial", 14)

Button(controlPanel, text="↑", font=arrow_font, width=3, height=1).grid(row=0, column=1, pady=5)
Button(controlPanel, text="←", font=arrow_font, width=3, height=1).grid(row=1, column=0, padx=5)
Button(controlPanel, text="Play", font=btn_font, width=6, height=2).grid(row=1, column=1, padx=5, pady=5)
Button(controlPanel, text="→", font=arrow_font, width=3, height=1).grid(row=1, column=2, padx=5)
Button(controlPanel, text="Stop", font=btn_font, width=6, height=2).grid(row=2, column=1, pady=5)
Button(controlPanel, text="↓", font=arrow_font, width=3, height=1).grid(row=3, column=1, pady=5)

#Video Stream 2
videoStream2.grid_rowconfigure(1, weight=1)
videoStream2.grid_columnconfigure(0, weight=1)

Label(videoStream2, text="Video Stream 2", font=("Arial", 16)).grid(row=0, column=0, pady=5, sticky="n")

#User log
userLog.grid_rowconfigure(1, weight=1)
userLog.grid_columnconfigure(0, weight=1)
Label(userLog, text="User Log", font=("Arial", 16)).grid(row=0, column=0, pady=5, sticky="n")



root.mainloop()
