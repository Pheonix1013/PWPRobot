from tkinter import *

def open_robot_gui(username):
    robot_window = Toplevel()
    robot_window.title(f"Robot GUI - signed into: {username}")
    robot_window.geometry("1920x1080")

    #main grid
    for r in range(2):
        robot_window.grid_rowconfigure(r, weight=1)
        robot_window.grid_columnconfigure(r, weight=1)

    # make the lines
    videoStream1 = Frame(robot_window, bd=2, relief="groove")
    controller = Frame(robot_window, bd=2, relief="groove")
    videoStream2 = Frame(robot_window, bd=2, relief="groove")
    userLog = Frame(robot_window, bd=2, relief="groove")

    videoStream1.grid(row=0, column=0, sticky="nsew")
    controller.grid(row=0, column=1, sticky="nsew")
    videoStream2.grid(row=1, column=0, sticky="nsew")
    userLog.grid(row=1, column=1, sticky="nsew")

    Label(videoStream1, text="Video Stream 1", font=("Arial", 16)).pack(pady=5)
    Label(controller, text="Controller", font=("Arial", 16)).pack(pady=5)
    Label(videoStream2, text="Video Stream 2", font=("Arial", 16)).pack(pady=5)
    Label(userLog, text="User Log", font=("Arial", 16)).pack(pady=5)

    controlPanel = Frame(controller)
    controlPanel.place(relx=0.5, rely=0.5, anchor="center")
    arrow_font = ("Arial", 28)

    upBtn    = Button(controlPanel, text="↑", font=arrow_font)
    leftBtn  = Button(controlPanel, text="←", font=arrow_font)
    playBtn  = Button(controlPanel, text="Play", font=arrow_font)
    rightBtn = Button(controlPanel, text="→", font=arrow_font)
    stopBtn  = Button(controlPanel, text="Stop", font=arrow_font)
    downBtn  = Button(controlPanel, text="↓", font=arrow_font)

    upBtn.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    leftBtn.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
    playBtn.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
    rightBtn.grid(row=1, column=2, sticky="nsew", padx=2, pady=2)
    stopBtn.grid(row=2, column=1, sticky="nsew", padx=2, pady=2)
    downBtn.grid(row=3, column=1, sticky="nsew", padx=2, pady=2)

    #put the buttons in a grid
    for i in range(4):
        controlPanel.grid_rowconfigure(i, weight=1)
    for i in range(3):
        controlPanel.grid_columnconfigure(i, weight=1)
