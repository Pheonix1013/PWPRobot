#import all libraries and other files
from tkinter import *
from tkinter import messagebox
import database
import robot_gui_window

#makes the login root
root = Tk()
root.title("Login / Signup")
root.geometry("1920x1080")


#function stuff
def makeAccount():
    """
    Make account display function
    
    Parameters:
    None
    
    Return:
    None
    """
    
    username = entry_username.get().strip()
    password = entry_password.get().strip()

    if not username or not password:
        messagebox.showwarning("Error", "Enter both username and password.")
        return

    if database.add_user(username, password):
        messagebox.showinfo("Success", f"Account created for {username}.")
    else:
        messagebox.showerror("Error", "Username already exists.")


def login():
    """
    Make login display
    
    Parameters:
    None
    
    Return:
    None
    """
    username = entry_username.get().strip()
    password = entry_password.get().strip()

    if database.authenticate(username, password):
        robot_gui_window.open_robot_gui(username)
        root.withdraw()
    else:
        retry = messagebox.askyesno("Failed", "Username or password not found. Create new account?")
        if retry:
            makeAccount()

#Making the text feilds
Label(root, text="Username:").pack(pady=(20, 5))
entry_username = Entry(root)
entry_username.pack(pady=5)

Label(root, text="Password:").pack(pady=5)
entry_password = Entry(root, show="*")
entry_password.pack(pady=5)

Button(root, text="Login", width=10, command=login).pack(pady=10)
Button(root, text="Signup", width=10, command=makeAccount).pack(pady=5)

#Contstantly updates the screen
root.mainloop()
