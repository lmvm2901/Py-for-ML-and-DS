# Esta es una calculadora, sirve de tutorial para mi
import tkinter as tk


def write_slogan():
    print("Tkinter is easy to use!")


root = tk.Tk()
root.geometry("400x400")
frame = tk.Frame(root)
frame.pack()


button = tk.Button(frame,
                   text="QUIT",
                   fg="red",
                   command=quit,
                   )
button.grid(row=2, column=0)


slogan = tk.Button(frame,
                   text="Hello",
                   command=write_slogan)
slogan.grid(row=2, column=1)

root.mainloop()
