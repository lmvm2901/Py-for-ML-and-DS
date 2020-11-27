# Esta es una calculadora, sirve de tutorial para mi
import tkinter as tk


def write_slogan():
    print("Tkinter is easy to use!")


root = tk.Tk()
root.geometry("400x400")
frame = tk.Frame(root)
frame.pack()

button1 = tk.Button(frame,
                    text="1",
                    fg="red",
                    command=quit,
                    )
button1.grid(row=2, column=0)

button2 = tk.Button(frame,
                    text="2",
                    command=write_slogan)
button2.grid(row=3, column=1)





root.mainloop()
