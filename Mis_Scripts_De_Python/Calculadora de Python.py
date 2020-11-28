# Esta es una calculadora, sirve de tutorial para mi
import tkinter as tk

expresion = ""


def press(num):
    global expresion
    print(num)
    expresion = expresion + str(num)
    print("La expresion es:" + expresion)


def limpiar():
    global expresion
    expresion = ""
    print(expresion)


def calculartodo():
    global expresion
    print("La solución es:" + str(eval(expresion)))
    return eval(expresion)


root = tk.Tk()
root.geometry("400x400")
frame = tk.Frame(root)
frame.pack()

button0 = tk.Button(frame, text="0", command=lambda: press(0), height=1, width=7)
button0.grid(row=5, column=0)

button1 = tk.Button(frame, text="1", fg="red", command=lambda: press(1), height=1, width=7)
button1.grid(row=2, column=0)

button2 = tk.Button(frame, text="2", command=lambda: press(2), height=1, width=7)
button2.grid(row=2, column=1)

button3 = tk.Button(frame, text="3", command=lambda: press(3), height=1, width=7)
button3.grid(row=2, column=2)

button4 = tk.Button(frame, text="4", command=lambda: press(4), height=1, width=7)
button4.grid(row=3, column=0)

button5 = tk.Button(frame, text="5", command=lambda: press(5), height=1, width=7)
button5.grid(row=3, column=1)

button6 = tk.Button(frame, text="6", command=lambda: press(6), height=1, width=7)
button6.grid(row=3, column=2)

button7 = tk.Button(frame, text="7", command=lambda: press(7), height=1, width=7)
button7.grid(row=4, column=0)

button8 = tk.Button(frame, text="8", command=lambda: press(8), height=1, width=7)
button8.grid(row=4, column=1)

button9 = tk.Button(frame, text="9", command=lambda: press(9), height=1, width=7)
button9.grid(row=4, column=2)

btnclear = tk.Button(frame, text="clear", command=lambda: limpiar(), height=1, width=7)
btnclear.grid(row=5, column=1)

btnsumar = tk.Button(frame, text="+", command=lambda: press("+"), height=1, width=7)
btnsumar.grid(row=2, column=3)

btnrestar = tk.Button(frame, text="-", command=lambda: press("-"), height=1, width=7)
btnrestar.grid(row=3, column=3)

btnmulti = tk.Button(frame, text="*", command=lambda: press("*"), height=1, width=7)
btnmulti.grid(row=4, column=3)

btndivis = tk.Button(frame, text="/", command=lambda: press("/"), height=1, width=7)
btndivis.grid(row=5, column=3)

btnigual = tk.Button(frame, text="ig", command=lambda: press("**"), height=1, width=7)
btnrestar.grid(row=5, column=2)

root.mainloop()
