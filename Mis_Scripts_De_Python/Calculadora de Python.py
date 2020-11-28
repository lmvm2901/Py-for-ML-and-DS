# Esta es una calculadora, sirve de tutorial para mi
import tkinter as tk

expresion = ""

# Funciones de la calculadora


def press(num):
    # Presionando alguno de los botones
    global expresion
    expresion = expresion + str(num)
    enpantalla.set(expresion)


def limpiar():
    # Limpiando la pantalla
    global expresion
    expresion = ""
    enpantalla.set("")


def calculartodo():
    # Haciendo el calculo final y manteniendo el resultado para otros calculos
    global expresion
    expresion = str(eval(expresion))
    enpantalla.set(expresion)


# Configuracion de la pantalla
root = tk.Tk()
root.geometry("400x400")
root.title("Mi Simple Calculadora")
root.configure(background="blue")
frame = tk.Frame(root)
frame.pack()


# Lo que se ve en pantalla
enpantalla = tk.StringVar()
Pantalla = tk.Entry(frame, textvariable=enpantalla)
Pantalla.grid(row=6, columnspan=3, ipadx=70)

# Todos los botones de la calculadora
button0 = tk.Button(frame, text="0", command=lambda: press(0), height=1, width=7)
button0.grid(row=3, column=0)

button1 = tk.Button(frame, text="1", command=lambda: press(1), height=1, width=7)
button1.grid(row=0, column=0)

button2 = tk.Button(frame, text="2", command=lambda: press(2), height=1, width=7)
button2.grid(row=0, column=1)

button3 = tk.Button(frame, text="3", command=lambda: press(3), height=1, width=7)
button3.grid(row=0, column=2)

button4 = tk.Button(frame, text="4", command=lambda: press(4), height=1, width=7)
button4.grid(row=1, column=0)

button5 = tk.Button(frame, text="5", command=lambda: press(5), height=1, width=7)
button5.grid(row=1, column=1)

button6 = tk.Button(frame, text="6", command=lambda: press(6), height=1, width=7)
button6.grid(row=1, column=2)

button7 = tk.Button(frame, text="7", command=lambda: press(7), height=1, width=7)
button7.grid(row=2, column=0)

button8 = tk.Button(frame, text="8", command=lambda: press(8), height=1, width=7)
button8.grid(row=2, column=1)

button9 = tk.Button(frame, text="9", command=lambda: press(9), height=1, width=7)
button9.grid(row=2, column=2)

btnclear = tk.Button(frame, text="clear", command=lambda: limpiar(), height=1, width=7)
btnclear.grid(row=5, columnspan=3)

btnsumar = tk.Button(frame, text="+", command=lambda: press("+"), height=1, width=7)
btnsumar.grid(row=3, column=2)

btnrestar = tk.Button(frame, text="-", command=lambda: press("-"), height=1, width=7)
btnrestar.grid(row=4, column=0)

btnmulti = tk.Button(frame, text="*", command=lambda: press("*"), height=1, width=7)
btnmulti.grid(row=4, column=1)

btndivis = tk.Button(frame, text="/", command=lambda: press("/"), height=1, width=7)
btndivis.grid(row=4, column=2)

btnigual = tk.Button(frame, text="=", command=lambda: calculartodo(), height=1, width=7)
btnigual.grid(row=3, column=1)

# Corriendo el programa para que no se cierre
root.mainloop()
