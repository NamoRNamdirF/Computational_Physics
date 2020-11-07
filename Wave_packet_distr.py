import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

plt.subplots_adjust(left=0.12, bottom=0.35)

k0 = 7  # номер центра волны гауссовского пакета
sigma = 1  # коэффициент распределения Гаусса
srx = 50  # положения по x

x = np.linspace(-srx / 3, 2 * srx / 3, srx * 20)
x2 = np.linspace(0, 20, 1000)

alpha = 1  ## alpha = hbar/m => v = alpha*k
is_color = False


def psi(t): # расчет волновой функции по времени
    global x
    v = alpha * k0
    omega = (k0 ** 2) * alpha / 2
    probCompl = (np.exp(-(1j)*(omega*t-k0*x)))*(np.exp(-((x - v) ** 2 / (2))) * (1 / (np.sqrt((np.sqrt(np.pi))))))
    return probCompl


ax_a = plt.axes([0.1, 0.05, 0.8, 0.03])
a_slider = Slider(ax_a, '$t$', 0, 10, valinit=0)  #Слайдер времен
a_slider.label.set_size(20)

ax_b = plt.axes([0.1, 0.15, 0.8, 0.03])
b_slider = Slider(ax_b, '$sigma$', 0.01, 5, valinit=sigma)  #слайдер а
b_slider.label.set_size(20)

ax_c = plt.axes([0.1, 0.25, 0.8, 0.03])
c_slider = Slider(ax_c, '$k_0$', 1, 15, valinit=k0)  # слайдер k0
c_slider.label.set_size(20)

rax = plt.axes([0.01, 0.45, 0.08, 0.1])  # кнопка управления цветом
check = CheckButtons(rax, ['Col'], [False])

def update_phase(val_):
    global a
    a = b_slider.val
    y = np.exp(- (a ** 2) * (k0 - x2) ** 2)  # распределение частот
    ax1.clear()
    ax1.set_title('LABA')
    ax1.plot(x2, y)

    fig.canvas.draw_idle()
    update_temps(0)

def update_temps(val_):  # отрисовка волновой функции

    probCompl = psi(a_slider.val)
    ax2.clear()
    ax2.set_xlim([-srx / 3, 2 * srx / 3])
    ax2.set_ylim([-4, 4])
    ax2.set_title('')

    if (is_color):  # цветной дисплей
        X = np.array([x, x])
        y0 = np.zeros(len(x))
        y = [abs(i) for i in probCompl]
        Y = np.array([y0, y])
        Z = np.array([probCompl, probCompl])
        C = np.angle(Z)
        ax2.pcolormesh(X, Y, C, cmap=cm.hsv, vmin=-np.pi, vmax=np.pi)
        ax2.plot(x, np.abs(probCompl), label='$|\psi|$', color='black')

    else:  # отображение реальных и мнимых частей
        ax2.plot(x, np.real(probCompl), label='$\operatorname{Re}(\psi)$')
        ax2.plot(x, np.imag(probCompl), label='$\operatorname{Im}(\psi)$')
        ax2.plot(x, np.absolute(probCompl) ** 2, label='$\psi \psi^{\dag} $')

    ax2.legend(fontsize=15)
    fig.canvas.draw_idle()

def update_k(val_):  # изменение k0
    global k0
    k0 = c_slider.val
    update_phase(0)
    update_temps(0)

def on_check(label):  # нажатие на кнопку
    global is_color
    is_color = not is_color
    update_temps(0)

update_phase(0)  

a_slider.on_changed(update_temps)
b_slider.on_changed(update_phase)
c_slider.on_changed(update_k)

check.on_clicked(on_check)

plt.show()

# Проверка нормировки
def fk(k, k0, s):
    return (abs(np.exp(-((k - k0) ** 2 / (2 * s ** 2))) * (1 / (np.sqrt(s * (np.sqrt(np.pi))))))) ** 2

from scipy.integrate import quad

p = quad(fk, -20, 20, args=(1, 5))
print(p) 

