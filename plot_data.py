from matplotlib import rc
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

try:
    rc('text', usetex=True)
    label_a = '$\\it{A}$'
    label_b = '$\\it{B}$'
    label_c = '$\\it{C}$'
    label_x = '$\\it{x}$'
    label_v = '$\\it{v}$'
    label_t = '$\\it{\\tau/2\\pi}$'
except RuntimeError:
    print("Warning! LaTeX not found!")
    label_a = 'A'
    label_b = 'B'
    label_c = 'C'
    label_x = 'x'
    label_v = 'v'
    label_t = 't'


def model_plot(iter_nums, a, b, c, graph_width=800, graph_height=600):
    plt.close('all')
    f = plt.figure(figsize=(graph_width / 100.0, graph_height / 100.0), dpi=100)
    axes = f.add_subplot(111)

    axes.plot(iter_nums, a, label=label_a)
    axes.plot(iter_nums, b, label=label_b)
    axes.plot(iter_nums, c, label=label_c)

    axes.set_xlabel('Iteration number', fontsize=20)
    axes.set_ylabel('Parameter value', fontsize=20)
    plt.legend()
    plt.show()


def plot_data(velocity, vel0):
    plt.close('all')
    f = plt.figure(figsize=(8.0, 6.0), dpi=100)
    axes = f.add_subplot(111)
    axes.plot(velocity.index, velocity, "b")
    axes.plot(vel0.index, vel0, 'gD')
    axes.set_xlabel(label_x, fontsize=20)
    axes.set_ylabel(label_v, fontsize=20)
    plt.show()


def get_plot_data(fit_instance):
    velocity = fit_instance.feed()
    velocity_0 = fit_instance.data
    return velocity, velocity_0


def phase_plot(v_t, v_t0, t):
    plt.close('all')
    f = plt.figure(figsize=(8.0, 6.0), dpi=100)
    axes = f.add_subplot(111)
    axes.plot(t, v_t.iloc[0])
    axes.plot(t, v_t0.iloc[0], 'D')
    axes.set_xlabel(label_t, fontsize=20)
    axes.set_ylabel(label_v, fontsize=20)
    axes.set_title('Check phase')
    plt.show()
