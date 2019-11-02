from tkinter import filedialog, Tk
from pickle import load, dump
from numpy import linspace, pi, exp, outer, sum, array
from pandas import read_csv, DataFrame, Series
from plot_data import *
from os.path import exists


def load_and_pivot_data(f_name):
    """
    Load text data file obtained by the Comsol.
    File must contain two columns 'x' and 'v' and 8 lines of the header
    :param f_name: String: the name of the data file
    :return: pandas.DataFrame(): A table with data.
    """
    data = read_csv(f_name, skiprows=8, header=None, delim_whitespace=True).rename(columns={0: 'x', 1: 'v'})
    data['t'] = (data['x'] == data['x'].iloc[0]).cumsum() - 1
    time = linspace(0, 2 * pi, max(data['t']) + 1)
    columns = {i: t for i, t in enumerate(time)}
    return data.pivot(index='x', columns='t', values='v'), columns


def m_ac(a, c):
    """
    A coefficient in further computations
    :param a: real number
    :param c: real number
    :return: complex number
    """
    return a * exp(-1j * c)


def v_1(x):
    """
    The function of simple viscous wave
    :param x: numpy.array(): Coordinate x
    :return: numpy.array(): math function
    """
    return exp((1j - 1) * x)


def func(x, a, b, c):
    """
    Our functions without a dependence on time
    :param x: numpy.array(): Coordinate x
    :param a: real number
    :param b: real number
    :param c: real number
    :return: numpy.array(): complex math function
    """
    return exp(-b * x) * ((1 - m_ac(a, c)) * v_1(x) + m_ac(a, c))


def sine_law(t, t2=-pi / 2):
    """
    The dependence on time for our functions
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param t2: real number: a phase shift
    :return: numpy.array(): complex math function
    """
    return exp(- 1j * (t + t2))


def multiply(t, function, *args, t2=-pi / 2):  # -pi / 2 ATTENTION!
    """
    A dot product of a function with a dependence on x and the function with the dependence on t
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param function: numpy.array(): Any math function that is dependns only on x
    :param args: tuple: arguments of the math function
    :param t2: real number: a phase shift
    :return: numpy.array() of shape len(x) x len(t) with real(!) numbers
    """
    return outer(function(*args), sine_law(t, t2)).real


def func_t(x, t, t2, a, b, c):
    """
    The math function for the approximation
    :param x: numpy.array(): Coordinate x
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param t2: real number: a phase shift
    :param a: real number
    :param b: real number
    :param c: real number
    :return: numpy.array() of shape len(x) x len(t) with real(!) numbers
    """
    return multiply(t, func, x, a, b, c, t2=t2)


def der_b(x, *args):
    """
    Ancillary math function for derivative computation
    :param x: numpy.array(): Coordinate x
    :param args: other args
    :return: numpy.array()
    """
    return -x * func(x, *args)


def derivative_b(x, t, t2, a, b, c):
    """
    Derivative d(func_t)/d(b)
    :param x: numpy.array(): Coordinate x
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param t2: real number: a phase shift
    :param a: real number
    :param b: real number
    :param c: real number
    :return: numpy.array() of shape len(x) x len(t) with real(!) numbers
    """
    return multiply(t, der_b, x, a, b, c, t2=t2)


def derivative_m_ac(x, b):
    """
    Ancillary math function for derivative computation
    :param x: numpy.array(): Coordinate x
    :param b: real number
    :return: numpy.array()
    """
    return exp(-b * x) * (1 - v_1(x))


def der_a(x, b, c):
    """
    Ancillary math function for derivative computation
    :param x: numpy.array(): Coordinate x
    :param b: real number
    :param c: real number
    :return: numpy.array()
    """
    return derivative_m_ac(x, b) * exp(-1j * c)


def derivative_a(x, t, t2, b, c):
    """
    Derivative d(func_t)/d(a)
    :param x: numpy.array(): Coordinate x
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param t2: real number: a phase shift
    :param b: real number
    :param c: real number
    :return: numpy.array() of shape len(x) x len(t) with real(!) numbers
    """
    return multiply(t, der_a, x, b, c, t2=t2)


def der_c(x, a, b, c):
    """
    Ancillary math function for derivative computation
    :param x: numpy.array(): Coordinate x
    :param a: real number
    :param b: real number
    :param c: real number
    :return: numpy.array()
    """
    return der_a(x, b, c) * (-1j * a)


def derivative_c(x, t, t2, a, b, c):
    """
    Derivative d(func_t)/d(c)
    :param x: numpy.array(): Coordinate x
    :param t: numpy.array(): time from 0 to 2pi or from pi/2 to pi/2+2pi
    :param t2: real number: a phase shift
    :param a: real number
    :param b: real number
    :param c: real number
    :return: numpy.array() of shape len(x) x len(t) with real(!) numbers
    """
    return multiply(t, der_c, x, a, b, c, t2=t2)


def sum_df(r, dv_da0k, scale=.01):
    """
    Compute a sum of gradients
    :param r: pandas.DataFrame(): residual
    :param dv_da0k: pandas.DataFrame(): a gradient
    :param scale: real number: scale of step
    :return: real number for backpropagation purposes
    """
    return -2 * sum(sum(r * dv_da0k)) * scale


class FitFunc:
    """
    Class for handling with the approximation
    """
    def __init__(self, f_name, a, b, c):
        """
        Load data from file specified by f_name and set other preferences
        :param f_name: String: path to the text file with data
        :param a: real number
        :param b: real number
        :param c: real number
        """
        self.data, self.t = load_and_pivot_data(f_name)
        self.v = DataFrame(0, index=self.data.index, columns=self.data.columns)
        self.dv_da = self.v.copy()
        self.dv_db = self.v.copy()
        self.dv_dc = self.v.copy()
        self.a = a
        self.b = b
        self.c = c
        self.aa = []
        self.bb = []
        self.cc = []
        self.iter_no = []
        self.scale = 0.001
        self.phase = -pi / 2

    def check_phase(self):
        """
        Check whether the phase shift t2 is correct
        Raise ValueError if t2 is wrong
        :return: None
        """
        t = Series(list(self.t.values()))
        value = ((sine_law(t, self.phase).real - self.data.iloc[0]) ** 2).max()
        if value > 1:
            self.phase = pi / 2
        elif value < 10e-4:
            self.phase = -pi / 2
        else:
            raise ValueError("Something wrong with the phase. "
                             "Specify the odd number of time nodes from n*T to (n+1)*T or from T/2+n*T to 3T/2+n*T. "
                             "Also, perhaps the X coordinate doesn't start form zero.")

    def residual(self):
        """
        Compute a residual
        :return: pandas.DataFrame()
        """
        return self.v - self.data

    def make_df(self, function, **kwargs):
        """
        Make a pandas.DataFrame() from numpy.array()
        :param function: numpy.array(): a math function
        :param kwargs: arguments of that function
        :return: pandas.DataFrame()
        """
        return DataFrame(function(x=array(self.v.index), t=array(list(self.t.values())),
                                  t2=self.phase, **kwargs), index=self.v.index)

    def feed(self):
        """
        A feedforward method
        which also prints the coefficient at every iteration (!)
        :return: pandas.DataFrame() with approximation
        """
        self.v = self.make_df(func_t, a=self.a, b=self.b, c=self.c)
        print("A = {:.8f},\tB = {:.8f},\tC = {:.8f}".format(self.a, self.b, self.c))
        return self.v

    def back(self, a_limit_positive=1.0, a_limit_negative=0.0, b_limit_positive=6.0):
        """
        A backpropagation method
        :param a_limit_positive: real number: max of a
        :param a_limit_negative: real number: min of a
        :param b_limit_positive: real number: max of b
        :return: None
        """
        self.dv_da = self.make_df(derivative_a, b=self.b, c=self.c)
        self.dv_db = self.make_df(derivative_b, a=self.a, b=self.b, c=self.c)
        self.dv_dc = self.make_df(derivative_c, a=self.a, b=self.b, c=self.c)

        if a_limit_negative <= self.a <= a_limit_positive:
            self.a += sum_df(self.residual(), self.dv_da, self.scale)
        elif self.a < a_limit_negative:
            self.a = a_limit_negative
        elif self.a > a_limit_positive:
            self.a = a_limit_positive

        if 0 <= self.b <= b_limit_positive:
            self.b += sum_df(self.residual(), self.dv_db, self.scale)
        elif self.b < 0:
            self.b = 0.0
        elif self.b > b_limit_positive:
            self.b = b_limit_positive

        self.c += sum_df(self.residual(), self.dv_dc)
        if self.c > pi:
            self.c -= 2 * pi
        elif self.c < -pi:
            self.c += 2 * pi
        elif (self.c > 2 * pi) or (self.c < -2 * pi):
            raise ValueError("correct c")

        self.aa.append(self.a)
        self.bb.append(self.b)
        self.cc.append(self.c)

    def predict(self, number_of_iterations, *args):
        """
        Organise the feedforward and backpropagation methods to the loop
        :param number_of_iterations: int: the number of the loop iterations
        :param args: arguments of the backpropagation method
        :return:
        """
        self.check_phase()
        for i in range(number_of_iterations):
            self.feed()
            self.back(*args)
            self.iter_no.append(i)


def approximate(text_filename, iterations, *args, **kwargs):
    """
    The main function
    :param text_filename: String: the name of the data file
    :param iterations: int: number of loop iterations
    :param args: limitation parameters
    :param kwargs: the coefficients a, b, and c
    :return: FitFunc() instance and loss value
    """
    fit = FitFunc(text_filename, **kwargs)
    fit.predict(iterations, *args)
    aa = fit.residual() ** 2
    return fit, aa


if __name__ == '__main__':
    pickle_file = 'config.pickle'
    config = None
    err = '\nIncorrect value! Step skipped. The default values are {}.\n'
    cond = True
    print("\n__________________________________________________________________________"
          "\n\nApproximation of viscous waves v1.0.\nDeveloped by Artem Pavlovskii\n"
          "https://github.com/Enolerobotti\nOctober 2019."
          "\n\nThis is free software that is distributed under the GNU General Public Licence.\n"
          "You can redistribute it and/or modify it under the terms of the GNU General\nPublic License as "
          "published by the Free Software Foundation.\n")

    while cond:
        if exists(pickle_file):
            with open(pickle_file, 'rb') as configuration:
                config = load(configuration)
        else:
            config = {'iterations': 500,
                      'a_init': 0.1,
                      'b_init': 0.2,
                      'c_init': 3.0,
                      'a_lim_p': 1.0,
                      'a_lim_n': 0.0,
                      'b_lim_p': 6.0}
        root = Tk()
        root.withdraw()

        print("__________________________________________________________________________"
              "\nPlease, select a text file with viscous wave velocity data in the popup. ")
        t_f = filedialog.askopenfilename(title="Select data file", filetypes=(("text files", "*.txt"),))
        root.destroy()

        try:
            print("\nHello! You have open file\n{}\n"
                  "We'll approximate it by dimensionless formula\n\n"
                  "\tv = Re((exp((j-1)x) + A*exp(-jC)(1-exp((j-1)x))*exp(-Bx-jt))"
                  "\n\nusing the least squares and gradient descent methods. We denote the imaginary\n"
                  "unit by j, the dimensionless coordinate by x and "
                  "dimensionless time by t. The A,\nB and C are unknown coefficients which we will try to predict.\n"
                  "Please, see the link https://link.springer.com/article/10.1134/S1063784216070185 for details.\n\n"
                  "Now you must specify the maximum number of iterations, "
                  "the coefficient's initial\nvalues and its limits. Let's start!".format(t_f))
            try:
                config['iterations'] = int(input("Set the maximum number of iterations (500 will be almost enough): "))
            except ValueError:
                print(err.format(config))
            if "y" in input("Would you like to change the initial parameters which are\n"
                            "A = {} and B = {}, and C = {} now? (y/n)\n".format(config['a_init'],
                                                                                config['b_init'],
                                                                                config['c_init'])).strip(" ").lower():
                try:
                    config['a_init'] = float(input("A = "))
                    config['b_init'] = float(input("B = "))
                    config['c_init'] = float(input("C = "))
                except ValueError:
                    print(err.format(config))
            if "y" in input("Would you like to change the initial parameters which are\n"
                            "{}<=A<={} , 0<=B<={} now? (y/n)\n".format(config['a_lim_n'],
                                                                       config['a_lim_p'],
                                                                       config['b_lim_p'])).strip(" ").lower():
                try:
                    config['a_lim_p'] = float(input("Set A positive limit: "))
                    config['a_lim_n'] = float(input("Set A negative limit: "))
                    config['b_lim_p'] = float(input("Set B positive limit: "))
                except ValueError:
                    print(err.format(config))

            with open(pickle_file, 'wb') as configuration:
                dump(config, configuration)

            fit_instance, loss = approximate(t_f, config['iterations'], config['a_lim_p'], config['a_lim_n'],
                                             config['b_lim_p'],
                                             a=config['a_init'], b=config['b_init'], c=config['c_init'])
            v, v0 = get_plot_data(fit_instance)
            print("\nFinally,"
                  "\nA = {}, B = {}, C = {}".format(fit_instance.a, fit_instance.b, fit_instance.c))
            print(t_f)
            print("Max RMSE is {}.".format(loss.max().max()))
            print("See the figures in the popup")
            #    phase_plot(fit_instance.v, fit_instance.data, pd.Series(list(fit_instance.t.values())) / pi / 2)
            plot_data(v, v0)
            model_plot(fit_instance.iter_no, fit_instance.aa, fit_instance.bb, fit_instance.cc)
        except FileNotFoundError:
            print('Wrong filename!')
        cond = 'r' in input("Type 'r' to repeat or just press Enter to exit ... ").lower()
