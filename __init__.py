import time
from datetime import timedelta
from functools import wraps, update_wrapper
from matplotlib import pyplot
import numbers
import numpy as np

def timed(in_function):
    @wraps(in_function)
    def timed_function(*args, **kwargs):
        start = time.perf_counter()
        ret = in_function(*args, **kwargs)
        return timedelta(seconds = time.perf_counter() - start), ret
    return timed_function

def numericRanged(in_function):
    @wraps(in_function)
    def wrapped_function(*args, **kwargs):
        ret = in_function(*args, **kwargs)
        if(isinstance(ret, numbers.Number)):
            return ret
        else:
            return float('nan')
    return wrapped_function

class RuntimeAnalyzer:
    """
    A function wrapper class for dimensional runtime analysis.
    """
    def __init__(self, in_function, input_metric = lambda *x, **kwx: x[0]):
        self.timed_func = timed(in_function)
        self.runtime_points = dict()
        safe_input_metric = numericRanged(input_metric)
        def new_call(*args, **kwargs):
            runtime, ret = self.timed_func(*args, **kwargs)
            size = safe_input_metric(*args, **kwargs)
            if (size in self.runtime_points):
                self.runtime_points[size].append(runtime.total_seconds() * 1000000)
            else:
                self.runtime_points[size] = [runtime.total_seconds() * 1000000]
            return ret

        self._fn = new_call

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def plot(self, xlim = None, ylim = None, dimension = None):
        """
        Plots input sizes by the runtime dimension and magnitude.

        Given an input size s and a runtime of T(s), we let
        T(s) := (2 ** magnitude(s)) * (s ** dimension(s))

        If the RuntimeAnalyzer was run with the same input size multiple times,
        T(s) is taken as the average runtime of those.
        """
        if(dimension == None):
            self._plot_without_fixed_dimension(xlim, ylim)
        else:
            self._plot_with_fixed_dimension(dimension, xlim, ylim)

    def _plot_without_fixed_dimension(self, xlim = None, ylim = None):
        sizes, mags, dims = self.size_to_dimension_and_magnitude()

        self._plot_panel(311, "Runtime", xlim, ylim, pyplot.loglog,\
            sizes, [np.mean(self.runtime_points[s]) for s in sizes],\
            'b.-', basex=10, basey=2)

        self._plot_panel(312, "Dimension", xlim, ylim,\
            pyplot.semilogx, sizes, dims, 'r.-', basex=10)

        self._plot_panel(313, "Magnitude", xlim, ylim,\
            pyplot.semilogx, sizes, mags, 'g.-', basex=10)

        pyplot.xlabel("Input Size")

        pyplot.show()

    def _plot_with_fixed_dimension(self, dimension, xlim=None, ylim=None):
        sizes = sorted([k for k in self.runtime_points.keys()])

        self._plot_panel(211, "Runtime", xlim, ylim, pyplot.loglog,\
            sizes, [np.mean(self.runtime_points[s]) for s in sizes],\
            'b.-', basex=10, basey=2)

        dims = [dimension for s in sizes]
        mags = self._size_and_dimension_to_magnitude(sizes, dims)

        self._plot_panel(212, "Magnitude\n(Dimension: "+ str(dimension)+")",\
            xlim, ylim, pyplot.semilogx, sizes, mags, 'g.-', basex=10)

        pyplot.xlabel("Input Size")


        pyplot.show()

    @staticmethod
    def _plot_panel(subplotCode, ylabel, xlim, ylim, plot_method, *args, **kwargs):
        pyplot.subplot(subplotCode)
        plot_method(*args, **kwargs)
        if(xlim != None):
            pyplot.xlim(xlim)
        pyplot.ylabel(ylabel)
        if(ylim != None):
            pyplot.ylim(ylim)

    def size_to_dimension_and_magnitude(self):
        """
        Returns a 2d array of [input_sizes, runtime_magnitudes, runtime_dimensions]

        Given an input size s and a runtime of T(s), we let
        T(s) := (2 ** magnitude(s)) * (s ** dimension(s))

        If the RuntimeAnalyzer was run with the same input size multiple times,
        T(s) is taken as the average runtime of those.
        """
        loglogPoints = [p for p in zip(*sorted([[np.log2(x), np.log2(np.mean(ys))]\
                            for x, ys in self.runtime_points.items()\
                            if (not np.isnan(x)) and (x > 0 and np.mean(ys) > 0)]))]
        x = sorted([k for k in self.runtime_points.keys()])
        x_log = np.array(loglogPoints[0])
        y_log = np.array(loglogPoints[1])
        sizes, dims = [x, np.gradient(y_log, edge_order = 2)/np.gradient(x_log, edge_order = 2)]
        mags = self._size_and_dimension_to_magnitude(sizes, dims)
        return [sizes, mags, dims]

    def _size_and_dimension_to_magnitude(self, sizes, dims):
        return [np.log2(np.mean(self.runtime_points[s])/np.power(s, d)) for s, d in zip(sizes, dims)]



def ex_0():
    def lin_func(n):
        """
        Just a simple function that runs in linear time.
        """
        ret = 0
        for i in range(n):
            ret += i
        return ret

    rt = RuntimeAnalyzer(lin_func)
    for i in range(1, 25):
        rt(2**i)

    rt.plot(dimension = 1)

def ex_1():
    def test_func(n):
        """
        This function is quadradic for n <= 400, linear for 400 < n <= 10000,
        and constant while 10000 < n
        """
        ret = 0
        for i in range(min(n, 10000)):
            for j in range(min(n, 400)):
                ret += (i*j)/n
        return ret

    rt = RuntimeAnalyzer(test_func)
    for t in range(2):
        for i in range(1, 60):
            rt(int(1.3**i))

    rt.plot()

def ex_2():
    def bootstrapped_func(n):
        """
        This function emulates bootstrapping up to n = 200 with a quadradic
        algorithm, and using a linear algorithm for larger n.
        """
        ret = 0
        for i in range(min(200, n)):
            for j in range(min(200, n)):
                ret += (i*j)/n

        if(n>200):
            for i in range(n):
                ret += i**2

        return ret

    rt = RuntimeAnalyzer(bootstrapped_func)
    for t in range(20):
        for i in range(1, 20):
            rt(2**i)

    rt.plot()
