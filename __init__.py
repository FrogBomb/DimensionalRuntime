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

def log2Bucketize(x, bucket_size = .2):
    if(x > 0):
        bucket_num = np.rint((np.log2(x))/bucket_size)
        return np.power(2, bucket_num*bucket_size)
    else:
        return 0.0

class RuntimeAnalyzer:
    """
    A function wrapper class for dimensional runtime analysis.
    """
    def __init__(self, in_function, input_metric = lambda *x, **kwx: x[0]):
        self.timed_func = timed(in_function)
        self.runtime_points = dict()
        safe_input_metric = numericRanged(input_metric)
        self._min_input = None
        self._max_input = None
        def new_call(*args, **kwargs):
            runtime, ret = self.timed_func(*args, **kwargs)
            size = log2Bucketize(safe_input_metric(*args, **kwargs))
            rt_microseconds = runtime.total_seconds() * 1000000
            if(self._min_input == None):
                self._max_input = size
                self._min_input = size
            elif(size < self._min_input):
                self._min_input = size
            elif(size > self._max_input):
                self._max_input = size

            if(size in self.runtime_points):
                self.runtime_points[size].append(rt_microseconds)
            else:
                self.runtime_points[size] = [rt_microseconds]
            return ret

        self._fn = new_call

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def plot(self, xlim = None, rt_lim = None, dim_lim = None, mag_lim = None, dimension = None):
        """
        Plots input sizes by the runtime dimension and magnitude.

        Given an input size s and a runtime of T(s), we let
        T(s) := (2 ** magnitude(s)) * (s ** dimension(s))

        If the RuntimeAnalyzer was run with the same input size multiple times,
        T(s) is taken as the average runtime of those.
        """
        if(xlim == None):
            xlim = self._min_and_max_x_for_plot()
        if(dimension == None):
            self._plot_without_fixed_dimension(xlim, rt_lim, dim_lim, mag_lim)
        else:
            self._plot_with_fixed_dimension(dimension, xlim, rt_lim, mag_lim)

    def _plot_without_fixed_dimension(self, xlim, rt_lim, dim_lim, mag_lim):
        rt_sizes = sorted([k for k in self.runtime_points.keys()])
        sizes, mags, dims = self.size_to_dimension_and_magnitude()

        self._plot_panel(311, "Runtime", xlim, rt_lim, pyplot.loglog,\
            rt_sizes, [np.median(self.runtime_points[s]) for s in rt_sizes],\
            'b.-', basex=10, basey=2)

        self._plot_panel(312, "Dimension", xlim, dim_lim,\
            pyplot.semilogx, sizes, dims, 'r.-', basex=10)

        self._plot_panel(313, "Magnitude", xlim, mag_lim,\
            pyplot.semilogx, sizes, mags, 'g.-', basex=10)

        pyplot.xlabel("Input Size")

        pyplot.show()

    def _plot_with_fixed_dimension(self, dimension, xlim, rt_lim, mag_lim):
        sizes = sorted([k for k in self.runtime_points.keys()])
        runtimes = [np.median(self.runtime_points[s]) for s in sizes]

        self._plot_panel(211, "Runtime", xlim, rt_lim, pyplot.loglog,\
            sizes, runtimes, 'b.-', basex=10, basey=2)

        dims = [dimension for s in sizes]
        mags = self._size_and_dimension_to_magnitude(runtimes, sizes, dims)

        self._plot_panel(212, "Magnitude\n(Dimension: "+ str(dimension)+")",\
            xlim, mag_lim, pyplot.semilogx, sizes, mags, 'g.-', basex=10)

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
        xy_points = [p for p in [(x, (np.median(ys)))\
                            for x, ys in self.runtime_points.items()]]
        sizes, dims, mags = RuntimeAnalyzer._x_to_smoothed_log_log_slope_with_mag(xy_points)
        return [sizes, mags, dims]

    @staticmethod
    def _x_to_smoothed_log_log_slope_with_mag(xy_points):
        (log_x_array, log_y_array) = zip(*sorted([(np.log2(x), np.log2(y)) for x, y in xy_points\
                                          if x > 0 and y > 0]))
        # Slope and point approximation kernels for 3rd order fit of 7 points
        weighed_point_kernal =  np.array([-2/21, 1/7, 2/7, 1/3, 2/7, 1/7, -2/21])
        smooth_slope_kernal = np.array([11/126, -67/252, -29/126, 0, 29/126, 67/252, -11/126])
        weighed_log_x = np.convolve(log_x_array, weighed_point_kernal, mode='valid')
        weighed_log_y = np.convolve(log_y_array, weighed_point_kernal, mode='valid')
        log_x_dir = np.convolve(log_x_array, smooth_slope_kernal, mode='valid')
        log_y_dir = np.convolve(log_y_array, smooth_slope_kernal, mode='valid')
        dims = log_y_dir/log_x_dir
        mags = weighed_log_y - (weighed_log_x * dims)
        return 2**weighed_log_x, dims, mags

    @staticmethod
    def _size_and_dimension_to_magnitude(runtimes, sizes, dims):
        return np.log2(runtimes/np.power(sizes, dims))

    def _min_and_max_x_for_plot(self):
        return self._min_input, self._max_input

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

    rt.plot()

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

def ex_3():
    """
    Asymptotically, this function is O(nlog(n)). From this, we know that the dimension
    will asymptotically be equal to 1.

    Running the analyzer on this function, we can see that the runtime has
    approximately dimension 1 with a noisy, but constant, magnitude.
    However, when we fix the dimension to 1, the magnitude appears to
    steadily increase in this range. This tells us the estimate of 1 is
    a little too low.

    If we fix the dimension to be 1.2, the magnitude appears to be remain
    constant in this range. Thus, we can conclude that it is more useful here
    to assume that this function runs in ~1.2 dimensional time.
    """
    import random
    rt = RuntimeAnalyzer(sorted, lambda x: len(x))
    seq = [i for i in range(1, 2**15)]
    random.shuffle(seq)

    for k in range(7):
        for i in range(300):
            rt(seq[:int(2**(i/20))])


    rt.plot()
    rt.plot(dimension = 1)
    rt.plot(dimension = 1.2)
