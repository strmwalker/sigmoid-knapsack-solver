__all__ = ["Curve", "MixedCurve", "Budget"]

from functools import partial

from numpy import array, exp, ndarray, linspace
from pandas import Series, DataFrame
from scipy.optimize import minimize


def basic(x, cap=1, ec50=0.5, steep=0, price=1, multiplier=1):
    """
    S-shaped curve similar to sigmoid function. Depending on parameters,
    can be fully concave given x > 0, or have convex and concave parts.

    :param x: input array
    :param cap: max level for function.
    :param ec50: half-efficiency point. Moves curve horizontally, and serves as bend point.
    :param steep: slope coefficient. If < 1, curve will be fully concave on interval (0, +Inf), if > 1, curve will be
    concave before ec50 and convex after.
    :param price: price
    :param multiplier: model coefficient for curve
    :return: float with same dimensions as x.
    """
    if isinstance(x, int) or isinstance(x, float):
        return (cap / (1 + (x / price / cap / ec50) ** (-steep))) * multiplier if x != 0 else 0
    elif isinstance(x, Series) or isinstance(x, ndarray):
        if 0 not in x:
            return array(cap / (1 + (x / price / cap / ec50) ** (-steep))) * multiplier
        else:
            return array([(cap / (1 + (y / price / cap / ec50) ** (-steep))) * multiplier if y != 0 else 0 for y in x])
    #
    # return (cap / (1 + (x / price / cap / ec50) ** (-steep))) * multiplier


def basic_derivative(x, cap, ec50, steep, price=1, multiplier=1):
    numerator = cap * steep * multiplier * (x / (cap * ec50 * price)) ** steep
    denominator = x * (1 + (x / (cap * ec50 * price)) ** steep) ** 2
    return numerator / denominator


def log(x, cap, ec50, steep, price=1, multiplier=1):
    """
     S-shaped curve based on exponential function

    :param x: input array
    :param cap: max level for function.
    :param ec50: half-efficiency point. Moves curve horizontally, and
    serves as bend point.
    :param steep: slope coefficient. If < 1, curve will be fully concave on
    interval (0, +Inf), if > 1, curve will be concave before ec50 and convex
    after.
    :param price:
    :param multiplier:
    :return: float with same dimensions as x.
    """
    return (cap / (1 + exp(-steep * x / price / cap - ec50)) - cap / (1 + exp(steep * ec50))) * multiplier


def log_derivative(x, cap, ec50, steep, price=1, multiplier=1):
    numerator = steep * multiplier * exp(steep * x / price / cap + ec50)
    denominator = (exp(steep * x / price / cap + ec50) + 1) ** 2
    return numerator / denominator


def art(x, a, b, multiplier=1):
    first_term = 100 / (1 + exp(a * exp(x * - a / b)))
    second_term = 100 / (1 + exp(a))
    return (first_term - second_term) * multiplier


def art_derivative(x, a, b, multiplier=1.0):
    numerator = -100 * a * multiplier * exp(a * exp(- (a / b) * x - (a / b)))
    denominator = (exp(a * exp(-a / b) * x) + 1) ** 2

    return numerator / denominator


class Curve(object):
    # TODO LaTeX curve equation rendering
    def __init__(self, cap, ec50, steep, multiplier=1, price=1, curve_type='basic'):
        """
        :param cap: maximum level for function
        :param ec50: half-efficiency point. Moves curve horizontally, and
        serves as bend point.
        :param steep: slope coefficient. If < 1, curve will be fully concave on
        interval (0, +Inf), if < 1, curve will be convex before ec50 and concave
        after.
        :param multiplier: model coefficient for curve
        :param curve_type: regular or logistic response curve, can be 'basic' or 'log'
        """
        self.cap = cap
        self.ec50 = ec50
        self.steep = steep
        self.multiplier = multiplier
        self.price = price
        self.type = curve_type

    @property
    def fun(self):
        if self.type == 'basic':
            return partial(basic,
                           cap=self.cap,
                           ec50=self.ec50,
                           steep=self.steep,
                           price=self.price,
                           multiplier=self.multiplier)
        elif self.type == 'log':
            return partial(log,
                           cap=self.cap,
                           ec50=self.ec50,
                           steep=self.steep,
                           price=self.price,
                           multiplier=self.multiplier)

    @property
    def derivative(self):
        if self.type == 'basic':
            return partial(basic_derivative,
                           cap=self.cap,
                           ec50=self.ec50,
                           steep=self.steep,
                           price=self.price,
                           multiplier=self.multiplier)
        elif self.type == 'log':
            return partial(log_derivative,
                           cap=self.cap,
                           ec50=self.ec50,
                           steep=self.steep,
                           price=self.price,
                           multiplier=self.multiplier)

    def __call__(self, x):
        """
        Calculate response
        :param x: budget
        :return:  float64
        """
        return self.fun(x)

    def plot(self, budget):
        DataFrame(
            {
                "media": self(linspace(0, budget, 1000))
            }
        ) \
            .plot(
            kind='line',
            grid=True,
            figsize=(12, 10)
        )


# noinspection PyMissingConstructor
class ArtyomCurve(object):
    def __init__(self, a, b, multiplier=1.0):
        self.a = a
        self.b = b
        self.multiplier = multiplier

    @property
    def fun(self):
        return partial(art, a=self.a, b=self.b, multiplier=self.multiplier)

    @property
    def derivative(self):
        return partial(art_derivative, a=self.a, b=self.b, multiplier=self.multiplier)

    def __call__(self, x):
        return self.fun(x)

        # def __str__(self):
        #     first_term = f'100 / (1 + exp({self.a} * exp(x * - {self.a} / {self.b})))'
        #     second_term = f'100 / (1 + exp({self.a}))'

        #     return first_term + ' - ' + second_term


class MixedCurve(object):
    """
    Mixed curve is designed for POEM and should represent response from one media. Constructed from a basic Curve objects.
    """

    def __init__(self, *curves):
        self.curves = curves

    def __call__(self, x):
        return self.fun(x)

    @property
    def fun(self):
        """
        Callable that can be passed further.
        """

        def f(x):
            return sum([curve(x) for curve in self.curves])

        return f

    @property
    def derivative(self):
        def d(x):
            return sum([curve.derivative(x) for curve in self.curves])

        return d


class Budget(object):
    """
    Optimization solver class.
    """
    notebook_mode = True

    def __init__(self, budget):
        """
        :param budget: total budget for problem
        """
        self.budget = budget
        self.solution = None
        self.__bounds = []
        self.__curves = []

    def get_curves(self):
        """
        :return: Curve objects assigned to this budget
        """
        return self.__curves

    def get_bounds(self):
        """
        :return: bounds for Curve objects assigned to this budget
        """
        return self.__bounds

    def add_curve(self, curve, lower=None, upper=None):
        """
        Add Curve (which essentially means media) to optimization problem.
        :param curve: Curve/MixedCurve instance
        :param upper: Upper bound for budget
        :param lower: Lower bound for budget
        :return:
        """
        if not lower:
            lower = 1
        if not upper:
            upper = self.budget
        self.__curves.append(curve)
        self.__bounds.append([lower, upper])

    @property
    def fun(self):
        def f(x, sign=1.0):
            impact = 0
            for i, curve in enumerate(self.__curves):
                impact += curve(x[i])

            return sign * impact

        return f

    def __call__(self, x):
        """
        Calculate response for given spends.
        :param x: int/float/numpy.ndarray/Series
        :return: float64
        """
        return self.fun(x)

    @property
    def mix(self):
        if self.solution:
            return self.solution.x
        else:
            return [0.0 for _ in self.__curves]

    @property
    def derivative(self):

        # despite most of the time sign == 1.0, this feature is needed if we want to minimize something
        def f(x, sign=1.0):
            return [sign * curve.derivative(x[i]) for i, curve in enumerate(self.__curves)]

        return f

    @property
    def constraints(self):
        """
        Generate callable constraints for SLSQP optimization.
        :return: dict{str, callable, callable}
        """

        def fun(x):
            spend = sum(x)
            return spend - self.budget

        def jac(x):
            return array([1.0 for _ in range(len(x))])

        constraints = (
            {
                'type': 'eq',
                'fun': fun,
                'jac': jac
            },
        )
        return constraints

    @property
    def constraints_cobyla(self):
        """
        Generate callable constraints for SLSQP optimization.
        :return: dict{str, callable, callable}
        """

        def fun(x):
            spend = sum(x)
            return spend - self.budget

        def jac(x):
            return array([1.0 for _ in range(len(x))])

        constraints = (
            {
                'type': 'ineq',
                'fun': fun,
                'jac': jac
            },
        )
        return constraints

    def solve(self, disp=True, maxiter=100):
        """
        Solve optimization problem for budget.
        :param disp: Set to True to print convergence messages
        :param maxiter: Maximum number of iterations to perform
        :return: numpy.array with corresponding budgets
        """
        constraints = self.constraints
        derivative = self.derivative
        x0 = array([bound[0] for bound in self.__bounds])

        self.solution = minimize(
            fun=self.fun,
            x0=x0,
            args=(-1.0,),
            method='SLSQP',
            jac=derivative,
            bounds=self.__bounds,
            constraints=constraints_cobyla,
            options={
                'disp': disp,
                'maxiter': maxiter
            }
        )

        return self.solution.x

    def solve_cobyla(self, disp=False, maxiter=100):
        """
        Solve optimization problem for budget.
        :param disp: Set to True to print convergence messages
        :param maxiter: Maximum number of iterations to perform
        :return: numpy.array with corresponding budgets
        """
        constraints = self.constraints_cobyla
        derivative = self.derivative
        x0 = array([bound[0] for bound in self.__bounds])

        self.solution = minimize(
            fun=self.fun,
            x0=x0,
            args=(-1.0,),
            method='COBYLA',
            jac=derivative,
            bounds=self.__bounds,
            constraints=constraints,
            options={
                'disp': disp,
                'maxiter': maxiter
            }
        )

        return self.solution.x

    def plot(self, names=None, budget=None, ext='png'):
        """
        Render all response curves to single plot. If ```notebook_mode``` is ```True```,
        return matplotlib subplot, else save image to file `plot.ext`.
        :param names: verbose names for plot
        :param budget: max x axis for plot
        :param ext: file extension if saving image to disk
        :return:
        """
        if budget:
            x = linspace(0, budget, 1000)
        else:
            x = linspace(0, self.budget + int(self.budget / 100), 1000)

        if names:
            data = {name: curve(x) for name, curve in zip(names, self.__curves)}
        else:
            data = {'y {0}'.format({i + 1}): curve(x) for i, curve in enumerate(self.__curves)}

        lines = DataFrame(
            data=data,
            index=x
        ) \
            .plot(
            kind='line',
            figsize=(12, 10),
            grid=True
        )

        if self.notebook_mode:
            return lines
        else:
            fig = lines.get_figure()
            fig.savefig("plot.".format(ext))


if __name__ == '__main__':
    pass
