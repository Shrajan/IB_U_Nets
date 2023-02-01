import torch
import math
from sympy import *

def DoG_IB_3D(x, y, z, center, gamma, radius):
    """compute weight at location (x, y, z) in the OOCS kernel with given parameters
        Parameters:
            x , y, z : position of the current weight
            center : position of the kernel center
            gamma : center to surround ratio
            radius : center radius

        Returns:
            excite and inhibit : calculated from Equation2 in the paper, without the coefficients A-c and A-s

    """
    # compute sigma from radius of the center and gamma(center to surround ratio)
    sigma = (radius / gamma) * (math.sqrt((1 - gamma ** 2) / (-6 * math.log(gamma))))
    excite = (1 / (gamma ** 3)) * math.exp(
        -1 * ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * ((gamma * sigma) ** 2)))
    inhibit = math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * (sigma ** 2)))

    return excite, inhibit


def IB_filters_3D(radius, gamma, in_channels, out_channels, off=false):
    """compute the kernel filters with given shape and parameters
        Parameters:
            gamma : center to surround ratio
            radius : center radius
            in_channels and out_channels: filter dimensions
            off(boolean) : if false, calculates on center kernel, and if true, off center

        Returns:
            kernel : On or Off center conv filters with requested shape

    """

    # size of the kernel
    kernel_size = int((radius / gamma) * 2 - 1)
    # center node index
    centerX = int((kernel_size + 1) / 2)

    posExcite = 0
    posInhibit = 0
    negExcite = 0
    negInhibit = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                excite, inhibit = DoG_IB_3D(i + 1, j + 1, k + 1, centerX, gamma, radius)
                if excite > inhibit:
                    posExcite += excite
                    posInhibit += inhibit
                else:
                    negExcite += excite
                    negInhibit += inhibit

    # Calculating A-c and A-s, with requiring the positive vlaues sum up to 1 and negative vlaues to -1
    x, y = symbols('x y')
    sum = 3.
    if kernel_size == 3:
        sum = 1.
    elif kernel_size == 5:
        sum = 3.
    solution = solve((x * posExcite + y * posInhibit - sum, negExcite * x + negInhibit * y + sum), x, y)
    A_c, A_s = float(solution[x].evalf()), float(solution[y].evalf())

    # making the On-center and Off-center conv filters
    kernel = torch.zeros([out_channels, in_channels, kernel_size, kernel_size, kernel_size], requires_grad=False)
    kernel_3D = torch.zeros([kernel_size, kernel_size, kernel_size])

    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                excite, inhibit = DoG_IB_3D(i + 1, j + 1, k + 1, centerX, gamma, radius)
                weight = excite * A_c + inhibit * A_s
                if off:
                    weight *= -1.
                kernel_3D[i][j][k] = weight

    # Creating all the necessary kernels based on the input and output channels.
    for i in range(out_channels):
        for j in range(in_channels):
            kernel[i][j] = kernel_3D

    return kernel.float()
