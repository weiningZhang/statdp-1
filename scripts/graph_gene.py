import matplotlib.pyplot as plt


def graph_gene(xlabel, ylabel, data, title, name):
    x = [i[0] for i in data]
    p1 = [i[1] for i in data]
    p2 = [i[2] for i in data]

    plt.plot(x, p1, 'o-', label='p1', markersize=2)
    plt.plot(x, p2, 'o-', label='p2', markersize=2)
    plt.axhline(y=0.5, color='black', linestyle='dashed')
    plt.axhline(y=0.05, color='green', linestyle='dashed', alpha=0.8)
    plt.axhline(y=0.01, color='red', linestyle='dashed', alpha=0.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=3)
    plt.savefig(name)
    plt.show()
    return


if __name__ == "__main__":
    data = [
        [3.79, 1.0, 1.0],
        [3.800000, 1.000000, 0.989902],
        [3.810000, 1.000000, 0.980416],
        [3.820000, 1.000000, 0.983674],
        [3.830000, 1.000000, 0.972858],
        [3.840000, 1.000000, 0.975320],
        [3.850000, 1.000000, 0.963962],
        [3.860000, 1.000000, 0.972260],
        [3.870000, 1.000000, 0.958232],
        [3.880000, 1.000000, 0.955138],
        [3.890000, 1.000000, 0.952834],
        [3.900000, 1.000000, 0.958624],
        [3.910000, 1.000000, 0.945942],
        [3.920000, 1.000000, 0.928616],
        [3.930000, 1.000000, 0.910860],
        [3.940000, 1.000000, 0.923700],
        [3.950000, 1.000000, 0.899720],
        [3.960000, 1.000000, 0.898666],
        [3.970000, 1.000000, 0.890928],
        [3.980000, 1.000000, 0.898246],
        [3.990000, 1.000000, 0.859422],
        [4.000000, 1.000000, 0.869486],
        [4.010000, 1.000000, 0.851590],
        [4.020000, 1.000000, 0.831466],
        [4.030000, 1.000000, 0.834780],
        [4.040000, 1.000000, 0.811870],
        [4.050000, 1.000000, 0.825710],
        [4.060000, 1.000000, 0.802406],
        [4.070000, 1.000000, 0.756360],
        [4.080000, 1.000000, 0.738950],
        [4.090000, 1.000000, 0.756250],
        [4.100000, 1.000000, 0.742090],
        [4.110000, 1.000000, 0.743192],
        [4.120000, 1.000000, 0.730008],
        [4.130000, 1.000000, 0.686216],
        [4.140000, 1.000000, 0.678688],
        [4.150000, 1.000000, 0.589160],
        [4.160000, 1.000000, 0.613242],
        [4.170000, 1.000000, 0.585780],
        [4.180000, 1.000000, 0.573098],
        [4.190000, 1.000000, 0.584422],
        [4.200000, 1.000000, 0.533300],
        [4.210000, 1.000000, 0.488452],
        [4.220000, 1.000000, 0.524784],
        [4.230000, 1.000000, 0.507360],
        [4.240000, 1.000000, 0.476022],
        [4.250000, 1.000000, 0.419788],
        [4.260000, 1.000000, 0.448522],
        [4.270000, 1.000000, 0.420242],
        [4.280000, 1.000000, 0.370778],
        [4.290000, 1.000000, 0.416700],
        [4.300000, 1.000000, 0.340320],
        [4.310000, 1.000000, 0.339554],
        [4.320000, 1.000000, 0.352354],
        [4.330000, 1.000000, 0.282626],
        [4.340000, 1.000000, 0.306846],
        [4.350000, 1.000000, 0.263174],
        [4.360000, 1.000000, 0.260858],
        [4.370000, 1.000000, 0.249936],
        [4.380000, 1.000000, 0.214436],
        [4.390000, 1.000000, 0.231644],
        [4.400000, 1.000000, 0.180198],
        [4.410000, 1.000000, 0.180610],
        [4.420000, 1.000000, 0.228670],
        [4.430000, 1.000000, 0.172696],
        [4.440000, 1.000000, 0.181156],
        [4.450000, 1.000000, 0.145254],
        [4.460000, 1.000000, 0.144986],
        [4.470000, 1.000000, 0.129600],
        [4.480000, 1.000000, 0.099216],
        [4.490000, 1.000000, 0.109894],
        [4.500000, 1.000000, 0.088588],
        [4.510000, 1.000000, 0.082842],
        [4.520000, 1.000000, 0.085642],
        [4.530000, 1.000000, 0.079374],
        [4.540000, 1.000000, 0.082686],
        [4.550000, 1.000000, 0.078552],
        [4.560000, 1.000000, 0.069900],
        [4.570000, 1.000000, 0.056216],
        [4.580000, 1.000000, 0.033792],
        [4.590000, 1.000000, 0.047806],
        [4.600000, 1.000000, 0.041450],
        [4.610000, 1.000000, 0.037628],
        [4.620000, 1.000000, 0.028058],
        [4.630000, 1.000000, 0.029854],
        [4.640000, 1.000000, 0.016612],
        [4.650000, 1.000000, 0.023142],
        [4.660000, 1.000000, 0.028556],
        [4.670000, 1.000000, 0.021478],
        [4.680000, 1.000000, 0.019308],
        [4.690000, 1.000000, 0.019902],
        [4.700000, 1.000000, 0.013640],
        [4.710000, 1.000000, 0.011914],
        [4.720000, 1.000000, 0.014848],
        [4.730000, 1.000000, 0.007244],
        [4.740000, 1.000000, 0.008202],
        [4.750000, 1.000000, 0.004214],
        [4.760000, 1.000000, 0.007142],
        [4.770000, 1.000000, 0.004152],
        [4.780000, 1.000000, 0.004556],
        [4.790000, 1.000000, 0.002684],
        [4.800000, 1.000000, 0.003202],
        [4.810000, 1.000000, 0.002386],
        [4.820000, 1.000000, 0.004244],
        [4.830000, 1.000000, 0.002508],
        [4.840000, 1.000000, 0.002420],
        [4.850000, 1.000000, 0.001586],
        [4.860000, 1.000000, 0.000340],
        [4.870000, 1.000000, 0.001804],
        [4.880000, 1.000000, 0.000758],
        [4.890000, 1.000000, 0.000980],
        [4.900000, 1.000000, 0.002074],
        [4.910000, 1.000000, 0.000990],
        [4.920000, 1.000000, 0.000550],
        [4.930000, 1.000000, 0.000408],
        [4.940000, 1.000000, 0.000542],
        [4.950000, 1.000000, 0.000780],
        [4.960000, 1.000000, 0.000272],
        [4.970000, 1.000000, 0.000182],
        [4.980000, 1.000000, 0.000108],
        [4.990000, 1.000000, 0.000134],
    ]

    graph_gene('Epsilon', 'P Value', data, 'NoisyMax P-Value with Option 1', 'result.pdf')
