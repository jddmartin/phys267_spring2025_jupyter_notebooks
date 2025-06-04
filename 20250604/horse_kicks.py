import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

df = pd.read_csv("hk.csv", header=None)

print(df.shape)

kicks = df.fillna(0).to_numpy().flatten()

print(f"{len(kicks)=} \n{np.mean(kicks)=} \n{np.std(kicks)=}"
      f"\n{np.std(kicks)**2=}")

bin_centers = list(range(0, 10))

bins = np.append(np.array(bin_centers) - 0.5,
                 bin_centers[-1] + 0.5)
bins = np.arange(-1, 10, 1) + 0.5
bin_centers = (bins[1:] + bins[:-1]) / 2
print(bin_centers)
h, _ = np.histogram(kicks, bins)

with mpl.rc_context(fname="physrev.mplstyle"):
    f, ax = plt.subplots()
    ax.bar(bin_centers, h, fill=None, edgecolor="blue")
    ax.set_xlabel("horse kicks / corp / year")
    ax.set_ylabel("occurences")
    f.savefig("part_a_generated.pdf")
    plt.close()

normalized_h, _ = np.histogram(kicks, bins, density=True)

with mpl.rc_context(fname="physrev.mplstyle"):
    f, ax = plt.subplots()

    ax.bar(bin_centers, normalized_h,  fill=None, edgecolor="blue",
           label="observed")
    ax.set_xlabel("horse kicks / corp / year")
    ax.set_ylabel("probability")

    def p(mean):
        def f(r):
            return 1/math.factorial(r) * mean**r * np.exp(-mean)
        return f

    nc = 10
    ax.bar(list(range(nc)), [p(np.mean(kicks))(i) for i in range(nc)],
           alpha=0.5,
           label="Poisson distribution")
    ax.legend()
    f.savefig("part_b_generated.pdf")



