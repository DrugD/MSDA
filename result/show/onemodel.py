
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================================================


def draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high):
    fig, ax1 = plt.subplots()
    plt.xlabel(" ", fontsize=14)
    plt.ylabel(" ", fontsize=14)
    # plt.rc('font', family='Times New Roman')

    ax1.bar([x-0.1 for x in X], Y2, color="#06415c",
            width=[0.045*len(Y2)], label="PCC", alpha=1)
    ax1.set_ylabel("PCC & SPC (%)")

    ax1.bar([x+0.1 for x in X], Y3, color="#b30200",
            width=[0.045*len(Y3)], label="SPC", alpha=1)
    plt.ylim(pcc_low, pcc_high)

    ax2 = ax1.twinx()
    ax2.plot(X, Y1, color="black", linewidth=3, marker='o',
             linestyle='--',  label="RMSE", alpha=0.8)
    ax2.set_ylabel("RMSE", fontsize=14)

    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xlabel("Number of target branches")

    plt.ylim(rmse_low, rmse_high)

    plt.title(Model, fontsize=16)

    fig.legend()
    plt.tight_layout()
    fig.savefig("/home/lk/project/MSDA_DRP/result/show/{}.jpg".format(Model))

# =================================


X = [3, 2, 1, 0]
Y1 = [0.079710601,
      0.091336021,
      0.120250926,
      0.193484435
      ]

Y2 = [35.7,      31.2,      25.8,      20.7,]
Y3 = [35.0,
      30.6,
      25.4,
      20.3,
      ]


pcc_low = 0
pcc_high = 40
rmse_low = 0.07
rmse_high = 0.20
Model = "tCNNs"
draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high)


# =================================

X = [3, 2, 1, 0]
Y1 = [0.0567,
      0.0572,
      0.0578,
      0.0587,

      ]

Y2 = [47.6,
      47.0,
      46.0,
      44.4,]


Y3 = [47.2,
      46.6,
      45.6,
      43.9,
      ]
pcc_low = 43
pcc_high = 48
rmse_low = 0.056
rmse_high = 0.059
Model = "DeepCDR"
draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high)


# =================================

X = [3, 2, 1, 0]
Y1 = [0.0604,
      0.0615,
      0.0630,
      0.0637,

      ]

Y2 = [46.4,
      45.6,
      44.6,
      44.0,]


Y3 = [46.8,
      45.9,
      45.0,
      44.5,

      ]

rmse_low = 0.06
rmse_high = 0.065

pcc_low = 43
pcc_high = 47

Model = "GraphDRP"


draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high)

# =================================

X = [3, 2, 1, 0]
Y1 = [0.0561,
      0.0575,
      0.0589,
      0.0595,

      ]

Y2 = [50.3,
      49.5,
      48.8,
      48.5,]


Y3 = [50.5,
      49.7,
      49.0,
      48.5,

      ]

rmse_low = 0.055
rmse_high = 0.060

pcc_low = 48
pcc_high = 51

Model = "GratransDRP"


draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high)


# =================================

X = [3, 2, 1, 0]
Y1 = [0.0573,
      0.0576,
      0.0588,
      0.0624,

      ]

Y2 = [53.7,
      53.1,
      51.8,
      50.6,]


Y3 = [53.7,
      53.3,
      51.8,
      50.4,

      ]

rmse_low = 0.057
rmse_high = 0.063

pcc_low = 50
pcc_high = 54

Model = "TransEDRP"


draw_combine(Model, X, Y1, Y2, Y3, rmse_low, rmse_high, pcc_low, pcc_high)
