import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

bMetadataSample = "Raw_Data/Brett_RawData/Brett_MetaData/device.csv"
lMetadataSample = "Raw_Data/Logan_data/Jumping_backpocket/meta/device.csv"
vMetadataSample = "Raw_Data/Vince_Data/Jump1_Hand/meta/device.csv"

v = pd.read_csv(vMetadataSample)
l = pd.read_csv(lMetadataSample)
b = pd.read_csv(bMetadataSample)

# Each CSV has a "property" column and a "value" column.
# Set "property" as the index so values can be accessed by row name.
v = v.set_index("property")["value"]
l = l.set_index("property")["value"]
b = b.set_index("property")["value"]

label_v = v["deviceModel"].title()
label_l = l["deviceModel"].title()
label_b = b["deviceModel"]

# Linear accelerometer rates (Hz)
# MinDelay (µs) → max rate;  MaxDelay (µs) → min rate

max_rate_v = 1_000_000 / float(v["linear_acceleration MinDelay"])
min_rate_v = 1_000_000 / float(v["linear_acceleration MaxDelay"])

max_rate_l = 1_000_000 / float(l["linear_acceleration MinDelay"])
min_rate_l = 1_000_000 / float(l["linear_acceleration MaxDelay"])

# Manufacturers

mfr_v = v["deviceManufacturer"].title()
mfr_l = l["deviceManufacturer"].title()
mfr_b = b["deviceBrand"]


# PLOT 1 – Linear accelerometer sampling rate (Android devices only)

fig1, ax1 = plt.subplots(figsize=(8, 5))

labels    = [label_v,    label_l   ]
min_rates = [min_rate_v, min_rate_l]
max_rates = [max_rate_v, max_rate_l]

x     = np.arange(len(labels))
width = 0.35

bars_min = ax1.bar(x - width / 2, min_rates, width,
                   label="Min Rate (Hz)", color="#5B9BD5", edgecolor="white", linewidth=0.8)
bars_max = ax1.bar(x + width / 2, max_rates, width,
                   label="Max Rate (Hz)", color="#ED7D31", edgecolor="white", linewidth=0.8)

for bar in list(bars_min) + list(bars_max):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + max(max_rates) * 0.01,
             f"{h:.1f}", ha="center", va="bottom", fontsize=10, color="#333")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel("Sampling Rate (Hz)", fontsize=11)
ax1.set_title("Linear Accelerometer Sampling Rate by Device", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_ylim(0, max(max_rates) * 1.20)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("plot1_lin_acc_rates.png", dpi=150, bbox_inches="tight")
plt.show()


# PLOT 2 – Device manufacturer overview (all 3 phones)

device_labels = [label_v, label_l, label_b]
manufacturers = [mfr_v,   mfr_l,   mfr_b  ]

unique_mfrs = list(dict.fromkeys(manufacturers))
palette     = ["#5B9BD5", "#ED7D31", "#70AD47"]
color_map   = {m: palette[i] for i, m in enumerate(unique_mfrs)}
bar_colors  = [color_map[m] for m in manufacturers]

fig2, ax2 = plt.subplots(figsize=(8, 4))

y      = np.arange(len(device_labels))
height = 0.45

ax2.barh(y, [1] * len(device_labels), height,
         color=bar_colors, edgecolor="white", linewidth=0.8)

for i, (model, mfr) in enumerate(zip(device_labels, manufacturers)):
    ax2.text(0.03, i, f"{model}  —  {mfr}",
             va="center", ha="left", fontsize=11, color="white", fontweight="bold")

legend_patches = [mpatches.Patch(color=color_map[m], label=m) for m in unique_mfrs]
ax2.legend(handles=legend_patches, title="Manufacturer",
           loc="lower right", fontsize=10, title_fontsize=10, framealpha=0.9)

ax2.set_yticks(y)
ax2.set_yticklabels(device_labels, fontsize=11)
ax2.set_xlim(0, 1)
ax2.set_xticks([])
ax2.set_title("Device Manufacturers", fontsize=13, fontweight="bold")
ax2.spines[["top", "right", "bottom", "left"]].set_visible(False)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig("plot2_manufacturers.png", dpi=150, bbox_inches="tight")
plt.show()
