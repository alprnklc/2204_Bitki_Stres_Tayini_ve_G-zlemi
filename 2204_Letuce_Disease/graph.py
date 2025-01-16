import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Şema için figür ve eksen oluştur
fig, ax = plt.subplots(figsize=(12, 10))

# Kutular ve bağlantılar için konumlar
steps = [
    {"name": "Başla", "xy": (0.5, 0.9)},
    {"name": "Kütüphaneleri Yükle\nEksikse Kur", "xy": (0.5, 0.8)},
    {"name": "Girdi Al\n(Sağlıklı, Hasta, GT)", "xy": (0.5, 0.7)},
    {"name": "Genetik Algoritma\nTanımları", "xy": (0.5, 0.6)},
    {"name": "Popülasyon Oluştur", "xy": (0.3, 0.5)},
    {"name": "Jenerasyon Döngüsü:\nFitness, Çaprazlama,\nMutasyon, Seçim", "xy": (0.7, 0.5)},
    {"name": "En İyi Bireyi Seç", "xy": (0.5, 0.4)},
    {"name": "Maske Üretimi", "xy": (0.5, 0.3)},
    {"name": "Sonuçları\nGörüntüle & Kaydet", "xy": (0.5, 0.2)},
    {"name": "Son", "xy": (0.5, 0.1)},
]

# Kutuları çiz
for step in steps:
    rect = patches.Rectangle(
        (step["xy"][0] - 0.15, step["xy"][1] - 0.05),
        0.3,
        0.1,
        edgecolor="black",
        facecolor="lightblue",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        step["xy"][0],
        step["xy"][1],
        step["name"],
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )

# Oklar için bağlantılar
arrows = [
    (steps[0]["xy"], steps[1]["xy"]),
    (steps[1]["xy"], steps[2]["xy"]),
    (steps[2]["xy"], steps[3]["xy"]),
    (steps[3]["xy"], steps[4]["xy"]),
    (steps[3]["xy"], steps[5]["xy"]),
    (steps[4]["xy"], steps[6]["xy"]),
    (steps[5]["xy"], steps[6]["xy"]),
    (steps[6]["xy"], steps[7]["xy"]),
    (steps[7]["xy"], steps[8]["xy"]),
    (steps[8]["xy"], steps[9]["xy"]),
]

# Okları çiz
for start, end in arrows:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

# Görünüm ayarları
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")  # Eksenleri gizle
plt.title("Genetik Algoritma Akış Şeması", fontsize=14, fontweight="bold")
plt.show()
