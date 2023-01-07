import math
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']


\mathbb{F}


\mathbb{R}

\mathbb{C}


\mathbb{N}


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
lu1 = mplt.image.imread("../images/lu-iteration.png")
ax.imshow(lu1)
ax.axis('off')
plt.savefig("../images/lu-iteration.svg", dpi=300, bbox_inches='tight')
