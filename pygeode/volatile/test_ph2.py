
import plot_wr_ph as pl
import numpy as np

reload(pl)

Ax = pl.plot(range(5))
Ax.setp(title='Ax 1', xlim=(0, 5), xlabel='bob')

Ax2 = pl.AxesWrapper(size=(3,2))
pl.plot(range(10), -np.arange(10), axes=Ax2)
Ax2.setp(title='Ax 2', xlim=(-1, 11), xlabel='bob')

Ax3 = pl.AxesWrapper(size=(3,4))
Ax3.plot(range(10), -np.sin(np.arange(10)))
Ax3.setp(title='Ax 3', xlim=(-1, 11), xlabel='sin')

Ax4 = pl.AxesWrapper(size=(3,4))
Ax4.plot(range(10), -np.sin(np.arange(10)))
Ax4.setp(title='Ax 4', xlim=(-1, 11), xlabel='bob')

AxG = pl.grid([[Ax, Ax2], [Ax3, Ax4]])
AxG.render(4)
