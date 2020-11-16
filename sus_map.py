#!/usr/bin/python

"""
Plot the generalized susceptibility chi^(nu nu') map
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sus

plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rc('font',**{'size':18})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath}')
plt.rc('lines', lw=2)
plt.rc('lines', markersize=5)
plt.rc('axes', linewidth=1.8)

# import filename U and beta susceptibility (real/imag) Niwf from command line
parser = argparse.ArgumentParser()
parser.add_argument('file',
                    help="select file or 'AL' for Atomic Limit")
parser.add_argument("U",
                    help="select interaction U",
                    type=float)
parser.add_argument("beta",
                    help="select beta",
                    type=float)
parser.add_argument("susceptibility",
                    help="select susceptibility",
                    choices=['c', 's', 'uu', 'ud'])
parser.add_argument("--Niwf",
                    help="number of positive matsubara frequencies",
                    type=int)
parser.add_argument("--complex",
                    help="select real or imaginary part",
                    nargs='?',
                    default='real',
                    choices=['real', 'imag'])
args = parser.parse_args()

filename = args.file
U = args.U
beta = args.beta
s = args.susceptibility
Niwf =args.Niwf
c = args.complex

if filename == 'AL':
    if Niwf is None:
        Niwf = 200
    data = sus.al(U, beta, Niwf)
    print('AL for Atomic Limit is selected')
else:
    try:
        data = sus.get(filename,beta, U=U)
        if Niwf is None or Niwf > data.Niwf:
            Niwf = data.Niwf
    except IOError:
        print("IOError: No such file or directory: '", filename, "'")
        exit()

if c == 'real':
    susz = np.real(getattr(data,s))
else:
    susz = np.imag(getattr(data,s))


print('filename =', filename, ', U=', U, ', beta =', beta)
print(', susceptibility =', s, ', complex =', c, ', Niwf =', Niwf)
# ------------------------------------------
# plot
max_pix_num = 15
max_plot_fq = (2*max_pix_num+1)*np.pi/data.beta

# colormap
cm = mpl.colors.LinearSegmentedColormap(name='my_map', segmentdata=sus.cdict,
                                        N=10000)
cm_max = np.amax(susz/data.beta**2)


fig, axs = plt.subplots()
#fig.suptitle(filename+r':  $U = %.2f, \beta = %.1f$'%(U, beta))

axs.set_xticks(np.linspace(-int(max_plot_fq),int(max_plot_fq),4))
axs.set_yticks(np.linspace(-int(max_plot_fq),int(max_plot_fq),4))
axs.set_xlim(-max_plot_fq, max_plot_fq)
axs.set_ylim(max_plot_fq, -max_plot_fq)



#axs.set_title(r'$\chi_{'+s+r'}^{\nu\nu^{\prime}}/\beta$')
axs.set_xlabel(r'$\nu$')
axs.set_ylabel(r'$\nu^{\prime}$')
c = axs.imshow(sus.sub_matrix((susz/data.beta**2), 2*max_pix_num),
                interpolation='nearest',
                cmap=cm,
                vmin=-cm_max, vmax=cm_max,
                extent=[-max_plot_fq, max_plot_fq, max_plot_fq, -max_plot_fq],
                aspect="equal")
fig.colorbar(c, ax=axs)

plt.tight_layout()									#that everything fits on the plot

plt.savefig("Xc_b{}_575.eps".format(beta))
plt.savefig("Xc_b{}_575.pdf".format(beta))
plt.show()
