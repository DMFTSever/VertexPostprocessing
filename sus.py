#!/usr/bin/python

"""
Python module for easier handling of the generalized
susceptibility matrices generated from
'w2dynamics' calculations.
"""

import numpy as np

class ChiCollector:
    """
    A class to read generalized susceptibility (Chi) out of the vert_chi file
    for zero bosonic Matsubara (Mat) frequency (fq).

    The vert_chi file is expected to be of the form:
    bosonic_Mat_fq  fermionic_Mat_fq  (fermionic_Mat_fq)'  Re(Chi_uu)  Im(Chi_uu)  Re(Chi_ud)  Im(Chi_ud)

    It is assumed that:
        number of fermionic_Mat_fq == number of (fermionic_Mat_fq)'

    Values
    ------
    infile : str
        the filename of the vert_chi file
    beta : float
        inverse temperature beta of Chi
    U : float
        interaction U

    Attributes
    ----------
    infile : str
        the filename of the vert_chi file
    beta : float
        beta of chi
    luu_real : list
        list of all Re(Chi_uu) values
    luu_imag : list
        list of all I(Chi_uu) values
    lud_real : list
        list of all Re(Chi_ud) values
    lud_imag : list
        list of all I(Chi_ud) values
    length : int
        number of lines in vert_chi file
    Niwf : int
        number of positive Matsubara frequencies
    uu : np.array( , )
        matrix of Complex(Chi_uu) values
    ud : np.array( , )
        matrix of Complex(Chi_ud) values
    c : np.array( , )
        matrix of Complex(Chi_uu)+Complex(Chi_ud) values
    s : np.array( , )
        matrix of Complex(Chi_uu)-Complex(Chi_ud) values
    nu : np.array( )
        array of Matsubara frequencies

    Functions
    ---------
    get_uu() :
        returns self.uu
    get_ud() :
        returns self.ud
    get_c() :
        returns self.c
    get_s() :
        returns self.s
    get_nu() :
        returns self.nu
    """
    def __init__(self, infile, beta, U='nan'):
        self.infile   = infile
        self.beta     = float(beta)
        self.U        = float(U)
        self.luu_real = []
        self.luu_img  = []
        self.lud_real = []
        self.lud_img  = []
        with open(infile)as f:
            for i, line in enumerate(f.readlines()):
                data=line.split()
                self.luu_real.append(float(data[3]))
                self.luu_img.append(float(data[4]))
                self.lud_real.append(float(data[5]))
                self.lud_img.append(float(data[6]))
        self.length = i + 1
        self.Niwf   = int(np.sqrt(float(self.length))/2) # even total Number of Mat-fg
                                                         # assumed
        self.uu     = np.reshape(np.array(self.luu_real) + 1j*np.array(self.luu_img),\
                                 (2*self.Niwf,2*self.Niwf))
        self.ud     = np.reshape(np.array(self.lud_real) + 1j*np.array(self.lud_img),\
                                 (2*self.Niwf,2*self.Niwf))
        self.c      = (self.uu + self.ud)
        self.s      = (self.uu - self.ud)
        self.nu     = np.linspace(-(2*(self.Niwf-1)+1)*np.pi/self.beta,\
                                  (2*(self.Niwf-1)+1)*np.pi/self.beta,num=2*self.Niwf)

    def get_uu(self):
        """returns uu"""
        return self.uu.copy()

    def get_ud(self):
        """returns ud"""
        return self.ud.copy()

    def get_c(self):
        """returns c"""
        return self.c.copy()

    def get_s(self):
        """returns s"""
        return self.s.copy()

    def get_nu(self):
        """returns nu"""
        return self.nu.copy()

class FCollector:
    """
    class to read full vertex F from F_DM file for zero bosonic Matsubara (Mat) frequency (fq).
    The file is expected to be of the form:
    bosonic_Mat_fq  fermionic_Mat_fq  (fermionic_Mat_fq)'  Re(F_c)  Im(F_c)  Re(F_m)  Im(F_m)

    Values
    ------
    infile : str
        the filename of the F_DM file
    beta : float
        inverse temperature beta of Chi
    U : float
        interaction U

    Attributes
    ----------
    infile : str
        the filename of the F_DM file
    beta : float
        beta
    luu_real : list
        list of all Re(Chi_uu) values
    luu_imag : list
        list of all I(Chi_uu) values
    lud_real : list
        list of all Re(Chi_ud) values
    lud_imag : list
        list of all I(Chi_ud) values
    length : int
        number of lines in vert_chi file
    Niwf : int
        number of positive Matsubara frequencies
    uu : np.array( , )
        matrix of Complex(Chi_uu) values
    ud : np.array( , )
        matrix of Complex(Chi_ud) values
    c : np.array( , )
        matrix of Complex(Chi_uu)+Complex(Chi_ud) values
    s : np.array( , )
        matrix of Complex(Chi_uu)-Complex(Chi_ud) values
    nu : np.array( )
        array of Matsubara frequencies

    Functions
    ---------
    get_uu() :
        returns self.uu
    get_ud() :
        returns self.ud
    get_c() :
        returns self.c
    get_s() :
        returns self.s
    get_nu() :
        returns self.nu
    """
    def __init__(self, infile, beta, U='nan'):
        self.infile = infile
        self.beta   = float(beta)
        self.U      = float(U)
        F           = np.genfromtxt(infile, comments='#', usecols=(3,4,5,6))
        self.length = F.shape[0]
        self.Niwf   = int(np.sqrt(float(self.length))/2.) # even total Number of Mat-fg
                                                          # assumed
        self.c      = np.reshape(F[:,0] + 1j*F[:,1], (2*self.Niwf,2*self.Niwf))
        self.s      = np.reshape(F[:,2] + 1j*F[:,3], (2*self.Niwf,2*self.Niwf))
        self.uu     = 0.5*(self.c + self.s)
        self.ud     = 0.5*(self.c - self.s)
        self.nu     = np.linspace(-(2*(self.Niwf-1)+1)*np.pi/self.beta,\
                                  (2*(self.Niwf-1)+1)*np.pi/self.beta,num=2*self.Niwf)

    def get_uu(self):
        """returns uu"""
        return self.uu.copy()

    def get_ud(self):
        """returns ud"""
        return self.ud.copy()

    def get_c(self):
        """returns c"""
        return self.c.copy()

    def get_s(self):
        """returns s"""
        return self.s.copy()

    def get_nu(self):
        """returns nu"""
        return self.nu.copy()

class GammaCollector:
    """
    Like getF only for Gamma_DM
    """
    def __init__(self, infile, beta, U='nan'):
        self.infile = infile
        self.beta   = float(beta)
        self.U      = float(U)
        G           = np.genfromtxt(infile, comments='#', usecols=(3,4,5,6))
        self.length = G.shape[0]
        self.Niwf   = int(np.sqrt(float(self.length))/2.) # even total Number of Mat-fg
                                                          # assumed
        self.c      = np.reshape(G[:,0] + 1j*G[:,1], (2*self.Niwf,2*self.Niwf))
        self.s      = np.reshape(G[:,2] + 1j*G[:,3], (2*self.Niwf,2*self.Niwf))
        self.uu     = 0.5*(self.c + self.s)
        self.ud     = 0.5*(self.c - self.s)
        self.nu     = np.linspace(-(2*(self.Niwf-1)+1)*np.pi/self.beta,\
                                  (2*(self.Niwf-1)+1)*np.pi/self.beta,num=2*self.Niwf)

    def get_uu(self):
        """returns uu"""
        return self.uu.copy()

    def get_ud(self):
        """returns ud"""
        return self.ud.copy()

    def get_c(self):
        """returns c"""
        return self.c.copy()

    def get_s(self):
        """returns s"""
        return self.s.copy()

    def get_nu(self):
        """returns nu"""
        return self.nu.copy()

class AtomicLimit:
    """
    A class to calculate Chi_c and Chi_s for zero bosonic Matsubara (Mat) frequency (fq) in the Atomic Limit (AL).
    And also F and Gamma of the AL in all the channels.

    Values
    ------
    U : float
        U of the AL
    beta : float
        beta of the AL
    Niwf : int
        number of positiv Mat-fq
    mult : float
        default = 2.; to get finer or broughter fq-grid then the Mat-fq grid chance mult acordingly
    Attributes
    ----------
    U : float
        U of the AL
    beta : float
        beta of the AL
    Niwf : int
        number of positiv Mat-fq
    mult : float
        multiplicator of frequency grid resolution
        default = 2.; to get finer or broughter Mat-fq grid chance mult acordingly
    Xc : np.array( , )
        matrix of Re(Chi_uu)+Re(Chi_ud) values
    Xs : np.array( , )
        matrix of Re(Chi_uu)-Re(Chi_ud) values
    Xuu : np.array( , )
        matrix of Re(Chi_uu) values
    Xud : np.array( , )
        matrix of Re(Chi_ud) values
    Fc : np.array( , )
        matrix of Re(F_uu)+Re(F_ud) values
    Fs : np.array( , )
        matrix of Re(F_uu)-Re(F_ud) values
    Fuu : np.array( , )
        matrix of Re(F_uu) values
    Fud : np.array( , )
        matrix of Re(F_ud) values
    Gc : np.array( , )
        matrix of Re(G_uu)+Re(G_ud) values
    Gs : np.array( , )
        matrix of Re(G_uu)-Re(G_ud) values
    Guu : np.array( , )
        matrix of Re(G_uu) values
    Gud : np.array( , )
        matrix of Re(G_ud) values
    nu : np.array( )
        array of Matsubara frequencies

    Functions
    ---------
    get_Xuu() :
        returns self.Xuu
    get_Xud() :
        returns self.Xud
    get_Xc() :
        returns self.Xc
    get_Xs() :
        returns self.Xs
    get_Fuu() :
        returns self.Fuu
    get_Fud() :
        returns self.Fud
    get_Fc() :
        returns self.Fc
    get_Fs() :
        returns self.Fs
    get_Guu() :
        returns self.Guu
    get_Gud() :
        returns self.Gud
    get_Gc() :
        returns self.Gc
    get_Gs() :
        returns self.Gs
    get_nu() :
        returns self.nu
    """
    def __init__(self, U, beta ,Niwf, mult=2.):
        self.U    = float(U)
        self.beta = float(beta)
        self.Niwf = int(Niwf)
        self.nu   = np.linspace(-(2*(float(Niwf)-1)+1)*np.pi/beta,\
                                (2*(float(Niwf)-1)+1)*np.pi/beta,\
                                num=float(mult)*float(Niwf))
        # calculation ---------------------
        U2_4   = (1/4. * self.U**2)
        bU_2   = (self.beta * self.U/2.)
        Embu_2 = np.exp(-bU_2)
        Ad2    = 3 * U2_4
        Am2    = (-U2_4)
        Bd2    = U2_4 * (-Embu_2 + 3) / (1 + Embu_2)
        Bm2    = U2_4 * (-1 + 3. * Embu_2) / (1 + Embu_2)
        Cd     = bU_2 * Embu_2/ (1 + Embu_2)
        Cm     = (-bU_2 * 1./ (1 + Embu_2))
        nu2    = np.square(self.nu)
        denom  = (np.square(nu2)  +  2 * nu2 * U2_4  +  U2_4**2)
        adnu   = self.beta/2. * np.divide((nu2 - Ad2), denom)
        amnu   = self.beta/2. * np.divide((nu2 - Am2), denom)
        b0dnu  = self.beta/2. * np.divide((nu2 - Bd2), denom)
        b0mnu  = self.beta/2. * np.divide((nu2 - Bm2), denom)
        b1dnu  = np.divide((nu2 - U2_4 * (1+Cd)/(1-Cd)), denom)
        b1mnu  = np.divide((nu2 - U2_4 * (1+Cm)/(1-Cm)), denom)
        b2nu   = np.divide(1, denom)

        # ------------------------X Calculation
        self.Xc  = (  np.diag(adnu) -  np.diag(adnu)[:,::0]\
                    + np.diag(b0dnu) +  np.diag(b0dnu)[:,::-1]\
                    - self.U*(1-Cd) * np.tensordot(b1dnu, b1dnu, 0)\
                    + U2_4*self.U**3/(1-Cd) * np.tensordot(b3nu, b2nu, 0) )

        self.Xs  = (  np.diag(amnu) - np.diag(amnu)[:,::-1]\
                    + np.diag(b0mnu) + np.diag(b0mnu)[:,::-1]
                    + self.U*(1-Cm)*np.tensordot(b1mnu, b1mnu, 0)\
                    - U2_4*self.U**3/(1-Cm)*np.tensordot(b2nu, b2nu, 0))
        self.Xuu = (self.Xc + self.Xs)/2.
        self.Xud = (self.Xc - self.Xs)/2.

	    #--------------------F Calculation - Taken from PRB 86, 2012 by Georg
        F0       = np.divide((nu2 + U2_4),nu2)
        F1a      = np.exp(-bU_2)/(1.+np.exp(-bU_2))
        F1b      = 1./(1.+np.exp(-bU_2))
        F1c      = (1.-np.exp(-bU_2))/(1.+np.exp(-bU_2))

        self.Fuu = self.beta*U2_4*np.tensordot(F0,F0,0)
        np.fill_diagonal(self.Fuu,0.0)

        ones     = np.ones(2*self.Niwf)

        self.Fud = (- self.U*np.ones((2*self.Niwf,2*self.Niwf)) \
                    + self.U*U2_4*(np.tensordot(ones,1.0/nu2,0)\
                    + np.tensordot(1.0/nu2,ones,0))\
                    + 3.0*self.U*((U2_4)**2)*np.tensordot(1.0/nu2,1.0/nu2,0)\
                    - self.beta*U2_4*F1c*np.tensordot(F0,F0,0)\
                    + 2.0*self.beta*U2_4*F1a*np.diag(F0**2)[:,::-1]\
                    - 2.0*self.beta*U2_4*F1b*np.diag(F0**2) )

        self.Fc  = self.Fuu + self.Fud
        self.Fs  = self.Fuu - self.Fud

        GAL      = -self.nu/(nu2+U2_4)	#imaginary part of GF of AL

        X0       = np.diag(self.beta*GAL*GAL)		#considering i from ImagGF and \Omega=0

        #------------------------Gamma Calculation
        # Fingers crossed I didn't screw it up... Taken from PRB 98 (2018) by Georg
	    #-----check for imaginary parts:
        if (-1.0+3.0*Embu_2)<=0.0:
                IMAG = True
        else:
                IMAG = False


        G0d   = self.beta/2. * Ad2 *np.divide(denom,(nu2-Ad2)*nu2)	#first term
        G1d   = self.beta/2. * Bd2 *np.divide(denom,(nu2-Bd2)*nu2)	#second
        G0m   = self.beta/2. * Am2 *np.divide(denom,(nu2-Am2)*nu2)	#first term
        G1m   = self.beta/2. * Bm2 *np.divide(denom,(nu2-Bm2)*nu2)	#second

        #ATTENTION - attracive case is not taken into account
        Gnomd = self.U*np.tan(self.beta/2.*np.sqrt(Bd2))/(2.*np.sqrt(Bd2)) + 1.

        #if imaginary tan has to be reformulated in tanh(x) = -i*tan(ix)
        if IMAG:
            Gnomm = self.U*np.tanh(self.beta/2.*np.sqrt(-Bm2))/(2.*np.sqrt(-Bm2)) - 1.
        else:
            Gnomm = self.U*np.tan(self.beta/2.*np.sqrt(Bm2))/(2.*np.sqrt(Bm2)) - 1.

        self.Gc = ( np.diag(G0d) - np.diag(G0d)[:,::-1]\
                   + np.diag(G1d) + np.diag(G1d)[:,::-1]\
                   - np.divide((self.U*(Bd2+U2_4)**2),Gnomd)\
                     *np.tensordot(1./(nu2-Bd2),1./(nu2-Bd2),0)\
                   + self.U*np.ones((2*self.Niwf,2*self.Niwf)) )

        if Gnomm == 0.0:
                self.Gs  = np.zeros((2*self.Niwf,2*self.Niwf))
                self.Guu = np.zeros((2*self.Niwf,2*self.Niwf))
                self.Gud = np.zeros((2*self.Niwf,2*self.Niwf))
        else:
                self.Gs  = ( np.diag(G0m) - np.diag(G0m)[:,::-1]\
                           + np.diag(G1m) + np.diag(G1m)[:,::-1]\
                  	       - np.divide((self.U*(Bm2+U2_4)**2),Gnomm)\
                             *np.tensordot(1./(nu2-Bm2),1./(nu2-Bm2),0)\
                           - self.U*np.ones((2*self.Niwf,2*self.Niwf)))
                self.Guu = (self.Gc + self.Gs)/2.
                self.Gud = (self.Gc - self.Gs)/2.


    def get_Xuu(self):
        """returns uu"""
        return self.Xuu.copy()

    def get_Xud(self):
        """returns ud"""
        return self.Xud.copy()

    def get_Xc(self):
        """returns c"""
        return self.Xc.copy()

    def get_Xs(self):
        """returns s"""
        return self.Xs.copy()

    def get_Fuu(self):
        """returns uu"""
        return self.Fuu.copy()

    def get_Fud(self):
        """returns ud"""
        return self.Fud.copy()

    def get_Fc(self):
        """returns c"""
        return self.Fc.copy()

    def get_Fs(self):
        """returns s"""
        return self.Fs.copy()

    def get_Guu(self):
        """returns uu"""
        return self.Guu.copy()

    def get_Gud(self):
        """returns ud"""
        return self.Gud.copy()

    def get_Gc(self):
        """returns c"""
        return self.Gc.copy()

    def get_Gs(self):
        """returns s"""
        return self.Gs.copy()

    def get_nu(self):
        """returns nu"""
        return self.nu.copy()

#functions
def sub_matrix(matrix,N):
    """
    Returns n x n  numpy.matrix around mid of quadratic numpy.matrix

    Exampe: matrix=
               [[ 1, 2, 3, 4],
                [ 5, 6, 7, 8],
                [ 9,10,11,12],
                [13,14,15,16]]

    sub_matrix(matrix,2)=
                  [[6 , 7],
                   [10,11]]
    """
    if type(matrix) is np.ndarray:
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            mid = matrix.shape[0]/2
            if int(N) > matrix.shape[0]:
                print('Error: shape of submatrix greater then input matrix')
                print('input N =', N, 'is set to', matrix.shape[0])
                N = matrix.shape[0]
            if matrix.shape[0]%2 == 0:
                n = (int(N)/2)*2
                if n <2:
                    n=2
                if N%2 != 0 or N<2:
                    print('even input matrix')
                    print('input N =', N, 'is set to', n)
                return matrix[ (int(mid)-int((n+1)/2)):(int(mid)+int(n/2)),\
                               (int(mid)-int((n+1)/2)):(int(mid)+int(n/2)) ]
            else:
                n = (int(N)/2)*2+1
                if n <1:
                    n=1
                if N%2 == 0 or N<1:
                    print('uneven input matrix')
                    print('input N =', N, 'is set to', n)
                return matrix[ (int(mid)-int((n)/2)):(int(mid)+int((n+1)/2)),\
                               (int(mid)-int((n)/2)):(int(mid)+int((n+1)/2)) ]
        else:
            print('Error: sub_matrix() expecting quadratic two-dimensional matrix')
    else:
        print('TypeError: sub_matrix() expecting argument of type numpy.ndarray')

def off_diag(matrix):
    """
    Returns off diagonal values of the upper left and lower right submatrix as numpy.matrix

    Exampe: matrix=
                   [[ 1, 2, 3, 4],
                    [ 5, 6, 7, 8],
                    [ 9,10,11,12],
                    [13,14,15,16]]

    off_diag(matrix)=
                   [[ 0, 2, 0, 0],
                    [ 5, 0, 0, 0],
                    [ 0, 0, 0,12],
                    [ 0, 0,15, 0]]
    """
    if type(matrix) is np.ndarray:
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]\
                            and matrix.shape[0]%2 == 0:
            end                 = matrix.shape[0]
            half                = end/2
            new                 = np.copy(matrix)
            new[:half,half:end] = 0
            new[half:end,:half] = 0
            np.fill_diagonal(new,0)
            return new
        else:
            print('Error: off_diag() expecting quadratic even two-dimensional matrix')
    else:
        raise TypeError('off_diag() expecting argument of type numpy.ndarray')


def off_counter(matrix):
    """
    Returns off diagonal values of the upper right and lower left submatrix
    along the counter diagonal as numpy.matrix

    Exampe: matrix=
                   [[ 1, 2, 3, 4],
                    [ 5, 6, 7, 8],
                    [ 9,10,11,12],
                    [13,14,15,16]]

    off_counter(matrix)=
                   [[ 0, 0, 3, 4],
                    [ 0, 0, 7, 8],
                    [ 9,10, 0, 0],
                    [13,14, 0, 0]]
    """
    if type(matrix) is np.ndarray:
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0]%2 == 0:
            end                    = matrix.shape[0]
            half                   = end/2
            new                    = np.copy(matrix)
            new[:half,:half]       = 0
            new[half:end,half:end] = 0
            return new
        else:
            print('Error: off_counter() expecting quadratic even two-dimensional matrix')
    else:
        raise TypeError('off_counter() expecting argument of type numpy.ndarray')

# -------------------------------------
# colormap inspired by Patrick Chalupa
# -------------------------------------
cdict = {'blue':  [[0.0, 0.6, 0.6],
                   [0.499, 1.0, 1.0],
                   [0.5, 0.0, 0.0],
                   #[0.5, 1.0, 1.0],
                   #[0.501, 0.0, 0.0],
                   [1.0, 0., 0.]],
         'green': [[0.0, 0.0, 0.0],
                   [0.02631578947368421, 7.673360394717657e-06, 7.673360394717657e-06],
                   [0.05263157894736842, 0.00012277376631548252, 0.00012277376631548252],
                   [0.07894736842105263, 0.0006215421919721302, 0.0006215421919721302],
                   [0.10526315789473684, 0.0019643802610477203, 0.0019643802610477203],
                   [0.13157894736842105, 0.004795850246698536, 0.004795850246698536],
                   [0.15789473684210525, 0.009944675071554084, 0.009944675071554084],
                   [0.18421052631578946, 0.018423738307717093, 0.018423738307717093],
                   [0.21052631578947367, 0.031430084176763524, 0.031430084176763524],
                   [0.23684210526315788, 0.050344917549742546, 0.050344917549742546],
                   [0.2631578947368421, 0.07673360394717657, 0.07673360394717657],
                   [0.2894736842105263, 0.11234566953906126, 0.11234566953906126],
                   [0.3157894736842105, 0.15911480114486534, 0.15911480114486534],
                   [0.3421052631578947, 0.21915884623353094, 0.21915884623353094],
                   [0.3684210526315789, 0.2947798129234735, 0.2947798129234735],
                   [0.39473684210526316, 0.3884638699825815, 0.3884638699825815],
                   [0.42105263157894735, 0.5028813468282164, 0.5028813468282164],
                   [0.4473684210526315, 0.6408867335272133, 0.6408867335272133],
                   [0.47368421052631576, 0.8055186807958807, 0.8055186807958807],
                   [0.499, 1.0, 1.0],
                   [0.5, 0.0, 0.0],
                   #[0.5, 1.0, 1.0],
                   [0.501, 1.0, 1.0],
                   [0.5263157894736843, 0.8055186807958807, 0.8055186807958807],
                   [0.5526315789473685, 0.6408867335272133, 0.6408867335272133],
                   [0.5789473684210527, 0.5028813468282164, 0.5028813468282164],
                   [0.6052631578947368, 0.3884638699825815, 0.3884638699825815],
                   [0.631578947368421, 0.2947798129234735, 0.2947798129234735],
                   [0.6578947368421053, 0.21915884623353094, 0.21915884623353094],
                   [0.6842105263157895, 0.15911480114486534, 0.15911480114486534],
                   [0.7105263157894737, 0.11234566953906126, 0.11234566953906126],
                   [0.736842105263158, 0.07673360394717657, 0.07673360394717657],
                   [0.7631578947368421, 0.050344917549742546, 0.050344917549742546],
                   [0.7894736842105263, 0.031430084176763524, 0.031430084176763524],
                   [0.8157894736842105, 0.018423738307717093, 0.018423738307717093],
                   [0.8421052631578947, 0.009944675071554084, 0.009944675071554084],
                   [0.868421052631579, 0.004795850246698536, 0.004795850246698536],
                   [0.8947368421052632, 0.0019643802610477203, 0.0019643802610477203],
                   [0.9210526315789473, 0.0006215421919721302, 0.0006215421919721302],
                   [0.9473684210526316, 0.00012277376631548252, 0.00012277376631548252],
                   [0.9736842105263158, 7.673360394717657e-06, 7.673360394717657e-06],
                   [1.0, 0.0, 0.0]],
         'red':   [[0.0, 0., 0.],
                   #[0.499, 0.0, 0.0],
                   #[0.5, 1.0, 1.0],
                   [0.5, 0.0, 0.0],
                   [0.501, 1.0, 1.0],
                   [1.0, 0.6, 0.6]]}

if __name__ == "__main__":
    exit()

