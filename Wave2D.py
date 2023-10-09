import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        xm = np.linspace(0, 1, N+1)
        ym = np.linspace(0, 1, N+1)
        mesh = np.meshgrid(xm, ym, indexing='ij', sparse = sparse)
        self.xij, self.yij = mesh
        self.N = N
        self.dx = 1/N
        return self.xij, self.yij, self.dx, self.N

    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*sp.pi
        ky = self.my*sp.pi
        return sp.sqrt(kx**2 + ky**2)*self.c

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Un, Unm1 = np.zeros((2, N+1, N+1))
        ue = self.ue(mx,my)
        U = ue.subs(t, 0)
        D = self.D2(N)/self.dx**2
        Unm1[:] = sp.lambdify((x, y), U)(self.xij, self.yij)
        Un[:] = Unm1[:] + 0.5*(self.c*self.dt)**2*(D @ Unm1 + Unm1 @ D.T)
        self.Unm1 = Unm1
        self.Un = Un
        return self.Unm1, self.Un

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.dx/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = self.ue(self.mx,self.my)
        ue = ue.subs(t,t0)
        return np.sqrt(self.dx**2*np.sum((u - sp.lambdify((x, y), ue)(self.xij, self.yij))**2))

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.mx = mx
        self.my = my
        D = self.D2(N)/self.dx**2
        self.cfl = cfl
        self.c = c
        dt = self.dt
        E = np.zeros(Nt+1)
        
    
        self.initialize(N,mx,my)
        self.Unp1 = np.zeros((N+1,N+1))
        E[0] = self.l2_error(self.Unm1, 0)
        E[1] = self.l2_error(self.Un, dt)
        plotdata = {0: self.Unm1.copy()}
        if store_data > 0:
            plotdata[1] = self.Un.copy()
        for n in range(1, Nt):
            self.Unp1 = 2*self.Un - self.Unm1 + (c*dt)**2 * (D @ self.Un + self.Un @ D.T)
            self.apply_bcs()
            self.Unm1 = self.Un
            self.Un = self.Unp1
            if n % store_data == 0: 
                plotdata[n] = self.Unm1.copy()
            E[n+1] = self.l2_error(self.Un, (n+1)*dt)
        if store_data > 0:
            return plotdata    
        if store_data == -1:
            h = 1/N
            return h, E

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :4] = -2, 2, 0, 0
        D2[-1, -4:] = 0, 0, 2, -2
        return D2

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    solN = Wave2D_Neumann()
    root2 = 2**(0.5)
    h, E = sol(N = 40, Nt = 10, cfl = 1/root2, mx=2, my=2)
    hN, EN = solN(N = 40, Nt = 10, cfl = 1/root2, mx=2, my=2)
    assert E.max() < 1e-15
    assert EN.max() < 1e-15