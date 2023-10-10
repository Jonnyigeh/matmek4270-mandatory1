import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij", sparse=sparse)
        self.dx = x[1] - x[0]
        self.dy = self.dx
        

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2

        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * sp.pi * sp.sqrt(self.mx ** 2 + self.my ** 2)         # Found using the dispersion relation

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
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))
        self.u_exact = sp.lambdify((x,y,t), self.ue(self.mx, self.my))

        self.Unm1[:] = self.u_exact(self.xij, self.yij, 0)
        self.Un[:] = self.Unm1 + 0.5 * (self.c * self.dt) **2 * (self.D @ self.Unm1 + self.Unm1 @ self.D.T)


    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        
        uexact = self.u_exact(self.xij, self.yij, t0)
        l2error = np.sqrt(self.dx * self.dy * np.sum((u - uexact) ** 2))

        return l2error
        

    def apply_bcs(self):
        """Applies Dirichlet boundary conditions (i.e 0 at the entire boundary)"""
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
        self.c = c
        self.cfl = cfl
        self.mx, self.my = mx, my
        

        self.create_mesh(N)
        self.D = self.D2(N=N) / self.dx ** 2
        self.initialize(N, mx, my)
        
        
        plotdata = {}
        l2error = []
        l2error.append(self.l2_error(self.Un, self.dt))         # Evaluated at time = dt which is one timestep into the solution.
       
        for i in range(1, Nt):
            self.Unp1[:] = 2 * self.Un - self.Unm1 + (self.c * self.dt) **2 * (self.D @ self.Un + self.Un @ self.D.T)
            self.apply_bcs()

            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if i % store_data == 0:
                plotdata[i] = self.Unm1.copy()        # Nødvendig å gange med self.dt her? Sjekk m convergence rate function
            l2error.append(self.l2_error(self.Un, (i+1) * self.dt)) # Add one timestep since Un is in reality Unp1
        if store_data == -1:
            return (self.dx, l2error[:])
        
        return self.xij, self.yij, plotdata


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
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        
        return D

    def ue(self, mx, my):
        return sp.cos( mx * sp.pi * x) * sp.cos( my * sp.pi * y) * sp.cos(self.w * t) # Eq. 1.5 in assignment

    def apply_bcs(self):
        pass            # The Von Neumann boundary conditions are handled in the D-matrix

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
    m = 2
    cfl = 1 / np.sqrt(2)
    _, E = sol(N=3000, Nt = 50, cfl = cfl, mx = m, my = m)
    _, E_N,  = solN(N=100, Nt = 50, cfl = cfl, mx = m, my = m)
    err = E[-1]
    err_N = E_N[-1]
    tol = 1e-15
    assert err < tol and err_N < tol
    



if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    
    create_movie = True
    if create_movie:
        import matplotlib.animation as animation
    
        sol = Wave2D_Neumann()
        m = 2
        cfl = 1 / np.sqrt(2)
        xij, yij, plotdata = sol(N=100, Nt=50, cfl=cfl, mx=m, my=m, store_data=1)

        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        frames = []
        for n, val in plotdata.items():
            frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2)
            frames.append([frame])
        
        
        ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
        ani.save("report/wavemovie2d_neumann.gif", writer="pillow", fps=5)
    pass