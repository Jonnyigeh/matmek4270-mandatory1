import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, self.L, N+1)
        y = np.linspace(0, self.L, N+1)
        self.dx = x[1] - x[0]                   # Grid spacing
        self.dy = self.dx
        self.h = self.dx
        self.xij, self.yij = np.meshgrid(x,y, indexing="ij")   # Stores the mesh in class
        

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D
        

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = (1 / self.dx**2) * self.D2()
        D2y = (1 / self.dy**2) * self.D2()
        return (sparse.kron(D2x, sparse.eye(self.N+1)) + 
                    sparse.kron(sparse.eye(self.N+1), D2y))

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1))
        B[1:-1, 1:-1] = 0
        self.boundary_indices = np.where(B.ravel() == 1)[0]
        

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        self.get_boundary_indices()    
        A = self.laplace().tolil()
        for i in self.boundary_indices:
            A[i] = 0
            A[i, i] = 1
        # A = A.tocsr()
        self.F = sp.lambdify((x,y), self.f)(self.xij, self.yij)
        b = self.F.ravel()
        self.u_exact = sp.lambdify((x,y), self.ue)(self.xij, self.yij)
        b[self.boundary_indices] = self.u_exact.ravel()[self.boundary_indices]            # boundary conditions

        return A, b

    def l2_error(self, u):
        """Return l2-error norm"""
        
        l2error = np.sqrt(self.dx * self.dy * np.sum((u - self.u_exact) ** 2))
        return l2error

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        self.N = N
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A.tocsr(), b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation by 4 point (bilinear) interpolation

        Returns
        -------
        The value of u(x, y)

        """

        nx = int(x // self.dx)         # Index of nearest gridpoint to x
        ny = int(y // self.dy)
        spacing_x = (x - nx * self.dx) / self.h
        spacing_y = (y - ny * self.dy) / self.h

        w1 = (1 - spacing_x) * (1 - spacing_y)
        w2 = (1 - spacing_y) * spacing_x
        w3 = (1 - spacing_x) * spacing_y
        w4 = spacing_x * spacing_y

        evaluation = ( w1 * self.U[nx, ny] + w2 * self.U[nx + 1, ny]
                        + w3 * self.U[nx, ny + 1] + w4 * self.U[nx +1, ny + 1]    
        )
        return evaluation
    
        

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation
    