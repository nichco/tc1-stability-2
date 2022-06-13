import csdl
import csdl_om
import numpy as np


class Eig_Lat(csdl.Model):

    def initialize(self):
        # size and value of A
        self.parameters.declare('size')
        self.parameters.declare('val')

    def define(self):
        # size and value of A
        size = self.parameters['size']
        shape = (size, size)
        val = self.parameters['val']

        # Create A matrix
        A_lat = self.create_input('A_lat', val=val)

        # custom operation insertion
        e_r, e_i = csdl.custom(A_lat, op=EigExplicit(size=size))

        # eigenvalues as output
        self.register_output('e_real_lat', e_r)
        self.register_output('e_imag_lat', e_i)


class EigExplicit(csdl.CustomExplicitOperation):

    def initialize(self):
        # size of A
        self.parameters.declare('size')

    def define(self):
        # size of A
        size = self.parameters['size']
        shape = (size, size)

        # Input: Matrix
        self.add_input('A_lat', shape=shape)

        # Output: Eigenvalues
        self.add_output('e_real_lat', shape=size)
        self.add_output('e_imag_lat', shape=size)

        self.declare_derivatives('e_real_lat', 'A_lat')
        self.declare_derivatives('e_imag_lat', 'A_lat')

    def compute(self, inputs, outputs):

        # Numpy eigenvalues
        w, v = np.linalg.eig(inputs['A_lat'])
        
        # lateral eigenvalue classification
        # First find the dutch roll, which is the only oscillatory mode
        # Search in the real part of the eig_val for a duplicate entry
        eig_vals_real = np.real(w)
        
        u, c = np.unique(eig_vals_real, return_counts=True)
        dup = u[c > 1]
        # Find the locations of that duplicate entry
        loc_DLam = np.where(eig_vals_real == dup)
        assert loc_DLam[0].shape[0] == 2
        dr_eig = w[loc_DLam[0][0]]

        # Delete the eigenvalues of the dutch roll mode
        temp = np.delete(w, loc_DLam[0])

        # Roll subsidence, identified as the highest damped mode
        loc_RLam = np.where(temp == np.amin(temp))  # Most damped
        rr_eig = temp[loc_RLam[0][0]]
        
        # Spiral mode, identified as the least damped mode
        loc_SLam = np.where(temp == np.amax(temp))
        ss_eig = temp[loc_SLam[0][0]]

        
        e_real_dr = np.real(dr_eig)
        e_imag_dr = np.imag(dr_eig)
        e_real_rr = np.real(rr_eig)
        e_imag_rr = np.imag(rr_eig)
        e_real_ss = np.real(ss_eig)
        e_imag_ss = np.imag(ss_eig)
        
        # order: dr, dr, rr, ss
        e_real = [e_real_dr, e_real_dr, e_real_rr, e_real_ss]
        e_imag = [e_imag_dr, e_imag_dr, e_imag_rr, e_imag_ss]
        
        outputs['e_real_lat'] = 1*e_real
        outputs['e_imag_lat'] = 1*e_imag

    def compute_derivatives(self, inputs, derivatives):
        size = self.parameters['size']
        shape = (size, size)

        # v are the eigenvectors in each columns
        w, v = np.linalg.eig(inputs['A_lat'])

        # v inverse transpose
        v_inv_T = (np.linalg.inv(v)).T

        # preallocate Jacobian: n outputs, n^2 inputs
        temp_r = np.zeros((size, size*size))
        temp_i = np.zeros((size, size*size))

        for j in range(len(w)):

            # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
            partial = np.outer(v[:, j], v_inv_T[:, j]).flatten(order='F')
            # Note that the order of flattening matters, hence argument in flatten()

            # Set jacobian rows
            temp_r[j, :] = np.real(partial)
            temp_i[j, :] = np.imag(partial)

        # Set Jacobian
        derivatives['e_real_lat', 'A_lat'] = temp_r
        derivatives['e_imag_lat', 'A_lat'] = temp_i