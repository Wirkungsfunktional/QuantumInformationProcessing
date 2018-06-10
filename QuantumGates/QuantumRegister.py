

class QbitRegister():
    def __init__(self, n, state):
        self.number_of_qbits = n
        self.qbits = state
        self.single_qbit_in_comp_basis = [np.array([1, 0]), np.array([0, 1])]

    def get_register_in_comp_basis(self):
        ret = self.single_qbit_in_comp_basis[self.qbits[0]]
        for i in range(1, self.number_of_qbits):
            ret = np.outer(self.single_qbit_in_comp_basis[self.qbits[i]], ret).flatten()
        return ret

    def get_desity_matrix_in_comp_basis(self):
        state_in_comp_basis = self.get_register_in_comp_basis()
        return np.outer(state_in_comp_basis, state_in_comp_basis)
