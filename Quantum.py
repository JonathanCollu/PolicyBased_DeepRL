import numpy as np
import qiskit
import torch
import warnings

warnings.filterwarnings("ignore")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

class Ansatz():
    def __init__(self, n_qubits, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.circuit.QuantumCircuit(n_qubits)
        all_qubits = np.arange(n_qubits).tolist()
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = qiskit.Aer.get_backend('aer_simulator')
        self.shots = shots

    def run(self, params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembled_qc = qiskit.assemble(self.circuit, shots=self.shots, parameter_binds = [{self.theta: param} for param in params])
            fxn()    
        result = self.backend.run(assembled_qc).result().get_counts()[0]
        # return probabilities for each action
        return torch.tensor(list(result.values())) / self.shots

class HybridFunction(torch.autograd.Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        ctx.shift = shift
        ctx.circuit = quantum_circuit
        result = ctx.circuit.run(input[0].tolist())
        ctx.save_for_backward(input, result)
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        input, _ = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = [ctx.circuit.run(shift_right[i]) - ctx.circuit.run(shift_left[i]) for i in range(len(input_list))]
        return torch.tensor(np.array([g.numpy() for g in gradients]).T).float() * grad_output.float(), None, None

class Hybrid(torch.nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, n_qubits, shots):
        super(Hybrid, self).__init__()
        self.quantum_circuit = Ansatz(n_qubits, shots)
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, np.pi / 2)