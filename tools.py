#Tools
import  numpy as np 
import itertools

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister 
from qiskit import Aer, execute
from qiskit.quantum_info.operators import Operator

from qiskit.aqua.algorithms import NumPyEigensolver
###########################################################################################################################
#Match gate tools:

# Pauli matrix
I=np.array([[1,0],[0,1]],dtype=np.complex128)
X=np.array([[0,1],[1,0]],dtype=np.complex128)
Y=np.array([[0,-1.0j],[1.0j,0]],dtype=np.complex128)
Z=np.array([[1,0],[0,-1]],dtype=np.complex128)

# We use these funcion to test if our 4x4 matrix are matchgate matrix.
def Match_gate_test(matrice):
    if matrice[0][1]!=0 or matrice[0][2]!=0 or matrice[1][0]!=0 or matrice[1][3]!=0 or matrice[2][0]!=0 or matrice[2][3]!=0 or matrice[3][1]!=0 or matrice[3][2]!=0 :
        print("Zero")
        return 0
    else:
        if (matrice[1][1]*matrice[2][2]-matrice[1][2]*matrice[2][1])==(matrice[0][0]*matrice[3][3]-matrice[0][3]*matrice[3][0]):
            return 1
        else:
            print("det(A)!= det(B)")
            return 0
############################################################################################################################

############################################################################################################################
#Expected value
# Pauli matrix
I=np.array([[1,0],[0,1]],dtype=np.complex128)
X=np.array([[0,1],[1,0]],dtype=np.complex128)
Y=np.array([[0,-1.0j],[1.0j,0]],dtype=np.complex128)
Z=np.array([[1,0],[0,-1]],dtype=np.complex128)

pauli=[I,X,Y,Z]
labels=['I','X','Y','Z']
indice=[0,1,2,3]

# Set signature:
# Z=|0><0|-|1><1|; ZZ=|00><00|-|01><01|-|10><10|+|11><11|   ZI=|00><00|+|01><01|-|10><10|-|11><11| IZ=|00><00|-|01><01|+|10><10|-|11><11|
def proces_counts( counts,z_index):

    z_index.sort(reverse=True) 
    new_counts = {}
    for key in counts:
        new_key = ''
        for index in z_index:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts

# Calculate expectation value from counts 
def expect_z(counts,shots,z_index=[]):
    
    if len(z_index)==0:
        #print("Zeroo")
        return 1
       
    else:
        z_counts=proces_counts(counts,z_index)
    #print(z_counts)
    expectation=0
    for key in z_counts:
        sign=-1
        #print(key)
        if key.count('1')%2==0:
            sign=1
        expectation= expectation+sign*z_counts[key]/shots
    return expectation

# IZXIIYII->IZZIIZII
def measure_qc(qc,Obs):
    m_qc=qc.copy()
    m_qc.barrier()
    for i in range(len(Obs)):
        if(Obs[i]=='Z')or(Obs[i]=='I'):
            m_qc.measure(i,i)
        if(Obs[i]=='X'):
            m_qc.h(i)
            m_qc.measure(i,i)
        if(Obs[i]=='Y'):
            m_qc.rx(np.pi/2,i)
            m_qc.measure(i,i) 
    return m_qc

# Return expected value of an Observable(Obs) for a state prepare by the circuit qc :
def expected(qc,Obs,shots,backend=Aer.get_backend('qasm_simulator')):
    mc=measure_qc(qc,Obs)
    counts=execute(mc,backend=backend,shots=shots).result().get_counts(mc)
    #print(counts)
    z_index=[]
    for i in range (len(Obs)):
        if(Obs[i]!='I'):
            z_index.append(i)
    return expect_z(counts,shots,z_index)

############################################################################################################################

############################################################################################################################
#Pauli decomposition:

# Pauli matrix
I=np.array([[1,0],[0,1]],dtype=np.complex128)
X=np.array([[0,1],[1,0]],dtype=np.complex128)
Y=np.array([[0,-1.0j],[1.0j,0]],dtype=np.complex128)
Z=np.array([[1,0],[0,-1]],dtype=np.complex128)

pauli=[I,X,Y,Z]
labels=['I','X','Y','Z']
indice=[0,1,2,3]

# Make Hilbert Schmid product betwen matrices mat1 and mat2
def HS(mat1,mat2): 
    return(np.dot(mat1.conjugate().transpose(), mat2)).trace()

# Decompose  observable in matrices create by kron. product o Pauli matrices (h-coeficient , h_label-kron product structure )
def decompose(O):
    size=len(O)
    nr_pauli=np.log2(size)
    norm_fact=1/(2**nr_pauli)

    elements=itertools.product(indice,repeat=int(nr_pauli))
    h_label=[]
    h=[]
    for i in elements:
        label=''
        matrice=pauli[i[0]]
        for  j in i :
            label=label+labels[indice[j]]
        for j in range(int(nr_pauli)-1):
            matrice=np.kron(matrice,pauli[i[j+1]])
        #print(matrice)
        h_label.append(label)
        h.append(norm_fact*HS(matrice,O))

    return h,h_label

############################################################################################################################

############################################################################################################################
#Compress:

#Pauli set:
I=np.array([[1,0],[0,1]],dtype=np.complex128)
X=np.array([[0,1],[1,0]],dtype=np.complex128)
Y=np.array([[0,-1.0j],[1.0j,0]],dtype=np.complex128)
Z=np.array([[1,0],[0,-1]],dtype=np.complex128)

# Operators for Jordan-Wigner mapping
def get_x(nrq):
    x=[]
    for i in range (nrq):
        x2=X
        x21=Y
        for k in range(i):
            x2=np.kron(Z,x2)
            x21=np.kron(Z,x21)
        for k in range(i+1,nrq):
            x2=np.kron(x2,I)
            x21=np.kron(x21,I)
        x.append(x2)
        x.append(x21)
    return x

def comutation(mat1,mat2):
    return np.matmul(mat1,mat2)-np.matmul(mat2,mat1)


def corelation(dens,op):
    nrq=int(np.log2(len(dens)))
    x=op.elements
    #print(x)
    C=[]
    for  i in range(2*nrq):
        li=[]
        for j in range(2*nrq):
            li.append((1.0j/2)*np.matmul(comutation(x[i],x[j]),dens).trace())
        C.append(li)
    return C

#Creat density matrix for the compress state  from initial density 
def new_state(dens,op):
    nrq=int(np.log2(len(dens)))
    ide=np.identity(2*nrq,dtype=np.complex128)
    C=corelation(dens,op)
    for i in range(len(C)):
        for j in range(len(C)):
            C[i][j]=1j*(C[i][j])
    return (1/(2*nrq))*(ide+C)

#############################################################################################################################

#############################################################################################################################
#Inefficient matchgate VQE

#Exact solver for reference:
def Exact_solver(qubitOp):
    ex = NumPyEigensolver(qubitOp)
    result = ex.run()
    ref = result['eigenvalues']
    return np.real(ref)[0]

class swap_t:
    def __init__(self,theata):
        self.theta=theata
        self.mat=np.array([[1,0,0,0],
                           [0,np.sin(self.theta),np.cos(self.theta),0],
                           [0,np.cos(self.theta),-1*np.sin(self.theta),0],
                           [0,0,0,-1]])
def U_t(theta):
    qc=QuantumCircuit(2,name='U_t('+str(theta)+')')
    qc.unitary(swap_t(theta).mat,[0,1])
    return qc.to_gate()

def ansatz_cell(qc,qo,nr_o, nr_e,thetas):
    
    #qo=QuantumRegister(nr_o,'qo')
    #qc=QuantumCircuit(qo,name='ansatz_cell')
    
    it=iter(thetas)
    start=nr_e-1
    limit=nr_o
    while start!=-1:
        cq=start
        tq=start+1
        while tq<limit:
            qc.append(U_t(next(it)),[cq,tq])
            cq=cq+1
            tq=tq+1

        start=start-1
        limit=limit-1
    return qc 

def var_circ(nr_o,nr_e,theta):
    qo=QuantumRegister(nr_o,'qo')
    cb=ClassicalRegister(nr_o,'cl')
    circ = QuantumCircuit(qo,cb)
    for i in range(nr_e):
        circ.x(i)
    ansatz_cell(circ,qo,nr_o, nr_e,theta)
    return circ

# Caluclate final expected value as sum of h[i]<psi|h_obs|psi> where h_obs-> h_label[i].

def value(h,h_label,circ,backend):
    
    val=0
    for i in range(len(h)):
        if h[i]!=0:
            exp=expected(circ,h_label[i],shots=100000,backend=backend)
            val=val+h[i]*exp
            #print('exp for {} ={}'.format(h_label[i],exp))
            
    return (val)


def cost(theta,weight,pauli,nr_o,nr_e,backend):
    circ=var_circ(nr_o,nr_e,theta)
    return value(weight,pauli,circ,backend)
#############################################################################################################################

#############################################################################################################################

