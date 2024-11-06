README for dataset p53


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Node Label Conversion === 

Node labels were converted to integer values using this map:

Component 0:
	0	I
	1	C
	2	N
	3	S
	4	O
	5	Cl
	6	Br
	7	P
	8	Hg
	9	B
	10	F
	11	Na
	12	Cu
	13	Fe
	14	Sn
	15	Cr
	16	Zn
	17	K
	18	Ca
	19	Se
	20	Co
	21	Ag
	22	Si
	23	Sb
	24	Li
	25	Pt
	26	Al
	27	As
	28	Bi
	29	Ba
	30	Au
	31	Ti
	32	Sr
	33	In
	34	Dy
	35	Ni
	36	Be
	37	Mg
	38	Nd
	39	Pd
	40	Mn
	41	Zr
	42	Pb
	43	Yb
	44	Mo
	45	Cd
	46	Ge



Edge labels were converted to integer values using this map:

Component 0:
	0	single
	1	double
	2	aromatic
	3	triple



=== References ===

Tox21 Data Challenge 2014, https://tripod.nih.gov/tox21/challenge/data.jsp
