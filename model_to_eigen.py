import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to saved model", type=str, required=True)
parser.add_argument("--dest", default="./network_weights.h", help="Path to destination header file", type=str)
args = parser.parse_args()

model = torch.load(args.model)
with open(args.dest,'w') as f:
	f.write("""#include <Eigen/Dense>
#include <cassert>
using namespace Eigen;

#define INPUT_SIZE """ + str(model.hidden.weight.shape[1]) + """
#define HIDDEN_SIZE """ + str(model.hidden.bias.shape[0]) + """
#define OUTPUT_SIZE """ + str(model.output.bias.shape[0]) + """

class Model 
{
private:
	Matrix<float, HIDDEN_SIZE, INPUT_SIZE> W_i_h;
	Matrix<float, OUTPUT_SIZE, HIDDEN_SIZE> W_h_o;
	Matrix<float, OUTPUT_SIZE, INPUT_SIZE> W;
	VectorXf b_h;
	VectorXf b_o;
	VectorXf b;
public:
	Model(): b_h(HIDDEN_SIZE), b_o(OUTPUT_SIZE), b(OUTPUT_SIZE)
	{
		W_i_h << """)
	for i in range(model.hidden.weight.shape[0]):
		for j in range(model.hidden.weight.shape[1]):
			f.write(str(model.hidden.weight[i][j].item()))
			if i == model.hidden.weight.shape[0] - 1 and j == model.hidden.weight.shape[1] - 1:
				f.write(";\n\t\tb_h << ")
			else:
				f.write(", ")
	for i in range(model.hidden.bias.shape[0]):
		f.write(str(model.hidden.bias[i].item()))
		if i == model.hidden.bias.shape[0] - 1:
			f.write(";\n\t\tW_h_o << ")
		else:
			f.write(", ")

	for i in range(model.output.weight.shape[0]):
		for j in range(model.output.weight.shape[1]):
			f.write(str(model.output.weight[i][j].item()))
			if i == model.output.weight.shape[0] - 1 and j == model.output.weight.shape[1] - 1:
				f.write(";\n\t\tb_o << ")
			else:
				f.write(", ")

	for i in range(model.output.bias.shape[0]):
		f.write(str(model.output.bias[i].item()))
		if i == model.output.bias.shape[0] - 1:
			f.write(";\n")
		else:
			f.write(", ")

	f.write("""
		W = (W_h_o*W_i_h).cast<float>();
		b = (W_h_o*b_h + b_o).cast<float>();
	}

	VectorXf operator()(VectorXf input)
	{
		assert(input.size()==INPUT_SIZE);
		return (W*input + b).cast<float>();
	}
};
	""")