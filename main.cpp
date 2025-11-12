/*

FORWARD PASSES

Hidden Layer

*subscript j - refers to 1 of the 2 neurons

1. Compute weighted sum for one hidden layer neuron, using inputs*
h j = (w1 * x1) + (w2 * x2) + b j

*inputs:
w - hidden layer weight, random -0.5 -> 0.5
x - input layer value, binary 0 or 1
b - bias, 0 or random small decimal

collect user input for x1 and x21

2. Compute activation output (sigmoid)
a j = 1 / (1 + e^(-(h j)))

e - Euler's number

note: to reuse sigmoid:
reference decimal parameter variable inside power for euler's number

note: use exp() function for euler's number


Output Layer

1. Get weighted sum for output neuron
o = (v1 * a1) + (v2 * a2) + b out

a1, a2 - activations for hidden layer 1st and 2nd neurons
v - output layer weight, DIFFERENT from hidden layer, random -0.5 -> 0.5
b out - bias, 0 or random small decimal

2. Get activation output (sigmoid)
y pred = 1 / (1 + e^(-(o)))

note: utilize reusable sigmoid function


MEAN SQUARED ERROR
L = (1 / 2) * (y - y pred)^2

y - expected XOR output, depends on which set of 1 and 0 input layer value (x) is currently computed
y pred - activation from output layer

note: determine how to select which XOR input pair to compute in one epoch

BACKPROPAGATION

Output Layer

note: one gradient = one hidden layer neuron = one corresponding hidden layer weight = one corresponding output layer activation = one corresponding hidden layer activation

1. Compute output error term
output error term = (y pred - y) * (y pred * (1 - y pred))

note: (y pred * (1 - y pred)) is derivative of y pred output layer activation (sigmoid)

2. Compute gradient for output weights
output weight gradient = output error term * a j

a j - activation of 1 of the 2 hidden layer neurons

3. Get gradient for output bias
output bias gradient = output error term


Hidden layer

note: one gradient = one output neuron = one corresponding hidden layer neuron = one corresponding hidden layer activation = one corresponding input layer value

1. Compute hidden error term
hidden error term = output error term * v j * (a j * (1 - a j))

2. Compute gradient for hidden weights
hidden weight gradient = hidden error term * x i

x i - input layer value

3. Get gradient for hidden bias
hidden bias gradient = hidden error term


UPDATE WEIGHTS AND BIASES

Output Layer

1. Update weight
v j = v j - (learning rate * output weight gradient)

2. Update bias
b out = b out - (learning rate * output error term)


Hidden Layer

1. Update weight
w j = w j - (learning rate * hidden weight gradient)

2. Update bias
b j = b j - (learning rate * hidden eror term) 

*/

#include <iostream> // import libraries
#include <string>
#include <vector>
#include <cmath>
#include <limits>

using std::cin; // namespace aliases
using std::cout;
using std::endl;
using std::numeric_limits;
using std::string;
using std::streamsize;

// inputs
int x1 = 0;
int x2 = 0;

double w1_x1 = 0.0;
double w1_x2 = 0.0;
double w2_x1 = 0.0;
double w2_x2 = 0.0;

double b1 = 0.0;
double b2 = 0.0;

double v1 = 0.0;
double v2 = 0.0;

double b_out = 0.0;

double learning_rate = 0.0;

int epochs = 0;

// hidden layer
double h1 = 0.0;
double h2 = 0.0;

double a1 = 0.0;
double a2 = 0.0;

// output layer
double o = 0.0;

double y_pred = 0.0;

// backpropagation
int y = 0;

double mse = 0.0;

double output_error_term = 0.0;

double v1_gradient = 0.0;
double v2_gradient = 0.0;

double b_out_gradient = 0.0;

double h1_error_term = 0.0;
double h2_error_term = 0.0;

double w1_x1_gradient = 0.0;
double w1_x2_gradient = 0.0;
double w2_x1_gradient = 0.0;
double w2_x2_gradient = 0.0;

double b1_gradient = 0.0;
double b2_gradient = 0.0;

double epoch_mse = 0.0;

bool is_user_retrying = true;

double compute_activation(const double& x) {
    return 1 / (1 + exp(-(x)));
}

int compute_xor(const int& x1, const int& x2) {
    return x1 != x2;
}

double compute_hj_error_term(const double& output_error_term, const double& v_j, const double& a_j) {
    return output_error_term * v_j * (a_j * (1.0 - a_j));
}

double compute_hj_wj_gradient(const double& hidden_error_term, const int& x_j) {
    return hidden_error_term * static_cast<double>(x_j);
}

void forward_pass_hidden_layer () {
    h1 = (w1_x1 * static_cast<double>(x1)) + (w1_x2 * static_cast<double>(x2)) + b1;
    h2 = (w2_x1 * static_cast<double>(x1)) + (w2_x2 * static_cast<double>(x2)) + b2;

    a1 = compute_activation(h1);
    a2 = compute_activation(h2);
}

void forward_pass_output_layer() {
    o = (v1 * static_cast<double>(a1)) + (v2 * static_cast<double>(a2)) + b_out;

    y_pred = compute_activation(o);
}

void compute_error() {
    y = compute_xor(x1, x2);

    mse = 0.5 * pow(((static_cast<double>(y) - y_pred)), 2);
}

void compute_output_layer_gradient() {
    output_error_term = (y_pred - y) * (y_pred * (1 - y_pred));
    v1_gradient = output_error_term * a1;
    v2_gradient = output_error_term * a2;
    b_out_gradient = output_error_term;
}

void compute_hidden_layer_gradient() {
    h1_error_term = compute_hj_error_term(output_error_term, v1, a1);
    h2_error_term = compute_hj_error_term(output_error_term, v2, a2);

    w1_x1_gradient = compute_hj_wj_gradient(h1_error_term, x1);
    w1_x2_gradient = compute_hj_wj_gradient(h1_error_term, x2);
    w2_x1_gradient = compute_hj_wj_gradient(h2_error_term, x1);
    w2_x2_gradient = compute_hj_wj_gradient(h2_error_term, x2);

    b1_gradient = h1_error_term;
    b2_gradient = h2_error_term;
}

void backpropagate_output_layer() {
    v1 = v1 - (learning_rate * v1_gradient);
    v2 = v2 - (learning_rate * v2_gradient);

    b_out = b_out - (learning_rate * b_out_gradient);
}

void backpropagate_hidden_layer() {
    w1_x1 = w1_x1 - (learning_rate * w1_x1_gradient);
    w1_x2 = w1_x2 - (learning_rate * w1_x2_gradient);
    w2_x1 = w2_x1 - (learning_rate * w2_x1_gradient);
    w2_x2 = w2_x2 - (learning_rate * w2_x2_gradient);

    b1 = b1 - (learning_rate * b1_gradient);
    b2 = b2 - (learning_rate * b2_gradient);
}

void run_xor_neural_network() {
    forward_pass_hidden_layer();
    forward_pass_output_layer();
    compute_error();
    compute_output_layer_gradient();
    compute_hidden_layer_gradient();
    backpropagate_output_layer();
    backpropagate_hidden_layer();

    epoch_mse += mse;
}

void show_results() {
    cout << "=== INPUT LAYER ===" << endl;
    cout << "INPUT 1: " << x1 << " | INPUT 2: " << x2 << endl;
    cout << endl;

    cout << "=== HIDDEN LAYER NEURONS ===" << endl;
    cout << "Weight 1 - Input 1: " << w1_x1 << " | Input 2: " << w1_x2 << endl;
    cout << "Weight 2 - Input 1: " << w2_x1 << " | Input 2: " << w2_x2 << endl;
    cout << "Bias 1: " << b1 << " | Bias 2: " << b2 << endl;
    cout << "HIDDEN NEURON 1 Weighted Sum: " << h1 << " | HIDDEN NEURON 2 Weighted Sum: " << h2 << endl;
    cout << "HIDDEN NEURON 1 Activation: " << a1 << " | HIDDEN NEURON 2 Activation: " << a2 << endl;
    cout << endl;

    cout << "=== OUTPUT LAYER NEURON ===" << endl;
    cout << "Weight 1: " << v1 << " | Weight 2: " << v2 << endl;
    cout << "Bias: " << b_out << endl;
    cout << "Weighted Sum: " << o << " | Activation: " << y_pred << endl;
    cout << endl;

    cout << "=== BACKPROPAGATION ===" << endl;
    cout << endl;
    cout << "EXPECTED XOR: " << y << endl;
    cout << endl;
    cout << "OUTPUT LAYER NEURON" << endl;
    cout << "Output Error Term: " << output_error_term << endl;
    cout << "Weight 1 Gradient: " << v1_gradient << " | Weight 2 Gradient: " << v2_gradient << endl;
    cout << "Bias Gradient: " << b_out_gradient << endl;
    cout << endl;
    cout << "HIDDEN LAYER NEURON" << endl;
    cout << "WEIGHT 1 - Input 1 Gradient: " << w1_x1_gradient << " | Input 2 Gradient: " << w1_x2_gradient << endl;
    cout << "WEIGHT 2 - Input 1 Gradient: " << w2_x1_gradient << " | Input 2 Gradient: " << w2_x2_gradient << endl;
    cout << "Bias 1: " << b1_gradient << " | Bias 2: " << b2_gradient << endl;
    cout << endl;
    cout << "MSE: " << mse << endl;
    cout << endl;
}

bool wants_to_retry() {
    return is_user_retrying;
}

template <typename T>
void validate_number_input(T& variable) {
    while (!(cin >> variable)) {
        cout << "Invalid input. Please re-enter a valid number:" << endl;
        cin.clear(); // clear error flag
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // discard invalid input
    }
}

int main() {
    while (wants_to_retry()) {
        cout << "Enter HIDDEN NEURON 1 WEIGHT for Input 1" << endl;
        validate_number_input(w1_x1);
        cout << "Enter HIDDEN NEURON 1 WEIGHT for Input 2" << endl;
        validate_number_input(w1_x2);
        cout << "Enter HIDDEN NEURON 2 WEIGHT for Input 1" << endl;
        validate_number_input(w2_x1);
        cout << "Enter HIDDEN NEURON 2 WEIGHT for Input 2" << endl;
        validate_number_input(w2_x2);
        cout << endl;

        cout << "Enter BIAS for NEURON 1" << endl;
        validate_number_input(b1);
        cout << "Enter BIAS for NEURON 2" << endl;
        validate_number_input(b2);
        cout << endl;

        cout << "Enter OUTPUT NEURON WEIGHT 1" << endl;
        validate_number_input(v1);
        cout << "Enter OUTPUT NEURON WEIGHT 2" << endl;
        validate_number_input(v2);
        cout << endl;

        cout << "Enter BIAS for OUTPUT NEURON" << endl;
        validate_number_input(b_out);
        cout << endl;

        cout << "Enter LEARNING RATE" << endl;
        validate_number_input(learning_rate);
        cout << "Enter NUMBER OF EPOCHS" << endl;
        validate_number_input(epochs);
        cout << endl;

        for (int i = 0; i < epochs; i++)
        {
            x1 = 0;
            x2 = 0;
            run_xor_neural_network();
            if (i % 100 == 0) {
                cout << "############ EPOCH " << i << " ############ " << endl;
                cout << endl;
                show_results();
            }

            x1 = 1;
            x2 = 0;
            run_xor_neural_network();
            if (i % 100 == 0) {
                show_results();
            }

            x1 = 0;
            x2 = 1;
            run_xor_neural_network();
            if (i % 100 == 0) {
                show_results();
            }

            x1 = 1;
            x2 = 1;
            run_xor_neural_network();
            if (i % 100 == 0) {
                show_results();
            }
        }

        epoch_mse /= 4.0;

        cout << "EPOCH MSE: " << epoch_mse << endl;

        cout << "Enter new set of parameters? [y/n]" << endl;
        
        char input = ' ';

        cin >> input;

        switch (input)
        {
        case 'y':
            w1_x1 = 0;
            w1_x2 = 0;
            w2_x1 = 0;
            w2_x2 = 0;

            b1 = 0;
            b2 = 0;

            v1 = 0;
            v2 = 0;

            b_out = 0;
            break;
        case 'n':
            is_user_retrying = false;
            break;
        default:
            is_user_retrying = false;
            break;
        }
    }
}