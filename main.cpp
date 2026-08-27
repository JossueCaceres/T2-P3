#include "Tensor.h"

#include <exception>
#include <iostream>

int main() {
    try {
        Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
        cout << "1. Entrada: "; input.print_shape(cout);

        Tensor flat = input.view({1000, 400});
        cout << "2. View: "; flat.print_shape(cout);

        Tensor w1 = Tensor::random({400, 100}, -0.05, 0.05);
        Tensor layer1 = matmul(flat, w1);
        cout << "3. Matmul W1: "; layer1.print_shape(cout);

        Tensor b1 = Tensor::zeros({1, 100});
        Tensor biased1 = layer1 + b1;
        cout << "4. Bias b1: "; biased1.print_shape(cout);

        Tensor activated1 = biased1.relu();
        cout << "5. ReLU: "; activated1.print_shape(cout);

        Tensor w2 = Tensor::random({100, 10}, -0.05, 0.05);
        Tensor layer2 = matmul(activated1, w2);
        cout << "6. Matmul W2: "; layer2.print_shape(cout);

        Tensor b2 = Tensor::zeros({1, 10});
        Tensor biased2 = layer2 + b2;
        cout << "7. Bias b2: "; biased2.print_shape(cout);

        Tensor output = biased2.sigmoid();
        cout << "8. Sigmoid: "; output.print_shape(cout);

        return 0;
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
