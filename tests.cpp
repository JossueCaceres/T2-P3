#include "Tensor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::ones({2, 2});
    assert((a + b).at({1, 1}) == 5);
    assert((a - b).at({0, 0}) == 0);
    assert((a * b).at({1, 0}) == 3);
    assert((a * 2.0).at({0, 1}) == 4);
    assert(dot(a, b).at({0}) == 10);

    Tensor viewed = a.view({4});
    assert(viewed.shape()[0] == 4 && a.shape()[0] == 2);
    Tensor column = viewed.unsqueeze(1);
    assert(column.shape()[0] == 4 && column.shape()[1] == 1);

    Tensor product = matmul(a, Tensor({2, 1}, {1, 2}));
    assert(product.shape()[0] == 2 && product.shape()[1] == 1);
    assert(product.at({1, 0}) == 11);

    Tensor joined0 = Tensor::concat({a, b}, 0);
    assert(joined0.shape()[0] == 4 && joined0.at({2, 0}) == 1);
    Tensor joined1 = Tensor::concat({a, b}, 1);
    assert(joined1.shape()[1] == 4 && joined1.at({1, 2}) == 1);

    Tensor bias({1, 2}, {10, 20});
    Tensor broadcast = a + bias;
    assert(broadcast.at({1, 0}) == 13 && broadcast.at({1, 1}) == 24);

    Tensor copy = a;
    assert(copy.at({0, 0}) == 1);
    Tensor moved(static_cast<Tensor&&>(copy));
    assert(moved.size() == 4 && copy.size() == 0);

    bool caught = false;
    try { (void)a.view({3}); } catch (const invalid_argument&) { caught = true; }
    assert(caught);

    Tensor stable({1}, {-1000});
    assert(isfinite(stable.sigmoid().at({0})));
    cout << "Todas las pruebas pasaron.\n";
}
