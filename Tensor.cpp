#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
vector<double> filled_values(const vector<size_t>& shape, double value) {
    if (shape.empty() || shape.size() > 3) {
        throw invalid_argument("La forma debe tener entre 1 y 3 dimensiones");
    }
    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0 || total > numeric_limits<size_t>::max() / dim) {
            throw invalid_argument("Dimensiones invalidas o demasiado grandes");
        }
        total *= dim;
    }
    return vector<double>(total, value);
}
}

void Tensor::validate_shape(const vector<size_t>& shape) {
    if (shape.empty() || shape.size() > 3) {
        throw invalid_argument("La forma debe tener entre 1 y 3 dimensiones");
    }
    for (size_t dimension : shape) {
        if (dimension == 0) {
            throw invalid_argument("Las dimensiones deben ser mayores que cero");
        }
    }
}

size_t Tensor::element_count(const vector<size_t>& shape) {
    validate_shape(shape);
    size_t total = 1;
    for (size_t dimension : shape) {
        if (total > numeric_limits<size_t>::max() / dimension) {
            throw overflow_error("El tensor es demasiado grande");
        }
        total *= dimension;
    }
    return total;
}

Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values)
    : shape_(shape), size_(element_count(shape)) {
    if (values.size() != size_) {
        throw invalid_argument("values no coincide con el producto de shape");
    }
    data_ = new double[size_];
    references_ = new size_t(1);
    copy(values.begin(), values.end(), data_);
}

Tensor::Tensor(initializer_list<size_t> shape, initializer_list<double> values)
    : Tensor(vector<size_t>(shape), vector<double>(values)) {}

Tensor::Tensor(const vector<size_t>& shape, double* data,
               size_t size, size_t* references, ShareTag)
    : shape_(shape), size_(size), data_(data), references_(references) {
    ++(*references_);
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_) {
    if (size_ != 0) {
        data_ = new double[size_];
        references_ = new size_t(1);
        copy(other.data_, other.data_ + size_, data_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : size_(other.size_), data_(other.data_), references_(other.references_) {
    shape_.swap(other.shape_);
    other.size_ = 0;
    other.data_ = nullptr;
    other.references_ = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        double* nuevos_datos = new double[other.size_];
        copy(other.data_, other.data_ + other.size_, nuevos_datos);

        release();
        shape_ = other.shape_;
        size_ = other.size_;
        data_ = nuevos_datos;
        references_ = new size_t(1);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        shape_.swap(other.shape_);
        size_ = other.size_;
        data_ = other.data_;
        references_ = other.references_;
        other.size_ = 0;
        other.data_ = nullptr;
        other.references_ = nullptr;
    }
    return *this;
}

Tensor::~Tensor() { release(); }

void Tensor::release() noexcept {
    // Solo se borra el arreglo cuando ningun tensor lo esta usando.
    if (references_ != nullptr && --(*references_) == 0) {
        delete[] data_;
        delete references_;
    }
    data_ = nullptr;
    references_ = nullptr;
    size_ = 0;
}

Tensor Tensor::zeros(const vector<size_t>& shape) {
    return Tensor(shape, filled_values(shape, 0.0));
}

Tensor Tensor::ones(const vector<size_t>& shape) {
    return Tensor(shape, filled_values(shape, 1.0));
}

Tensor Tensor::random(const vector<size_t>& shape,
                      double min, double max) {
    if (!(min < max)) {
        throw invalid_argument("random requiere min < max");
    }
    vector<double> values = filled_values(shape, 0.0);
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<double> distribution(min, max);
    for (double& value : values) value = distribution(generator);
    return Tensor(shape, values);
}

Tensor Tensor::arange(double start, double end) {
    if (!(start < end)) {
        throw invalid_argument("arange requiere start < end");
    }
    const double distance = end - start;
    if (distance > static_cast<double>(numeric_limits<size_t>::max())) {
        throw overflow_error("Rango demasiado grande");
    }
    const size_t count = static_cast<size_t>(ceil(distance));
    vector<double> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = start + i;
    return Tensor({count}, values);
}

bool Tensor::is_row_bias_for(const Tensor& target) const noexcept {
    return shape_.size() == 2 && target.shape_.size() == 2 &&
           shape_[0] == 1 && shape_[1] == target.shape_[1];
}

Tensor Tensor::elementwise(const Tensor& rhs, char operation) const {
    // Tambien se permite sumar un bias de una fila a una matriz.
    const bool same_shape = shape_ == rhs.shape_;
    const Tensor* bias = nullptr;
    const Tensor* target = this;
    if (!same_shape) {
        if (rhs.is_row_bias_for(*this)) bias = &rhs;
        else if (is_row_bias_for(rhs)) { bias = this; target = &rhs; }
        else throw invalid_argument("Formas incompatibles para operacion elemento a elemento");
    }
    vector<double> result(target->size_);
    for (size_t i = 0; i < target->size_; ++i) {
        double left = same_shape ? data_[i] : target->data_[i];
        double right = same_shape ? rhs.data_[i] : bias->data_[i % bias->shape_[1]];
        if (!same_shape && target == &rhs) swap(left, right);
        if (operation == '+') result[i] = left + right;
        else if (operation == '-') result[i] = left - right;
        else result[i] = left * right;
    }
    return Tensor(target->shape_, result);
}

Tensor Tensor::operator+(const Tensor& rhs) const { return elementwise(rhs, '+'); }
Tensor Tensor::operator-(const Tensor& rhs) const { return elementwise(rhs, '-'); }
Tensor Tensor::operator*(const Tensor& rhs) const { return elementwise(rhs, '*'); }

Tensor Tensor::operator*(double scalar) const {
    vector<double> result(size_);
    for (size_t i = 0; i < size_; ++i) result[i] = data_[i] * scalar;
    return Tensor(shape_, result);
}

Tensor operator*(double scalar, const Tensor& tensor) { return tensor * scalar; }

Tensor Tensor::view(const vector<size_t>& new_shape) const {
    if (element_count(new_shape) != size_) {
        throw invalid_argument("view debe conservar el numero de elementos");
    }
    return Tensor(new_shape, data_, size_, references_, ShareTag{});
}

Tensor Tensor::unsqueeze(size_t axis) const {
    if (shape_.size() == 3 || axis > shape_.size()) {
        throw invalid_argument("Eje invalido o limite de 3 dimensiones excedido");
    }
    vector<size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + static_cast<ptrdiff_t>(axis), 1);
    return Tensor(new_shape, data_, size_, references_, ShareTag{});
}

Tensor Tensor::relu() const {
    vector<double> result(size_);
    for (size_t i = 0; i < size_; ++i) result[i] = max(0.0, data_[i]);
    return Tensor(shape_, result);
}

Tensor Tensor::sigmoid() const {
    vector<double> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        if (data_[i] >= 0) result[i] = 1.0 / (1.0 + exp(-data_[i]));
        else { const double e = exp(data_[i]); result[i] = e / (1.0 + e); }
    }
    return Tensor(shape_, result);
}

Tensor Tensor::concat(const vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty()) throw invalid_argument("concat requiere al menos un tensor");
    const vector<size_t>& base = tensors.front().shape_;
    if (axis >= base.size()) throw invalid_argument("Eje de concatenacion invalido");
    vector<size_t> output_shape = base;
    output_shape[axis] = 0;
    for (const Tensor& tensor : tensors) {
        if (tensor.shape_.size() != base.size()) throw invalid_argument("Rangos incompatibles");
        for (size_t d = 0; d < base.size(); ++d) {
            if (d != axis && tensor.shape_[d] != base[d])
                throw invalid_argument("Formas incompatibles para concat");
        }
        output_shape[axis] += tensor.shape_[axis];
    }
    size_t inner = 1;
    for (size_t d = axis + 1; d < base.size(); ++d) {
        inner *= base[d];
    }

    size_t outer = 1;
    for (size_t d = 0; d < axis; ++d) {
        outer *= base[d];
    }
    vector<double> values(element_count(output_shape));
    size_t destination = 0;
    for (size_t o = 0; o < outer; ++o) {
        for (const Tensor& tensor : tensors) {
            const size_t block = tensor.shape_[axis] * inner;
            const size_t source = o * block;
            copy(tensor.data_ + source, tensor.data_ + source + block,
                 values.begin() + static_cast<ptrdiff_t>(destination));
            destination += block;
        }
    }
    return Tensor(output_shape, values);
}

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape_ != b.shape_) throw invalid_argument("dot requiere formas iguales");
    double result = 0.0;
    for (size_t i = 0; i < a.size_; ++i) result += a.data_[i] * b.data_[i];
    return Tensor({1}, {result});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2 || a.shape_[1] != b.shape_[0]) {
        throw invalid_argument("matmul requiere matrices 2D compatibles");
    }
    const size_t rows = a.shape_[0], shared = a.shape_[1], cols = b.shape_[1];
    vector<double> values(rows * cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t k = 0; k < shared; ++k) {
            const double aik = a.data_[i * shared + k];
            for (size_t j = 0; j < cols; ++j)
                values[i * cols + j] += aik * b.data_[k * cols + j];
        }
    return Tensor({rows, cols}, values);
}

size_t Tensor::size() const noexcept { return size_; }
size_t Tensor::ndim() const noexcept { return shape_.size(); }
const vector<size_t>& Tensor::shape() const noexcept { return shape_; }

double Tensor::at(const vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) throw out_of_range("Numero de indices incorrecto");
    size_t offset = 0;
    for (size_t d = 0; d < shape_.size(); ++d) {
        if (indices[d] >= shape_[d]) throw out_of_range("Indice fuera de rango");
        offset = offset * shape_[d] + indices[d];
    }
    return data_[offset];
}

void Tensor::print_shape(ostream& out) const {
    out << "shape=(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) out << ", ";
        out << shape_[i];
    }
    out << "), size=" << size_ << '\n';
}
