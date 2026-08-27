#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <vector>

using namespace std;

class Tensor {
public:
    Tensor(const vector<size_t>& shape, const vector<double>& values);
    Tensor(initializer_list<size_t> shape, initializer_list<double> values);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape,
                         double min, double max);
    static Tensor arange(double start, double end);
    static Tensor concat(const vector<Tensor>& tensors, size_t axis);

    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;
    Tensor operator*(double scalar) const;
    friend Tensor operator*(double scalar, const Tensor& tensor);

    Tensor view(const vector<size_t>& new_shape) const;
    Tensor unsqueeze(size_t axis) const;
    Tensor relu() const;
    Tensor sigmoid() const;

    size_t size() const noexcept;
    size_t ndim() const noexcept;
    const vector<size_t>& shape() const noexcept;
    double at(const vector<size_t>& indices) const;
    void print_shape(ostream& out) const;

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

private:
    vector<size_t> shape_;
    size_t size_ = 0;
    double* data_ = nullptr;
    size_t* references_ = nullptr;

    struct ShareTag {};
    Tensor(const vector<size_t>& shape, double* data,
           size_t size, size_t* references, ShareTag);

    static size_t element_count(const vector<size_t>& shape);
    static void validate_shape(const vector<size_t>& shape);
    void release() noexcept;
    bool is_row_bias_for(const Tensor& target) const noexcept;
    Tensor elementwise(const Tensor& rhs, char operation) const;
};

#endif
