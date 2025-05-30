#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <matplot/matplot.h>
#include <pybind11/complex.h>
#include <vector>
#include <cmath>
#include <complex>
#include<random>

namespace py = pybind11;
namespace plt = matplot;
constexpr double M_PI = 3.141592653589793;


// Dodawanie szumu Gaussowskiego do sygnału
std::vector<double> add_noise(const std::vector<double>& signal, double noise_stddev) {
    std::vector<double> noisy_signal(signal.size());
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(0.0, noise_stddev);

    for (size_t i = 0; i < signal.size(); ++i) {
        noisy_signal[i] = signal[i] + distribution(generator);
    }
    return noisy_signal;
}

// Pomocnicza funkcja linspace
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i)
        result[i] = start + i * step;
    return result;
}

// Generowanie sygnałów (sin, cos, kwadrat, piłokształtny)
std::vector<double> generate_sine(double freq, double start, double end, int samples) {
    auto t = linspace(start, end, samples);
    std::vector<double> result(samples);
    for (int i = 0; i < samples; ++i)
        result[i] = sin(2 * M_PI * freq * t[i]);
    return result;
}

std::vector<double> generate_cosine(double freq, double start, double end, int samples) {
    auto t = linspace(start, end, samples);
    std::vector<double> result(samples);
    for (int i = 0; i < samples; ++i)
        result[i] = cos(2 * M_PI * freq * t[i]);
    return result;
}

std::vector<double> generate_square(double freq, double start, double end, int samples) {
    auto t = linspace(start, end, samples);
    std::vector<double> result(samples);
    for (int i = 0; i < samples; ++i)
        result[i] = sin(2 * M_PI * freq * t[i]) >= 0 ? 1.0 : -1.0;
    return result;
}

std::vector<double> generate_sawtooth(double freq, double start, double end, int samples) {
    auto t = linspace(start, end, samples);
    std::vector<double> result(samples);
    for (int i = 0; i < samples; ++i)
        result[i] = 2.0 * (t[i] * freq - floor(t[i] * freq + 0.5));
    return result;
}

// Rysowanie sygnału
void plot_signal(const std::vector<double>& y) {
    std::vector<double> x(y.size());
    for (size_t i = 0; i < y.size(); ++i)
        x[i] = i;
    plt::plot(x, y);
    plt::show();
}

// DFT i IDFT
std::vector<std::complex<double>> dft(const std::vector<double>& input) {
    size_t N = input.size();
    std::vector<std::complex<double>> output(N);
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum = 0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            sum += input[n] * std::complex<double>(cos(angle), sin(angle));
        }
        output[k] = sum;
    }
    return output;
}

std::vector<double> idft(const std::vector<std::complex<double>>& input) {
    size_t N = input.size();
    std::vector<double> output(N);
    for (size_t n = 0; n < N; ++n) {
        std::complex<double> sum = 0;
        for (size_t k = 0; k < N; ++k) {
            double angle = 2.0 * M_PI * k * n / N;
            sum += input[k] * std::complex<double>(cos(angle), sin(angle));
        }
        output[n] = sum.real() / N;
    }
    return output;
}

// Filtracja 1D (konwolucja)
std::vector<double> filter_1d(const std::vector<double>& signal, const std::vector<double>& kernel) {
    int n = signal.size();
    int k = kernel.size();
    int half_k = k / 2;
    std::vector<double> result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < k; ++j) {
            int idx = i + j - half_k;
            if (idx >= 0 && idx < n)
                sum += signal[idx] * kernel[j];
        }
        result[i] = sum;
    }
    return result;
}

// Typ alias na macierz (2D)
using Matrix = std::vector<std::vector<double>>;

std::vector<std::vector<double>> convolve_2d(const std::vector<std::vector<double>>& image,
                                             const std::vector<std::vector<double>>& kernel) {
    int rows = image.size();
    int cols = image[0].size();
    int krows = kernel.size();
    int kcols = kernel[0].size();
    int kcenterX = kcols / 2;
    int kcenterY = krows / 2;

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (int m = 0; m < krows; ++m) {
                for (int n = 0; n < kcols; ++n) {
                    int x = j + n - kcenterX;
                    int y = i + m - kcenterY;
                    if (x >= 0 && x < cols && y >= 0 && y < rows) {
                        sum += image[y][x] * kernel[m][n];
                    }
                }
            }
            result[i][j] = sum;
        }
    }
    return result;
}

std::vector<double> convolve_1d(const std::vector<double>& signal, const std::vector<double>& kernel) {
    int n = signal.size();
    int k = kernel.size();
    int pad = k / 2;
    std::vector<double> result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int idx = i + j - pad;
            if (idx >= 0 && idx < n) {
                result[i] += signal[idx] * kernel[j];
            }
        }
    }
    return result;
}


PYBIND11_MODULE(example, m) {
    m.def("generate_sine", &generate_sine, "Generuj sygnał sinusoidalny",
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("samples"));
    m.def("convolve_1d", &convolve_1d, "Filtracja 1D");
    m.def("generate_cosine", &generate_cosine, "Generuj sygnał cosinusoidalny",
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("samples"));
    m.def("generate_square", &generate_square, "Generuj sygnał prostokątny",
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("samples"));
    m.def("generate_sawtooth", &generate_sawtooth, "Generuj sygnał piłokształtny",
          py::arg("freq"), py::arg("start"), py::arg("end"), py::arg("samples"));
    m.def("plot_signal", &plot_signal, "Rysuj sygnał");

    m.def("dft", &dft, "Oblicz dyskretną transformatę Fouriera");
    m.def("idft", &idft, "Oblicz odwrotną DFT");

    m.def("filter_1d", &filter_1d, "Filtracja 1D",
          py::arg("signal"), py::arg("kernel"));


    m.def("add_noise", &add_noise, "Dodaj szum Gaussowski do sygnału",
      py::arg("signal"), py::arg("noise_stddev"));
    m.def("convolve_2d", &convolve_2d, "Filtracja 2D obrazu");

}
