import example
import matplotlib.pyplot as plt
import numpy as np

# Parametry sygnału
fs = 1000      # Częstotliwość próbkowania (Hz)
f = 5          # Częstotliwość sygnału (Hz)
start = 0.0    # Początek sygnału (sekundy)
end = 1.0      # Koniec sygnału (sekundy)
samples = 1000 # Liczba próbek

# Przykładowa macierz obrazu 10x10 z "jasnym punktem" na środku
image = [[0]*10 for _ in range(10)]
image[5][5] = 100

# Jądro rozmywające (średnia z sąsiadów)
kernel = [[1/9]*3 for _ in range(3)]

# Filtracja
filtered = example.convolve_2d(image, kernel)

# Wizualizacja
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Oryginał")

plt.subplot(1, 2, 2)
plt.imshow(filtered, cmap='gray')
plt.title("Po filtracji")

plt.show()

print("Generowanie i rysowanie sygnałów:")

print("-> Sinusoida")
fala_sinus = example.generate_sine(f, start, end, samples)
example.plot_signal(fala_sinus)

print("-> Cosinusoida")
fala_cosinus = example.generate_cosine(f, start, end, samples)
example.plot_signal(fala_cosinus)

print("-> Prostokątny")
fala_kwadrat = example.generate_square(f, start, end, samples)
example.plot_signal(fala_kwadrat)

print("-> Piłokształtny")
fala_pila = example.generate_sawtooth(f, start, end, samples)
example.plot_signal(fala_pila)

# Przykładowy sygnał do DFT
signal = example.generate_sine(10, start, end, 100)

# DFT
print("-> DFT")
spectrum = example.dft(signal)

# Rysowanie widma
magnitudes = [abs(c) for c in spectrum]
plt.figure()
plt.plot(magnitudes)
plt.title("Widmo amplitudowe (DFT)")
plt.xlabel("Próbka")
plt.ylabel("Amplituda")
plt.grid()
plt.show()

# IDFT
print("-> IDFT")
restored = example.idft(spectrum)
plt.figure()
plt.plot(restored, label="odtworzony")
plt.plot(signal, label="oryginalny", linestyle='dashed')
plt.legend()
plt.title("Sygnał: Oryginalny vs. Odtworzony z IDFT")
plt.grid()
plt.show()

# FILTRACJA 1D – np. wygładzanie prostokątnego
print("-> Filtracja 1D")
kernel = [1/10, 1/10, 1/10]  # prosty filtr uśredniający
filtered = example.convolve_1d(fala_kwadrat, kernel)
plt.figure()
plt.plot(fala_kwadrat, label="oryginalny")
plt.plot(filtered, label="przefiltrowany", linestyle='dashed')
plt.legend()
plt.title("Filtracja 1D (średnia ruchoma)")
plt.grid()
plt.show()

# FILTRACJA 2D – opcjonalnie, jeśli masz 2D w C++
try:
    print("-> Filtracja 2D")
    # Przykład: mała macierz 2D
    image = [[i + j for j in range(10)] for i in range(10)]
    kernel2d = [[1/9]*3 for _ in range(3)]  # uśredniający 3x3
    filtered_image = example.convolve_2d(image, kernel2d)
    print("Filtracja 2D zakończona (nie rysuję, bo to nie obraz)")
except AttributeError:
    print("Filtracja 2D niedostępna w module")

print("-> Zaszumianie sygnału (szum Gaussowski)")
noisy = example.add_noise(fala_sinus, 0.5)  # 0.5 to stddev szumu – może Pan zmieniać :) zmieniać
example.plot_signal(noisy)
