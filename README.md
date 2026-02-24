# GammaSpike-AGI
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
dt = 0.1          # passo de tempo em ms
T = 500           # duração total em ms
time = np.arange(0, T, dt)
tau = 10.0        # constante de tempo do neurônio (ms)
V_th = 1.0        # limiar de disparo
V_reset = 0.0

V = np.zeros(len(time))   # potencial de membrana
spikes = []               # lista de horários de spike

# Input com frequência gamma (40 Hz) + bias
freq_gamma = 40.0  # Hz
I_gamma = 0.6 * np.sin(2 * np.pi * freq_gamma * time / 1000) + 0.8

for i in range(1, len(time)):
    # Equação do LIF
    dV = (-V[i-1] + I_gamma[i-1]) * dt / tau
    V[i] = V[i-1] + dV
    
    if V[i] >= V_th:
        spikes.append(time[i])
        V[i] = V_reset  # reset

# Plot lindo
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, I_gamma, 'orange', label='Input Gamma 40Hz')
plt.ylabel('Corrente (I)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, V, 'b', label='Potencial de Membrana')
plt.scatter(spikes, [1.1]*len(spikes), color='red', marker='^', label='Spikes')
plt.axhline(V_th, color='red', linestyle='--', alpha=0.5)
plt.ylabel('V (mV)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
# FFT pra ver o pico gamma
fft = np.abs(np.fft.rfft(V))
freqs = np.fft.rfftfreq(len(V), d=dt/1000)
plt.plot(freqs[:200], fft[:200])
plt.xlabel('Frequência (Hz)')
plt.ylabel('Potência')
plt.title('Espectro mostrando pico em ~40 Hz (gamma)')
plt.grid()

plt.tight_layout()
plt.show()

print(f"Disparos: {len(spikes)} spikes | Frequência média: {len(spikes)/(T/1000):.1f} Hz")
