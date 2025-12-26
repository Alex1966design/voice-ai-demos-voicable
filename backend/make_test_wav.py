import wave
import struct


def main():
    # Создаём 1 секунду тишины
    duration_seconds = 1.0
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)

    with wave.open("test.wav", "w") as w:
        w.setnchannels(1)       # моно
        w.setsampwidth(2)       # 16 бит
        w.setframerate(sample_rate)

        for _ in range(num_samples):
            w.writeframes(struct.pack("<h", 0))  # тишина

    print("Создан файл test.wav в текущей папке")


if __name__ == "__main__":
    main()
