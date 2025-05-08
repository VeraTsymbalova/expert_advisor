import os
import time
import numpy as np
import onnxruntime as ort
import errno

# 🔧 Укажи путь к папке Files агента тестера стратегий
base_path = r"C:\Users\vera-\AppData\Roaming\MetaQuotes\Tester\5FFA568149E88FCD5B44D926DCFEAA79\Agent-127.0.0.1-3000\MQL5\Files"
input_file = os.path.join(base_path, "ppo_input.txt")
output_file = os.path.join(base_path, "ppo_action.txt")
model_path = "ppo_forex_model.onnx"
log_file = "ppo_log.txt"
error_log = "ppo_error.log"
expected_feature_count = 13  # Укажи точное число признаков

# Загрузка ONNX-модели
try:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("✅ PPO inference script started.")
except Exception as e:
    with open(error_log, "a") as f:
        f.write(f"Ошибка загрузки модели: {str(e)}\n")
    raise e

# Основной цикл
while True:
    if os.path.exists(input_file):
        try:
            with open(input_file, "r") as f:
                line = f.read().strip()

            line = line.encode("ascii", errors="ignore").decode()  # Удалить мусорные символы

            if not line:
                raise ValueError("Файл пустой.")

            print(f"📥 Прочитан файл: {line}")
            features = np.array([float(x) for x in line.split()], dtype=np.float32).reshape(1, -1)

            if features.shape[1] != expected_feature_count:
                raise ValueError(f"Неверное число признаков: {features.shape[1]}, ожидалось {expected_feature_count}")

            inputs = {session.get_inputs()[0].name: features}
            outputs = session.run(None, inputs)
            action = int(np.argmax(outputs[0]))

            import tempfile  # в начало файла, если ещё не импортировано

            # Атомарная запись в output_file
            try:
                with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(output_file), delete=False, encoding="utf-8") as tmp:
                    tmp.write(str(action))
                    temp_name = tmp.name
                os.replace(temp_name, output_file)
            except Exception as e:
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(f"Ошибка при атомарной записи: {str(e)}\n")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✔ Action: {action}\n")

            print(f"✔ Признаки обработаны. Action: {action}")

        except IOError as e:
            if e.errno == errno.EACCES:
                print("⌛ Файл занят другим процессом. Повтор через 0.5 сек.")
            else:
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(f"IOError: {str(e)}\n")
        except Exception as e:
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"Ошибка: {str(e)}\n")
            print("⚠ Ошибка при обработке входа:", e)
    else:
        print("⌛ Ожидание появления ppo_input.txt...")

    time.sleep(0.5)
