from preprocess_data import preprocess_data
from add_trend_indicators import add_trend_indicators
import import_data  # запускается автоматически при импорте
import train_evaluate
import hyperparameter_optimization
import visualize_results

from environment import ForexTradingEnv

from stable_baselines3 import PPO

def main():
    print("\n=== Шаг 1: Импорт и сохранение данных ===")
    # import_data запускается при импорте и сохраняет EURUSD-H1.csv

    print("\n=== Шаг 2: Предобработка и добавление индикаторов ===")
    df_train, df_test, df_val = preprocess_data("EURUSD-H1.csv")

    df_train = add_trend_indicators(df_train)
    df_test = add_trend_indicators(df_test)
    df_val = add_trend_indicators(df_val)

    print(f"Train: {df_train.shape}, Test: {df_test.shape}, Validation: {df_val.shape}")

    df_train.to_csv("data_train.csv", index=False)
    df_test.to_csv("data_test.csv", index=False)
    df_val.to_csv("data_val.csv", index=False)
    print("Данные сохранены.")

    print("\n=== Шаг 3: Оптимизация гиперпараметров ===")
    hyperparameter_optimization.objective  # просто для явного указания использования
    if __name__ == '__main__':
        hyperparameter_optimization.study = hyperparameter_optimization.optuna.create_study(direction='maximize')
        hyperparameter_optimization.study.optimize(hyperparameter_optimization.objective, n_trials=100)

    print("\n=== Шаг 4: Обучение модели ===")
    env_train = ForexTradingEnv(df_train)
    env_test = ForexTradingEnv(df_test)

    model = train_evaluate.train_model(env_train)
    log = train_evaluate.evaluate_model(model, env_test)

    print("\n=== Шаг 5: Визуализация результатов ===")
    visualize_results.main()

if __name__ == '__main__':
    main()