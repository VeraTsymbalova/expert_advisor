from preprocess_data import preprocess_data

print("\n=== Начинаем предобработку данных ===")
df_train, df_test, df_val = preprocess_data("EURUSD-H1.csv")

# Вывод размеров
print(f"\n📊 Размеры выборок:")
print(f"Train: {df_train.shape}")
print(f"Test: {df_test.shape}")
print(f"Validation: {df_val.shape}")

# Вывод первых строк
print("\n🔍 Первые строки обучающей выборки:")
print(df_train.head())

print("\n🔍 Первые строки тестовой выборки:")
print(df_test.head())

print("\n🔍 Первые строки валидационной выборки:")
print(df_val.head())