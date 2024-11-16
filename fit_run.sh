#!/bin/bash

# Список конфигурационных файлов
CONFIG_FILES=(
"configs/lstm_dual_attn_3_196_fold_8879.yaml"
"configs/lstm_dual_attn_3_196_fold_8880.yaml"
"configs/lstm_dual_attn_3_196_fold_8881.yaml"
"configs/lstm_dual_attn_3_196_fold_8882.yaml"
"configs/lstm_dual_attn_2_512_fold_8878.yaml"
"configs/lstm_dual_attn_2_512_fold_8879.yaml"
"configs/lstm_dual_attn_2_512_fold_8880.yaml"
"configs/lstm_dual_attn_2_512_fold_8882.yaml"
"configs/lstm_dual_attn_2_512_fold_8883.yaml"
"configs/lstm_dual_attn_4_256_fold_8881.yaml"
"configs/lstm_dual_attn_4_256_fold_8883.yaml"
"configs/lstm_dual_attn_2_1024_fold_8883.yaml"
"configs/big_bs_lstm_dual_attn_2_256_fold_8881.yaml"
)

# Папка для сохранения логов
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

for config in "${CONFIG_FILES[@]}"; do
    config_name=$(basename "$config" .yaml)
    
    echo "Запуск обучения с конфигурацией: $config"
    
    python train.py --hparams "$config" > "$LOG_DIR/${config_name}.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Ошибка при выполнении обучения с конфигурацией $config. Проверьте логи."
        exit 1
    fi
    
    echo "Обучение с конфигурацией $config завершено успешно. Логи сохранены в $LOG_DIR/${config_name}.log"
done

echo "Все запуски завершены."
