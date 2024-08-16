PART=dev
SHOT=5
TRAIN_DATA="--train_data_file ./result/Spider/train.json --train_schema_file ./dataset/Spider/tables.json --train_database_path ./dataset/Spider/database"
for MODEL in Llama3-chat Deepseek-Coder-chat gpt-3.5-turbo; do
    for SCALE in - 6.7b 8b 33b 70b; do
        for DATASET in Spider Bird KaggleDBQA; do
            if [[ "$MODEL" != "gpt-3.5-turbo" || "$SCALE" != "-" ]] && [ ! -d "./model/$MODEL/$SCALE" ]; then
                continue
            fi
            if [ "$MODEL" == "gpt-3.5-turbo" ]; then
                MODEL_NAME_OR_PATH=gpt-3.5-turbo
            else
                MODEL_NAME_OR_PATH=./model/$MODEL/$SCALE
            fi

            if [ ! -f "./result/$DATASET/$MODEL/$SCALE/initialize/$PART.json" ]; then
                python3 generate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./result/$DATASET/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./result/$DATASET/$MODEL/$SCALE/initialize/$PART.json \
                    --shot $SHOT
            fi

            if [ ! -f "./result/$DATASET/$MODEL/$SCALE/align/$PART.json" ]; then
                python3 align.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./result/$DATASET/$MODEL/$SCALE/initialize/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./result/$DATASET/$MODEL/$SCALE/align/$PART.json \
                    --shot $SHOT
            fi

            if [ ! -f "./result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json" ]; then
                python3 hallucinate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./result/$DATASET/$MODEL/$SCALE/align/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json \
                    --shot $SHOT
            fi

            if [ ! -f "./result/$DATASET/$MODEL/$SCALE/generate/$PART.json" ]; then
                python3 generate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./result/$DATASET/$MODEL/$SCALE/generate/$PART.json \
                    --shot $SHOT \
                    --aligned
            fi

            if [ ! -f "./result/$DATASET/$MODEL/$SCALE/debug/$PART.json" ]; then
                python3 debug.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    --data_file ./result/$DATASET/$MODEL/$SCALE/generate/$PART.json \
                    --schema_file ./dataset/$DATASET/tables.json \
                    --database_path ./dataset/$DATASET/database \
                    --dump_file ./result/$DATASET/$MODEL/$SCALE/debug/$PART.json \
                    --temperature 0.8
            fi
        done
    done
done
