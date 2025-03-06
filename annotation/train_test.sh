DATA_PATH="/media02/nthuy/data/entube/EnTube/data"
OUTPUT_TRAIN_PATH="data_short/EnTube_10m_train_short.json"
OUTPUT_TEST_PATH="data_short/EnTube_10m_test_short.json"

python annotation/train_test.py \
--data $DATA_PATH \
--output_train_file $OUTPUT_TRAIN_PATH \
--output_test_file $OUTPUT_TEST_PATH \
--min_duration 1 \
--max_duration 600
