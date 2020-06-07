python3 model/bert_inference.py \
    --batch_size=512 \
    --activation_dir="data/imdb_embeddings.npy" \
    --train_dir="data/imdb-fragments.pkl" \
    --bert_weights="model/imdb_weights"
