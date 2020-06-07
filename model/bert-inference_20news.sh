python3 model/bert_inference.py \
    --batch_size=512 \
    --activation_dir="data/20news_embeddings.npy" \
    --train_dir="data/news_train_fragments.pkl" \
    --bert_weights="model/news_weights"
