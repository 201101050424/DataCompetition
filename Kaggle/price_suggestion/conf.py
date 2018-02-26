IS_DEBUG = True
# IS_DEBUG = False

data_origin = './train.tsv'
if IS_DEBUG:
    data_origin = './train_debug.tsv'
split_per = 0.8
train_data_preprocess = './train_preprocess'
test_data_preprocess = './test_preprocess'

# train
epoch_num = 100
model_dir = './model'
lr = 1e-2
# embedding_dim_name = 50
# embedding_dim_item_condition_id = 5
# embedding_dim_category_name = 50
# embedding_dim_brand_name = 50
# embedding_dim_shipping = 2

# hidden_size_name = 100
# hidden_size_category_name = 100
