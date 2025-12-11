image_encoder <- keras_model_sequential() %>%
  layer_conv_2d(32, 3, activation='relu', input_shape=c(img_size,img_size,3)) %>%
  layer_max_pooling_2d(2) %>%
  layer_conv_2d(64, 3, activation='relu') %>%
  layer_max_pooling_2d(2) %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(64, activation='relu')
