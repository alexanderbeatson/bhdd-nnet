library (tidyverse)
library (tensorflow)
library (reticulate)
library (keras)
library (reshape2)

#Not run this code-chunk ----
reticulate::py_run_string( 
  "from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)")

  #Ubuntu sleep issue sudo rmmod nvidia_uvm ; sudo modprobe nvidia_uvm
#-------------------------Data pre-processing--------------------------------

pd <- import("pandas")
pickle_data <- pd$read_pickle("./BHDD-master/data.pkl")

#Train data pre-processing
train_x <- NULL
train_y <- NULL
for (i in 1:length(pickle_data$trainDataset)) {
  train_x[[i]] <- pickle_data$trainDataset[[i]]$image/255
  train_y[i] <- pickle_data$trainDataset[[i]]$label
}
train_x <- array_reshape(train_x, dim = c(60000,28,28,1))
train_y <- to_categorical(train_y)
#Test data pre-processing
test_x <- NULL
test_y <- NULL
for (i in 1:length(pickle_data$testDataset)) {
  test_x[[i]] <- pickle_data$testDataset[[i]]$image/255
  test_y[i] <- pickle_data$testDataset[[i]]$label
}
test_x <- array_reshape(test_x, dim = c(27561,28,28,1))
test_y <- to_categorical(test_y)

#---------------------------Data pre-processing ended----------------------
#Testing
model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), activation = "relu", filters = 64, kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_dropout(0.3) %>% layer_conv_2d(activation = "relu", kernel_size = c(3,3), filters = 128) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_dropout(0.3) %>% layer_flatten() %>% layer_dense(units = 128, activation = "relu") %>% layer_dropout(0.3) %>% layer_dense(units = 10, activation = "softmax")
model %>% compile(loss = "categorical_crossentropy", optimizer = "rmsprop" , metrics = "accuracy")

model %>% fit(
  train_x, train_y,
  epoch = 30,
  validation_split = 0.2,
  batch_size = 128
)

ev <- model %>% evaluate(test_x, test_y)

#First layer neuron choosing
#-------------------------------------------------------------------------------------------
#First CNN layer number of filters
viz_loss1 <- data.table::data.table()
viz_acc1 <- data.table::data.table()
viz_loss1$epoch <- 1:10
viz_acc1$epoch <- 1:10
filters <- c(16,32,48,64,80)
for (i in filters) {
  model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = i, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
  model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 384)
  history <- model[["history"]]
  viz_loss1[,paste0("F",i)] <- unlist(history$history$val_loss)
  viz_acc1[,paste0("F",i)] <- unlist(history$history$val_accuracy)
}
viz_loss1 <- melt(as.data.frame(viz_loss1), id.vars = "epoch")
viz_acc1 <- melt(as.data.frame(viz_acc1), id.vars = "epoch")
viz_loss1 %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc1 %>% ggplot() + geom_line(aes(epoch, value, col = variable))
#-------------------------------------------------------------------------------------------
#Second CNN layer number of filters
viz_loss2 <- data.table::data.table()
viz_acc2 <- data.table::data.table()
viz_loss2$epoch <- 1:10
viz_acc2$epoch <- 1:10
filters <- c(128,144,160,178)
for (i in filters) {
  model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_conv_2d(filters = i, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
  model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 384)
  history <- model[["history"]]
  viz_loss2[,paste0("F",i)] <- unlist(history$history$val_loss)
  viz_acc2[,paste0("F",i)] <- unlist(history$history$val_accuracy)
}
viz_loss2 <- melt(as.data.frame(viz_loss2), id.vars = "epoch")
viz_acc2 <- melt(as.data.frame(viz_acc2), id.vars = "epoch")
viz_loss2 %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc2 %>% ggplot() + geom_line(aes(epoch, value, col = variable))
#-------------------------------------------------------------------------------------------
#Max-pool or batch-normalization or both?
viz_loss3 <- data.table::data.table()
viz_acc3 <- data.table::data.table()
viz_loss3$epoch <- 1:10
viz_acc3$epoch <- 1:10

#Max-pool
model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 256)
history <- model[["history"]]
viz_loss3[,"MP"] <- unlist(history$history$val_loss)
viz_acc3[,"MP"] <- unlist(history$history$val_accuracy)

#Batch-normalization
model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_batch_normalization() %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 256)
history <- model[["history"]]
viz_loss3[,"BN"] <- unlist(history$history$val_loss)
viz_acc3[,"BN"] <- unlist(history$history$val_accuracy)

#Max-pool + Batch-normalization
model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 256)
history <- model[["history"]]
viz_loss3[,"MPBN"] <- unlist(history$history$val_loss)
viz_acc3[,"MPBN"] <- unlist(history$history$val_accuracy)

viz_loss3 <- melt(as.data.frame(viz_loss3), id.vars = "epoch")
viz_acc3 <- melt(as.data.frame(viz_acc3), id.vars = "epoch")
viz_loss3 %>% filter (epoch >= 4) %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc3 %>% filter (epoch >= 4) %>% ggplot() + geom_line(aes(epoch, value, col = variable))

#-------------------------------------------------------------------------------------------
#Adding another CNN layer is better or not?
viz_loss4 <- data.table::data.table()
viz_acc4 <- data.table::data.table()
viz_loss4$epoch <- 1:10
viz_acc4$epoch <- 1:10
filters <- c(224,256,282)

for (i in filters) {
  model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_conv_2d(filters = i, activation = "relu", kernel_size = c(3,3)) %>% layer_flatten() %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
  model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 32)
  history <- model[["history"]]
  viz_loss4[,paste0("F",i)] <- unlist(history$history$val_loss)
  viz_acc4[,paste0("F",i)] <- unlist(history$history$val_accuracy)
}

viz_loss4 <- melt(as.data.frame(viz_loss4), id.vars = "epoch")
viz_acc4 <- melt(as.data.frame(viz_acc4), id.vars = "epoch")
viz_loss4 %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc4 %>% ggplot() + geom_line(aes(epoch, value, col = variable))

#Adding another CNN layer does not make any improvement!
#It makes sense. You are just getting back all the filters (3,3,2xx). So, you see no improvement.
#-------------------------------------------------------------------------------------------
#Adding another dense layer
viz_loss5 <- data.table::data.table()
viz_acc5 <- data.table::data.table()
viz_loss5$epoch <- 1:10
viz_acc5$epoch <- 1:10
dunits = c(240,160,80)

for (i in dunits) {
  model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_flatten() %>% layer_dense(units = i, activation = "relu") %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
  model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 128)
  history <- model[["history"]]
  viz_loss5[,paste0("U",i)] <- unlist(history$history$val_loss)
  viz_acc5[,paste0("U",i)] <- unlist(history$history$val_accuracy)
}

viz_loss5 <- melt(as.data.frame(viz_loss5), id.vars = "epoch")
viz_acc5 <- melt(as.data.frame(viz_acc5), id.vars = "epoch")
viz_loss5 %>% filter(epoch >= 4) %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc5 %>% filter(epoch >= 4) %>% ggplot() + geom_line(aes(epoch, value, col = variable))

#-------------------------------------------------------------------------------------------
#Considering dropout
#I will use unique dropout rate because it will take over a day on CPU to compute all possible combinations.
viz_loss6 <- data.table::data.table()
viz_acc6 <- data.table::data.table()
viz_loss6$epoch <- 1:10
viz_acc6$epoch <- 1:10
drate = c(0.3,0.4,0.5,0.6,0.7)

for (i in drate) {
  model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_dropout(i) %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_dropout(i) %>% layer_flatten() %>% layer_dense(units = 80, activation = "relu") %>% layer_dropout(i) %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
  model %>% fit (train_x, train_y, epoch = 10, validation_split = 0.2, batch_size = 128)
  history <- model[["history"]]
  viz_loss6[,paste0("U",i)] <- unlist(history$history$val_loss)
  viz_acc6[,paste0("U",i)] <- unlist(history$history$val_accuracy)
}

viz_loss6 <- melt(as.data.frame(viz_loss6), id.vars = "epoch")
viz_acc6 <- melt(as.data.frame(viz_acc6), id.vars = "epoch")
viz_loss6 %>% filter(epoch >= 2) %>% ggplot() + geom_line(aes(epoch, value, col = variable))
viz_acc6 %>% filter(epoch >= 2) %>% ggplot() + geom_line(aes(epoch, value, col = variable))

#-------------------------------------------------------------------------------------------
#Finalizaion
#Augmentation
datagen <- image_data_generator(
  rotation_range = 10,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  zoom_range = 0.1,
  horizontal_flip = FALSE
)
data_gen <- flow_images_from_data(
  train_x,
  generator = datagen
)
#Building model
model <- keras_model_sequential() %>% layer_conv_2d(input_shape = c(28,28,1), filters = 80, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_dropout(0.4) %>% layer_conv_2d(filters = 160, activation = "relu", kernel_size = c(3,3)) %>% layer_max_pooling_2d(pool_size = c(2,2)) %>% layer_batch_normalization() %>% layer_dropout(0.4) %>% layer_flatten() %>% layer_dense(units = 80, activation = "relu") %>% layer_dropout(0.4) %>% layer_dense(units = 10, activation = "softmax")
model %>% compile (loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
model %>% fit (data_gen$x, train_y, epoch = 100, validation_split = 0.2, batch_size = 128)
history <- model[["history"]]
history <- history$history
model %>% evaluate(test_x, test_y)
acc_plot <- unlist(history$val_accuracy) %>% data.frame(acc = .)
acc_plot[51:100,] %>% ggplot(aes(acc)) + geom_density(fill = "skyblue") #Network get stable (no noticable improvement nor overfitting) after 50 epoch

save_model_hdf5(model, "model.h5")