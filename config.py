class DefaultConfigs(object):
    #1.string parameters
    train_data = "/root/pycharm/ResNet_RockRecognition_keras/data/underglass_rock_recognization/train/"
    #test_data = "/home/yxq/桌面/石头/val/7/"
    val_data = "/root/pycharm/ResNet_RockRecognition_keras/data/underglass_rock_recognization/val/"
    model_name = "densenet121"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0,1"
    augmen_level = "medium"  # "light","hard","hard2"

    #2.numeric parameters
    epochs = 40
    batch_size = 10
    img_height = 320
    img_weight = 320
    num_classes = 30
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4
    plot_every = 10

config = DefaultConfigs()
