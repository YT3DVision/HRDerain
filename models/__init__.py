import logging


# 读取配置选项 opt 中的 "model" 字段来决定创建哪种类型的模型，并记录模型创建的日志
def create_model(opt, logger):
    model = opt["model_type"]

    if model == "HRDerainModel":
        from models.BaseModel import BaseModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    model = M(opt)
    logger.info("Model [{:s}] is created.".format(M.__class__.__name__))
    print("Model [{:s}] is created.".format(M.__class__.__name__))
    return model
