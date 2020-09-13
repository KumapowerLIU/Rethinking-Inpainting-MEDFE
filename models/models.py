from .MEDFE import MEDFE
def create_model(opt):
    model = MEDFE(opt)
    print("model [%s] was created" % (model.name()))
    return model

