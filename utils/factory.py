def get_model(model_name, args):
    name = model_name.lower()
    if name == "sema":
        from models.sema import Learner
    else:
        raise NotImplementedError("Model {} is not supported in Jittor version. "
                                  "Only 'sema' is currently migrated.".format(model_name))
    return Learner(args)
