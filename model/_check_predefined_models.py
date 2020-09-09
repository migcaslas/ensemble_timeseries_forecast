class _CheckPredefinedModels(object):

    @classmethod
    def check_defined_model(cls, model):
        if model is None:
            return False
        else:
            return cls.check_valid_model(model)

    @staticmethod
    def check_valid_model(model):
        # Check if cluster, classifier and regression model are ok
        if "fit" not in dir(model):
            return False
        if "predict" not in dir(model):
            return False
        return True
