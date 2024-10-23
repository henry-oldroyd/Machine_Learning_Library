from ml_exceptions import ObjectNotCompiledEror, MethodSequencingError

def requires_compilation(input_funcion):
    def wrapper(self, *args, **kwargs):
        if not self._is_compiled: 
            raise ObjectNotCompiledEror(f"Cannot execute this method without object being compiled")
        return input_funcion(self, *args, **kwargs)

    # name is not just functional as it is important for abstract base classes
    wrapper.__name__ = input_funcion.__name__
    wrapper.__annotations__ = input_funcion.__annotations__

    return wrapper


# def setter_necessitates_parameter_reset_and_recompilation(input_funcion):
#     def wrapper(self, *args, **kwargs):
#         input_funcion(self, *args, **kwargs)
#         self._keep_parameters_on_compilation = False
#         self._is_compiled = False
#         return self

#     # name is not just functional as it is important for abstract base classes
#     wrapper.__name__ = input_funcion.__name__
#     wrapper.__annotations__ = input_funcion.__annotations__

#     return wrapper

# def setter_necessitates_recompilation(input_funcion):
#     def wrapper(self, *args, **kwargs):
#         input_funcion(self, *args, **kwargs)
#         self._is_compiled = False
#         return self

#     # name is not just functional as it is important for abstract base classes
#     wrapper.__name__ = input_funcion.__name__
#     wrapper.__annotations__ = input_funcion.__annotations__

#     return wrapper

def if_full_validation(input_funcion):
    def wrapper(self, *args, **kwargs):
        if self._full_validation: 
            return input_funcion(self, *args, **kwargs)

    # name is not just functional as it is important for abstract base classes
    wrapper.__name__ = input_funcion.__name__
    wrapper.__annotations__ = input_funcion.__annotations__

    return wrapper

def requires_activation_computation(input_funcion):
    def wrapper(self, *args, **kwargs):
        if not self._activation_is_computed: 
            MethodSequencingError("Cannot execute this method before the activations have been computed")
        return input_funcion(self, *args, **kwargs)

    # name is not just functional as it is important for abstract base classes
    wrapper.__name__ = input_funcion.__name__
    wrapper.__annotations__ = input_funcion.__annotations__

    return wrapper


def requires_loss_computation(input_funcion):
    def wrapper(self, *args, **kwargs):
        if not self._loss_computed: 
            MethodSequencingError("Cannot execute this method before the loss has been computed")
        return input_funcion(self, *args, **kwargs)

    # name is not just functional as it is important for abstract base classes
    wrapper.__name__ = input_funcion.__name__
    wrapper.__annotations__ = input_funcion.__annotations__

    return wrapper
