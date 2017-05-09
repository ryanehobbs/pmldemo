import types
from enum import Enum

class Models(Enum):
    LINEAR='linear'
    LOGISTIC='logistic'

class DataTypes(Enum):
    MATLAB = 'matlab'
    CSV = 'csv'

    def __init__(self, dtype):

        self.dtype=dtype

    @property
    def DataType(self):
        return self.dtype

class Delegate:
    """Create a Delegate class. Delegates are pointers to functions they are a type that holds
    references to methods with a specific parameter signature and return type."""

    # In python there is no internal delegate class like in C# this class is constructed using a common
    # observable pattern.

    def __init__(self):
        """Represents a delegate, which is a data structure that
        refers to a static method or to a class instance and an
        instance method of that class."""

        # list containing pointers to methods, these are the observers
        self.__delegates = []

    def __iadd__(self, pointer_callback):
        """Augment arithmetic assignment ( += )
        http://docs.python.org/2/reference/datamodel.html#object.__iadd__
        Arguments:
        pointer_callback -- Add method or function to delegates list"""

        # With delegates we can add, subtract or process all methods on a Delegate

        # append reference to the __delegates list
        self.__delegates.append(pointer_callback)
        return self

    def __isub__(self, pointer_callback):
        """Augment arithmetic assignment ( -= )
        http://docs.python.org/2/reference/datamodel.html#object.__isub__
        Arguments:
        pointer_callback -- Remove method or function from delegates list"""

        # http://docs.python.org/2/reference/datamodel.html
        # When a user-defined method object is created by retrieving a class method
        # object from a class or instance, its im_self attribute is the class itself,
        # and its im_func attribute is the function object underlying the class method.

        # With delegates we can add, subtract or process all methods on a Delegate

        # Iterate through list of callbacks if pointer_callback is a class instance and found in list remove it
        self.__delegates = [ptr for ptr in self.__delegates if getattr(ptr, 'im_self', None) != pointer_callback]
        # check if method is callable
        if callable(pointer_callback):
            # iterate through list backwards if callback is in list of delegates remove first one found
            for i in range(len(self.__delegates) - 1, -1, -1):
                if self.__delegates[i] == pointer_callback:
                    del self.__delegates[i]
                    return self

        return self

    def __call__(self, *args, **kwargs):
        """Called when Delegate instance is "called" as a function Delegate(*args, **kwargs).
        Arguments:
        *args    -- List of functions/methods to add as delegates
        **kwargs -- Dictionary of functions/methods to add as delegates"""

        # With delegates we can add, subtract or process all methods on a Delegate

        # call all methods / functions in list of delegates and return results
        return [ptr(*args, **kwargs) for ptr in self.__delegates]


class Event(object):
    """Event class registers a method and or function in a Produce to a callback in a Consumer

    Example:
        import Event

        @Event
        def OnStateChange(self, mdl_object):
            pass

        # Add delegate for cost function, wire up events
        self.OnStateChange += lambda obj: self.statechange(obj)

        # call this method when OnStateChange fires
        def statechange(self, mdl_object):
            pass

        def __main__()
            self.OnStateChange(self)  #<-- Fire event
    """

    _shared_state = None
    _func_ptrs = {}
    def __init__(self, callback_func=None, *args, **kwargs):
        """Create a Event decorator, used to register a method or function as a callback
        Arguments:
        callback_func -- Method or function to call when the event is fired"""

        if not Event._shared_state:
            Event._shared_state = self.__dict__
        else:
            self.__dict__ = Event._shared_state

        # each callback function is a unique key. If specified use the identity
        # of the callback method/function else use identity of this Event object
        if callback_func:
            self._key = '_CallPtr_' + str(id(callback_func.__name__))
            if args or kwargs:
                self._func_ptrs[self._key] = {'args': args, 'kwargs': kwargs}
        else:
            self._key = '_CallPtr_' + str(id(self))

    def __get__(self, obj, cls):
        """Get the attribute of the owner class (cls) or the get the attribute of the instance class (obj)
        Arguments:
        obj -- Instance
        cls -- Owner"""

        try:
            # return the callback method or function to caller
            return obj.__dict__[self._key]
        except KeyError:
            # if not found create a new delegate and assign callback_func to list of delegates
            event = obj.__dict__[self._key] = Delegate()
            return event

    def __call__(self, *args, **kwargs):
        """ """

        # retrieve the call back function or method name
        callback_func = args[0]
        # convert func or method name to unique ID
        callback_func_id = '_CallPtr_' + str(id(callback_func.__name__))

        if hasattr(callback_func, self._key):  # if function and key matches call function
            getattr(callback_func, self._key)(callback_func, *args, **kwargs)
        elif callback_func_id in self._func_ptrs:  # method call direct
            return callback_func(*self._func_ptrs[callback_func_id]['args'], **self._func_ptrs[callback_func_id]['kwargs'])

    def call(self, callback_func, *args, **kwargs):
        """ """
        if hasattr(callback_func, self._key):
            getattr(callback_func, self._key)(callback_func, *args, **kwargs)

class DeferredExec(object):
    """Deferred exec will delay execution of a method and allow you pack *args and **kwargs into
    a delayed call"""

    _func = None
    _callback = None


    def __init__(self, *args, **kwargs):

        self._func = args[0]  # first arg is method of function
        # create delegate call back for deferred execution
        self._callback = Event(self._func, *args[1:], **kwargs)

    @staticmethod
    def call_func(func, *args, **kwargs):

        if isinstance(func, DeferredExec):
            return func()
        else:
            return func(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        # call method or function
        return self._callback(self._func, *args, **kwargs)


