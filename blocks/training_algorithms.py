from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class TrainingAlgorithm(object):
    """The interface for training algorithms."""
    __metaclass__ = ABCMeta

    def __init__(self, model):
        """Initialize the given training algorithm.

        Parameters
        ----------
        model : object
            blocks.model

        """
        self.model = model

    @abstractmethod
    def get_train_updates(self):
        """Return the updates to the model constituting a single update."""
        # TODO Should this return inputs as well? Does an algorithm ever
        # need inputs beyond the model inputs?
        pass

    def get_monitoring_channels(self):
        """Return monitoring channels.

        Examples would be the values for momentum, the learning rate.

        Notes
        -----
        There is no need to return the cost here, this is added to the
        monitor channels by the :class:`block.train.Train` object.

        """
        return OrderedDict()
