"""
Used for creating a graph of attention over a fixed number of logits over a
sequence. E.g., attention over an input sequence while generating an output
sequence.
"""
import matplotlib.pyplot as plt
import numpy as np


class AttentionGraph():
    """Creates a graph showing attention distributions for inputs and outputs.

    Attributes:
      keys (list of str): keys over which attention is done during generation.
      generated_values (list of str): keeps track of the generated values.
      attentions (list of list of float): keeps track of the probability
          distributions.
    """

    def __init__(self, keys):
        """
        Initializes the attention graph.

        Args:
          keys (list of string): a list of keys over which attention is done
            during generation.
        """
        if not keys:
            raise ValueError("Expected nonempty keys for attention graph.")

        self.keys = keys
        self.generated_values = []
        self.attentions = []

    def add_attention(self, gen_value, probabilities):
        """
        Adds attention scores for all item in `self.keys`.

        Args:
          gen_value (string): a generated value for this timestep.
          probabilities (np.array): probability distribution over the keys. Assumes
            the order of probabilities corresponds to the order of the keys.

        Raises:
          ValueError if `len(probabilities)` is not the same as `len(self.keys)`
          ValueError if `sum(probabilities)` is not 1
        """
        if len(probabilities) != len(self.keys):
            raise ValueError("Length of attention keys is " +
                             str(len(self.keys)) +
                             " but got probabilities of length " +
                             str(len(probabilities)))
#        if sum(probabilities) != 1.0:
#            raise ValueError("Probabilities sum to " +
#                             str(sum(probabilities)) + "; not 1.0")

        self.generated_values.append(gen_value)
        self.attentions.append(probabilities)

    def render(self, filename):
        """
        Renders the attention graph over timesteps.

        Args:
          filename (string): filename to save the figure to.
        """
        figure, axes = plt.subplots()
        graph = np.stack(self.attentions)
        axes.imshow(graph, cmap=plt.cm.gray, interpolation="nearest")
        axes.xaxis.tick_top()
        axes.set_xticks(range(len(self.keys)))
        axes.set_xticklabels(self.keys)
        plt.setp(axes.get_xticklabels(), rotation=90)
        axes.set_yticks(range(len(self.generated_values)))
        axes.set_yticklabels(self.generated_values)

        figure.savefig(filename)
