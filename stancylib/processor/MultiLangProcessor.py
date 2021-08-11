import os

from .DataProcessor import DataProcessor
from .InputExample import InputExample


class MultiLangProcessor(DataProcessor):
    """Processor for the Multi-Language data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), quotechar='"'), "test")

    def get_labels(self):
        """See base class."""
        return ["supports", "refutes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            question = line[1]
            evidence = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=question, text_b=evidence, label=label))
        return examples

