import torch

class F1Score():

    def __init__(self, number_of_classes):
        self.number_of_classes = number_of_classes
        self.true_positive  = [0 for current_class in range(number_of_classes)]
        self.false_positive = [0 for current_class in range(number_of_classes)]
        self.false_negative = [0 for current_class in range(number_of_classes)]

    def truePositive(self, positive_index, predictions, targets):
        _, predictions = torch.topk(predictions, 1, 1)
        predictions = predictions.view(-1)
        true = predictions.eq(targets)
        positives = targets.eq(positive_index)
        true_and_positive = torch.mul(true, positives)
        return true_and_positive.float().sum(0).numpy()[0]

    def falsePositive(self, positive_index, predictions, targets):
        _, predictions = torch.topk(predictions, 1, 1)
        predictions = predictions.view(-1)
        false = predictions.ne(targets)
        positives = targets.eq(positive_index)
        false_and_positive = torch.mul(false, positives)
        return false_and_positive.float().sum(0).numpy()[0]

    def falseNegative(self, positive_index, predictions, targets):
        _, predictions = torch.topk(predictions, 1, 1)
        predictions = predictions.view(-1)
        false = predictions.ne(targets)
        negative = targets.ne(positive_index)
        false_and_negative = torch.mul(false, negative)
        return false_and_negative.float().sum(0).numpy()[0]

    def recall(self, positive_index):
        if (self.true_positive[positive_index] + self.false_negative[positive_index]) > 0:
            return self.true_positive[positive_index] / (self.true_positive[positive_index] + self.false_negative[positive_index])
        else:
            return 0.0

    def precision(self, positive_index):
        if (self.true_positive[positive_index] + self.false_positive[positive_index]) > 0:
            return self.true_positive[positive_index] / (self.true_positive[positive_index] + self.false_positive[positive_index])
        else:
            return 0.0

    def add(self, predictions, targets):
        for positive_index in range(0, self.number_of_classes):
            self.true_positive[positive_index]  = self.true_positive[positive_index]  + self.truePositive(positive_index, predictions, targets)
            self.false_positive[positive_index] = self.false_positive[positive_index] + self.falsePositive(positive_index, predictions, targets)
            self.false_negative[positive_index] = self.false_negative[positive_index] + self.falseNegative(positive_index, predictions, targets)

    def compute(self):
        f1_score_sum = 0
        for positive_index in range(0, self.number_of_classes):
            current_precision = self.precision(positive_index)
            current_recall = self.recall(positive_index)
            if(current_precision + current_recall) > 0.0:
                f1_score_sum = f1_score_sum + (2.0 * current_precision * current_recall / (current_precision + current_recall))
        return f1_score_sum / self.number_of_classes
