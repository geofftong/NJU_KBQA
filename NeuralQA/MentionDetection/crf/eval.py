import sys


def get_span(label):
    start, end = 0, 0
    flag = False
    span = []
    for k, l in enumerate(label):
        if l == 'I' and not flag:
            start = k
            flag = True
        if l != 'I' and flag:
            flag = False
            en = k
            span.append((start, en))
            start, end = 0, 0
    if start != 0 and end == 0:
        end = len(label) + 1  # bug fixed: geoff
        span.append((start, end))
    return span


def evaluation(filename):
    fin = open(filename, 'r')
    pred = []
    gold = []
    right = 0
    predicted = 0
    total_en = 0
    for line in fin.readlines():
        if line == '\n':
            gold_span = get_span(gold)
            pred_span = get_span(pred)
            total_en += len(gold_span)
            predicted += len(pred_span)
            for item in pred_span:
                if item in gold_span:
                    right += 1
            gold = []
            pred = []
        else:
            word, gold_label, pred_label = line.strip().split()
            gold.append(gold_label)
            pred.append(pred_label)

    if gold != [] or pred != []:
        gold_span = get_span(gold)
        pred_span = get_span(pred)
        total_en += len(gold_span)
        predicted += len(pred_span)
        for item in pred_span:
            if item in gold_span:
                right += 1

    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print("Precision", precision, "Recall", recall, "F1", f1, "right", right, "predicted", predicted, "total", total_en)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Need to specify the file")
    filename = sys.argv[1]
    evaluation(filename)
