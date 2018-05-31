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


def evaluation(gold, pred, index2tag, type):
    right = 0
    predicted = 0
    total_en = 0
    # fout = open('log.valid', 'w')
    for i in range(len(gold)):
        gold_batch = gold[i]
        pred_batch = pred[i]
        for j in range(len(gold_batch)):
            gold_label = gold_batch[j]
            pred_label = pred_batch[j]
            gold_span = get_span(gold_label, index2tag, type)
            pred_span = get_span(pred_label, index2tag, type)
            # fout.write('{}\n{}'.format(gold_span, pred_span))
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
    # fout.flush()
    # fout.close()
    return precision, recall, f1
