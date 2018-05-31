from torchtext import data


class SimpleQADataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field, path,
               train='train.txt', validation='valid.txt', test='test.txt'):
        return super(SimpleQADataset, cls).splits(
            path=path, train=train, validation=validation, test=test,
            format='TSV', fields=[('id', None), ('sub', None), ('entity', None), ('relation', None),
                                  ('obj', None), ('text', text_field), ('ed', label_field)]
        )


class SimpleQaRelationDataset(data.TabularDataset):  # train.txt  train_relation
    @classmethod
    def splits(cls, text_field, label_field, path,
               train='train.txt', validation='valid.txt', test='test.txt'):
        return super(SimpleQaRelationDataset, cls).splits(
            path, '', train, validation, test,
            format='TSV', fields=[('id', None), ('sub', None), ('entity', None), ('relation', label_field),
                                  ('obj', None), ('text', text_field), ('ed', None)]
        )
