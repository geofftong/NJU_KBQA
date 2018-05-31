from torchtext import data


class SimpleQuestionsDataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field, path,  # train.txt
               train='train_relation', validation='valid_relation', test='test_relation'):
        return super(SimpleQuestionsDataset, cls).splits(
            path, '', train, validation, test,
            format='TSV', fields=[('id', None), ('sub', None), ('entity', None), ('relation', label_field),
                                  ('obj', None), ('text', text_field), ('ed', None)]
        )
