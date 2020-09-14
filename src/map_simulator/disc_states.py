from enum import Enum

import operator


class DiscreteStates(Enum):

    #   KEY   | Val |   Text     |    Color        |  Label
    UNDEFINED =  1  , 'undefined', 'lightsteelblue', 'Undef.'
    UNIFORM   =  2  , 'uniform'  , 'violet'        , 'Unif. (0, 1)'
    BIMODAL   =  3  , 'bimodal'  , 'mediumpurple'  , 'BiMod. {0, 1}'

    def __new__(cls, v, text, color, label):
        obj = object.__new__(cls)
        obj._value_ = v
        obj.text = text
        obj.color = color
        obj.label = label
        return obj

    @classmethod
    def list_all(cls):
        ds_list = [s for s in cls]
        return ds_list

    @classmethod
    def sort_ds_list(cls, ds_list):
        ordered_ds_list = sorted(ds_list, key=lambda x : x.value)
        return ordered_ds_list

    @classmethod
    def get_colors(cls, ds_list):
        ds_list = cls.sort_ds_list(ds_list)
        return [state.color for state in ds_list]

    @classmethod
    def get_values(cls, ds_list):
        ds_list = cls.sort_ds_list(ds_list)
        return [state.value for state in ds_list]

    @classmethod
    def get_labels(cls, ds_list):
        ds_list = cls.sort_ds_list(ds_list)
        return [state.label for state in ds_list]


if __name__ == "__main__":

    obj_type = type(DiscreteStates.UNIFORM)
    print('Type Test:', obj_type)

    value = DiscreteStates.UNIFORM.value
    print('Value Test:', value)

    test_all_list = DiscreteStates.list_all()
    print('All Test:', test_all_list)

    test_ds_list = [DiscreteStates.UNIFORM, DiscreteStates.UNDEFINED]
    print('DS List Test:', test_ds_list)

    ordered_test_ds_list = DiscreteStates.sort_ds_list(test_ds_list)
    print('Ordered DSList Test:', ordered_test_ds_list)

    colors = DiscreteStates.get_colors(test_ds_list)
    print('Get Colors Test:', colors)

    labels = DiscreteStates.get_labels(test_ds_list)
    print('Get Labels Test:', labels)

    values = DiscreteStates.get_values(test_ds_list)
    print('Get Values Test:', values)
