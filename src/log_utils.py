# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring


def gen_error_msg(exception):
    template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
    message = template.format(type(exception).__name__, exception.args)
    return message
