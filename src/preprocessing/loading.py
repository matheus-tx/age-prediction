import os
import re


def make_labels(set_: str, path: str, what: str) -> list[int]:
    filenames: str = list(os.walk(f'{path}/{set_}'))[0][2]
    filenames.sort()

    match what:
        case 'age':
            label_index: int = 1
        case 'gender':
            label_index = 2
        case 'race':
            label_index = 3

    labels: list[int] = [float(re.findall(r'\d+', filename)[label_index])
                         for filename in filenames]

    return labels
