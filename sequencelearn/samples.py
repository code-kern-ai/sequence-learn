import json
import os
from typing import Optional, Tuple, List


def get_entities_data(
    num_samples: Optional[int] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Loads sample data for tutorials. With this corpus, you can predict entities such as people within texts.

    Args:
        num_samples (Optional[int], optional): Specify the number of samples you want to load (there are 1,000 records available). Defaults to None.

    Returns:
        Tuple[List[str], List[List[str]]]: Sentences and plain labels
    """
    path_to_file = os.path.join("sequencelearn", "data", "sample_entities.json")
    with open(path_to_file, "r") as file:
        content = json.load(file)
    if num_samples is not None:
        return content["sentences"][:num_samples], content["labels"][:num_samples]
    else:
        return content["sentences"], content["labels"]
