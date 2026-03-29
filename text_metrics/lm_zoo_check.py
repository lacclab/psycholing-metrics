import lm_zoo as zoo
import pandas as pd


def _get_lmzoo_surprisal(text: str, model: str) -> pd.DataFrame:
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    """

    df = zoo.get_surprisals(zoo.get_registry()[model], [text])
    df.columns = ["Word", "Surprisal"]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = _get_lmzoo_surprisal(text="hello, how are you?", model="gpt2")
    print(df)
