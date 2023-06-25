import os
import pandas as pd
from datetime import datetime
from src.parse import parse_opt
from gogoowl import GoModel

opt = parse_opt()


def save_results(results: pd.DataFrame):
    source_dir = (
        opt.source if os.path.isdir(opt.source) else os.path.dirname(opt.source)
    )
    dirname = os.path.basename(source_dir)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(source_dir, "%s_%s.csv" % (dirname, time))

    results.to_csv(path, index=False)


def main():
    model = GoModel(opt.model)
    result = model.detect(opt.source, bs=opt.batch)
    result = result.pandas(conf=opt.threshold, min_duration=opt.min_duration)
    save_results(result)


if __name__ == "__main__":
    main()
