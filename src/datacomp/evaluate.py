import argparse
import json
import os
import time
import yaml

from datacomp.eval_utils.fairness_eval import (
    evaluate_dollar_street_dataset,
    evaluate_fairface_dataset,
    evaluate_geode_dataset,
)
from datacomp.eval_utils.retr_eval import evaluate_retrieval_dataset
from datacomp.eval_utils.wds_eval import evaluate_webdataset
from datacomp.eval_utils.wilds_eval import evaluate_wilds_dataset
from datacomp.eval_utils.wino_eval import evaluate_winogavil_dataset

import pandas as pd

def evaluate_model(task_key, train_info, data_root, dataset_size, batch_size=64, num_workers=4, model_dict=None):
    if task_key.startswith("retrieval/"):
        metrics = evaluate_retrieval_dataset(
            task_key,
            train_info,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            model_dict=model_dict,
        )
    elif task_key.startswith("wilds/"):
        metrics = evaluate_wilds_dataset(
            task_key,
            train_info,
            data_root=data_root,
            dataset_len=dataset_size,
            batch_size=batch_size,
            num_workers=1,
            model_dict=model_dict,
        )
    elif task_key.startswith("fairness/"):
        eval_fn = {
            "fairness/dollar_street": evaluate_dollar_street_dataset,
            "fairness/geode": evaluate_geode_dataset,
            "fairness/fairface": evaluate_fairface_dataset,
            "fairness/utkface": evaluate_fairface_dataset,
        }.get(task_key)
        if eval_fn is not None:
            metrics = eval_fn(
                task_key,
                train_info,
                data_root=data_root,
                dataset_len=dataset_size,
                batch_size=batch_size,
                num_workers=num_workers,
                model_dict=model_dict,
            )
        else:
            metrics = {}
    elif task_key.startswith("misc/"):
        if task_key == "misc/winogavil":
            metrics = evaluate_winogavil_dataset(
                train_info,
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                model_dict=model_dict,
            )
        else:
            metrics = {}
    else:
        metrics = evaluate_webdataset(
            task_key,
            train_info,
            data_root=data_root,
            dataset_len=dataset_size,
            batch_size=batch_size,
            num_workers=1,
            model_dict=model_dict,  
        )
    return metrics


def evaluate(path_to_results, data_dir=None, train_info=None, model_dict=None, eval_ret_only=False, batch_size=64):
    assert train_info or model_dict, "Either train_info or model_dict must be provided."
    
    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f:
        tasks = yaml.safe_load(f)

    if eval_ret_only:
        tasks = {
            k: v for k, v in tasks.items() if k.startswith("retrieval/")
        }
    
    starttime = int(time.time())
    results = {}   
    # Read existing results
    if os.path.exists(path_to_results):
        with open(path_to_results, "r") as f:
            lines = [json.loads(s) for s in f.readlines()]
            for line in lines:
                if line["key"] not in tasks:
                    continue
                results[line["dataset"]] = line
        print(f"Found {len(results)} eval result(s) in {path_to_results}.") 
    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(
                f"Skipping {task_name} since results are already in {path_to_results}"
            )
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                train_info,
                data_dir,
                tasks[task_key].get("size"),
                batch_size=batch_size,
                model_dict=model_dict,
            )
            metrics["main_metric"] = metrics.get(
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(path_to_results, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
            # also print "image_retrieval_recall@1" and "text_retrieval_recall@1" if available
            if "image_retrieval_recall@1" in results[task_name]["metrics"]:
                print(f"Image retrieval recall@1: {results[task_name]['metrics']['image_retrieval_recall@1']:.4f} | Text retrieval recall@1: {results[task_name]['metrics']['text_retrieval_recall@1']:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")
    
    return results


def convert_to_csv(path_to_results, output_file):
    # Define dataset ordering and category labels
    dataset_order = [
        "ImageNet 1k", "ImageNet Sketch", "ImageNet v2", "ImageNet-A", "ImageNet-O", "ImageNet-R", "ObjectNet", "AVG",
        "Retrieval", "Flickr", "MSCOCO", "WinoGAViL", "AVG",
        "Classification", "Caltech-101", "CIFAR-10", "CIFAR-100", "CLEVR Counts", "CLEVR Distance", "Country211",
        "Describable Textures", "EuroSAT", "FGVC Aircraft", "Food-101", "GTSRB", "KITTI Vehicle Distance", "MNIST",
        "Oxford Flowers-102", "Oxford-IIIT Pet", "Pascal VOC 2007", "PatchCamelyon", "Rendered SST2", "RESISC45",
        "Stanford Cars", "STL-10", "SUN397", "SVHN", "iWildCam", "Camelyon17", "FMoW", "Dollar Street", "GeoDE", "AVG"
    ]

    # Load JSON file
    data = []
    with open(path_to_results, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Extract relevant fields into a DataFrame
    df = pd.DataFrame([
        {"Dataset": item["dataset"], "Main Metric": item["metrics"]["main_metric"]}
        for item in data
    ])

    # Compute AVG values for three groups
    def compute_avg(subset):
        return df[df["Dataset"].isin(subset)]["Main Metric"].mean()

    # Sort DataFrame based on the predefined order
    df["Sort Order"] = df["Dataset"].apply(lambda x: dataset_order.index(x) if x in dataset_order else float("inf"))
    df = df.sort_values(by="Sort Order").drop(columns=["Sort Order"])

    # Save to CSV (optional)
    df.to_csv(output_file, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_output_dir",
        required=True,
        help="Path to output directory from training.",
    )
    parser.add_argument(
        "--data_dir",
        help="(Optional) Path to directory containing downloaded evaluation datasets.",
        default=None,
    )
    parser.add_argument(
        "--eval_ret_only",
        action="store_true",
        help="Evaluate only retrieval tasks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--epoch", default=33, type=int, help="Epoch number to evaluate.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")

    args = parser.parse_args()


    train_info = {
        "scale_config": {"model": args.model}, 
        "checkpoint": f"{args.train_output_dir}/checkpoints/epoch_{args.epoch}.pt"
    }

    path_to_results = f"{args.train_output_dir}/epoch{args.epoch}_eval_results.jsonl"

    print("Evaluating")
    results = evaluate(
        path_to_results,
        data_dir=args.data_dir,
        train_info=train_info, 
        model_dict=None,  
        eval_ret_only=args.eval_ret_only,
        batch_size=args.batch_size
    )
    
    path_to_csv = f"{args.train_output_dir}/epoch{args.epoch}_eval_results.csv"
    print("Converting to CSV")
    convert_to_csv(path_to_results, path_to_csv)
    
    print("Done")