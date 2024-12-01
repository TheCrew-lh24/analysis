import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def indices_of_value(sorted_array, value):
    start_index = np.searchsorted(sorted_array, value, side="left")
    end_index = np.searchsorted(sorted_array, value, side="right")
    return start_index, end_index


def match_with_base(true_ids, assigned_ids):
    sorted_pairs = sorted(zip(true_ids, assigned_ids), key=lambda x: x[0])
    sorted_true_ids = np.array([pair[0] for pair in sorted_pairs])
    assigned_ids_sorted_pair = np.array([pair[1] for pair in sorted_pairs])
    num_all_total_pairs = 0
    num_all_matched_pairs = 0
    for true_id in np.unique(sorted_true_ids):
        start_index, end_index = indices_of_value(sorted_true_ids, true_id)
        num_true_occurrences = end_index - start_index
        num_total_pairs = num_true_occurrences * (num_true_occurrences - 1) // 2
        num_matched_pairs = 0
        sliced_assigned_ids = assigned_ids_sorted_pair[start_index:end_index]
        for assigned_id in np.unique(sliced_assigned_ids):
            num_assigned_occurrences = (sliced_assigned_ids == assigned_id).sum()
            num_matched_pairs += (
                num_assigned_occurrences * (num_assigned_occurrences - 1) // 2
            )
        num_all_total_pairs += num_total_pairs
        num_all_matched_pairs += num_matched_pairs
    return (
        float(num_all_matched_pairs / num_all_total_pairs)
        if num_all_total_pairs > 0
        else 0
    )


def f1_score(true_ids, assigned_ids):
    rc = match_with_base(true_ids, assigned_ids)
    pr = match_with_base(assigned_ids, true_ids)
    return 2 * rc * pr / (rc + pr) if rc + pr > 0 else 0, rc, pr


def pipeline(df, pipeline):
    for step in pipeline:
        df = step(df)
    return df


def init_df(df):
    df = df.copy(deep=True)
    df["assigned_id"] = np.arange(len(df))
    return df


def merge_ids(parties_df, ids_to_merge, col_name="assigned_id"):
    id = np.array(ids_to_merge).min()
    parties_df.loc[parties_df[col_name].isin(ids_to_merge), col_name] = id
    return parties_df


def merge_two_plans(assigned_id_to_plan_id_map, plan_id_to_assigned_id_map, plan_ids):
    merged_plan_id = min(plan_ids)
    plan_assigned_ids = set()
    for plan_id in plan_ids:
        plan_assigned_ids = plan_assigned_ids.union(plan_id_to_assigned_id_map[plan_id])
        del plan_id_to_assigned_id_map[plan_id]
    plan_id_to_assigned_id_map[merged_plan_id] = plan_assigned_ids
    for assigned_id in plan_assigned_ids:
        assigned_id_to_plan_id_map[assigned_id] = merged_plan_id
    return assigned_id_to_plan_id_map, plan_id_to_assigned_id_map, merged_plan_id


def add_new_plan(
    assigned_id_to_plan_id_map,
    plan_id_to_assigned_id_map,
    plan_id,
    plan_members,
):
    for plan_member in plan_members:
        assigned_id_to_plan_id_map[plan_member] = plan_id
    plan_id_to_assigned_id_map[plan_id] = plan_id_to_assigned_id_map[plan_id].union(
        plan_members
    )
    return assigned_id_to_plan_id_map, plan_id_to_assigned_id_map


def create_super_merging_plan(merging_plan):
    assigned_id_to_plan_id_map = dict()
    plan_id_to_assigned_id_map = dict()
    plan_id_serial = 0

    for plan in tqdm(merging_plan, desc="crete final merging plans"):
        plan_ids = set()
        for assigned_id in plan:
            if assigned_id in assigned_id_to_plan_id_map:
                plan_ids.add(assigned_id_to_plan_id_map[assigned_id])

        if len(plan_ids) == 0:
            plan_id = plan_id_serial
            plan_id_to_assigned_id_map[plan_id] = set()
            plan_id_serial += 1
        elif len(plan_ids) == 1:
            plan_id = list(plan_ids)[0]
        else:
            assigned_id_to_plan_id_map, plan_id_to_assigned_id_map, plan_id = (
                merge_two_plans(
                    assigned_id_to_plan_id_map,
                    plan_id_to_assigned_id_map,
                    plan_ids,
                )
            )

        assigned_id_to_plan_id_map, plan_id_to_assigned_id_map = add_new_plan(
            assigned_id_to_plan_id_map,
            plan_id_to_assigned_id_map,
            plan_id,
            plan,
        )

    plans = plan_id_to_assigned_id_map.values()
    plans = [list(plan) for plan in plans if len(plan) > 1]
    return plans


def get_merging_plans_on_batches(df, col, start_index, end_index):
    start_i = start_index
    merging_plans = []
    with tqdm(total=end_index - start_index) as pbar:
        pbar.set_description(f"processing {col}")
        while start_i < end_index:
            end_i = start_i + 1
            while end_i < end_index and (df[col].iloc[start_i] == df[col].iloc[end_i]):
                end_i += 1
            if end_i - start_i > 1:
                assigned_ids = df["assigned_id"].iloc[start_i:end_i].tolist()
                merging_plans.append(assigned_ids)
            pbar.update(end_i - start_i)
            start_i = end_i
    return merging_plans


def create_batches(df, col, num_batches):
    average_batch_size = len(df) // num_batches
    start_i = 0
    batches = []
    while start_i < len(df):
        end_i = start_i + average_batch_size
        if end_i > len(df):
            end_i = len(df)
        while end_i < len(df) and (df[col].iloc[end_i - 1] == df[col].iloc[end_i]):
            end_i += 1
        batches.append((start_i, end_i))
        start_i = end_i
    return batches


STAGES = [
    "init",
    "eval",
    "match_iban",
    "match_phone",
    "match_name",
    "make_submission",
]

ACTIONS = [
    "split",
    "execute",
    "merge",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=Path, required=True)
    parser.add_argument("-s", "--stage", choices=STAGES, required=True)
    parser.add_argument("-a", "--action", choices=ACTIONS, required=False)
    parser.add_argument("-n", "--batch", type=int, required=False)
    parser.add_argument("-o", "--output_file", type=Path, required=False)
    args = parser.parse_args()

    input_path = args.input_file
    stage = args.stage
    action = args.action
    n = args.batch
    output_path = args.output_file

    if stage == "init":
        df = pd.read_csv(input_path, low_memory=False)
        if "external_id" in df.columns:
            df = df.dropna(subset=["external_id"])
            df["external_id"] = df["external_id"].astype(int)
        df = init_df(df)
        df.to_csv(output_path)
    elif stage == "eval":
        df = pd.read_csv(input_path, low_memory=False)
        print(f1_score(df["external_id"], df["assigned_id"]))
    elif stage in ["match_iban", "match_phone", "match_name"]:
        if stage == "match_iban":
            col_name = "party_iban"
        elif stage == "match_phone":
            col_name = "party_phone"
        elif stage == "match_name":
            col_name = "parsed_name"
        else:
            raise KeyError(f"Unknown stage: {stage}")

        if action == "split":
            df = pd.read_csv(input_path, low_memory=False)
            df = df.sort_values(col_name, inplace=False)
            batches = create_batches(df, col_name, n)
            for i, (start_i, end_i) in enumerate(batches):
                df.iloc[start_i:end_i].to_csv(output_path / f"batch_in_{i}.csv")
        elif action == "execute":
            df = pd.read_csv(input_path / f"batch_in_{n}.csv", low_memory=False)
            merging_plans = get_merging_plans_on_batches(df, col_name, 0, len(df))
            with open(output_path / f"batch_out_{n}.pkl", "wb") as f:
                pickle.dump(merging_plans, f)
        elif action == "merge":
            merging_plans = []
            for i in range(n):
                with open(output_path / f"batch_out_{i}.pkl", "rb") as f:
                    merging_plans.extend(pickle.load(f))
            merging_plans = create_super_merging_plan(merging_plans)
            df = pd.read_csv(input_path, low_memory=False)
            for plan in tqdm(merging_plans, desc="executing merging plans"):
                df = merge_ids(df, plan)
            df.to_csv(output_path / "merged.csv")
        else:
            raise KeyError(f"Unknown action: {action}")
    elif stage == "make_submission":
        df = pd.read_csv(input_path, low_memory=False)
        df = df.sort_values("assigned_id")
        df = df[["transaction_reference_id", "assigned_id"]]
        df.columns = ["transaction_reference_id", "external_id"]
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_path, index=False)
    else:
        raise KeyError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
