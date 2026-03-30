import os
from eval_scripts.common.ansi_escapes import ANSIEscapes, ansiesc_print
from typing import TypeVar


T = TypeVar("T")


def select_task_subset_if_slurm(
    items: list[T],
) -> list[T]:
    """
    Returns a subset of the given list of items based on SLURM array task environment variables.
    If SLURM_ARRAY_TASK_ID is not set, returns the original items.
    Args:
        items: any list of items to split among SLURM array tasks.
    Returns:
        A subset of the input items assigned to the current SLURM array task.
    """
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    task_id_min_str = os.environ.get("SLURM_ARRAY_TASK_MIN", None)
    array_size_str = os.environ.get("SLURM_ARRAY_TASK_COUNT", None)

    if task_id_str is None:
        return items

    if array_size_str is None or task_id_min_str is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is set, but SLURM_ARRAY_TASK_COUNT or SLURM_ARRAY_TASK_MIN is not!"
        )

    task_id, task_id_min, array_size = (
        int(task_id_str),
        int(task_id_min_str),
        int(array_size_str),
    )

    job_ix = task_id - task_id_min
    ansiesc_print(
        f"Running in SLURM: {job_ix=} {array_size=} {task_id=} {task_id_min=}",
        ANSIEscapes.YELLOW,
    )

    base_tasks_per_job = len(items) // array_size
    remaining_tasks = len(items) - base_tasks_per_job * array_size

    num_tasks_per_job = [base_tasks_per_job for _ in range(array_size)]
    for i in range(remaining_tasks):
        num_tasks_per_job[i] += 1

    tasks_before_this_job = sum(num_tasks_per_job[:job_ix])
    this_job_tasks = (
        base_tasks_per_job + 1 if job_ix < remaining_tasks else base_tasks_per_job
    )

    ansiesc_print(
        f"This job will run {this_job_tasks} combinations - [{tasks_before_this_job}, {tasks_before_this_job + this_job_tasks - 1}].",
        ANSIEscapes.YELLOW,
    )
    return items[tasks_before_this_job : tasks_before_this_job + this_job_tasks]
