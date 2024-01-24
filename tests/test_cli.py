"""Tests for unified CLI entrypoint."""

import pytest
from kinesis.cli import build_parser, handle_check_docs, handle_data


def test_cli_parser_commands():
    parser = build_parser()
    
    # Check collect
    args_collect = parser.parse_args(["collect", "--task", "cartpole_swingup", "--episodes", "2", "--seed", "123"])
    assert args_collect.command == "collect"
    assert args_collect.task == "cartpole_swingup"
    assert args_collect.episodes == 2
    assert args_collect.seed == 123

    # Check train
    args_train = parser.parse_args(["train", "--task", "walker_walk", "--epochs", "5"])
    assert args_train.command == "train"
    assert args_train.task == "walker_walk"
    assert args_train.epochs == 5

    # Check evaluate
    args_eval = parser.parse_args(["evaluate", "--shift-study"])
    assert args_eval.command == "evaluate"
    assert args_eval.shift_study is True

    # Check bench
    args_bench = parser.parse_args(["bench", "--short-reps", "2", "--long-reps", "1"])
    assert args_bench.command == "bench"
    assert args_bench.short_reps == 2

    # Check check-docs
    args_check = parser.parse_args(["check-docs"])
    assert args_check.command == "check-docs"


def test_cli_data_validation():
    parser = build_parser()
    args = parser.parse_args(["data", "--validate"])
    ret = handle_data(args)
    assert ret == 0


def test_cli_check_docs_clean():
    parser = build_parser()
    args = parser.parse_args(["check-docs"])
    ret = handle_check_docs(args)
    assert ret == 0
