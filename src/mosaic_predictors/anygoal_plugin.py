"""Register MoPWM with AnyGoal's model catalog."""

from __future__ import annotations


def model_spec():
    from anygoal.registry import ModelSpec

    return ModelSpec(
        id='mopwm',
        aliases=(),
        capabilities=frozenset({'plan'}),
        families=frozenset({'pusht', 'cube', 'tworoom', 'insert_peg'}),
        evaluation_id='mopwm',
        trainer='mosaic_predictors.anygoal_plugin',
        rollout_runner='mosaic_predictors.anygoal_plugin',
        planner='anygoal.runners.plan.planning_mopwm',
        training_config_dir=None,
        training_config_suffix='',
        trainer_config_style='hydra',
        training_profiles={},
        planning_config_dir='lewm',
        planning_config_suffix='',
        planner_config_style='hydra',
        planner_seed_style='override',
        planning_checkpoint_key='model.checkpoint',
        planning_training_config_key=None,
    )
